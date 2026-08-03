"""预算门禁（preflight + runtime hard limit）。

## 设计目标

1. **Preflight**：在真实运行首次发送外部视觉请求前检查
   - ``--confirm-api-call`` 是否需要
   - ``estimated_cost`` 是否超 ``--max-estimated-cost``
   - ``max_api_calls`` 是否被预计最低调用数超出

2. **Runtime hard limit**：每次实际外部视觉请求前后守卫
   - 达到 ``max_api_calls`` 后阻止下次请求
   - retry 是否计为一次 API 调用明确定义（**每次 record_call 计一次外部调用**）
   - Mock / localhost / cache hit 不计入外部调用
   - 超过限制时：已完成 Batch 保留 + 输出 partial

异常：

* ``ApiCallConfirmationRequiredError``：缺 --confirm-api-call 且 unknown cost
* ``EstimatedCostLimitExceededError``：estimated_cost > max_estimated_cost
* ``ApiCallLimitExceededError``：调用次数上限
* ``RuntimeCostLimitExceededError``：运行时累计费用超限
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from video_analyst.exceptions import VideoAnalystError
from video_analyst.telemetry.models import ProviderCallRecord

log = logging.getLogger(__name__)


class ApiCallConfirmationRequiredError(VideoAnalystError):
    """API cost 未知但未授权 --confirm-api-call。"""

    def __init__(
        self, provider: str, model: str, message: str | None = None
    ) -> None:
        msg = message or (
            f"API cost cannot be estimated for {provider}/{model}. "
            "Re-run with --confirm-api-call to authorize external requests."
        )
        super().__init__(msg)
        self.provider = provider
        self.model = model


class EstimatedCostLimitExceededError(VideoAnalystError):
    """preflight：estimated_cost > max_estimated_cost。"""

    def __init__(
        self, estimated_cost: float, max_estimated_cost: float, currency: str | None
    ) -> None:
        msg = (
            f"estimated cost {estimated_cost:.4f} {currency or 'USD'} "
            f"> max_estimated_cost {max_estimated_cost:.4f} {currency or 'USD'}"
        )
        super().__init__(msg)
        self.estimated_cost = estimated_cost
        self.max_estimated_cost = max_estimated_cost
        self.currency = currency


class ApiCallLimitExceededError(VideoAnalystError):
    """运行时：实际外部调用次数已达 max_api_calls。"""

    def __init__(
        self, made: int, max_api_calls: int | None, message: str | None = None
    ) -> None:
        limit = "unbounded" if max_api_calls is None else f"{max_api_calls}"
        msg = message or (
            f"API call limit reached: made {made} calls (max_api_calls={limit}). "
            "Already-completed batches preserved; remaining work skipped as partial."
        )
        super().__init__(msg)
        self.made = made
        self.max_api_calls = max_api_calls


class RuntimeCostLimitExceededError(VideoAnalystError):
    """运行时：累计已知费用超 max_estimated_cost。"""

    def __init__(
        self, accumulated: float, max_estimated_cost: float, currency: str | None
    ) -> None:
        msg = (
            f"runtime cost {accumulated:.4f} {currency or 'USD'} "
            f"> max_estimated_cost {max_estimated_cost:.4f} {currency or 'USD'}"
        )
        super().__init__(msg)
        self.accumulated = accumulated
        self.max_estimated_cost = max_estimated_cost
        self.currency = currency


class VisionModelNotConfiguredError(VideoAnalystError):
    """Anthropic-compatible 必须显式配置模型名；缺省时显式失败。"""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"{provider} requires an explicit model name. "
            "Set --vision-model or VIDEO_ANALYST_ANTHROPIC_MODEL, "
            "or configure vision.model in config.yaml."
        )
        self.provider = provider


@dataclass
class BudgetState:
    """运行时累计外部调用计数与费用。

    Mock / local_mock / cache_hit 不计入 external_call_count。
    网络 scope 在 ProviderCallRecord 里查。
    """

    external_call_count: int = 0
    accumulated_cost: float = 0.0
    accumulated_currency: str | None = None


def _is_external_scope(record: ProviderCallRecord) -> bool:
    return (
        record.network_scope == "external_provider"
        and record.network_request_sent
    )


@dataclass
class PreflightInput:
    """preflight 决策所需输入。"""

    provider: str
    model: str
    cost_status: str  # unknown / known / mock
    estimated_cost: float | None
    currency: str | None
    minimum_pending_calls: int  # 最低预期调用数（不可写 0）
    max_api_calls: int | None
    max_estimated_cost: float | None
    confirm_api_call: bool


def preflight(input_: PreflightInput) -> None:
    """运行前门禁；任何违反则抛对应异常。

    规则：

    * provider=mock → 不要求 --confirm-api-call；不检查 cost/call
    * 外部 provider ＋ cost_status=unknown → 必须 --confirm-api-call
    * 外部 provider ＋ cost known + estimated_cost > max → 阻止
    * minimum_pending_calls > max_api_calls → 阻止
    """
    if input_.provider == "mock":
        # Mock provider 不发网络，无需预算门禁
        return

    # external provider 路径
    if input_.cost_status == "unknown":
        if not input_.confirm_api_call:
            raise ApiCallConfirmationRequiredError(input_.provider, input_.model)

    if (
        input_.cost_status == "known"
        and input_.estimated_cost is not None
        and input_.max_estimated_cost is not None
        and input_.estimated_cost > input_.max_estimated_cost
    ):
        raise EstimatedCostLimitExceededError(
            input_.estimated_cost,
            input_.max_estimated_cost,
            input_.currency,
        )

    if (
        input_.max_api_calls is not None
        and input_.minimum_pending_calls > input_api_calls_helper(input_.max_api_calls)
    ):
        raise ApiCallLimitExceededError(
            made=0,
            max_api_calls=input_.max_api_calls,
            message=(
                f"projected minimum calls {input_.minimum_pending_calls} "
                f"already exceeds --max-api-calls {input_.max_api_calls}"
            ),
        )


def input_api_calls_helper(max_api_calls: int) -> int:
    return max_api_calls


def before_external_call(
    state: BudgetState,
    *,
    max_api_calls: int | None,
) -> None:
    """每次实际外部 Provider 调用前调用。throw 阻止本次调用。

    ``state`` 由调用方维护；本函数只读取。
    """
    if max_api_calls is not None and state.external_call_count >= max_api_calls:
        raise ApiCallLimitExceededError(
            made=state.external_call_count, max_api_calls=max_api_calls
        )


def record_external_call(
    state: BudgetState, record: ProviderCallRecord
) -> None:
    """每次 record_call 完成后调用，更新 BudgetState。"""
    if not _is_external_scope(record):
        return
    state.external_call_count += 1
    if record.estimated_cost is not None:
        state.accumulated_cost += record.estimated_cost
        state.accumulated_currency = state.accumulated_currency or record.currency


def after_external_call(
    state: BudgetState,
    record: ProviderCallRecord,
    *,
    max_estimated_cost: float | None,
) -> None:
    """外部调用后：累计 + 费用检查。"""
    record_external_call(state, record)
    if (
        max_estimated_cost is not None
        and state.accumulated_cost > max_estimated_cost
    ):
        raise RuntimeCostLimitExceededError(
            state.accumulated_cost,
            max_estimated_cost,
            state.accumulated_currency,
        )


__all__ = [
    "ApiCallConfirmationRequiredError",
    "ApiCallLimitExceededError",
    "BudgetState",
    "EstimatedCostLimitExceededError",
    "PreflightInput",
    "RuntimeCostLimitExceededError",
    "VisionModelNotConfiguredError",
    "after_external_call",
    "before_external_call",
    "preflight",
    "record_external_call",
]  # type: ignore[attr-defined]  # ensure all exported