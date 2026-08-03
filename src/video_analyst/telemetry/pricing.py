"""定价快照（Epic 6）。

不硬编码实时价格作为绝对真值。价格来源优先级：
1. 用户配置
2. Provider 响应直接返回
3. 内置带版本和日期的快照

没有可靠价格时 estimated_cost = None，不输出虚假费用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelPricing:
    """单个模型的定价（每百万 token）。"""

    input_per_million_tokens: float | None = None
    output_per_million_tokens: float | None = None
    currency: str = "USD"


@dataclass
class PricingSnapshot:
    """带版本和日期的价格快照。"""

    version: str = "builtin-v1"
    snapshot_date: str = "2026-07-22"
    source: str = "builtin"
    models: dict[str, ModelPricing] = field(default_factory=dict)

    def get(self, provider: str, model: str) -> ModelPricing | None:
        """按 provider:model 或 model 查找。"""
        key = f"{provider}:{model}"
        if key in self.models:
            return self.models[key]
        if model in self.models:
            return self.models[model]
        return None


def _builtin_snapshot() -> PricingSnapshot:
    """内置保守价格快照（仅用于估算，非真值）。

    这些是公开文档中的常见价格区间下限，仅用于 dry-run 估算；
    实际费用应以 Provider 账单为准。
    """
    snap = PricingSnapshot(
        version="builtin-v1",
        snapshot_date="2026-07-22",
        source="builtin",
    )
    # OpenAI 兼容常见模型（保守估算，可能过时）
    snap.models["openai-compatible:gpt-4o"] = ModelPricing(
        input_per_million_tokens=2.5,
        output_per_million_tokens=10.0,
    )
    snap.models["openai-compatible:gpt-4o-mini"] = ModelPricing(
        input_per_million_tokens=0.15,
        output_per_million_tokens=0.6,
    )
    snap.models["openai-compatible:gpt-4-vision-preview"] = ModelPricing(
        input_per_million_tokens=10.0,
        output_per_million_tokens=30.0,
    )
    snap.models["openai-compatible:gpt-4-turbo"] = ModelPricing(
        input_per_million_tokens=10.0,
        output_per_million_tokens=30.0,
    )
    return snap


def load_pricing_snapshot(
    user_config_path: Path | None = None,
) -> PricingSnapshot:
    """加载价格快照。

    优先级：用户配置 > 内置快照。
    用户配置文件格式（YAML）：

        version: user-v1
        snapshot_date: "2026-07-22"
        source: user
        models:
          openai-compatible:gpt-4o-mini:
            input_per_million_tokens: 0.15
            output_per_million_tokens: 0.6
            currency: USD
    """
    if user_config_path is not None and user_config_path.is_file():
        try:
            data = yaml.safe_load(user_config_path.read_text(encoding="utf-8"))
            return _parse_pricing_dict(data, source="user")
        except Exception:
            pass
    return _builtin_snapshot()


def _parse_pricing_dict(data: Any, *, source: str) -> PricingSnapshot:
    if not isinstance(data, dict):
        return _builtin_snapshot()
    snap = PricingSnapshot(
        version=str(data.get("version") or "user-v1"),
        snapshot_date=str(data.get("snapshot_date") or "2026-07-22"),
        source=source,
    )
    models = data.get("models") or {}
    if isinstance(models, dict):
        for key, val in models.items():
            if not isinstance(val, dict):
                continue
            snap.models[str(key)] = ModelPricing(
                input_per_million_tokens=val.get("input_per_million_tokens"),
                output_per_million_tokens=val.get("output_per_million_tokens"),
                currency=str(val.get("currency") or "USD"),
            )
    return snap


def estimate_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    snapshot: PricingSnapshot,
) -> tuple[float | None, str | None, str | None]:
    """估算单次调用费用。

    Returns: (estimated_cost, currency, pricing_source)
    无可靠价格时返回 (None, None, None)。
    """
    pricing = snapshot.get(provider, model)
    if pricing is None:
        return (None, None, None)
    if pricing.input_per_million_tokens is None or pricing.output_per_million_tokens is None:
        return (None, None, None)
    if input_tokens is None or output_tokens is None:
        return (None, pricing.currency, snapshot.source)
    cost = (
        input_tokens * pricing.input_per_million_tokens / 1_000_000
        + output_tokens * pricing.output_per_million_tokens / 1_000_000
    )
    return (round(cost, 6), pricing.currency, snapshot.source)


__all__ = [
    "ModelPricing",
    "PricingSnapshot",
    "estimate_cost",
    "load_pricing_snapshot",
]