"""Terminal fine analysis (Epic 5).

复用 Epic 4 fine batch_runner 的全套逻辑：
- FineVisionProvider + OpenAI-compatible 传输层
- Fine Vision Request builder
- response parser
- evidence validator
- event deduplication
- batch runner + cache

约束（Epic 5 终局规则）：
- 第二轮补帧 Plan 校验：max_additional_rounds == 1, terminal_after_this_round == true
- Provider 返回的 second_refinement_requests 必须被清空（终局不允许生成新请求）
- 输出至 analysis/terminal_fine/，不覆盖 analysis/fine/
- terminal_fine/terminal_fine_analysis.json（替代 fine_video_analysis.json）
- 终局 events 与 round1 events 合并由 final/merge 完成
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import FineAnalysisConfig
from video_analyst.exceptions import (
    FineVisionProviderUnavailableError,
    TerminalFineAnalysisError,
    TerminalRefinementPlanValidationError,
)
from video_analyst.fine.batch_runner import run_fine_analysis
from video_analyst.fine.models import FineVisionRequest
from video_analyst.fine.registry import create_fine_vision_provider
from video_analyst.logging.setup import get_logger

log = get_logger("fine.terminal")


@dataclass
class TerminalFineAnalysisResult:
    status: str  # completed / completed_no_batches / partial / failed
    total_batches: int = 0
    succeeded_batches: int = 0
    failed_batches: int = 0
    cached_batches: int = 0
    total_events: int = 0
    manifest_path: Path | None = None
    temporal_events_path: Path | None = None
    fine_analysis_path: Path | None = None
    failed_batch_ids: list[str] = None  # type: ignore[assignment]
    warnings: list[str] = None  # type: ignore[assignment]


def validate_terminal_refinement_plan(plan_path: Path) -> dict[str, Any]:
    """校验 second_refinement_plan.json 是否是合法的「终局」计划。

    Raises:
        TerminalRefinementPlanValidationError: 如果违反 max_additional_rounds/terminal_after_this_round
    """
    if not plan_path.is_file():
        raise TerminalRefinementPlanValidationError(
            f"second_refinement_plan.json missing at {plan_path}"
        )
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TerminalRefinementPlanValidationError(
            f"second_refinement_plan.json unreadable: {exc}"
        ) from exc
    max_rounds = int(data.get("max_additional_rounds") or 0)
    terminal = bool(data.get("terminal_after_this_round") or False)
    if max_rounds != 1:
        raise TerminalRefinementPlanValidationError(
            f"max_additional_rounds must be 1, got {max_rounds}"
        )
    if not terminal:
        raise TerminalRefinementPlanValidationError(
            "terminal_after_this_round must be true"
        )
    return data


def run_terminal_fine_analysis(
    *,
    layout: OutputLayout,
    config: FineAnalysisConfig,
    provider_name: str,
    vision_config: Any,
    user_request: str = "",
    refresh: bool = False,
    refresh_batches: list[str] | None = None,
    allow_partial: bool = False,
) -> TerminalFineAnalysisResult:
    """运行终局精细分析。

    复用 Epic 4 `run_fine_analysis`，但：
    - 输入从 `analysis/terminal_refinement/visual_inputs/fine_batches.json`
    - 输出至 `analysis/terminal_fine/`
    - 第二轮补帧请求被强制清空
    - 不允许生成第三轮（覆盖默认 prompts + 检查 second_refinement_requests 必空）
    """
    # 1. 读取终局 fine_batches.json
    fb_path = layout.terminal_refinement_fine_batches_json
    if not fb_path.is_file():
        log.info("terminal fine_batches.json missing; nothing to analyze")
        return TerminalFineAnalysisResult(
            status="completed_no_batches",
            failed_batch_ids=[],
            warnings=["terminal fine_batches missing; Epic 5 terminal round produced no intervals"],
        )

    # 2. 验证 second_refinement_plan
    plan = validate_terminal_refinement_plan(layout.fine_second_refinement_plan_json)
    intervals_count = len(plan.get("intervals") or [])
    deferred_count = len(plan.get("deferred_intervals") or [])
    log.info(
        "terminal refinement plan: %d intervals, %d deferred",
        intervals_count,
        deferred_count,
    )

    # 3. 构造 provider
    try:
        provider = create_fine_vision_provider(provider=provider_name, config=vision_config)
    except FineVisionProviderUnavailableError as exc:
        raise TerminalFineAnalysisError(
            f"terminal fine provider unavailable: {exc}"
        ) from exc

    # 4. 复用 run_fine_analysis（用覆盖 layout 路径的方式）
    #    通过 monkey patch OutputLayout 路径属性不可行；改为直接构造临时 layout
    #    更优雅：提取 batch_runner 核心逻辑 → 但 spec 要求复用，所以直接 copy paths
    fake_layout = _terminal_layout(layout)

    # 构造具备 fingerprint 方法的 AppConfig-like 对象
    from video_analyst.config import AppConfig

    full_cfg = AppConfig()
    full_cfg.vision = vision_config
    full_cfg.fine_analysis = config

    try:
        result = run_fine_analysis(
            layout=fake_layout,
            config=full_cfg,
            provider=provider,
            user_request=user_request,
            refresh=refresh,
            refresh_batches=refresh_batches,
            allow_partial=allow_partial,
            build_second_plan=False,  # 终局不再生成下一轮
        )
    except Exception as exc:
        raise TerminalFineAnalysisError(f"terminal fine analysis failed: {exc}") from exc

    # 5. 清除终局 batches 的 second_refinement_requests（终局不允许生成新请求）
    _strip_terminal_second_requests(fake_layout)

    # 6. 写 terminal_fine_analysis.json（覆盖 fine_video_analysis.json 角色）
    src = fake_layout.fine_video_analysis_json
    dst = layout.terminal_fine_analysis_json
    if src.is_file():
        cache = CacheStore(layout)
        cache.write_json(dst, cache.read_json(src))

    return TerminalFineAnalysisResult(
        status=result.status,
        total_batches=result.total_batches,
        succeeded_batches=result.succeeded_batches,
        failed_batches=result.failed_batches,
        cached_batches=result.cached_batches,
        total_events=result.total_events,
        manifest_path=fake_layout.fine_manifest_json,
        temporal_events_path=fake_layout.fine_temporal_events_json,
        fine_analysis_path=layout.terminal_fine_analysis_json,
        failed_batch_ids=list(result.failed_batch_ids),
        warnings=list(result.warnings),
    )


def _terminal_layout(real_layout: OutputLayout) -> OutputLayout:
    """构造一个虚拟 layout，把 fine/* 路径替换为 terminal_fine/*。

    不修改磁盘，只是让 batch_runner 把产出写到 terminal_fine/ 下。
    """
    import dataclasses

    # 使用 dataclasses.replace 不可行（Pydantic frozen=True）
    # 直接用 object.__setattr__ hack frozen 实例
    fake = object.__new__(OutputLayout)
    fake.__dict__.update(real_layout.__dict__)
    # 重新指向 terminal_fine 路径
    fake.__dict__["fine_dir"] = real_layout.terminal_fine_dir
    fake.__dict__["fine_batches_out_dir"] = real_layout.terminal_fine_batches_out_dir
    fake.__dict__["fine_raw_dir"] = real_layout.terminal_fine_raw_dir
    fake.__dict__["fine_manifest_json"] = real_layout.terminal_fine_manifest_json
    fake.__dict__["fine_temporal_events_json"] = real_layout.terminal_fine_temporal_events_json
    fake.__dict__["fine_video_analysis_json"] = real_layout.terminal_fine_analysis_json
    fake.__dict__["fine_second_refinement_plan_json"] = (
        real_layout.terminal_fine_dir / "second_refinement_plan.json"
    )
    fake.__dict__["refinement_fine_batches_json"] = (
        real_layout.terminal_refinement_fine_batches_json
    )
    # Ensure dirs exist
    for d in (
        fake.terminal_fine_dir,
        fake.terminal_fine_batches_out_dir,
        fake.terminal_fine_raw_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return fake


def _strip_terminal_second_requests(fake_layout: OutputLayout) -> None:
    """把 terminal_fine/ 下的每个 fine batch 的 second_refinement_requests 清空。

    终局分析返回的「新补帧请求」全部丢弃；记录为 terminal_unresolved_request。
    """
    cache = CacheStore(fake_layout)
    for fb_path in fake_layout.terminal_fine_batches_out_dir.glob("fine_*.json"):
        try:
            data = cache.read_json(fb_path)
        except Exception:
            continue
        second = data.get("second_refinement_requests") or []
        if not second:
            continue
        # 转为 unresolved_items 记录
        warnings = data.get("warnings") or []
        for req in second:
            warnings.append(
                f"terminal_unresolved_request: {req.get('request_id', '?')} dropped "
                "(Epic 5 disallows new requests in terminal round)"
            )
        data["second_refinement_requests"] = []
        data["warnings"] = warnings
        cache.write_json(fb_path, data)


__all__ = [
    "TerminalFineAnalysisResult",
    "run_terminal_fine_analysis",
    "validate_terminal_refinement_plan",
]