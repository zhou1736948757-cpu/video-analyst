"""Dry-run + 预算/调用上限保护（Epic 6）。

Dry Run 不发送任何网络请求，只统计：
- 预计粗分析 Batch 数
- Fine Batch 数
- Terminal Batch 数
- 图片总数
- 预计 API 调用次数
- 已有缓存
- 尚需调用的 Batch
- 费用能否估算

预算限制：
- 预计费用 > max_estimated_cost → 阻止
- 无法估算费用时要求 --confirm-api-call
- 运行中调用次数达 max_api_calls → 停止新请求，保存已完成结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig
from video_analyst.logging.setup import get_logger

log = get_logger("pipeline.dry_run")


@dataclass
class DryRunEstimate:
    """Dry-run 估算结果。

    字段说明：

    * ``estimated_cost``：None = 未知（**绝不**写成 0）；float = 估算值
    * ``cost_status``：``known`` / ``unknown`` / ``mock``
    * ``pricing_source``：价格来源标识（snapshot / mock_provider_no_real_cost /
      no_price_snapshot）
    """

    coarse_batches_total: int = 0
    coarse_batches_cached: int = 0
    coarse_batches_pending: int = 0

    fine_batches_total: int = 0
    fine_batches_cached: int = 0
    fine_batches_pending: int = 0

    terminal_batches_total: int = 0
    terminal_batches_cached: int = 0
    terminal_batches_pending: int = 0

    total_images_pending: int = 0
    total_api_calls_pending: int = 0

    estimated_cost: float | None = None
    cost_estimable: bool = False
    cost_status: str = "unknown"  # known / unknown / mock
    pricing_source: str | None = None
    currency: str | None = None

    provider_mode: str = "mock"
    model_resolved: str | None = None
    known_model_limits: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    @property
    def would_invoke_real_api(self) -> bool:
        return self.provider_mode != "mock" and self.total_api_calls_pending > 0


def estimate_dry_run(
    layout: OutputLayout,
    config: AppConfig,
) -> DryRunEstimate:
    """估算一次 run 的 API 调用规模，不发送网络请求。"""
    cache = CacheStore(layout)
    est = DryRunEstimate()
    est.provider_mode = config.vision.provider

    # coarse batches
    if layout.phase2_batches_json.is_file():
        try:
            data = cache.read_json(layout.phase2_batches_json)
            est.coarse_batches_total = len(data.get("batches") or [])
        except Exception:
            pass
    # coarse cached
    if layout.coarse_manifest_json.is_file():
        try:
            data = cache.read_json(layout.coarse_manifest_json)
            est.coarse_batches_cached = sum(
                1 for b in (data.get("batches") or []) if b.get("status") == "ok"
            )
        except Exception:
            pass
    est.coarse_batches_pending = max(0, est.coarse_batches_total - est.coarse_batches_cached)

    # fine batches
    if layout.refinement_fine_batches_json.is_file():
        try:
            data = cache.read_json(layout.refinement_fine_batches_json)
            est.fine_batches_total = len(data.get("batches") or [])
        except Exception:
            pass
    if layout.fine_manifest_json.is_file():
        try:
            data = cache.read_json(layout.fine_manifest_json)
            est.fine_batches_cached = sum(
                1 for b in (data.get("batches") or []) if b.get("status") in ("ok", "cached")
            )
        except Exception:
            pass
    est.fine_batches_pending = max(0, est.fine_batches_total - est.fine_batches_cached)

    # terminal batches
    if layout.fine_second_refinement_plan_json.is_file():
        try:
            data = cache.read_json(layout.fine_second_refinement_plan_json)
            est.terminal_batches_total = len(data.get("intervals") or [])
        except Exception:
            pass
    if layout.terminal_fine_manifest_json.is_file():
        try:
            data = cache.read_json(layout.terminal_fine_manifest_json)
            est.terminal_batches_cached = sum(
                1 for b in (data.get("batches") or []) if b.get("status") in ("ok", "cached")
            )
        except Exception:
            pass
    est.terminal_batches_pending = max(0, est.terminal_batches_total - est.terminal_batches_cached)

    # 图片估算（粗: 每批 ~20 张上限；精细: 每批 contact sheets + 2 detail）
    est.total_images_pending = (
        est.coarse_batches_pending * min(20, config.vision.batch_max_frames)
        + est.fine_batches_pending * 6
        + est.terminal_batches_pending * 6
    )

    est.total_api_calls_pending = (
        est.coarse_batches_pending + est.fine_batches_pending + est.terminal_batches_pending
    )

    # 费用估算
    if est.provider_mode == "mock":
        est.estimated_cost = 0.0
        est.cost_estimable = True
        est.currency = "USD"
        est.cost_status = "known"
        est.pricing_source = "mock_provider_no_real_cost"
    else:
        from video_analyst.model_resolution import resolve_vision_model
        from video_analyst.telemetry.pricing import load_pricing_snapshot, estimate_cost

        snapshot = load_pricing_snapshot()
        # 严格不跨 provider 回退（之前会回退到 gpt-4o-mini）
        model = resolve_vision_model(est.provider_mode, config=config.vision)
        est.model_resolved = model
        # 保守估算每批 input ~2000 tokens, output ~1500 tokens
        cost, currency, src = estimate_cost(
            provider=est.provider_mode,
            model=model,
            input_tokens=2000 * est.total_api_calls_pending,
            output_tokens=1500 * est.total_api_calls_pending,
            snapshot=snapshot,
        )
        if cost is not None:
            est.estimated_cost = cost
            est.cost_estimable = True
            est.currency = currency
            est.cost_status = "known"
            est.pricing_source = src or "snapshot_unknown"
        else:
            # 费用未知：不伪造为 0；标记 unknown；记录价格来源缺失
            est.estimated_cost = None
            est.cost_estimable = False
            est.cost_status = "unknown"
            est.pricing_source = "no_price_snapshot"
            est.warnings.append(
                f"cannot estimate cost for {est.provider_mode}/{model}; "
                "use --confirm-api-call to proceed (unknown cost will not block dry run)"
            )

    if est.provider_mode != "mock":
        est.known_model_limits.append(
            f"max_frames_per_batch={config.vision.batch_max_frames}"
        )

    return est


def check_budget(
    est: DryRunEstimate,
    *,
    max_estimated_cost: float | None,
    max_api_calls: int | None,
    confirm_api_call: bool,
) -> tuple[bool, str]:
    """检查预算。返回 (allowed, reason)。"""
    if est.provider_mode == "mock":
        return True, "mock provider; no real API cost"
    if max_api_calls is not None and est.total_api_calls_pending > max_api_calls:
        return False, (
            f"pending API calls {est.total_api_calls_pending} > max_api_calls {max_api_calls}"
        )
    if max_estimated_cost is not None and est.cost_estimable:
        if est.estimated_cost is not None and est.estimated_cost > max_estimated_cost:
            return False, (
                f"estimated cost {est.estimated_cost:.4f} > max_estimated_cost {max_estimated_cost}"
            )
    if not est.cost_estimable and not confirm_api_call:
        return False, (
            "cost not estimable and --confirm-api-call not set; "
            "refusing to invoke real API with unknown cost"
        )
    return True, "budget ok"


def format_dry_run_report(est: DryRunEstimate) -> str:
    """格式化 dry-run 报告（CLI 输出）。"""
    lines = [
        "=== Dry Run Estimate ===",
        f"Provider mode:       {est.provider_mode}",
        f"Model resolved:      {est.model_resolved or '(unknown)'}",
        f"Coarse batches:      {est.coarse_batches_total} total, {est.coarse_batches_cached} cached, {est.coarse_batches_pending} pending",
        f"Fine batches:        {est.fine_batches_total} total, {est.fine_batches_cached} cached, {est.fine_batches_pending} pending",
        f"Terminal batches:    {est.terminal_batches_total} total, {est.terminal_batches_cached} cached, {est.terminal_batches_pending} pending",
        f"Total pending calls: {est.total_api_calls_pending}",
        f"Total pending images: {est.total_images_pending}",
    ]
    # 费用状态：明确区分 known / unknown / mock；绝不把 unknown 写成 0
    if est.cost_status == "mock":
        lines.append(
            f"Estimated cost:      0.0000 {est.currency or 'USD'} "
            f"(cost_status={est.cost_status}; pricing_source={est.pricing_source})"
        )
    elif est.cost_status == "known" and est.estimated_cost is not None:
        lines.append(
            f"Estimated cost:      {est.estimated_cost:.4f} {est.currency or 'USD'} "
            f"(cost_status=known; pricing_source={est.pricing_source})"
        )
    else:
        # unknown：保持 None / null，绝不伪造为 0
        lines.append(
            f"Estimated cost:      null (cost_status=unknown; "
            f"pricing_source={est.pricing_source or 'no_price_snapshot'})"
        )
    if est.known_model_limits:
        lines.append("Known model limits:  " + "; ".join(est.known_model_limits))
    if est.warnings:
        lines.append("Warnings:")
        for w in est.warnings:
            lines.append(f"  - {w}")
    if est.would_invoke_real_api:
        lines.append(">>> Would invoke REAL vision API. Use --confirm-api-call to proceed.")
    else:
        lines.append(">>> No real API calls needed (mock or fully cached).")
    return "\n".join(lines)


__all__ = [
    "DryRunEstimate",
    "check_budget",
    "estimate_dry_run",
    "format_dry_run_report",
]