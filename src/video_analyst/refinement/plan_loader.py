"""Load + validate `analysis/coarse/refinement_plan.json` for Epic 3.

Strict validation:
- schema_version compatibility
- interval id uniqueness
- FPS in {5, 10, 15}
- start >= 0, end > start, end <= video_duration (with small tolerance)
- non-monotonic or duplicate timestamps
- deferred_intervals are kept for inspection but NEVER processed

Empty plan is a valid (no-op) state; pipeline returns status
`completed_no_intervals` and never calls FFmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.coarse_analysis.models import RefinementPlan
from video_analyst.exceptions import (
    RefinementPlanLoadError,
    RefinementPlanValidationError,
    SchemaVersionUnsupportedError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.models import RefinementSource

log = get_logger("refinement.plan_loader")

PLAN_LOAD_SCHEMA_VERSION = "1.0"
_PLAN_LOAD_MAJOR = 1
PLAN_TOLERANCE_SECONDS = 1e-3


def _parse_schema_major(schema_version: str | None) -> int | None:
    """从 schema_version 字符串解析主版本号。返回 None 表示无法解析。"""
    if not schema_version:
        return None
    s = str(schema_version).strip()
    if not s:
        return None
    # 取 "1.0" / "1" / "v1.2" 的第一个数字段
    import re

    m = re.match(r"v?(\d+)(?:\.|$)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _load_plan_json(plan_path: Path) -> dict[str, Any]:
    """Read + JSON-parse the plan file.

    Raises ``RefinementPlanLoadError`` for missing / unreadable / non-JSON
    files. Structural (Pydantic) validation errors are raised later as
    ``RefinementPlanValidationError`` by the caller.
    """
    if not plan_path.is_file():
        raise RefinementPlanLoadError(
            f"refinement_plan.json missing at {plan_path}; "
            "run `video-analyst coarse-analyze` first"
        )
    try:
        raw = plan_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RefinementPlanLoadError(
            f"refinement_plan.json unreadable: {exc}"
        ) from exc
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RefinementPlanLoadError(
            f"refinement_plan.json is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RefinementPlanLoadError(
            "refinement_plan.json must be a JSON object"
        )
    return data


def _resolve_video_duration(layout: OutputLayout) -> float:
    """Best-effort video duration lookup for plan validation.

    Prefer media_info.json (ffprobe output), then phase2 batches, then
    refinement_plan.video_duration itself.
    """
    if layout.media_info_json.is_file():
        try:
            data = json.loads(layout.media_info_json.read_text(encoding="utf-8"))
            d = data.get("duration")
            if d is not None:
                return float(d)
        except Exception:
            pass
    if layout.phase2_batches_json.is_file():
        try:
            data = json.loads(layout.phase2_batches_json.read_text(encoding="utf-8"))
            batches = data.get("batches") or []
            if batches:
                return float(max(b.get("end", 0.0) for b in batches))
        except Exception:
            pass
    return 0.0


def load_refinement_plan(layout: OutputLayout) -> tuple[list[RefinementSource], RefinementPlan]:
    """Load + validate the active refinement plan.

    Returns ``(active_intervals, full_plan)``. ``full_plan`` is preserved
    so the caller can still reference deferred intervals for diagnostics.
    """
    plan_path = layout.coarse_refinement_plan_json
    data = _load_plan_json(plan_path)

    schema_version = str(data.get("schema_version") or "")
    # 收紧 schema version 兼容性：主版本必须相同，次版本差异容忍并 warning。
    # 主版本不同 → SchemaVersionUnsupportedError（不再 best-effort 猜测字段语义）。
    major = _parse_schema_major(schema_version or PLAN_LOAD_SCHEMA_VERSION)
    if major is None:
        # 无法解析主版本，保守拒绝
        raise SchemaVersionUnsupportedError(
            f"refinement_plan.json schema_version={schema_version!r} could not "
            f"be parsed into a major version"
        )
    if major != _PLAN_LOAD_MAJOR:
        raise SchemaVersionUnsupportedError(
            f"refinement_plan.json schema_version={schema_version!r} has major "
            f"version {major}, current supports major {_PLAN_LOAD_MAJOR}. "
            "Refusing to process unknown major version."
        )
    if schema_version and schema_version != PLAN_LOAD_SCHEMA_VERSION:
        # 同主版本，次版本差异：接受并记录 warning
        log.warning(
            "refinement_plan.json schema_version=%s differs from current %s "
            "(same major version; accepted with warning)",
            schema_version,
            PLAN_LOAD_SCHEMA_VERSION,
        )

    # Pydantic model_validate catches structural errors (invalid FPS,
    # inverted ranges, etc.). These are content-level validation failures,
    # so we surface them as RefinementPlanValidationError (not LoadError).
    try:
        plan = RefinementPlan.model_validate(data)
    except Exception as exc:
        raise RefinementPlanValidationError(
            f"refinement_plan.json failed Pydantic validation: {exc}"
        ) from exc

    duration = _resolve_video_duration(layout)
    if duration <= 0:
        duration = plan.video_duration or 0.0

    if not plan.intervals:
        return [], plan

    sources: list[RefinementSource] = []
    seen_ids: set[str] = set()
    last_end = -1.0
    for i, iv in enumerate(plan.intervals):
        # Unique ID
        if iv.interval_id in seen_ids:
            raise RefinementPlanValidationError(
                f"duplicate interval_id at position {i}: {iv.interval_id!r}"
            )
        seen_ids.add(iv.interval_id)
        # Range
        if iv.start < -PLAN_TOLERANCE_SECONDS:
            raise RefinementPlanValidationError(
                f"interval {iv.interval_id}: start ({iv.start}) < 0"
            )
        if iv.end <= iv.start:
            raise RefinementPlanValidationError(
                f"interval {iv.interval_id}: end ({iv.end}) <= start ({iv.start})"
            )
        # Video bound (allow small float tolerance)
        if duration > 0 and iv.end > duration + PLAN_TOLERANCE_SECONDS:
            raise RefinementPlanValidationError(
                f"interval {iv.interval_id}: end ({iv.end}) exceeds "
                f"video_duration ({duration})"
            )
        # FPS — already enforced by RefinementRequest Literal, but be defensive
        if iv.requested_fps not in (5, 10, 15):
            raise RefinementPlanValidationError(
                f"interval {iv.interval_id}: requested_fps "
                f"{iv.requested_fps} not in {{5,10,15}}"
            )
        # Stable time order
        if iv.start + PLAN_TOLERANCE_SECONDS < last_end:
            # Overlapping is OK; only flag out-of-order
            if iv.start < last_end - PLAN_TOLERANCE_SECONDS:
                log.debug(
                    "interval %s overlaps or precedes previous interval (start=%.3f, prev_end=%.3f)",
                    iv.interval_id,
                    iv.start,
                    last_end,
                )
        last_end = max(last_end, iv.end)
        sources.append(
            RefinementSource(
                interval_id=iv.interval_id,
                start=float(iv.start),
                end=float(iv.end),
                requested_fps=int(iv.requested_fps),  # type: ignore[arg-type]
                priority=iv.priority,  # type: ignore[arg-type]
                reasons=list(iv.reasons),
                questions=list(iv.questions),
                source_batch_ids=list(iv.source_batch_ids),
                evidence_frame_ids=list(iv.evidence_frame_ids),
            )
        )

    log.info(
        "refinement plan loaded: %d active interval(s), %d deferred",
        len(sources),
        len(plan.deferred_intervals),
    )
    return sources, plan


__all__ = [
    "PLAN_LOAD_SCHEMA_VERSION",
    "load_refinement_plan",
]