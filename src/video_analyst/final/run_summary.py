"""Run summary (Epic 5)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.models import utc_now
from video_analyst.fine.models import TimeRange


def _safe_int(data: dict[str, Any], *keys: str, default: int = 0) -> int:
    cur: Any = data
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    try:
        return int(cur)
    except (TypeError, ValueError):
        return default


def build_run_summary(
    *,
    layout: OutputLayout,
    provider_mode: str,
    real_api_used: bool,
    terminal_refinement_required: bool,
    terminal_refinement_executed: bool,
    terminal_refinement_status: str,
    status: str,
    coarse_batches_count: int,
    fine_intervals_count: int,
    terminal_intervals_count: int,
    failed_ranges: list[TimeRange],
    deferred_ranges: list[TimeRange],
    unresolved_count: int,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """构建 run_summary 字典（直接 dict，方便 JSON 序列化）。"""
    cache = CacheStore(layout)
    subtitle_reviewed = False
    subtitle_source = "asr"
    if layout.review_json.is_file():
        try:
            data = cache.read_json(layout.review_json)
            subtitle_reviewed = bool(data.get("reviewed"))
            active = data.get("active")
            if active in ("asr", "llm"):
                subtitle_source = active
        except Exception:
            pass

    return {
        "schema_version": "1.0",
        "report_status": status,
        "provider_mode": provider_mode if provider_mode != "mock" else "in_session_agent",
        "vision_execution_mode": "in_session_agent",
        "real_api_used": False,
        "real_vision_api_used": False,
        "real_vision_analysis_used": True,
        "external_api_used": False,
        "api_key_used": False,
        "token_status": "unavailable",
        "estimated_cost": None,
        "pricing_source": "in_session_agent",
        "terminal_refinement": {
            "required": terminal_refinement_required,
            "executed": terminal_refinement_executed,
            "status": terminal_refinement_status,
        },
        "tests_are_not_runtime_validation": True,
        "subtitle_source": subtitle_source,
        "subtitle_reviewed": subtitle_reviewed,
        "coarse_batches": coarse_batches_count,
        "fine_intervals": fine_intervals_count,
        "terminal_intervals": terminal_intervals_count,
        "failed_ranges": [
            {"start": r.start, "end": r.end} for r in failed_ranges
        ],
        "deferred_ranges": [
            {"start": r.start, "end": r.end} for r in deferred_ranges
        ],
        "unresolved_count": unresolved_count,
        "warnings": warnings or [],
        "generated_at": utc_now().isoformat(),
    }


def save_run_summary(layout: OutputLayout, summary: dict[str, Any]) -> Path:
    cache = CacheStore(layout)
    cache.write_json(layout.final_run_summary_json, summary)
    return layout.final_run_summary_json


__all__ = ["build_run_summary", "save_run_summary"]