"""Shared fixtures for Epic 5 (final report) tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout

from tests._epic4_fixtures import (
    has_pillow,
    make_fine_visual_batch,
    seed_fine_batches,
)


def add_coarse_to_epic4_layout(layout: OutputLayout, *, video_duration: float = 60.0) -> None:
    """在 Epic 4 fixtures 之上补充 coarse 结构（manifest + structure + refinement_plan）。"""
    cache = CacheStore(layout)
    cache.write_json(
        layout.coarse_manifest_json,
        {
            "schema_version": "1.0",
            "video_id": "v",
            "provider": "mock",
            "model": "mock-vision-1",
            "prompt_version": "coarse-vision-v1",
            "user_request": "",
            "analysis_dimensions": [],
            "created_at": "2026-07-22T00:00:00Z",
            "updated_at": "2026-07-22T00:00:00Z",
            "batches": [],
            "success_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "total_batches": 0,
            "allow_partial": True,
            "config_fingerprint": "",
        },
    )
    cache.write_json(
        layout.coarse_structure_json,
        {
            "schema_version": "1.0",
            "video_start": 0.0,
            "video_end": video_duration,
            "overall_summary": f"test structure for {video_duration}s video",
            "sections": [
                {
                    "section_id": "section_0001",
                    "start": 0.0,
                    "end": video_duration / 2,
                    "title": "First Half",
                    "summary": "first half summary",
                    "source_batch_ids": ["batch_0001"],
                    "important_moments": [],
                    "confidence": 0.6,
                },
                {
                    "section_id": "section_0002",
                    "start": video_duration / 2,
                    "end": video_duration,
                    "title": "Second Half",
                    "summary": "second half summary",
                    "source_batch_ids": ["batch_0002"],
                    "important_moments": [],
                    "confidence": 0.6,
                },
            ],
            "recurring_subjects": [],
            "recurring_visual_patterns": [],
            "subtitle_visual_notes": [],
            "unresolved_questions": ["coarse_q"],
            "failed_batches": [],
            "gaps": [],
        },
    )


def add_fine_video_analysis(
    layout: OutputLayout,
    *,
    analyzed_intervals: list[dict[str, float]] | None = None,
    analysis_gaps: list[dict[str, float]] | None = None,
    terminal_executed: bool = False,
) -> None:
    """添加 fine_video_analysis.json 到 layout。"""
    cache = CacheStore(layout)
    cache.write_json(
        layout.fine_video_analysis_json,
        {
            "schema_version": "1.0",
            "video_start": 0.0,
            "video_end": 60.0,
            "analyzed_intervals": analyzed_intervals
            or [{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 60.0}],
            "analysis_gaps": analysis_gaps or [],
            "temporal_events": [],
            "subject_motion_summary": [],
            "camera_motion_summary": [],
            "text_animation_summary": [],
            "graphic_animation_summary": [],
            "transition_summary": [],
            "answered_questions": [],
            "unresolved_questions": [],
            "important_events": [],
            "second_refinement_required": False,
            "second_refinement_plan_path": None,
            "confidence": 0.6,
            "warnings": [],
        },
    )


def add_fine_temporal_events(
    layout: OutputLayout,
    *,
    events: list[dict[str, Any]] | None = None,
) -> None:
    cache = CacheStore(layout)
    cache.write_json(
        layout.fine_temporal_events_json,
        {"events": events or []},
    )


def add_second_refinement_plan(
    layout: OutputLayout,
    *,
    intervals: list[dict[str, Any]] | None = None,
    deferred_intervals: list[dict[str, Any]] | None = None,
) -> None:
    cache = CacheStore(layout)
    cache.write_json(
        layout.fine_second_refinement_plan_json,
        {
            "schema_version": "1.0",
            "video_duration": 60.0,
            "round1_refinement_duration": 10.0,
            "total_requested_duration": 0.0,
            "budget_duration": 3.0,
            "max_additional_rounds": 1,
            "terminal_after_this_round": True,
            "intervals": intervals or [],
            "deferred_intervals": deferred_intervals or [],
            "generation_warnings": [],
        },
    )


def seed_epic5_inputs(tmp_path: Path) -> OutputLayout:
    """生成完整的 Epic 5 输入（Epic 1–4 输出）。"""
    layout = seed_fine_batches(tmp_path)
    add_coarse_to_epic4_layout(layout, video_duration=60.0)
    add_fine_video_analysis(layout)
    add_fine_temporal_events(layout)
    return layout


__all__ = [
    "add_coarse_to_epic4_layout",
    "add_fine_temporal_events",
    "add_fine_video_analysis",
    "add_second_refinement_plan",
    "has_pillow",
    "seed_epic5_inputs",
]