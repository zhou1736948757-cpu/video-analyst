"""Tests for final coverage (Epic 5)."""

from __future__ import annotations

import json

import pytest

from video_analyst.cache.store import CacheStore
from video_analyst.final.coverage import build_coverage_report, save_coverage_report

from tests._epic5_fixtures import (
    add_coarse_to_epic4_layout,
    add_fine_video_analysis,
    add_second_refinement_plan,
    seed_epic5_inputs,
)


def test_coverage_basic(tmp_path):
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(
        layout,
        analyzed_intervals=[{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 60.0}],
    )
    cov = build_coverage_report(layout, video_duration=60.0)
    assert cov.video_duration == 60.0
    assert cov.fine_coverage_ratio > 0.99
    assert cov.coarse_coverage_ratio > 0.99
    assert cov.providers_used == ["in_session_agent"]
    assert not cov.real_vision_api_used
    assert any("in_session_agent" in w for w in cov.warnings)


def test_coverage_with_gaps(tmp_path):
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(
        layout,
        analyzed_intervals=[{"start": 0.0, "end": 30.0}],
        analysis_gaps=[{"start": 30.0, "end": 60.0}],
    )
    cov = build_coverage_report(layout, video_duration=60.0)
    assert len(cov.failed_ranges) == 1
    assert cov.failed_ranges[0].start == 30.0
    assert cov.failed_ranges[0].end == 60.0
    assert cov.fine_coverage_ratio < 0.55  # 30/60


def test_coverage_with_deferred(tmp_path):
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(layout)
    add_second_refinement_plan(
        layout,
        intervals=[],
        deferred_intervals=[{"start": 5.0, "end": 7.0}],
    )
    cov = build_coverage_report(layout, video_duration=60.0)
    assert len(cov.deferred_ranges) == 1
    assert any("deferred" in w for w in cov.warnings)


def test_coverage_never_flags_external_api_as_product_default(tmp_path):
    """Even legacy manifest provider strings do not enable external vision API flags.

    The corresponding runtime mode was intentionally removed from the product.
    """
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(layout)
    CacheStore(layout).write_json(
        layout.coarse_manifest_json,
        {
            "schema_version": "1.0",
            "video_id": "v",
            "provider": "openai-compatible",
            "model": "gpt-4o-mini",
            "batches": [],
            "total_batches": 0,
            "success_count": 0,
            "failed_count": 0,
        },
    )
    cov = build_coverage_report(layout, video_duration=60.0)
    assert cov.real_vision_api_used is False
    assert "in_session_agent" in cov.providers_used


def test_coverage_subtitle_status(tmp_path):
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(layout)
    CacheStore(layout).write_json(
        layout.review_json,
        {
            "schema_version": "1.0",
            "version": "1.0",
            "source_video": "/tmp/v.mp4",
            "active": "llm",
            "reviewed": True,
            "needs_review": False,
            "asr": {}, "llm": None, "diffs": [],
            "asr_meta": {}, "llm_meta": {},
            "source_hashes": {},
            "summary": {"total_entries": 0, "unchanged": 0, "accepted": 0, "rejected": 0, "needs_review": 0},
            "created_at": "2026-07-22T00:00:00Z",
            "updated_at": "2026-07-22T00:00:00Z",
        },
    )
    cov = build_coverage_report(layout, video_duration=60.0)
    assert cov.subtitle_reviewed is True
    assert "human-reviewed" not in " ".join(cov.warnings)


def test_save_coverage_report(tmp_path):
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(layout)
    cov = build_coverage_report(layout, video_duration=60.0)
    save_coverage_report(layout, cov)
    assert layout.final_analysis_coverage_json.is_file()
    data = json.loads(layout.final_analysis_coverage_json.read_text(encoding="utf-8"))
    assert data["video_duration"] == 60.0


def test_unexamined_ranges_filled(tmp_path):
    """fine 覆盖 < 全片 → 未分析区间填充。"""
    layout = seed_epic5_inputs(tmp_path)
    add_fine_video_analysis(
        layout,
        analyzed_intervals=[{"start": 10.0, "end": 20.0}],
    )
    cov = build_coverage_report(layout, video_duration=60.0)
    assert len(cov.unexamined_ranges) >= 2  # [0, 10] 和 [20, 60]