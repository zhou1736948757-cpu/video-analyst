"""Tests for refinement plan loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_analyst.cache.store import OutputLayout
from video_analyst.exceptions import (
    RefinementPlanLoadError,
    RefinementPlanValidationError,
    SchemaVersionUnsupportedError,
)
from video_analyst.refinement.plan_loader import (
    PLAN_LOAD_SCHEMA_VERSION,
    load_refinement_plan,
)

from tests._epic3_fixtures import make_refinement_plan_json, seed_refinement_layout


def test_loads_normal_plan(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    intervals, plan = load_refinement_plan(layout)
    assert len(intervals) == 2
    assert intervals[0].interval_id == "interval_0001"
    assert intervals[0].start == 5.0
    assert intervals[0].requested_fps == 10
    assert plan.video_duration == 60.0


def test_empty_intervals_returns_empty(tmp_path: Path):
    layout = seed_refinement_layout(
        tmp_path, refinement_intervals=[]
    )
    intervals, plan = load_refinement_plan(layout)
    assert intervals == []


def test_deferred_intervals_not_in_active(tmp_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        refinement_intervals=[
            {
                "interval_id": "interval_0001",
                "start": 5.0,
                "end": 8.0,
                "requested_fps": 10,
            }
        ],
    )
    # Append a deferred interval manually
    make_refinement_plan_json(
        layout.coarse_refinement_plan_json,
        duration=60.0,
        intervals=[{"interval_id": "interval_0001", "start": 5.0, "end": 8.0, "requested_fps": 10}],
        deferred_intervals=[
            {"interval_id": "interval_def_1", "start": 30.0, "end": 32.0, "requested_fps": 5, "priority": "low"}
        ],
    )
    intervals, plan = load_refinement_plan(layout)
    assert len(intervals) == 1
    assert intervals[0].interval_id == "interval_0001"
    # Deferred is preserved on the plan object
    assert len(plan.deferred_intervals) == 1
    assert plan.deferred_intervals[0].interval_id == "interval_def_1"


def test_invalid_fps_rejected(tmp_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 5.0, "end": 8.0, "requested_fps": 10}
        ],
    )
    # Overwrite the plan with raw JSON carrying an invalid FPS (7) so the
    # file is structurally valid JSON but semantically invalid. We bypass
    # RefinementInterval construction (which would reject FPS=7 itself) to
    # exercise the loader's own validation path.
    layout.coarse_refinement_plan_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "video_duration": 60.0,
                "total_requested_duration": 3.0,
                "budget_duration": 0.0,
                "intervals": [
                    {
                        "interval_id": "interval_0001",
                        "start": 5.0,
                        "end": 8.0,
                        "requested_fps": 7,
                        "priority": "medium",
                        "reasons": [],
                        "questions": [],
                        "source_batch_ids": [],
                        "evidence_frame_ids": [],
                        "estimated_frame_count": 0,
                    }
                ],
                "deferred_intervals": [],
                "generation_warnings": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RefinementPlanValidationError):
        load_refinement_plan(layout)


def test_inverted_time_range_rejected(tmp_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 8.0, "end": 5.0, "requested_fps": 10}
        ],
    )
    with pytest.raises(RefinementPlanValidationError):
        load_refinement_plan(layout)


def test_exceeds_video_duration_rejected(tmp_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        duration=10.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 5.0, "end": 25.0, "requested_fps": 10}
        ],
    )
    with pytest.raises(RefinementPlanValidationError):
        load_refinement_plan(layout)


def test_duplicate_interval_id_rejected(tmp_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 5.0, "end": 8.0, "requested_fps": 10},
            {"interval_id": "interval_0001", "start": 10.0, "end": 12.0, "requested_fps": 10},
        ],
    )
    with pytest.raises(RefinementPlanValidationError):
        load_refinement_plan(layout)


def test_missing_plan_file_raises(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    layout.coarse_refinement_plan_json.unlink()
    with pytest.raises(RefinementPlanLoadError):
        load_refinement_plan(layout)


def test_malformed_plan_raises(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    layout.coarse_refinement_plan_json.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(RefinementPlanLoadError):
        load_refinement_plan(layout)


def test_unsupported_schema_version_rejected(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    data = json.loads(layout.coarse_refinement_plan_json.read_text(encoding="utf-8"))
    data["schema_version"] = "2.0"
    layout.coarse_refinement_plan_json.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(SchemaVersionUnsupportedError):
        load_refinement_plan(layout)


def test_same_major_minor_version_accepted_with_warning(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    data = json.loads(layout.coarse_refinement_plan_json.read_text(encoding="utf-8"))
    data["schema_version"] = "1.1"
    layout.coarse_refinement_plan_json.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    intervals, _ = load_refinement_plan(layout)
    assert len(intervals) == 2  # same major → accepted


def test_missing_schema_version_accepted(tmp_path: Path):
    # When schema_version is missing, we fall back to the current major (1)
    layout = seed_refinement_layout(tmp_path)
    data = json.loads(layout.coarse_refinement_plan_json.read_text(encoding="utf-8"))
    data.pop("schema_version", None)
    layout.coarse_refinement_plan_json.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    intervals, _ = load_refinement_plan(layout)
    assert len(intervals) == 2


def test_load_constant_exists():
    assert PLAN_LOAD_SCHEMA_VERSION == "1.0"