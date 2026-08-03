"""Tests for the Epic 2 cache integration (cache/store.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.cache.store import CacheStore, OutputLayout


def test_output_layout_has_epic2_paths(tmp_path: Path):
    layout = OutputLayout.create(tmp_path)
    # New Epic 2 path keys must exist
    assert hasattr(layout, "analysis_dir")
    assert hasattr(layout, "coarse_dir")
    assert hasattr(layout, "coarse_batches_dir")
    assert hasattr(layout, "coarse_raw_dir")
    assert hasattr(layout, "coarse_manifest_json")
    assert hasattr(layout, "coarse_structure_json")
    assert hasattr(layout, "coarse_refinement_plan_json")
    # and they must all be unique files under analysis_dir/coarse/
    assert layout.analysis_dir.is_dir()
    assert layout.coarse_dir.is_dir()
    assert layout.coarse_batches_dir.is_dir()
    assert layout.coarse_raw_dir.is_dir()


def test_clear_epic2_tree_removes_only_coarse(tmp_path: Path):
    layout = OutputLayout.create(tmp_path)
    # Write a marker file
    marker = layout.coarse_batches_dir / "batch_0001.json"
    marker.write_text("{}", encoding="utf-8")
    assert marker.is_file()
    store = CacheStore(layout)
    store.clear_epic2_tree()
    assert not marker.is_file()
    # parent dir should still be usable
    assert layout.analysis_dir.is_dir()
    # other layout dirs (e.g. subtitles) must remain
    assert layout.subtitles_dir.is_dir()


def test_pipeline_state_includes_coarse_paths(tmp_path: Path):
    from video_analyst.models import PipelineState

    state = PipelineState(video_path="/tmp/v.mp4", output_dir=str(tmp_path))
    assert hasattr(state, "coarse_manifest_path")
    assert hasattr(state, "coarse_structure_path")
    assert hasattr(state, "refinement_plan_path")


def test_pipeline_stage_includes_vision_stages():
    from video_analyst.models import PipelineStage

    # Make sure Epic 2 stages are present
    assert PipelineStage.VISION_REQUEST_BUILD.value == "vision_request_build"
    assert PipelineStage.VISION_BATCH_ANALYSIS.value == "vision_batch_analysis"
    assert PipelineStage.COARSE_AGGREGATION.value == "coarse_aggregation"
    assert PipelineStage.REFINEMENT_PLAN.value == "refinement_plan"