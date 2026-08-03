"""Tests for Epic 3 cache integration (OutputLayout)."""

from __future__ import annotations

from pathlib import Path

from video_analyst.cache.store import CacheStore, OutputLayout


def test_output_layout_has_epic3_paths(tmp_path: Path):
    layout = OutputLayout.create(tmp_path)
    # All Epic 3 keys must exist
    assert hasattr(layout, "refinement_dir")
    assert hasattr(layout, "refinement_frames_dir")
    assert hasattr(layout, "refinement_contact_sheets_dir")
    assert hasattr(layout, "refinement_visual_inputs_dir")
    assert hasattr(layout, "refinement_batches_dir")
    assert hasattr(layout, "refinement_intervals_dir")
    assert hasattr(layout, "refinement_manifest_json")
    assert hasattr(layout, "refinement_fine_batches_json")
    # and they must all be unique directories
    for d in (
        layout.refinement_dir,
        layout.refinement_frames_dir,
        layout.refinement_contact_sheets_dir,
        layout.refinement_visual_inputs_dir,
        layout.refinement_batches_dir,
        layout.refinement_intervals_dir,
    ):
        assert d.is_dir()


def test_clear_epic3_tree_removes_only_refinement(tmp_path: Path):
    layout = OutputLayout.create(tmp_path)
    # Write markers
    (layout.refinement_batches_dir / "fine_interval_0001.json").write_text("{}", encoding="utf-8")
    (layout.refinement_frames_dir / "interval_0001").mkdir()
    (layout.refinement_frames_dir / "interval_0001" / "frame_00000001.jpg").write_bytes(b"x")
    # Write Epic 2 marker
    (layout.coarse_manifest_json).write_text("{}", encoding="utf-8")

    store = CacheStore(layout)
    store.clear_epic3_tree()

    # Epic 3 cleaned
    assert not (layout.refinement_batches_dir / "fine_interval_0001.json").is_file()
    assert not (layout.refinement_frames_dir / "interval_0001").is_dir()
    # Epic 2 preserved
    assert layout.coarse_manifest_json.is_file()


def test_clear_epic3_interval_removes_only_one_interval(tmp_path: Path):
    layout = OutputLayout.create(tmp_path)
    (layout.refinement_frames_dir / "interval_0001").mkdir()
    (layout.refinement_frames_dir / "interval_0001" / "frame.jpg").write_bytes(b"x")
    (layout.refinement_frames_dir / "interval_0002").mkdir()
    (layout.refinement_frames_dir / "interval_0002" / "frame.jpg").write_bytes(b"x")
    (layout.refinement_batches_dir / "fine_interval_0001.json").write_text("{}", encoding="utf-8")
    (layout.refinement_batches_dir / "fine_interval_0002.json").write_text("{}", encoding="utf-8")
    (layout.refinement_intervals_dir / "interval_0001.json").write_text("{}", encoding="utf-8")

    store = CacheStore(layout)
    store.clear_epic3_interval("interval_0001")

    assert not (layout.refinement_frames_dir / "interval_0001").is_dir()
    assert (layout.refinement_frames_dir / "interval_0002").is_dir()
    assert not (layout.refinement_batches_dir / "fine_interval_0001.json").is_file()
    assert (layout.refinement_batches_dir / "fine_interval_0002.json").is_file()
    assert not (layout.refinement_intervals_dir / "interval_0001.json").is_file()


def test_pipeline_state_includes_epic3_paths(tmp_path: Path):
    from video_analyst.models import PipelineState

    state = PipelineState(video_path="/tmp/v.mp4", output_dir=str(tmp_path))
    assert hasattr(state, "refinement_manifest_path")
    assert hasattr(state, "fine_batches_path")


def test_pipeline_stage_includes_epic3_stages():
    from video_analyst.models import PipelineStage

    assert PipelineStage.REFINEMENT_PLAN_LOAD.value == "refinement_plan_load"
    assert PipelineStage.REFINEMENT_FRAME_EXTRACT.value == "refinement_frame_extract"
    assert PipelineStage.REFINEMENT_TIMESTAMP_PARSE.value == "refinement_timestamp_parse"
    assert PipelineStage.REFINEMENT_DEDUPLICATE.value == "refinement_deduplicate"
    assert PipelineStage.TEMPORAL_CONTACT_SHEET.value == "temporal_contact_sheet"
    assert PipelineStage.DETAIL_FRAME_SELECT.value == "detail_frame_select"
    assert PipelineStage.FINE_BATCH_BUILD.value == "fine_batch_build"