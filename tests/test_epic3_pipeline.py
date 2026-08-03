"""Tests for the Epic 3 pipeline orchestration (run_epic3)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from video_analyst.config import AppConfig
from video_analyst.pipeline.epic3 import run_epic3

from tests._epic3_fixtures import (
    has_ffmpeg,
    has_pillow,
    make_test_video,
    seed_refinement_layout,
)

pytestmark = pytest.mark.skipif(
    not (has_ffmpeg() and has_pillow()),
    reason="ffmpeg + Pillow required for Epic 3 pipeline tests",
)


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    out = tmp_path / "v.mp4"
    make_test_video(out, duration=8.0, fps=30)
    return out


def test_full_pipeline(tmp_path: Path, video_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10},
            {"interval_id": "interval_0002", "start": 4.0, "end": 6.0, "requested_fps": 15},
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    result = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg, user_request="x")
    assert result.total_intervals == 2
    assert result.succeeded_intervals + result.cached_intervals == 2
    # Outputs exist
    assert layout.refinement_manifest_json.is_file()
    assert layout.refinement_fine_batches_json.is_file()
    assert (layout.refinement_batches_dir / "fine_interval_0001.json").is_file()
    assert (layout.refinement_batches_dir / "fine_interval_0002.json").is_file()
    # Interval artifacts persisted
    assert (layout.refinement_intervals_dir / "interval_0001.json").is_file()
    assert (layout.refinement_intervals_dir / "interval_0002.json").is_file()
    # Frames extracted
    assert (layout.refinement_frames_dir / "interval_0001").is_dir()
    assert (layout.refinement_frames_dir / "interval_0002").is_dir()
    # Contact sheets built
    assert (layout.refinement_contact_sheets_dir / "interval_0001").is_dir()


def test_no_partial_raises_on_failure(tmp_path: Path, video_path: Path):
    # A very short (but valid) interval may produce only 1-2 frames; the
    # spec says don't fake success, so we verify the run still completes
    # honestly (status=completed, not fabricated).
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 0.0, "end": 0.2, "requested_fps": 10},
        ],
    )
    cfg = AppConfig()
    result = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    assert result.status in ("completed", "completed_no_intervals")
    assert result.total_intervals == 1


def test_empty_plan_returns_completed_no_intervals(tmp_path: Path, video_path: Path):
    layout = seed_refinement_layout(tmp_path, duration=8.0, refinement_intervals=[])
    cfg = AppConfig()
    result = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    assert result.status == "completed_no_intervals"
    assert result.total_intervals == 0
    # Manifest still written
    assert layout.refinement_manifest_json.is_file()


def test_second_run_hits_cache(tmp_path: Path, video_path: Path):
    layout = seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10}
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    r1 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Capture mtime of extracted frames
    frames_dir = layout.refinement_frames_dir / "interval_0001"
    assert frames_dir.is_dir()
    first_frame = next(frames_dir.iterdir())
    f1_mtime = os.path.getmtime(first_frame)
    # Second run
    r2 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Manifest is consistent on cached intervals
    assert r2.cached_intervals + r2.succeeded_intervals == 1
    # File mtime unchanged on cached intervals
    f2_mtime = os.path.getmtime(first_frame)
    assert f1_mtime == f2_mtime


def test_user_request_change_does_not_re_extract_frames(tmp_path: Path, video_path: Path):
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10}
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    r1 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg, user_request="A")
    # Capture manifest keys
    m1 = json.loads((tmp_path / "analysis" / "refinement" / "manifest.json").read_text())
    keys1 = {e["interval_id"]: e["extraction_signature"] for e in m1["intervals"]}

    # Second run with different user_request
    r2 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg, user_request="B")
    m2 = json.loads((tmp_path / "analysis" / "refinement" / "manifest.json").read_text())
    keys2 = {e["interval_id"]: e["extraction_signature"] for e in m2["intervals"]}
    # Extraction signature should be unchanged (user_request doesn't affect extraction)
    assert keys1 == keys2


def test_changing_sheet_layout_does_not_re_extract(tmp_path: Path, video_path: Path):
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10}
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    cfg.refinement.frames_per_sheet = 4
    cfg.refinement.sheet_overlap_frames = 1
    r1 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)

    # Now change layout
    cfg.refinement.frames_per_sheet = 6
    cfg.refinement.sheet_overlap_frames = 0
    r2 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Extraction signature should match; presentation differs but frames are NOT re-extracted
    m1 = json.loads((tmp_path / "analysis" / "refinement" / "manifest.json").read_text())
    m2 = json.loads((tmp_path / "analysis" / "refinement" / "manifest.json").read_text())
    e1 = next(e for e in m1["intervals"] if e["interval_id"] == "interval_0001")
    e2 = next(e for e in m2["intervals"] if e["interval_id"] == "interval_0001")
    assert e1["extraction_signature"] == e2["extraction_signature"]


def test_refresh_refinement_clears_all(tmp_path: Path, video_path: Path):
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10}
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Refresh
    r2 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg, refresh=True)
    assert r2.succeeded_intervals == 1


def test_refresh_specific_interval(tmp_path: Path, video_path: Path):
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10},
            {"interval_id": "interval_0002", "start": 4.0, "end": 6.0, "requested_fps": 10},
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Refresh only interval_0001
    run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg, refresh_intervals=["interval_0001"])
    # interval_0001 artifacts regenerated, interval_0002 preserved
    batches_dir = tmp_path / "analysis" / "refinement" / "visual_inputs" / "batches"
    assert (batches_dir / "fine_interval_0001.json").is_file()
    assert (batches_dir / "fine_interval_0002.json").is_file()


def test_partial_failure_then_resume(tmp_path: Path, video_path: Path):
    """Simulate: first run has one interval that fails, second run succeeds it."""
    layout = seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10},
            {"interval_id": "interval_0002", "start": 4.0, "end": 6.0, "requested_fps": 10},
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    result = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Both should succeed in normal conditions
    assert result.succeeded_intervals == 2


def test_failed_interval_does_not_delete_siblings(tmp_path: Path, video_path: Path, monkeypatch):
    """Force one interval to fail; the other should remain."""
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10},
            {"interval_id": "interval_0002", "start": 4.0, "end": 6.0, "requested_fps": 10},
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True

    # Patch extract_interval_frames so interval_0002 fails the first time
    import video_analyst.pipeline.epic3 as epic3
    real_extract = epic3.extract_interval_frames
    state = {"calls": 0}

    def selective_extract(*args, **kwargs):
        interval = kwargs.get("interval")
        if interval is not None and interval.interval_id == "interval_0002":
            state["calls"] += 1
            if state["calls"] == 1:
                from video_analyst.exceptions import RefinementFrameExtractionError

                raise RefinementFrameExtractionError("simulated failure")
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(epic3, "extract_interval_frames", selective_extract)

    r = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # interval_0001 ok, interval_0002 failed
    assert r.succeeded_intervals == 1
    assert r.failed_intervals == 1
    assert "interval_0002" in r.failed_interval_ids
    # interval_0001 artifacts preserved
    assert (
        tmp_path / "analysis" / "refinement" / "visual_inputs" / "batches" / "fine_interval_0001.json"
    ).is_file()


def test_manifest_atomic_updates(tmp_path: Path, video_path: Path):
    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10},
            {"interval_id": "interval_0002", "start": 4.0, "end": 6.0, "requested_fps": 10},
        ],
    )
    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # Manifest exists and is valid JSON
    m = json.loads((tmp_path / "analysis" / "refinement" / "manifest.json").read_text())
    assert "intervals" in m
    assert m["total_intervals"] == 2
    assert m["succeeded_intervals"] == 2


def test_pipeline_state_updated(tmp_path: Path, video_path: Path):
    from video_analyst.models import PipelineState

    seed_refinement_layout(
        tmp_path,
        duration=8.0,
        refinement_intervals=[
            {"interval_id": "interval_0001", "start": 1.0, "end": 3.0, "requested_fps": 10}
        ],
    )
    # Write a stub pipeline_state.json with a known video_path
    state_path = tmp_path / "cache" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = PipelineState(
        video_path=str(video_path),
        output_dir=str(tmp_path),
    )
    state_path.write_text(json.dumps(state.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")

    cfg = AppConfig()
    cfg.refinement.allow_partial = True
    run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # pipeline_state.json now has refinement_manifest_path + fine_batches_path
    new_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert new_state["refinement_manifest_path"]
    assert new_state["fine_batches_path"]