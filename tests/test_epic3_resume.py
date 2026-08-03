"""Epic 3 resume / partial-recovery tests.

Covers the spec's resume requirements:
- Mid-run interruption resumes from the next un-finished interval
- A failed interval does not delete already-succeeded intervals
- Refresh of a single interval preserves the other intervals' artifacts
- refresh=True reruns all intervals
"""

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
    reason="ffmpeg + Pillow required for Epic 3 resume tests",
)


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    out = tmp_path / "v.mp4"
    make_test_video(out, duration=8.0, fps=30)
    return out


def test_resume_after_partial_failure(tmp_path: Path, video_path: Path, monkeypatch):
    """First run: interval_0002 fails (simulated). Second run: succeeds."""
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

    import video_analyst.pipeline.epic3 as epic3

    real_extract = epic3.extract_interval_frames
    state = {"calls": 0}

    def selective_extract(*args, **kwargs):
        interval = kwargs.get("interval")
        if interval is not None and interval.interval_id == "interval_0002":
            state["calls"] += 1
            if state["calls"] == 1:
                from video_analyst.exceptions import RefinementFrameExtractionError

                raise RefinementFrameExtractionError("simulated failure on first attempt")
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(epic3, "extract_interval_frames", selective_extract)

    r1 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    assert r1.succeeded_intervals == 1
    assert r1.failed_intervals == 1
    assert "interval_0002" in r1.failed_interval_ids

    # Second run — no failure injected (state.calls already > 1)
    monkeypatch.undo()
    r2 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    # interval_0001 should be cached; interval_0002 should now succeed
    assert r2.succeeded_intervals + r2.cached_intervals == 2
    assert r2.failed_intervals == 0
    # Both fine batches now on disk
    batches_dir = tmp_path / "analysis" / "refinement" / "visual_inputs" / "batches"
    assert (batches_dir / "fine_interval_0001.json").is_file()
    assert (batches_dir / "fine_interval_0002.json").is_file()


def test_successful_interval_survives_subsequent_failure(
    tmp_path: Path, video_path: Path, monkeypatch
):
    """interval_0001 succeeds, interval_0002 fails; interval_0001's frames
    and fine batch must remain on disk."""
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

    import video_analyst.pipeline.epic3 as epic3

    real_extract = epic3.extract_interval_frames

    def always_fail_0002(*args, **kwargs):
        interval = kwargs.get("interval")
        if interval is not None and interval.interval_id == "interval_0002":
            from video_analyst.exceptions import RefinementFrameExtractionError

            raise RefinementFrameExtractionError("always fails")
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(epic3, "extract_interval_frames", always_fail_0002)

    r = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    assert r.succeeded_intervals == 1
    assert r.failed_intervals == 1

    # interval_0001 artifacts must be on disk
    assert (
        tmp_path / "analysis" / "refinement" / "frames" / "interval_0001"
    ).is_dir()
    assert (
        tmp_path
        / "analysis"
        / "refinement"
        / "visual_inputs"
        / "batches"
        / "fine_interval_0001.json"
    ).is_file()
    # interval_0002 must NOT have a fine batch
    assert not (
        tmp_path
        / "analysis"
        / "refinement"
        / "visual_inputs"
        / "batches"
        / "fine_interval_0002.json"
    ).is_file()


def test_refresh_single_interval_keeps_others(tmp_path: Path, video_path: Path):
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

    # Capture interval_0002 fine batch content
    fb2 = (
        tmp_path
        / "analysis"
        / "refinement"
        / "visual_inputs"
        / "batches"
        / "fine_interval_0002.json"
    )
    fb2_before = fb2.read_text(encoding="utf-8")

    # Refresh only interval_0001
    run_epic3(
        video_path=video_path,
        output_dir=tmp_path,
        config=cfg,
        refresh_intervals=["interval_0001"],
    )

    # interval_0002 fine batch unchanged (cached)
    fb2_after = fb2.read_text(encoding="utf-8")
    assert fb2_before == fb2_after


def test_refresh_all_reruns_every_interval(tmp_path: Path, video_path: Path):
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
    r1 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)

    # Capture mtime of an interval_0001 frame
    frames_dir = tmp_path / "analysis" / "refinement" / "frames" / "interval_0001"
    first_frame = next(frames_dir.iterdir())
    mtime_before = os.path.getmtime(first_frame)

    # Refresh all
    r2 = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg, refresh=True)
    # All intervals succeeded (re-extracted)
    assert r2.succeeded_intervals == 2
    # The frame file was rewritten (mtime changed)
    mtime_after = os.path.getmtime(first_frame)
    assert mtime_after != mtime_before


def test_manifest_records_status_per_interval(tmp_path: Path, video_path: Path, monkeypatch):
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

    import video_analyst.pipeline.epic3 as epic3

    real_extract = epic3.extract_interval_frames

    def fail_0002_once(*args, **kwargs):
        interval = kwargs.get("interval")
        if interval is not None and interval.interval_id == "interval_0002":
            from video_analyst.exceptions import RefinementFrameExtractionError

            raise RefinementFrameExtractionError("first-attempt failure")
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(epic3, "extract_interval_frames", fail_0002_once)
    run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)

    manifest = json.loads(
        (tmp_path / "analysis" / "refinement" / "manifest.json").read_text()
    )
    statuses = {e["interval_id"]: e["status"] for e in manifest["intervals"]}
    assert statuses["interval_0001"] in ("ok", "cached")
    assert statuses["interval_0002"] == "failed"


def test_empty_plan_writes_empty_fine_batches_index(tmp_path: Path, video_path: Path):
    seed_refinement_layout(tmp_path, duration=8.0, refinement_intervals=[])
    cfg = AppConfig()
    result = run_epic3(video_path=video_path, output_dir=tmp_path, config=cfg)
    assert result.status == "completed_no_intervals"

    fb_index = json.loads(
        (tmp_path / "analysis" / "refinement" / "visual_inputs" / "fine_batches.json").read_text()
    )
    assert fb_index["total_batches"] == 0
    assert fb_index["batches"] == []
