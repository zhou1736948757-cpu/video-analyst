"""Tests for detail-frame selection."""

from __future__ import annotations

from pathlib import Path

from video_analyst.config import RefinementConfig
from video_analyst.refinement.deduplicate import DeduplicatedFrame
from video_analyst.refinement.detail_frames import select_detail_frames


def _df(
    t: float,
    *,
    interval_id: str = "interval_0001",
    path: str | None = None,
    sha: str = "abcd",
) -> DeduplicatedFrame:
    return DeduplicatedFrame(
        interval_id=interval_id,
        local_index=-1,
        frame_id=f"{interval_id}__f{int(t * 1000):08d}",
        path=Path(path or f"/tmp/{interval_id}/{int(t * 1000):08d}.jpg"),
        timestamp_seconds=t,
        relative_timestamp_seconds=t,
        requested_fps=10,
        width=320,
        height=240,
        size_bytes=1024,
        sha256=sha,
        source_video_sha256="vsha",
    )


def test_picks_first_and_last():
    frames = [_df(t) for t in (5.0, 5.1, 5.2, 5.3, 5.4)]
    cfg = RefinementConfig()
    refs = select_detail_frames(frames, interval_id="interval_0001", interval_start=5.0, interval_end=5.4, cfg=cfg)
    assert len(refs) == 2
    assert refs[0].selection_reason == "interval_start"
    assert refs[1].selection_reason == "interval_end"


def test_single_frame_interval_yields_one():
    frames = [_df(5.0)]
    cfg = RefinementConfig()
    refs = select_detail_frames(
        frames, interval_id="interval_0001", interval_start=5.0, interval_end=5.0, cfg=cfg
    )
    # First == last → keep only first
    assert len(refs) == 1
    assert refs[0].frame_id == frames[0].frame_id


def test_empty_frames():
    refs = select_detail_frames([], interval_id="interval_0001", interval_start=5.0, interval_end=6.0, cfg=RefinementConfig())
    assert refs == []


def test_max_per_interval_caps():
    # The default config has detail_max_per_interval=2; we don't increase it
    # in this test (we can't call select with cfg.max>2 easily without
    # changing defaults). Instead: confirm the cap is honored.
    frames = [_df(t) for t in (1.0, 1.5, 2.0)]
    cfg = RefinementConfig(detail_max_per_interval=2)
    refs = select_detail_frames(frames, interval_id="interval_0001", interval_start=1.0, interval_end=2.0, cfg=cfg)
    assert len(refs) == 2


def test_selection_reason_for_single_frame():
    frames = [_df(5.0)]
    cfg = RefinementConfig()
    refs = select_detail_frames(
        frames, interval_id="interval_0001", interval_start=5.0, interval_end=5.0, cfg=cfg
    )
    assert len(refs) == 1
    # The first selection should still be 'interval_start'
    assert refs[0].selection_reason == "interval_start"


def test_paths_preserved():
    frames = [_df(5.0, path="/tmp/a.jpg"), _df(5.1, path="/tmp/b.jpg")]
    cfg = RefinementConfig()
    refs = select_detail_frames(frames, interval_id="interval_0001", interval_start=5.0, interval_end=5.1, cfg=cfg)
    assert str(refs[0].path) == "/tmp/a.jpg"
    assert str(refs[1].path) == "/tmp/b.jpg"