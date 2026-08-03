"""Tests for the deterministic frame deduplicator."""

from __future__ import annotations

from pathlib import Path

from video_analyst.refinement.deduplicate import (
    DeduplicatedFrame,
    deduplicate_intervals,
)


def _ef(
    *,
    path: str,
    t: float,
    sha: str,
    rel_t: float = 0.0,
    fps: int = 10,
    width: int = 320,
    height: int = 240,
) -> "ExtractedFrameLike":  # type: ignore[name-defined]
    """Local stand-in for ExtractedFrame for tests."""
    from video_analyst.refinement.extractor import ExtractedFrame
    from video_analyst.refinement.timestamp_parser import ShowinfoEntry

    return ExtractedFrame(
        showinfo=ShowinfoEntry(pts=int(t * 1000), pts_time=rel_t),
        path=Path(path),
        width=width,
        height=height,
        size_bytes=1024,
        sha256=sha,
        absolute_timestamp=t,
        relative_timestamp=rel_t,
    )


def test_intra_interval_dedup_same_time_same_sha():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    b = _ef(path="/tmp/interval_0001/b.jpg", t=5.002, sha="aaaa", fps=10)
    c = _ef(path="/tmp/interval_0001/c.jpg", t=5.5, sha="cccc", fps=10)
    report = deduplicate_intervals(
        {"interval_0001": [a, b, c]},
        time_threshold_seconds=0.005,
        requested_fps_by_interval={"interval_0001": 10},
    )
    assert len(report.canonical_frames) == 2  # b dropped
    assert report.stats["intra_dropped"] == 1
    kept_times = [d.timestamp_seconds for d in report.canonical_frames]
    assert 5.0 in kept_times
    assert 5.5 in kept_times


def test_intra_interval_keeps_different_sha_at_close_time():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    b = _ef(path="/tmp/interval_0001/b.jpg", t=5.001, sha="bbbb", fps=10)
    report = deduplicate_intervals(
        {"interval_0001": [a, b]},
        time_threshold_seconds=0.005,
        requested_fps_by_interval={"interval_0001": 10},
    )
    # Both kept (different content)
    assert len(report.canonical_frames) == 2


def test_intra_interval_keeps_legal_consecutive_frames():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    b = _ef(path="/tmp/interval_0001/b.jpg", t=5.1, sha="bbbb", fps=10)
    c = _ef(path="/tmp/interval_0001/c.jpg", t=5.2, sha="cccc", fps=10)
    report = deduplicate_intervals(
        {"interval_0001": [a, b, c]},
        time_threshold_seconds=0.005,
        requested_fps_by_interval={"interval_0001": 10},
    )
    assert len(report.canonical_frames) == 3


def test_cross_interval_alias_same_time_same_sha():
    a1 = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    a2 = _ef(path="/tmp/interval_0002/a.jpg", t=5.003, sha="aaaa", fps=15)
    report = deduplicate_intervals(
        {"interval_0001": [a1], "interval_0002": [a2]},
        time_threshold_seconds=0.005,
    )
    # canonical keeps only one (the one with lower interval_id after sorting)
    assert len(report.canonical_frames) == 1
    # The other interval's frame is aliased to the canonical frame_id
    assert len(report.cross_aliases) == 1
    (alias_key, alias_target) = next(iter(report.cross_aliases.items()))
    assert alias_key[0] == "interval_0002"
    assert alias_target == report.canonical_frames[0].frame_id


def test_cross_interval_does_not_alias_different_sha():
    a1 = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    a2 = _ef(path="/tmp/interval_0002/a.jpg", t=5.001, sha="xxxx", fps=15)
    report = deduplicate_intervals(
        {"interval_0001": [a1], "interval_0002": [a2]},
        time_threshold_seconds=0.005,
    )
    assert len(report.canonical_frames) == 2
    assert report.cross_aliases == {}


def test_cross_interval_does_not_alias_far_time():
    a1 = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    a2 = _ef(path="/tmp/interval_0002/a.jpg", t=10.0, sha="aaaa", fps=15)
    report = deduplicate_intervals(
        {"interval_0001": [a1], "interval_0002": [a2]},
        time_threshold_seconds=0.005,
    )
    assert len(report.canonical_frames) == 2


def test_local_index_re_assigned_kept_order():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    b = _ef(path="/tmp/interval_0001/b.jpg", t=5.1, sha="bbbb", fps=10)
    c = _ef(path="/tmp/interval_0001/c.jpg", t=5.2, sha="cccc", fps=10)
    report = deduplicate_intervals(
        {"interval_0001": [a, b, c]},
        time_threshold_seconds=0.005,
        requested_fps_by_interval={"interval_0001": 10},
    )
    indices = [d.local_index for d in report.canonical_frames]
    assert indices == [0, 1, 2]


def test_source_video_sha_propagated():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    report = deduplicate_intervals(
        {"interval_0001": [a]},
        time_threshold_seconds=0.005,
        source_video_sha256="video123",
    )
    assert report.canonical_frames[0].source_video_sha256 == "video123"


def test_requested_fps_attached():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    report = deduplicate_intervals(
        {"interval_0001": [a]},
        time_threshold_seconds=0.005,
        requested_fps_by_interval={"interval_0001": 15},
    )
    assert report.canonical_frames[0].requested_fps == 15


def test_stats_keys():
    a = _ef(path="/tmp/interval_0001/a.jpg", t=5.0, sha="aaaa", fps=10)
    report = deduplicate_intervals({"interval_0001": [a]})
    assert "input_frames" in report.stats
    assert "after_intra" in report.stats
    assert "after_cross" in report.stats
    assert report.stats["input_frames"] == 1


def test_dedup_no_input():
    report = deduplicate_intervals({})
    assert report.canonical_frames == []