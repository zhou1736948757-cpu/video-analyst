"""Tests for FFmpeg showinfo pts_time parser."""

from __future__ import annotations

import pytest

from video_analyst.exceptions import RefinementTimestampParseError
from video_analyst.refinement.timestamp_parser import (
    ShowinfoEntry,
    compute_actual_fps,
    expected_interval_seconds,
    pair_with_files,
    parse_showinfo_stderr,
)


def _sample_stderr() -> str:
    return (
        "ffmpeg version 6.0 ...\n"
        "  Stream #0:0: Video: h264, 320x240, 30 fps\n"
        "[Parsed_showinfo_1 @ 0x...] n:0 pts:0 pts_time:0.000000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:1 pts:30000 pts_time:0.500000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:2 pts:60000 pts_time:1.000000 pos:48\n"
        "frame=    3 fps=0.0 q=28.0 size=N/A\n"
    )


def test_parse_basic_pts_time():
    entries = parse_showinfo_stderr(_sample_stderr())
    assert len(entries) == 3
    assert entries[0].pts_time == 0.0
    assert entries[1].pts_time == 0.5
    assert entries[2].pts_time == 1.0
    assert entries[0].pts == 0
    assert entries[1].pts == 30000
    assert entries[2].pts == 60000


def test_parse_keeps_order():
    text = (
        "[Parsed_showinfo_1 @ 0x...] n:0 pts:0 pts_time:0.000000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:1 pts:100000 pts_time:3.333000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:2 pts:50000 pts_time:1.666000 pos:48\n"
    )
    entries = parse_showinfo_stderr(text)
    assert [e.pts_time for e in entries] == [0.0, 3.333, 1.666]


def test_parse_empty_stderr_raises():
    with pytest.raises(RefinementTimestampParseError):
        parse_showinfo_stderr("")


def test_parse_no_pts_time_raises():
    with pytest.raises(RefinementTimestampParseError):
        parse_showinfo_stderr("ffmpeg ran but no pts_time lines here")


def test_pair_with_files_match_count():
    files = [type("F", (), {"__str__": lambda s: f"f{i}.jpg"})() for i in range(3)]
    entries = parse_showinfo_stderr(_sample_stderr())
    pairs = pair_with_files(entries, files)
    assert len(pairs) == 3
    # Order matches
    assert pairs[0][1] is files[0]
    assert pairs[2][1] is files[2]


def test_pair_mismatch_raises():
    files = [type("F", (), {})() for _ in range(2)]
    entries = parse_showinfo_stderr(_sample_stderr())
    with pytest.raises(RefinementTimestampParseError):
        pair_with_files(entries, files)


def test_compute_actual_fps():
    entries = parse_showinfo_stderr(
        "[Parsed_showinfo_1 @ 0x...] n:0 pts:0 pts_time:0.000000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:1 pts:100000 pts_time:1.000000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:2 pts:200000 pts_time:2.000000 pos:48\n"
    )
    assert compute_actual_fps(entries) == pytest.approx(1.0)


def test_compute_actual_fps_too_few_frames():
    e1 = ShowinfoEntry(pts=0, pts_time=0.0)
    assert compute_actual_fps([e1]) is None


def test_expected_interval_seconds():
    assert expected_interval_seconds(5, 1.0) == pytest.approx(0.2)
    assert expected_interval_seconds(10, 1.0) == pytest.approx(0.1)
    assert expected_interval_seconds(15, 1.0) == pytest.approx(1 / 15)


def test_expected_interval_seconds_invalid_fps():
    with pytest.raises(ValueError):
        expected_interval_seconds(0, 1.0)


def test_parse_handles_negative_pts():
    # Some streams have negative start pts; pts_time is always wall-clock.
    text = (
        "[Parsed_showinfo_1 @ 0x...] n:0 pts:-1000 pts_time:-0.100000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:1 pts:0 pts_time:0.000000 pos:48\n"
    )
    entries = parse_showinfo_stderr(text)
    assert entries[0].pts_time == pytest.approx(-0.1)
    assert entries[1].pts_time == pytest.approx(0.0)


def test_parse_skips_lines_without_pts_time():
    text = (
        "frame=    1 fps=0.0 q=28.0 size=N/A\n"
        "[Parsed_showinfo_1 @ 0x...] n:0 pts:0 pts_time:0.000000 pos:48\n"
        "[Parsed_showinfo_1 @ 0x...] n:1 pts:30000 pts_time:0.500000 pos:48\n"
    )
    entries = parse_showinfo_stderr(text)
    assert len(entries) == 2