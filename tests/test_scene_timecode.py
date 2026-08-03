"""Focused tests: scene frame timecode contract + provider normalization contract.

No network/provider calls.  No real media files.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from video_analyst.asr.groq_whisper import normalize_provider_response
from video_analyst.models import (
    ASREngine,
    SubtitleDocument,
    SubtitleMetadata,
    SubtitleSegment,
)
from video_analyst.phase2.prepare import build_visual_batches


# ---------------------------------------------------------------------------
# Provider normalization contract
# ---------------------------------------------------------------------------


def test_normalize_segments_get_valid_times():
    """null/negative start/end are clamped to valid floats."""
    raw = {
        "text": "hello world",
        "duration": 10.0,
        "language": "en",
        "segments": [
            {"id": 0, "start": None, "end": None, "text": "hello"},
            {"id": 1, "start": -1.0, "end": -0.5, "text": "world"},
        ],
    }
    result = normalize_provider_response(raw)
    for seg in result["segments"]:
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert seg["start"] >= 0.0
        assert seg["end"] >= seg["start"]


def test_normalize_drops_empty_segments():
    """Segments with blank text are removed; re-indexing stays sequential."""
    raw = {
        "text": "a",
        "duration": 5.0,
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "a"},
            {"id": 5, "start": 1.0, "end": 2.0, "text": ""},
            {"id": 9, "start": 2.0, "end": 3.0, "text": "  "},
        ],
    }
    result = normalize_provider_response(raw)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["id"] == 0
    assert result["segments"][0]["text"] == "a"


def test_normalize_creates_segments_from_text_when_missing():
    """Fallback: one synthetic segment from payload text if segments list empty."""
    raw = {"text": "single utterance", "duration": 3.0}
    result = normalize_provider_response(raw)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 3.0
    assert result["segments"][0]["text"] == "single utterance"


def test_normalize_supplies_provider_markers():
    """_engine and _model are always present after normalization."""
    raw = {"text": "test", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "test"}]}
    result = normalize_provider_response(raw)
    assert "_engine" in result
    assert "_model" in result
    assert result["_engine"] == ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value


def test_normalize_end_never_less_than_start():
    """If end < start, end is clamped to start."""
    raw = {
        "text": "test",
        "segments": [{"id": 0, "start": 5.0, "end": 2.0, "text": "backwards"}],
    }
    result = normalize_provider_response(raw)
    assert result["segments"][0]["end"] >= result["segments"][0]["start"]


def test_normalize_text_stripped():
    """Text is stripped of leading/trailing whitespace."""
    raw = {
        "text": "  padded  ",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "  padded  "}],
    }
    result = normalize_provider_response(raw)
    assert result["segments"][0]["text"] == "padded"


# ---------------------------------------------------------------------------
# Scene frame timecode contract
# ---------------------------------------------------------------------------


def test_build_visual_batches_scene_frames_with_timecode():
    """Scene frames with valid time are placed in the correct batch window."""
    doc = SubtitleDocument(
        source_video="/tmp/v.mp4",
        engine=ASREngine.GROQ_WHISPER_LARGE_V3_TURBO,
        duration=40.0,
        origin="llm",
        segments=[
            SubtitleSegment(index=1, start=0.0, end=5.0, text="first block"),
            SubtitleSegment(index=2, start=25.0, end=30.0, text="second block"),
        ],
    )
    meta = SubtitleMetadata()
    fixed = [{"path": "/tmp/0.jpg", "time": 0.0, "kind": "fixed_fps", "fps": 1.0}]
    scene = [
        {"path": "/tmp/s0.jpg", "time": 2.5, "kind": "scene", "index": 0},
        {"path": "/tmp/s1.jpg", "time": 18.0, "kind": "scene", "index": 1},
        {"path": "/tmp/s2.jpg", "time": 28.0, "kind": "scene", "index": 2},
    ]
    batches = build_visual_batches(
        doc=doc,
        meta=meta,
        fixed_frames=fixed,
        scene_frames=scene,
        batch_seconds=20.0,
        duration=40.0,
    )
    assert len(batches) == 2
    # batch 0: 0-20s → scene frames at 2.5s and 18.0s
    batch0_times = {f["time"] for f in batches[0].frames}
    assert 2.5 in batch0_times
    assert 18.0 in batch0_times
    # batch 1: 20-40s → scene frame at 28.0s
    batch1_times = {f["time"] for f in batches[1].frames}
    assert 28.0 in batch1_times
    assert 2.5 not in batch1_times


def test_extract_scene_frames_never_returns_time_none():
    """Verify the internal contract: scene frame dict 'time' is never None.

    Uses mocked ffmpeg with showinfo-style output on stderr.
    """
    from video_analyst.env.detect import FFmpegEnvironment
    from video_analyst.phase2.prepare import extract_scene_frames

    env = FFmpegEnvironment(
        ffmpeg_path=Path("/mock/ffmpeg"),
        ffprobe_path=Path("/mock/ffprobe"),
        ffmpeg_version="mock",
        ffprobe_version="mock",
    )
    stderr_output = (
        "[Parsed_showinfo_1 @ 0x1] n:   1 pts: 12345 pts_time:3.456  pos:100 fmt:yuv420p\n"
        "[Parsed_showinfo_1 @ 0x2] n:   2 pts: 67890 pts_time:12.789 pos:500 fmt:yuv420p\n"
    )

    def fake_run(*_args, **_kwargs):
        from video_analyst.process import CommandResult

        return CommandResult(
            args=["ffmpeg"],
            returncode=0,
            stdout="",
            stderr=stderr_output,
        )

    with (
        patch("video_analyst.phase2.prepare.run_command", side_effect=fake_run),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.glob", return_value=[
            Path("/out/scene_000001.jpg"),
            Path("/out/scene_000002.jpg"),
        ]),
        patch("pathlib.Path.unlink"),
    ):
        frames = extract_scene_frames(
            Path("/mock/video.mp4"),
            Path("/out"),
            env,
            threshold=0.3,
        )

    assert len(frames) == 2
    for f in frames:
        assert f["time"] is not None
        assert isinstance(f["time"], float)

    assert frames[0]["time"] == 3.456
    assert frames[1]["time"] == 12.789
    assert frames[0]["kind"] == "scene"
    assert frames[1]["kind"] == "scene"
