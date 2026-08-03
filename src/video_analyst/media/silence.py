"""Detect silence gaps for multilingual chunked ASR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import AudioExtractError
from video_analyst.logging.setup import get_logger
from video_analyst.process import run_command

log = get_logger("media.silence")

_SILENCE_START = re.compile(r"silence_start:\s*([-\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([-\d.]+)")


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_speech_ranges(
    audio_path: Path,
    env: FFmpegEnvironment,
    *,
    duration: float | None = None,
    noise_db: float = -32.0,
    min_silence: float = 0.55,
    min_speech: float = 0.35,
    pad: float = 0.15,
) -> list[TimeRange]:
    """
    Return speech ranges by inverting ffmpeg silencedetect gaps.

    Used only as a *language-detection* helper: short mid-sentence breaths may
    still create multiple ranges; callers should merge same-language results
    afterward. Falls back to a single full-range span when no silence is found.
    """
    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise AudioExtractError(f"Audio not found: {path}")

    if duration is None:
        duration = _probe_duration(path, env)

    af = f"silencedetect=noise={noise_db}dB:d={min_silence}"
    result = run_command(
        [env.ffmpeg_path, "-i", path, "-af", af, "-f", "null", "-"],
        timeout=120,
    )
    # silencedetect writes to stderr
    text = (result.stderr or "") + "\n" + (result.stdout or "")

    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in text.splitlines():
        m_start = _SILENCE_START.search(line)
        if m_start:
            start = float(m_start.group(1))
            continue
        m_end = _SILENCE_END.search(line)
        if m_end and start is not None:
            end = float(m_end.group(1))
            if end > start:
                silences.append((start, end))
            start = None

    # Trailing silence without end
    if start is not None and duration > start:
        silences.append((start, duration))

    speech: list[TimeRange] = []
    cursor = 0.0
    for s0, s1 in silences:
        if s0 > cursor + 1e-3:
            speech.append(TimeRange(start=cursor, end=s0))
        cursor = max(cursor, s1)
    if cursor < duration - 1e-3:
        speech.append(TimeRange(start=cursor, end=duration))

    if not speech:
        speech = [TimeRange(start=0.0, end=duration)]

    # pad and merge short gaps already handled by silence; clamp + drop tiny
    padded: list[TimeRange] = []
    for r in speech:
        a = max(0.0, r.start - pad)
        b = min(duration, r.end + pad)
        if b - a < min_speech:
            # keep short utterances if they have any length
            if b - a < 0.15:
                continue
        if padded and a <= padded[-1].end + 0.05:
            prev = padded[-1]
            padded[-1] = TimeRange(start=prev.start, end=max(prev.end, b))
        else:
            padded.append(TimeRange(start=a, end=b))

    if not padded:
        padded = [TimeRange(start=0.0, end=duration)]

    log.info(
        "Speech ranges: %d (duration=%.2fs silence_gaps=%d)",
        len(padded),
        duration,
        len(silences),
    )
    return padded


def extract_audio_segment(
    audio_path: Path,
    output_path: Path,
    env: FFmpegEnvironment,
    *,
    start: float,
    end: float,
) -> Path:
    """Cut [start, end) from audio into a new mp3 for chunked ASR."""
    src = audio_path.expanduser().resolve()
    dst = output_path.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, end - start)
    args = [
        env.ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        src,
        "-acodec",
        "libmp3lame",
        "-b:a",
        "64k",
        "-ar",
        "16000",
        "-ac",
        "1",
        dst,
    ]
    result = run_command(args, timeout=120)
    if not result.ok or not dst.is_file() or dst.stat().st_size == 0:
        detail = (result.stderr or result.stdout or "")[-400:]
        raise AudioExtractError(f"Failed to cut audio segment: {detail}")
    return dst


def _probe_duration(path: Path, env: FFmpegEnvironment) -> float:
    result = run_command(
        [
            env.ffprobe_path,
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=30,
    )
    if not result.ok:
        raise AudioExtractError(f"ffprobe duration failed for {path}")
    try:
        return float((result.stdout or "").strip())
    except ValueError as exc:
        raise AudioExtractError(f"Invalid duration for {path}") from exc
