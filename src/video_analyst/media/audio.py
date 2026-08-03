"""FFmpeg audio separation and compression."""

from __future__ import annotations

from pathlib import Path

from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import AudioExtractError
from video_analyst.logging.setup import get_logger
from video_analyst.process import run_command

log = get_logger("media.audio")


def extract_audio(
    video_path: Path,
    output_audio: Path,
    env: FFmpegEnvironment,
    *,
    audio_bitrate: str = "64k",
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """
    Extract and compress audio for ASR.

    Default: mono 16 kHz MP3 @ 64k — good size/quality tradeoff for Whisper.
    """
    src = video_path.expanduser().resolve()
    dst = output_audio.expanduser().resolve()
    if not src.is_file():
        raise AudioExtractError(f"Video file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    args: list[str | Path] = [
        env.ffmpeg_path,
        "-y",
        "-i",
        src,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        audio_bitrate,
        "-ar",
        str(sample_rate),
    ]
    if mono:
        args.extend(["-ac", "1"])
    args.append(dst)

    log.info("Extracting audio → %s", dst)
    result = run_command(args, timeout=600)
    if not result.ok:
        detail = (result.stderr or result.stdout or "").strip()[-800:]
        raise AudioExtractError(f"ffmpeg audio extract failed ({result.returncode}): {detail}")

    if not dst.is_file() or dst.stat().st_size == 0:
        raise AudioExtractError(f"Audio output missing or empty: {dst}")

    log.info("Audio extracted: %s (%d bytes)", dst, dst.stat().st_size)
    return dst
