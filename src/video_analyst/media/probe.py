"""ffprobe-based media metadata reading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import MediaProbeError
from video_analyst.logging.setup import get_logger
from video_analyst.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_analyst.process import run_command

log = get_logger("media.probe")


def _to_float(value: Any) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def probe_media(video_path: Path, env: FFmpegEnvironment) -> MediaInfo:
    path = video_path.expanduser().resolve()
    if not path.is_file():
        raise MediaProbeError(f"Video file not found: {path}")

    args = [
        env.ffprobe_path,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    log.info("Probing media: %s", path)
    result = run_command(args, timeout=120)
    if not result.ok:
        detail = (result.stderr or result.stdout or "").strip()[:800]
        raise MediaProbeError(f"ffprobe failed ({result.returncode}): {detail}")

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned invalid JSON: {exc}") from exc

    fmt = raw.get("format") or {}
    video_streams: list[VideoStreamInfo] = []
    audio_streams: list[AudioStreamInfo] = []

    for stream in raw.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            video_streams.append(
                VideoStreamInfo(
                    index=int(stream.get("index", len(video_streams))),
                    codec_name=stream.get("codec_name"),
                    width=_to_int(stream.get("width")),
                    height=_to_int(stream.get("height")),
                    avg_frame_rate=stream.get("avg_frame_rate"),
                    r_frame_rate=stream.get("r_frame_rate"),
                    duration=_to_float(stream.get("duration")),
                    bit_rate=_to_int(stream.get("bit_rate")),
                    pix_fmt=stream.get("pix_fmt"),
                )
            )
        elif codec_type == "audio":
            tags = stream.get("tags") or {}
            audio_streams.append(
                AudioStreamInfo(
                    index=int(stream.get("index", len(audio_streams))),
                    codec_name=stream.get("codec_name"),
                    sample_rate=_to_int(stream.get("sample_rate")),
                    channels=_to_int(stream.get("channels")),
                    channel_layout=stream.get("channel_layout"),
                    duration=_to_float(stream.get("duration")),
                    bit_rate=_to_int(stream.get("bit_rate")),
                    language=tags.get("language"),
                )
            )

    info = MediaInfo(
        path=path,
        format_name=fmt.get("format_name"),
        format_long_name=fmt.get("format_long_name"),
        duration=_to_float(fmt.get("duration")),
        size=_to_int(fmt.get("size")),
        bit_rate=_to_int(fmt.get("bit_rate")),
        video_streams=video_streams,
        audio_streams=audio_streams,
        raw=raw,
    )
    log.info(
        "Probe OK: duration=%s video_streams=%d audio_streams=%d",
        info.duration,
        len(info.video_streams),
        len(info.audio_streams),
    )
    return info
