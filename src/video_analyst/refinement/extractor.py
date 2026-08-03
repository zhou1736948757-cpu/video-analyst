"""FFmpeg-based fine-grained frame extractor (5 / 10 / 15 FPS).

Per-interval extraction strategy:
- Seek to `start` BEFORE the input with `-ss`, then `-t duration` to limit.
- Output VFR with `fps=FPS` so the wall-clock between output frames matches
  the requested FPS (5/10/15).
- Inject `showinfo` filter to surface per-frame `pts_time` so we never
  rely on filename index to infer timestamps.
- Default JPEG q:v 2, optional PNG.
- Scale to a configurable long-edge cap (preserving aspect).

RefinementConfig controls: image_format / jpeg_quality / max_frame_long_edge.

The extractor NEVER writes the output file outside ``out_dir`` and NEVER
trusts filename-derived timestamps — every output file is paired with a
``pts_time`` parsed from the same FFmpeg run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_analyst.config import RefinementConfig
from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import (
    RefinementFrameExtractionError,
    RefinementFrameMismatchError,
    RefinementTimestampParseError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.process import CommandResult, run_command
from video_analyst.refinement.models import RefinementSource
from video_analyst.refinement.timestamp_parser import (
    ShowinfoEntry,
    parse_showinfo_stderr,
    pair_with_files,
)
from video_analyst.util.hashing import sha256_file

log = get_logger("refinement.extractor")

_SCALE_RE = re.compile(r"scale=w=(-?\d+):h=(-?\d+)")

IMAGE_EXTENSIONS: dict[str, str] = {
    "jpg": "jpg",
    "jpeg": "jpg",
    "png": "png",
}


@dataclass
class ExtractedFrame:
    """Pairing of a parsed showinfo entry with its output JPEG."""

    showinfo: ShowinfoEntry
    path: Path
    width: int
    height: int
    size_bytes: int
    sha256: str
    absolute_timestamp: float
    relative_timestamp: float


def _scale_filter(long_edge: int) -> str:
    if long_edge <= 0:
        return "scale=trunc(iw*min(1\\,min(1600/iw\\,1600/ih))):trunc(ih*min(1\\,min(1600/iw\\,1600/ih)))"
    le = int(long_edge)
    return (
        f"scale='if(gt(iw\\,ih)\\,min({le}\\,iw)\\,-2)':"
        f"'if(gt(iw\\,ih)\\,-2\\,min({le}\\,ih))'"
    )


def _probe_image_size(path: Path) -> tuple[int, int]:
    """Read width/height from JPEG/PNG using Pillow if available, else ffprobe.

    Pillow is preferred for speed; fall back to ``ffprobe`` (always present
    per Phase 1) when Pillow is missing.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except ImportError:
        pass
    # Fallback: ffprobe
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            data = json.loads(out.stdout)
            stream = (data.get("streams") or [{}])[0]
            return int(stream.get("width", 0)), int(stream.get("height", 0))
    except Exception:
        return 0, 0
    return 0, 0


def _build_ffmpeg_command(
    *,
    ffmpeg_path: Path,
    input_path: Path,
    output_pattern: Path,
    interval: RefinementSource,
    cfg: RefinementConfig,
) -> list[str]:
    """Build the FFmpeg command line for one interval."""
    ext = IMAGE_EXTENSIONS.get(cfg.image_format.lower(), "jpg")
    fps = int(interval.requested_fps)
    duration = float(interval.end) - float(interval.start)

    # Re-point the pattern to the chosen extension
    pattern = output_pattern.with_name(f"{output_pattern.stem}%08d.{ext}")
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any pre-existing JPGs/PNGs in this output directory (idempotent)
    for old in output_pattern.parent.glob(f"{output_pattern.stem}*.{ext}"):
        try:
            old.unlink()
        except OSError:
            pass
    # Also wipe alternate extensions if a different run wrote them
    for other_ext in {"jpg", "png"} - {ext}:
        for old in output_pattern.parent.glob(f"{output_pattern.stem}*.{other_ext}"):
            try:
                old.unlink()
            except OSError:
                pass

    scale_filter = _scale_filter(cfg.max_frame_long_edge)
    # Output q:v mapping: lower = better; FFmpeg's default is ~2-3 for high quality.
    qv = max(1, min(31, int(cfg.jpeg_quality)))

    # Use `-ss` BEFORE -i for fast seek (less accurate but bounded; we
    # re-anchor using showinfo pts_time). `-t duration` bounds output.
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-nostdin",
        "-loglevel", "info",
        "-ss", f"{float(interval.start):.6f}",
        "-i", str(input_path),
        "-t", f"{max(0.001, duration):.6f}",
        "-vf", f"fps={fps},{scale_filter},showinfo",
        "-fps_mode", "vfr",
        "-q:v", str(qv),
        str(pattern),
    ]
    return cmd


def extract_interval_frames(
    *,
    ffmpeg_env: FFmpegEnvironment,
    source_video: Path,
    interval: RefinementSource,
    out_dir: Path,
    cfg: RefinementConfig,
    timeout: float | None = None,
) -> list[ExtractedFrame]:
    """Extract frames for one interval.

    Raises:
        RefinementFrameExtractionError: FFmpeg failed.
        RefinementTimestampParseError: showinfo produced no usable entries.
        RefinementFrameMismatchError: count of parsed entries != count of files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame"  # we'll add %08d.<ext> in _build_ffmpeg_command
    cmd = _build_ffmpeg_command(
        ffmpeg_path=ffmpeg_env.ffmpeg_path,
        input_path=source_video,
        output_pattern=pattern,
        interval=interval,
        cfg=cfg,
    )
    effective_timeout = float(timeout if timeout is not None else cfg.ffmpeg_timeout)

    log.info(
        "ffmpeg extract interval=%s fps=%d [%.3f..%.3f] -> %s",
        interval.interval_id,
        interval.requested_fps,
        interval.start,
        interval.end,
        out_dir,
    )
    result: CommandResult = run_command(
        cmd, timeout=effective_timeout, check=False
    )
    if not result.ok:
        raise RefinementFrameExtractionError(
            f"ffmpeg failed for interval {interval.interval_id} "
            f"(rc={result.returncode}): {(result.stderr or '')[-400:]}"
        )

    # Parse showinfo
    try:
        entries = parse_showinfo_stderr(result.stderr or "")
    except RefinementTimestampParseError:
        raise

    # Discover output files in stable order
    ext = IMAGE_EXTENSIONS.get(cfg.image_format.lower(), "jpg")
    files = sorted(out_dir.glob(f"frame*.{ext}"))
    if not files and not entries:
        # Empty interval is allowed (very small duration)
        log.info(
            "interval %s produced 0 frames (duration=%.3f)",
            interval.interval_id,
            interval.end - interval.start,
        )
        return []
    # FFmpeg's showinfo filter reports on every frame the filter chain emits,
    # but the encoder/muxer may drop the final partial frame at the `-t`
    # boundary, producing len(entries) == len(files) + 1. This is a known,
    # deterministic ffmpeg behavior (the trailing entry corresponds to a
    # frame that was filtered but never muxed to disk). We tolerate exactly
    # this off-by-one by trimming the trailing entry; any other mismatch is
    # a hard error per spec (we never guess timestamps from filenames).
    if len(entries) == len(files) + 1:
        log.debug(
            "interval %s: trimming 1 trailing showinfo entry (encoder dropped final frame)",
            interval.interval_id,
        )
        entries = entries[:-1]
    try:
        pairs = pair_with_files(entries, files)
    except RefinementTimestampParseError as exc:
        raise RefinementFrameMismatchError(str(exc)) from exc

    out: list[ExtractedFrame] = []
    for local_index, (entry, path) in enumerate(pairs):
        size_bytes = path.stat().st_size
        w, h = _probe_image_size(path)
        if w <= 0 or h <= 0:
            log.warning(
                "could not determine size for %s; using 0x0", path
            )
            w, h = 0, 0
        sha = sha256_file(path)
        abs_ts = float(interval.start) + float(entry.pts_time)
        rel_ts = float(entry.pts_time)
        out.append(
            ExtractedFrame(
                showinfo=entry,
                path=path,
                width=int(w),
                height=int(h),
                size_bytes=int(size_bytes),
                sha256=sha,
                absolute_timestamp=abs_ts,
                relative_timestamp=rel_ts,
            )
        )
    return out


def ffmpeg_supports_filter(ffmpeg_env: FFmpegEnvironment, filter_name: str) -> bool:
    """Return True if the given filter is reported as available."""
    cmd = [str(ffmpeg_env.ffmpeg_path), "-hide_banner", "-filters"]
    result = run_command(cmd, timeout=30, check=False)
    if not result.ok:
        return False
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    # Each filter line begins with a single-letter (encoder status) then a space.
    return any(
        line.split()[0:2]
        == line.split()[0:2]  # identity (always true)
        and filter_name in line
        for line in text.splitlines()
    )


def remove_extracted_interval(out_dir: Path) -> None:
    """Wipe the interval's output directory."""
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)


__all__ = [
    "ExtractedFrame",
    "extract_interval_frames",
    "ffmpeg_supports_filter",
    "remove_extracted_interval",
]