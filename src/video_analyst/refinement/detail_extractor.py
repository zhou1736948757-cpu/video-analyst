"""高分辨率 detail frame 提取器（Epic 4 遗留修复）。

Epic 3 原实现只是把普通 refinement frame（长边 ≤1600）当作 detail frame，
没有真正的高分辨率版本。本模块从源视频按 detail frame 时间戳单独抽取，
默认不执行 1600 缩放；可配置 `detail_frame_max_long_edge` 上限。

resolution_mode 标记：
- ``source_original``：完全未缩放，输出与源视频分辨率一致
- ``capped_high_resolution``：用户配置了上限，按上限缩放（仍高于 1600）

输出至 ``analysis/refinement/detail_frames/<interval_id>/``。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_analyst.config import RefinementConfig
from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import RefinementFrameExtractionError
from video_analyst.logging.setup import get_logger
from video_analyst.process import CommandResult, run_command
from video_analyst.refinement.models import DetailFrameReference
from video_analyst.util.hashing import sha256_file

log = get_logger("refinement.detail_extractor")

IMAGE_EXTENSIONS: dict[str, str] = {
    "jpg": "jpg",
    "jpeg": "jpg",
    "png": "png",
}


@dataclass
class ExtractedDetailFrame:
    frame_id: str
    timestamp_seconds: float
    path: Path
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    scaled: bool
    resolution_mode: str
    sha256: str


def _probe_image_size(path: Path) -> tuple[int, int]:
    """读取 JPEG/PNG 宽高，优先用 Pillow，缺失则用 ffprobe。"""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except ImportError:
        pass
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
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


def _probe_source_dimensions(
    ffmpeg_env: FFmpegEnvironment, source_video: Path
) -> tuple[int, int]:
    """用 ffprobe 读取源视频宽高。"""
    out = subprocess.run(
        [
            str(ffmpeg_env.ffprobe_path),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(source_video),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        shell=False,
        check=False,
    )
    if out.returncode != 0 or not out.stdout:
        return 0, 0
    try:
        data = json.loads(out.stdout)
        stream = (data.get("streams") or [{}])[0]
        return int(stream.get("width", 0)), int(stream.get("height", 0))
    except Exception:
        return 0, 0


def _scale_filter_for_detail(long_edge: int) -> str:
    """detail frame 的 scale 滤镜：long_edge=0 时不缩放。"""
    if long_edge <= 0:
        return "scale=iw:ih"
    le = int(long_edge)
    return (
        f"scale='if(gt(iw\\,ih)\\,min({le}\\,iw)\\,-2)':"
        f"'if(gt(iw\\,ih)\\,-2\\,min({le}\\,ih))'"
    )


def extract_detail_frame(
    *,
    ffmpeg_env: FFmpegEnvironment,
    source_video: Path,
    frame_id: str,
    timestamp_seconds: float,
    out_dir: Path,
    cfg: RefinementConfig,
    selection_reason: str,
    timeout: float | None = None,
) -> ExtractedDetailFrame:
    """从源视频按时间戳单独抽取一张高分辨率 detail frame。

    使用 input-side `-ss` seek 定位到目标时间，输出单张 JPEG。
    不依赖帧文件序号推算时间。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = IMAGE_EXTENSIONS.get(cfg.image_format.lower(), "jpg")
    # 清理可能存在的旧文件（同 frame_id 不同格式）
    for other in ("jpg", "png"):
        old = out_dir / f"{frame_id}.{other}"
        if old.is_file():
            try:
                old.unlink()
            except OSError:
                pass
    out_path = out_dir / f"{frame_id}.{ext}"

    long_edge = int(cfg.detail_frame_max_long_edge)
    scale_filter = _scale_filter_for_detail(long_edge)
    qv = max(1, min(31, int(cfg.detail_frame_jpeg_quality)))

    cmd = [
        str(ffmpeg_env.ffmpeg_path),
        "-y",
        "-nostdin",
        "-loglevel", "info",
        "-ss", f"{float(timestamp_seconds):.6f}",
        "-i", str(source_video),
        "-frames:v", "1",
        "-vf", scale_filter,
        "-q:v", str(qv),
        str(out_path),
    ]
    effective_timeout = float(timeout if timeout is not None else cfg.ffmpeg_timeout)

    log.info(
        "ffmpeg detail frame=%s ts=%.3f long_edge=%d -> %s",
        frame_id,
        timestamp_seconds,
        long_edge,
        out_path,
    )
    result: CommandResult = run_command(
        cmd, timeout=effective_timeout, check=False
    )
    if not result.ok:
        raise RefinementFrameExtractionError(
            f"ffmpeg failed for detail frame {frame_id} "
            f"(rc={result.returncode}): {(result.stderr or '')[-400:]}"
        )
    if not out_path.is_file():
        raise RefinementFrameExtractionError(
            f"ffmpeg produced no output for detail frame {frame_id}"
        )

    out_w, out_h = _probe_image_size(out_path)
    src_w, src_h = _probe_source_dimensions(ffmpeg_env, source_video)
    scaled = long_edge > 0 and (
        out_w > 0 and (out_w != src_w or out_h != src_h)
    )
    if long_edge <= 0:
        resolution_mode = "source_original"
    else:
        resolution_mode = "capped_high_resolution" if scaled else "source_original"
    sha = sha256_file(out_path)

    return ExtractedDetailFrame(
        frame_id=frame_id,
        timestamp_seconds=round(float(timestamp_seconds), 6),
        path=out_path,
        source_width=int(src_w),
        source_height=int(src_h),
        output_width=int(out_w),
        output_height=int(out_h),
        scaled=bool(scaled),
        resolution_mode=resolution_mode,
        sha256=sha,
    )


def to_detail_frame_reference(
    extracted: ExtractedDetailFrame,
    *,
    selection_reason: str,
) -> DetailFrameReference:
    """把提取结果转换为持久化的 DetailFrameReference。"""
    return DetailFrameReference(
        frame_id=extracted.frame_id,
        timestamp_seconds=extracted.timestamp_seconds,
        path=extracted.path,
        selection_reason=selection_reason,
        source_width=extracted.source_width,
        source_height=extracted.source_height,
        output_width=extracted.output_width,
        output_height=extracted.output_height,
        scaled=extracted.scaled,
        resolution_mode=extracted.resolution_mode,
        sha256=extracted.sha256,
    )


__all__ = [
    "ExtractedDetailFrame",
    "extract_detail_frame",
    "to_detail_frame_reference",
]