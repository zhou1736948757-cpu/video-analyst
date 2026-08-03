"""Phase 2 input preparation: coarse frames, scene frames, subtitle mapping, batches.

Does NOT call visual models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import Phase2Config
from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import Phase2PrepareError
from video_analyst.logging.setup import get_logger
from video_analyst.models import (
    Phase2PrepareResult,
    SubtitleDocument,
    SubtitleMetadata,
    SubtitleReviewPackage,
    VisualBatch,
)
from video_analyst.process import run_command

log = get_logger("phase2.prepare")


def load_active_subtitles_for_phase2(
    layout: OutputLayout,
) -> tuple[SubtitleDocument, SubtitleMetadata]:
    """Phase 2 may only read active subtitles + review metadata."""
    store = CacheStore(layout)
    if not layout.subtitles_json.is_file():
        raise Phase2PrepareError("active subtitles.json missing; run Phase 1 first")
    doc = SubtitleDocument.model_validate(store.read_json(layout.subtitles_json))
    meta = SubtitleMetadata(active_source=doc.origin if doc.origin != "active" else "asr")
    if layout.review_json.is_file():
        try:
            from video_analyst.llm.review import load_review

            review = load_review(layout)
            meta = SubtitleMetadata(
                active_source=review.active.value,
                reviewed=review.reviewed,
                needs_review=review.needs_review,
                asr_model=review.asr_meta.model if review.asr_meta else None,
                calibration_model=review.llm_meta.model if review.llm_meta else None,
            )
        except Exception as exc:
            log.warning("Could not load review.json for phase2 meta: %s", exc)
    return doc, meta


def extract_fixed_fps_frames(
    video_path: Path,
    out_dir: Path,
    env: FFmpegEnvironment,
    *,
    fps: float = 1.0,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear old jpgs
    for p in out_dir.glob("frame_*.jpg"):
        p.unlink()
    pattern = str(out_dir / "frame_%06d.jpg")
    result = run_command(
        [
            env.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "3",
            pattern,
        ],
        timeout=600,
    )
    if not result.ok:
        raise Phase2PrepareError(
            f"1fps frame extract failed: {(result.stderr or '')[-400:]}"
        )
    frames: list[dict[str, Any]] = []
    for i, path in enumerate(sorted(out_dir.glob("frame_*.jpg"))):
        t = i / fps if fps > 0 else float(i)
        frames.append(
            {
                "path": str(path),
                "time": round(t, 3),
                "kind": "fixed_fps",
                "fps": fps,
            }
        )
    log.info("Extracted %d fixed frames @ %.2f fps", len(frames), fps)
    return frames


def extract_scene_frames(
    video_path: Path,
    out_dir: Path,
    env: FFmpegEnvironment,
    *,
    threshold: float = 0.3,
    max_frames: int = 200,
) -> list[dict[str, Any]]:
    """Scene-change protective frames via ffmpeg select=gt(scene\\,T).

    Extract scene frames and capture exact pts_time via showinfo filter so
    every frame contract carries a traceable timecode (never None).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("scene_*.jpg"):
        p.unlink()
    pattern = str(out_dir / "scene_%06d.jpg")
    # showinfo emits pts_time per output frame on stderr for reliable timecodes
    vf = f"select=gt(scene\\,{threshold}),showinfo"
    result = run_command(
        [
            env.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-vsync",
            "vfr",
            "-q:v",
            "3",
            pattern,
        ],
        timeout=600,
    )
    if not result.ok:
        log.warning("Scene frame extract soft-failed: %s", (result.stderr or "")[-200:])
        return []

    # Parse pts_time from showinfo stderr lines
    pts_times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", result.stderr)]

    frames: list[dict[str, Any]] = []
    for i, path in enumerate(sorted(out_dir.glob("scene_*.jpg"))):
        if i >= max_frames:
            path.unlink(missing_ok=True)
            continue
        # Use parsed pts_time if available; fall back to estimation from index
        if i < len(pts_times):
            t = pts_times[i]
        else:
            # Fallback: derive from frame index (bounded by video duration later)
            t = float(i) * 0.5  # rough 0.5s spacing placeholder
        frames.append(
            {
                "path": str(path),
                "time": round(t, 3),
                "kind": "scene",
                "index": i,
            }
        )
    log.info(
        "Extracted %d scene frames (threshold=%.2f, pts_times=%d)",
        len(frames),
        threshold,
        len(pts_times),
    )
    return frames


def map_subtitles_to_window(
    doc: SubtitleDocument,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    out = []
    for s in doc.segments:
        if s.end < start or s.start > end:
            continue
        out.append(s.model_dump(mode="json"))
    return out


def build_visual_batches(
    *,
    doc: SubtitleDocument,
    meta: SubtitleMetadata,
    fixed_frames: list[dict[str, Any]],
    scene_frames: list[dict[str, Any]],
    batch_seconds: float = 20.0,
    user_request: str = "",
    duration: float | None = None,
) -> list[VisualBatch]:
    if duration is None:
        duration = doc.duration or 0.0
        if doc.segments:
            duration = max(duration, max(s.end for s in doc.segments))
        if fixed_frames:
            duration = max(duration, max(float(f.get("time") or 0) for f in fixed_frames))
    if duration <= 0:
        duration = batch_seconds

    batches: list[VisualBatch] = []
    t = 0.0
    idx = 1
    while t < duration - 1e-6:
        end = min(duration, t + batch_seconds)
        frames = [
            f
            for f in fixed_frames
            if f.get("time") is not None and t <= float(f["time"]) < end
        ]
        scene_in_window = [
            f
            for f in scene_frames
            if f.get("time") is not None and t <= float(f["time"]) < end
        ]
        frames = frames + scene_in_window
        subs = map_subtitles_to_window(doc, t, end)
        batches.append(
            VisualBatch(
                batch_id=f"batch_{idx:04d}",
                start=round(t, 3),
                end=round(end, 3),
                frames=frames,
                subtitles=[  # type: ignore[arg-type]
                    __import__("video_analyst.models", fromlist=["SubtitleSegment"]).SubtitleSegment.model_validate(s)
                    for s in subs
                ],
                subtitle_metadata=meta,
                user_request=user_request,
            )
        )
        idx += 1
        t = end
    return batches


def prepare_phase2(
    video_path: Path,
    layout: OutputLayout,
    env: FFmpegEnvironment,
    *,
    cfg: Phase2Config | None = None,
    user_request: str = "",
    resume: bool = True,
) -> Phase2PrepareResult:
    cfg = cfg or Phase2Config()
    if not cfg.enabled:
        raise Phase2PrepareError("Phase 2 preparation disabled in config")

    store = CacheStore(layout)
    doc, meta = load_active_subtitles_for_phase2(layout)

    # Fixed frames
    if resume and any(layout.phase2_fixed_frames_dir.glob("frame_*.jpg")):
        log.info("Resume: fixed frames from cache")
        fixed = []
        for i, path in enumerate(sorted(layout.phase2_fixed_frames_dir.glob("frame_*.jpg"))):
            fixed.append(
                {
                    "path": str(path),
                    "time": round(i / cfg.coarse_fps, 3),
                    "kind": "fixed_fps",
                    "fps": cfg.coarse_fps,
                }
            )
    else:
        fixed = extract_fixed_fps_frames(
            video_path,
            layout.phase2_fixed_frames_dir,
            env,
            fps=cfg.coarse_fps,
        )

    # Scene frames
    scene_ts_file = layout.phase2_scene_frames_dir / "_timestamps.json"
    if resume and any(layout.phase2_scene_frames_dir.glob("scene_*.jpg")):
        log.info("Resume: scene frames from cache")
        if scene_ts_file.is_file():
            ts_map: dict[str, float] = json.loads(scene_ts_file.read_text(encoding="utf-8"))
        else:
            ts_map = {}
        scene = []
        for i, p in enumerate(sorted(layout.phase2_scene_frames_dir.glob("scene_*.jpg"))):
            if i >= cfg.max_scene_frames:
                p.unlink(missing_ok=True)
                continue
            t = ts_map.get(p.name)
            if t is None:
                t = float(i) * 0.5  # rough fallback spacing
            scene.append(
                {"path": str(p), "time": round(t, 3), "kind": "scene", "index": i}
            )
    else:
        scene = extract_scene_frames(
            video_path,
            layout.phase2_scene_frames_dir,
            env,
            threshold=cfg.scene_threshold,
            max_frames=cfg.max_scene_frames,
        )
        # Persist timestamps for reliable resume
        if scene:
            ts_map = {Path(f["path"]).name: f["time"] for f in scene}
            scene_ts_file.write_text(
                json.dumps(ts_map, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    # Subtitle mapping + batches always rebuild if missing (cheap; depends on active subs)
    batches = build_visual_batches(
        doc=doc,
        meta=meta,
        fixed_frames=fixed,
        scene_frames=scene,
        batch_seconds=cfg.batch_seconds,
        user_request=user_request,
        duration=doc.duration,
    )
    mapping = {
        "schema_version": "1.0",
        "active_source": meta.active_source,
        "reviewed": meta.reviewed,
        "needs_review": meta.needs_review,
        "windows": [
            {
                "batch_id": b.batch_id,
                "start": b.start,
                "end": b.end,
                "subtitle_indices": [s.index for s in b.subtitles],
            }
            for b in batches
        ],
    }
    store.write_json(layout.phase2_mapping_json, mapping)
    store.write_json(
        layout.phase2_batches_json,
        {
            "schema_version": "1.0",
            "batches": [b.model_dump(mode="json") for b in batches],
            "subtitle_metadata": meta.model_dump(mode="json"),
            "note": "Prep only — no visual model calls yet",
        },
    )

    return Phase2PrepareResult(
        fixed_frames_dir=str(layout.phase2_fixed_frames_dir),
        scene_frames_dir=str(layout.phase2_scene_frames_dir),
        fixed_frame_count=len(fixed),
        scene_frame_count=len(scene),
        batches_path=str(layout.phase2_batches_json),
        batch_count=len(batches),
        subtitle_metadata=meta,
    )
