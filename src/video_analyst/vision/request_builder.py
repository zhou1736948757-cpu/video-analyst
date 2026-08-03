"""Request builder for vision batch analysis.

Reads from the existing Phase 2 prep output (visual_batches.json) and the
active subtitle review package, NEVER from raw asr/llm subtitle files.

Outputs:
- VisionBatchRequest (unified)
- VisionBatchSelection (audit trail of which frames were selected)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.exceptions import Phase2PrepareError, SubtitleReviewError
from video_analyst.models import (
    SubtitleReviewPackage,
    SubtitleSegment,
    SubtitleVersion,
)
from video_analyst.vision.models import (
    SCHEMA_VERSION,
    PreviousBatchContext,
    SubtitleContext,
    SubtitleMetadata,
    VisionBatchRequest,
    VisionBatchSelection,
    VisionFrame,
)
from video_analyst.vision.prompts import DEFAULT_ANALYSIS_DIMENSIONS


def _timecode(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _read_visual_batches(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase2PrepareError(
            f"visual_batches.json missing at {path}; run 'video-analyst analyze' first"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Phase2PrepareError(f"Failed to read {path}: {exc}") from exc


def _read_subtitle_metadata(layout: OutputLayout) -> tuple[SubtitleMetadata, SubtitleReviewPackage | None]:
    """Load subtitle metadata from review.json (active source).

    Falls back to active subtitles.json only when review.json is missing.
    Never reads subtitles_asr.json or subtitles_llm.json for this metadata.
    """
    cache = CacheStore(layout)
    if layout.review_json.is_file():
        try:
            from video_analyst.llm.review import load_review

            pkg = load_review(layout)
        except Exception as exc:
            raise SubtitleReviewError(f"Failed to load review.json: {exc}") from exc
        meta = SubtitleMetadata(
            active_source=pkg.active.value,
            reviewed=pkg.reviewed,
            needs_review=pkg.needs_review,
            asr_model=pkg.asr_meta.model if pkg.asr_meta else None,
            calibration_model=(
                pkg.llm_meta.model if pkg.llm_meta and pkg.llm_meta.enabled else None
            ),
        )
        return meta, pkg
    # No review package — read active subtitles.json to learn active_source
    if layout.subtitles_json.is_file():
        try:
            doc = json.loads(layout.subtitles_json.read_text(encoding="utf-8"))
            origin = str(doc.get("origin") or "asr")
            if origin == "active":
                origin = "asr"
            meta = SubtitleMetadata(active_source=origin)
            return meta, None
        except Exception as exc:
            raise SubtitleReviewError(f"Failed to read subtitles.json: {exc}") from exc
    raise SubtitleReviewError("No review.json or active subtitles.json found")


def _frame_id_for(batch_id: str, kind: str, ts: float) -> str:
    return f"{batch_id}__{kind}__{int(round(ts * 1000)):08d}"


def _select_frames_for_batch(
    *,
    batch_id: str,
    start: float,
    end: float,
    raw_frames: list[dict[str, Any]],
    max_frames: int,
) -> tuple[list[VisionFrame], VisionBatchSelection]:
    """Select frames for one batch, preserving scene-protection frames and
    capping to max_frames while preserving time-order."""
    if max_frames <= 0:
        max_frames = 1

    # Pull candidate frames inside window
    candidates: list[dict[str, Any]] = []
    for f in raw_frames:
        t = f.get("time")
        if t is None:
            continue
        try:
            t_f = float(t)
        except (TypeError, ValueError):
            continue
        if t_f < start or t_f >= end:
            continue
        candidates.append(f)

    # Sort by time
    candidates.sort(key=lambda f: float(f["time"]))

    # Split fixed vs scene-protection
    fixed = [f for f in candidates if f.get("kind") == "fixed_fps"]
    scenes = [f for f in candidates if f.get("kind") == "scene"]

    # Build VisionFrame objects for scenes first
    selected: list[VisionFrame] = []
    seen_ids: set[str] = set()
    preserved_scene_ids: list[str] = []

    for f in scenes:
        t = float(f["time"])
        path = Path(f["path"])
        fid = _frame_id_for(batch_id, "scene", t)
        if fid in seen_ids:
            continue
        selected.append(
            VisionFrame(
                frame_id=fid,
                timestamp_seconds=t,
                timecode=_timecode(t),
                path=path,
                sources=["scene"],
                scene_score=f.get("score"),
            )
        )
        seen_ids.add(fid)
        preserved_scene_ids.append(fid)

    # If we are still under the budget, add fixed frames in order
    for f in fixed:
        t = float(f["time"])
        path = Path(f["path"])
        fid = _frame_id_for(batch_id, "fixed", t)
        if fid in seen_ids:
            continue
        if len(selected) >= max_frames:
            break
        selected.append(
            VisionFrame(
                frame_id=fid,
                timestamp_seconds=t,
                timecode=_timecode(t),
                path=path,
                sources=["fixed_fps"],
                scene_score=None,
            )
        )
        seen_ids.add(fid)

    # If still under budget but we skipped fixed frames to keep under cap, do
    # time-uniform sampling of remaining fixed frames.
    if len(selected) < max_frames:
        remaining = [f for f in fixed if _frame_id_for(batch_id, "fixed", float(f["time"])) not in seen_ids]
        if remaining:
            step = max(1, len(remaining) // (max_frames - len(selected)))
            for i in range(0, len(remaining), step):
                f = remaining[i]
                t = float(f["time"])
                fid = _frame_id_for(batch_id, "fixed", t)
                if fid in seen_ids or len(selected) >= max_frames:
                    continue
                selected.append(
                    VisionFrame(
                        frame_id=fid,
                        timestamp_seconds=t,
                        timecode=_timecode(t),
                        path=Path(f["path"]),
                        sources=["fixed_fps"],
                        scene_score=None,
                    )
                )
                seen_ids.add(fid)

    selected.sort(key=lambda vf: vf.timestamp_seconds)

    available_ids = {_frame_id_for(batch_id, "fixed", float(f["time"])) for f in fixed}
    available_ids.update(_frame_id_for(batch_id, "scene", float(f["time"])) for f in scenes)
    selected_ids = {vf.frame_id for vf in selected}
    omitted = sorted(available_ids - selected_ids)

    selection = VisionBatchSelection(
        batch_id=batch_id,
        total_available=len(available_ids),
        selected=len(selected),
        omitted_frame_ids=omitted,
        preserved_scene_frame_ids=preserved_scene_ids,
    )
    return selected, selection


def _subtitle_segments_for_window(
    pkg: SubtitleReviewPackage | None,
    segments_iter: Iterable[SubtitleSegment],
    start: float,
    end: float,
) -> list[SubtitleContext]:
    """Return subtitle context entries overlapping [start, end).

    Only uses the active document. Never reads raw asr/llm files.
    """
    out: list[SubtitleContext] = []
    if pkg is not None:
        doc = pkg.active_document()
        source_iter = doc.segments
    else:
        source_iter = list(segments_iter)

    for s in source_iter:
        try:
            seg_start = float(s.start)
            seg_end = float(s.end)
        except (TypeError, ValueError):
            continue
        if seg_end < start or seg_start > end:
            continue
        text = (s.text or "").strip()
        if not text:
            continue
        out.append(
            SubtitleContext(
                index=int(getattr(s, "index", 0) or 0),
                start=seg_start,
                end=seg_end,
                text=text,
                language=getattr(s, "language", None),
            )
        )
    return out


def build_request_for_batch(
    batch_index: int,
    layout: OutputLayout,
    *,
    user_request: str,
    video_id: str,
    max_frames: int,
    analysis_dimensions: list[str] | None = None,
    previous_context: PreviousBatchContext | None = None,
    active_subtitle_version: SubtitleVersion | str | None = None,
) -> tuple[VisionBatchRequest, VisionBatchSelection]:
    """Build a VisionBatchRequest for the batch at batch_index (0-based).

    Reads:
    - layout.phase2_batches_json (Phase 2 prep output)
    - layout.review.json / layout.subtitles.json (active subtitle source)
    """
    raw = _read_visual_batches(layout.phase2_batches_json)
    batches = raw.get("batches") or []
    if batch_index < 0 or batch_index >= len(batches):
        raise Phase2PrepareError(
            f"batch_index {batch_index} out of range (0..{len(batches) - 1})"
        )

    batch = batches[batch_index]
    batch_id = str(batch.get("batch_id"))
    start = float(batch.get("start"))
    end = float(batch.get("end"))

    meta, pkg = _read_subtitle_metadata(layout)
    if active_subtitle_version is not None:
        # Allow override (e.g. force ASR view)
        override = str(active_subtitle_version)
        if override in ("asr", "llm"):
            meta.active_source = override  # type: ignore[assignment]

    raw_frames = batch.get("frames") or []
    selected, selection = _select_frames_for_batch(
        batch_id=batch_id,
        start=start,
        end=end,
        raw_frames=raw_frames,
        max_frames=max_frames,
    )

    segments_in_batch = [
        SubtitleSegment.model_validate(s) for s in (batch.get("subtitles") or [])
    ]
    subtitles = _subtitle_segments_for_window(pkg, segments_in_batch, start, end)

    request = VisionBatchRequest(
        schema_version=SCHEMA_VERSION,
        batch_id=batch_id,
        video_id=video_id,
        batch_start=start,
        batch_end=end,
        frames=selected,
        subtitles=subtitles,
        subtitle_metadata=meta,
        user_request=user_request or "",
        analysis_dimensions=list(analysis_dimensions) if analysis_dimensions else list(DEFAULT_ANALYSIS_DIMENSIONS),
        previous_context=previous_context,
    )
    return request, selection


def build_all_batch_requests(
    layout: OutputLayout,
    *,
    user_request: str,
    video_id: str,
    max_frames: int,
    analysis_dimensions: list[str] | None = None,
) -> list[tuple[VisionBatchRequest, VisionBatchSelection]]:
    """Build all batch requests in time order."""
    raw = _read_visual_batches(layout.phase2_batches_json)
    batches = raw.get("batches") or []
    out: list[tuple[VisionBatchRequest, VisionBatchSelection]] = []
    for i in range(len(batches)):
        req, sel = build_request_for_batch(
            i,
            layout,
            user_request=user_request,
            video_id=video_id,
            max_frames=max_frames,
            analysis_dimensions=analysis_dimensions,
        )
        out.append((req, sel))
    return out


__all__ = [
    "build_all_batch_requests",
    "build_request_for_batch",
]