"""Build per-interval FineVisualBatch and the global fine_batches index.

Subtitles are sourced ONLY from active subtitles.json + review.json.
NEVER from subtitles_asr.json / subtitles_llm.json.

The fine batch is the only contract Epic 4 will read. It includes:
- the original interval
- canonical frame records (with sha256)
- contact sheet refs
- detail frame refs
- subtitles overlapping the interval ± context window
- subtitle metadata
- user request / refinement questions
- coarse-context pointer to source_batch_ids
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import RefinementConfig
from video_analyst.exceptions import (
    FineBatchBuildError,
    SubtitleReviewError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.deduplicate import DeduplicatedFrame
from video_analyst.refinement.models import (
    DetailFrameReference,
    EPIC3_SCHEMA_VERSION,
    FineBatchesIndex,
    FineVisualBatch,
    RefinementFrame,
    RefinementSource,
    TemporalContactSheet,
)
from video_analyst.vision.models import SubtitleContext, SubtitleMetadata

log = get_logger("refinement.batch_builder")


def load_active_subtitles_for_interval(
    layout: OutputLayout,
    *,
    interval_start: float,
    interval_end: float,
    context_before: float,
    context_after: float,
) -> tuple[list[SubtitleContext], SubtitleMetadata]:
    """Read active subtitle document + review metadata, slice to window.

    Never reads subtitles_asr.* or subtitles_llm.* directly.
    """
    cache = CacheStore(layout)
    meta = SubtitleMetadata(active_source="asr")
    pkg = None
    if layout.review_json.is_file():
        try:
            from video_analyst.llm.review import load_review

            pkg = load_review(layout)
            meta = SubtitleMetadata(
                active_source=pkg.active.value,
                reviewed=pkg.reviewed,
                needs_review=pkg.needs_review,
                asr_model=pkg.asr_meta.model if pkg.asr_meta else None,
                calibration_model=(
                    pkg.llm_meta.model if pkg.llm_meta and pkg.llm_meta.enabled else None
                ),
            )
        except SubtitleReviewError as exc:
            log.warning("could not load review.json for refinement subtitles: %s", exc)
    elif layout.subtitles_json.is_file():
        try:
            doc = json.loads(layout.subtitles_json.read_text(encoding="utf-8"))
            meta = SubtitleMetadata(active_source=str(doc.get("origin") or "asr"))
        except Exception as exc:
            log.warning("could not read subtitles.json: %s", exc)
    else:
        # No active subtitles; return empty
        return [], meta

    if pkg is not None:
        doc = pkg.active_document()
    else:
        try:
            from video_analyst.models import SubtitleDocument

            doc = SubtitleDocument.model_validate(
                json.loads(layout.subtitles_json.read_text(encoding="utf-8"))
            )
        except Exception:
            doc = None

    if doc is None:
        return [], meta

    win_start = max(0.0, interval_start - max(0.0, context_before))
    win_end = interval_end + max(0.0, context_after)
    out: list[SubtitleContext] = []
    for s in doc.segments:
        try:
            seg_start = float(s.start)
            seg_end = float(s.end)
        except (TypeError, ValueError):
            continue
        if seg_end < win_start or seg_start > win_end:
            continue
        text = (s.text or "").strip()
        if not text:
            continue
        out.append(
            SubtitleContext(
                index=int(getattr(s, "index", 0) or 0),
                start=round(seg_start, 3),
                end=round(seg_end, 3),
                text=text,
                language=getattr(s, "language", None),
            )
        )
    return out, meta


def _frame_record(d: DeduplicatedFrame, is_detail: bool = False, *, fallback_index: int = 0) -> RefinementFrame:
    # DeduplicatedFrame may carry local_index=-1 when constructed directly
    # (e.g. in tests or before intra-dedup re-assignment). RefinementFrame
    # requires local_index >= 0, so fall back to the caller-supplied index.
    local_index = d.local_index if d.local_index >= 0 else max(0, fallback_index)
    return RefinementFrame.make(
        interval_id=d.interval_id,
        local_index=local_index,
        timestamp_seconds=d.timestamp_seconds,
        relative_timestamp_seconds=d.relative_timestamp_seconds,
        requested_fps=d.requested_fps,  # type: ignore[arg-type]
        path=d.path,  # type: ignore[arg-type]
        width=d.width,
        height=d.height,
        size_bytes=d.size_bytes,
        sha256=d.sha256,
        source_video_sha256=d.source_video_sha256,
        is_detail=is_detail,
        frame_id=d.frame_id,
    )


def build_fine_batch(
    *,
    interval: RefinementSource,
    frames: list[DeduplicatedFrame],
    contact_sheets: list[TemporalContactSheet],
    detail_frames: list[DetailFrameReference],
    layout: OutputLayout,
    cfg: RefinementConfig,
    user_request: str,
    refinement_questions: list[str] | None = None,
    coarse_analysis_paths: dict[str, str] | None = None,
) -> FineVisualBatch:
    """Assemble one FineVisualBatch."""
    subtitles, meta = load_active_subtitles_for_interval(
        layout,
        interval_start=interval.start,
        interval_end=interval.end,
        context_before=cfg.subtitle_context_before,
        context_after=cfg.subtitle_context_after,
    )
    detail_ids = {df.frame_id for df in detail_frames}

    frame_records: list[RefinementFrame] = []
    for i, d in enumerate(frames):
        frame_records.append(
            _frame_record(d, is_detail=d.frame_id in detail_ids, fallback_index=i)
        )

    return FineVisualBatch(
        schema_version=EPIC3_SCHEMA_VERSION,
        fine_batch_id=f"fine_{interval.interval_id}",
        interval=interval,
        frames=frame_records,
        contact_sheets=list(contact_sheets),
        detail_frames=list(detail_frames),
        subtitles=subtitles,
        subtitle_metadata=meta,
        user_request=user_request or "",
        refinement_questions=list(refinement_questions or interval.questions),
        coarse_context={
            "source_batch_ids": list(interval.source_batch_ids),
            "evidence_frame_ids": list(interval.evidence_frame_ids),
            "coarse_analysis_paths": dict(coarse_analysis_paths or {}),
        },
    )


def persist_fine_batch(
    layout: OutputLayout,
    batch: FineVisualBatch,
) -> Path:
    """Atomic write of a single fine batch."""
    cache = CacheStore(layout)
    path = layout.refinement_batches_dir / f"{batch.fine_batch_id}.json"
    cache.write_json(path, batch.model_dump(mode="json"))
    return path


def build_fine_batches_index(
    batches: Iterable[FineVisualBatch],
    *,
    video_id: str = "video",
) -> FineBatchesIndex:
    items = list(batches)
    return FineBatchesIndex(
        schema_version=EPIC3_SCHEMA_VERSION,
        video_id=video_id,
        total_batches=len(items),
        batches=items,
    )


def persist_fine_batches_index(layout: OutputLayout, index: FineBatchesIndex) -> Path:
    cache = CacheStore(layout)
    cache.write_json(layout.refinement_fine_batches_json, index.model_dump(mode="json"))
    return layout.refinement_fine_batches_json


def load_fine_batches_index(layout: OutputLayout) -> FineBatchesIndex | None:
    if not layout.refinement_fine_batches_json.is_file():
        return None
    try:
        data = json.loads(layout.refinement_fine_batches_json.read_text(encoding="utf-8"))
        return FineBatchesIndex.model_validate(data)
    except Exception as exc:
        log.warning("could not load fine_batches.json: %s", exc)
        return None


__all__ = [
    "build_fine_batch",
    "build_fine_batches_index",
    "load_active_subtitles_for_interval",
    "load_fine_batches_index",
    "persist_fine_batch",
    "persist_fine_batches_index",
]