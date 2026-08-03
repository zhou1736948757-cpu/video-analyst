"""Epic 3 manifest, cache key construction, and interval persistence.

Cache signatures (per spec):
- Extraction signature: source_video_sha, interval start/end, FPS, FFmpeg
  version, scale (max_frame_long_edge), image format/quality, timestamp
  parser version, schema version.
- Presentation signature: per-interval frame sha-256 set,
  frames_per_sheet, overlap_frames, sheet columns, label format version,
  subtitle hash, review hash, user_request, refinement_questions,
  schema version.

Cache invalidation table:
- source video change     → invalidate everything
- interval start/end/FPS  → invalidate only that interval's extraction +
                             downstream
- contact-sheet layout    → invalidate only contact sheets + fine batch
- active subtitle source  → invalidate subtitles + fine batch
- user_request/questions  → invalidate fine batch only
- detail-frame rules      → invalidate detail frames + fine batch
- deferred intervals      → no effect on already-processed intervals
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import RefinementConfig
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.deduplicate import DeduplicatedFrame
from video_analyst.refinement.models import (
    EPIC3_SCHEMA_VERSION,
    Epic3Manifest,
    IntervalArtifact,
    IntervalManifestEntry,
    RefinementSource,
)
from video_analyst.util.hashing import sha256_file, sha256_json

log = get_logger("refinement.manifest")


def _read_subtitle_hash(layout: OutputLayout) -> str:
    """Hash of the active subtitle sources (subs.json + review.json)."""
    parts: list[str] = []
    for p in (layout.subtitles_json, layout.review_json):
        if p.is_file():
            try:
                parts.append(sha256_file(p))
            except Exception:
                parts.append(f"missing:{p}")
    return sha256_json(parts)


def build_extraction_signature(
    *,
    interval: RefinementSource,
    source_video_sha: str,
    ffmpeg_version: str,
    cfg: RefinementConfig,
) -> str:
    payload = {
        "source_video_sha": source_video_sha,
        "interval_id": interval.interval_id,
        "start": interval.start,
        "end": interval.end,
        "requested_fps": interval.requested_fps,
        "ffmpeg_version": ffmpeg_version,
        "max_frame_long_edge": cfg.max_frame_long_edge,
        "image_format": cfg.image_format,
        "jpeg_quality": cfg.jpeg_quality,
        "timestamp_parser_version": cfg.timestamp_parser_version,
        "schema_version": cfg.schema_version,
    }
    return sha256_json(payload)


def build_presentation_signature(
    *,
    frames: list[DeduplicatedFrame],
    cfg: RefinementConfig,
    user_request: str,
    refinement_questions: list[str],
    layout: OutputLayout,
) -> str:
    payload = {
        "frame_sha256s": [f.sha256 for f in frames],
        "frames_per_sheet": cfg.frames_per_sheet,
        "sheet_overlap_frames": cfg.sheet_overlap_frames,
        "sheet_columns": cfg.sheet_columns,
        "contact_sheet_max_edge": cfg.contact_sheet_max_edge,
        "contact_sheet_label_format_version": cfg.contact_sheet_label_format_version,
        "sheets_enabled": cfg.sheets_enabled,
        "no_contact_sheets": cfg.no_contact_sheets,
        "subtitle_hash": _read_subtitle_hash(layout),
        "subtitle_context_before": cfg.subtitle_context_before,
        "subtitle_context_after": cfg.subtitle_context_after,
        "user_request": user_request,
        "refinement_questions": list(refinement_questions),
        "schema_version": cfg.schema_version,
    }
    return sha256_json(payload)


# ---- Manifest load / save ----


def load_manifest(layout: OutputLayout) -> Epic3Manifest | None:
    if not layout.refinement_manifest_json.is_file():
        return None
    try:
        data = json.loads(layout.refinement_manifest_json.read_text(encoding="utf-8"))
        return Epic3Manifest.model_validate(data)
    except Exception as exc:
        log.warning("could not load Epic3 manifest: %s", exc)
        return None


def save_manifest(layout: OutputLayout, manifest: Epic3Manifest) -> None:
    cache = CacheStore(layout)
    cache.write_json(
        layout.refinement_manifest_json, manifest.model_dump(mode="json")
    )


# ---- Interval artifact ----


def persist_interval_artifact(
    layout: OutputLayout,
    artifact: IntervalArtifact,
) -> Path:
    cache = CacheStore(layout)
    path = layout.refinement_intervals_dir / f"{artifact.interval_id}.json"
    cache.write_json(path, artifact.model_dump(mode="json"))
    return path


def load_interval_artifact(
    layout: OutputLayout, interval_id: str
) -> IntervalArtifact | None:
    path = layout.refinement_intervals_dir / f"{interval_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return IntervalArtifact.model_validate(data)
    except Exception as exc:
        log.warning("could not load interval artifact %s: %s", interval_id, exc)
        return None


def upsert_manifest_entry(
    manifest: Epic3Manifest,
    entry: IntervalManifestEntry,
) -> Epic3Manifest:
    """Replace or insert a manifest entry by interval_id."""
    new_intervals = [
        e for e in manifest.intervals if e.interval_id != entry.interval_id
    ]
    new_intervals.append(entry)
    new_intervals.sort(key=lambda e: e.interval_id)
    return manifest.model_copy(update={"intervals": new_intervals})


def _interval_id_to_path(layout: OutputLayout, interval_id: str) -> Path:
    return layout.refinement_intervals_dir / f"{interval_id}.json"


__all__ = [
    "build_extraction_signature",
    "build_presentation_signature",
    "load_interval_artifact",
    "load_manifest",
    "persist_interval_artifact",
    "save_manifest",
    "upsert_manifest_entry",
]