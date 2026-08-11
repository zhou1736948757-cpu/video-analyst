"""Deterministic, pixel-only near-duplicate removal for coarse inputs.

The coarse stage must not decide whether a frame is semantically useful.  This
module therefore only compares small grayscale thumbnails and keeps the
selection auditable.  Scene frames and explicitly pinned frames are protected
from removal, as are the first and last candidates.

The implementation deliberately lives outside the refinement deduplication
module: refinement deduplication is timestamp/content based, while this pass
is a visual-input reduction step for the coarse agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

THUMBNAIL_SIZE = 16
DEFAULT_THRESHOLD = 2.0
ALGORITHM_VERSION = "gray-thumbnail-delta-v1"
_TIMESTAMP_TOLERANCE = 1e-3


@dataclass
class DedupeResult:
    """Selection and audit data for one deterministic candidate set."""

    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    threshold: float = DEFAULT_THRESHOLD
    enabled: bool = True
    algorithm_version: str = ALGORITHM_VERSION
    thumbnail_size: int = THUMBNAIL_SIZE

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary (paths are left as strings by callers)."""

        return {
            "enabled": self.enabled,
            "algorithm_version": self.algorithm_version,
            "thumbnail_size": self.thumbnail_size,
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "kept_count": self.kept_count,
            "dropped_count": self.dropped_count,
            "retained_frames": list(self.kept),
            "dropped_candidates": list(self.dropped),
        }


def _timestamp(frame: dict[str, Any]) -> float:
    """Read the common timestamp spellings used by coarse and inspection data."""

    value = frame.get("time")
    if value is None:
        value = frame.get("timestamp")
    if value is None:
        value = frame.get("timestamp_seconds")
    try:
        result = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        result = 0.0
    return result if math.isfinite(result) else 0.0


def _frame_kind(frame: dict[str, Any]) -> str:
    return str(frame.get("kind") or frame.get("reason") or "candidate")


def _is_pinned(frame: dict[str, Any], pinned_timestamps: set[float]) -> bool:
    if bool(frame.get("pinned")):
        return True
    if _frame_kind(frame) in {"pinned", "user_pinned", "transcript-cue"}:
        return True
    timestamp = _timestamp(frame)
    return any(abs(timestamp - pin) <= _TIMESTAMP_TOLERANCE for pin in pinned_timestamps)


def _thumbnail(path: Path, size: int) -> bytes | None:
    """Decode ``path`` as a stable grayscale thumbnail.

    Pillow is already a required runtime dependency for contact sheets.  A
    decode error is fail-open: retaining a frame costs tokens but never loses
    evidence or makes preparation fail.
    """

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            gray = image.convert("L")
            # Resize to an exact square so aspect-ratio differences cannot
            # produce variable byte lengths.  This is intentionally pixel-only.
            gray = gray.resize((size, size), Image.Resampling.BILINEAR)
            return gray.tobytes()
    except Exception:
        return None


def frame_delta(left: bytes | None, right: bytes | None) -> float:
    """Mean absolute grayscale pixel difference in the 0..255 range."""

    if not left or not right or len(left) != len(right):
        return float("inf")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _normalise_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy and stably sort candidates by absolute source timestamp."""

    copied: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item.setdefault("candidate_index", index)
        copied.append(item)
    # The original index is the deterministic tie-breaker.  Do not infer
    # anything from a filename or from the image content.
    copied.sort(key=lambda item: (_timestamp(item), int(item.get("candidate_index", 0))))
    return copied


def deduplicate_frames(
    candidates: Iterable[dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    enabled: bool = True,
    pinned_timestamps: Iterable[float] = (),
    algorithm_version: str = ALGORITHM_VERSION,
    thumbnail_size: int = THUMBNAIL_SIZE,
) -> DedupeResult:
    """Greedily remove near-duplicates against the previous kept frame.

    The function never performs semantic inference and never deletes files.
    Callers can decide whether dropped source files should remain available for
    audit.  Every dropped candidate contains ``compared_to_candidate_index``
    and ``kept_candidate_index`` so the decision is reproducible.
    """

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("dedupe threshold must be a finite non-negative number") from exc
    if not math.isfinite(threshold_value) or threshold_value < 0:
        raise ValueError("dedupe threshold must be a finite non-negative number")
    try:
        thumb_size = int(thumbnail_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("thumbnail_size must be a positive integer") from exc
    if thumb_size <= 0:
        raise ValueError("thumbnail_size must be a positive integer")

    items = _normalise_candidates(candidates)
    result = DedupeResult(
        candidate_count=len(items),
        threshold=threshold_value,
        enabled=bool(enabled),
        algorithm_version=str(algorithm_version),
        thumbnail_size=thumb_size,
    )
    if not items:
        return result

    pinned = set()
    for value in pinned_timestamps:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            pinned.add(numeric)

    # Precompute thumbnails once; this keeps the pass deterministic and avoids
    # decoding the same image repeatedly when a held frame follows a held frame.
    thumbnails = {
        int(item["candidate_index"]): _thumbnail(Path(str(item["path"])), thumb_size)
        for item in items
        if item.get("path")
    }

    kept: list[dict[str, Any]] = []
    last_kept_thumbnail: bytes | None = None
    last_kept_index: int | None = None

    for position, candidate in enumerate(items):
        candidate_index = int(candidate["candidate_index"])
        timestamp = _timestamp(candidate)
        force_reasons: list[str] = []
        if position == 0:
            force_reasons.append("first")
        if position == len(items) - 1:
            force_reasons.append("last")
        if _frame_kind(candidate) in {"scene", "scene_change", "scene-change"} or bool(
            candidate.get("scene_change")
        ):
            force_reasons.append("scene_change")
        if _is_pinned(candidate, pinned):
            force_reasons.append("user_pinned")

        current_thumbnail = thumbnails.get(candidate_index)
        delta = frame_delta(current_thumbnail, last_kept_thumbnail)
        is_near_duplicate = (
            enabled
            and last_kept_index is not None
            and delta <= threshold_value
        )

        if not enabled or not is_near_duplicate or force_reasons:
            item = dict(candidate)
            item["timestamp"] = round(timestamp, 6)
            item["retained_reason"] = force_reasons or ["pixel_change"]
            item["dedupe_delta"] = None if last_kept_index is None else round(delta, 6)
            kept.append(item)
            last_kept_index = candidate_index
            last_kept_thumbnail = current_thumbnail
            continue

        # The candidate is dropped only because of a pixel delta.  Keep its
        # source information in the report; do not make a semantic claim.
        result.dropped.append(
            {
                "candidate_index": candidate_index,
                "timestamp": round(timestamp, 6),
                "path": str(candidate.get("path") or ""),
                "reason": "near_duplicate_pixel_delta",
                "delta": round(delta, 6),
                "compared_to_candidate_index": last_kept_index,
                "kept_candidate_index": last_kept_index,
            }
        )

    result.kept = kept
    return result


# Aliases make the low-level algorithm easy to discover for callers and tests.
dedupe_coarse_frames = deduplicate_frames
dedupe_perceptual = deduplicate_frames


__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_THRESHOLD",
    "DedupeResult",
    "THUMBNAIL_SIZE",
    "dedupe_coarse_frames",
    "dedupe_perceptual",
    "deduplicate_frames",
    "frame_delta",
]
