"""Deterministic deduplication of fine-grained frames.

Two deduplication dimensions:
1. **Intra-interval** — same interval, two frames at nearly-identical
   timestamps. We keep the earliest (lowest `local_index`) record whose
   timestamp and content both match.
2. **Cross-interval** — two intervals may produce frames at the same
   absolute timestamp (overlap boundary, FP rounding, plan re-runs). We
   deduplicate by **absolute timestamp ± threshold** AND sha256 equality.
   Both intervals may keep a reference, but the physical file is
   referenced via a canonical alias.

The function returns:
- a list of canonical frames (one per unique absolute timestamp)
- an alias map: ``(interval_id, frame_id) -> canonical_frame_id``

Determinism: input order is preserved (sort by interval then local_index);
ties on time+sha are broken by source order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from video_analyst.refinement.extractor import ExtractedFrame


@dataclass
class DeduplicatedFrame:
    interval_id: str
    local_index: int
    frame_id: str
    path: object  # Path; kept loose to avoid coupling
    timestamp_seconds: float
    relative_timestamp_seconds: float
    requested_fps: int
    width: int
    height: int
    size_bytes: int
    sha256: str
    source_video_sha256: str = ""

    @property
    def is_detail(self) -> bool:
        return False


@dataclass
class DeduplicationReport:
    canonical_frames: list[DeduplicatedFrame] = field(default_factory=list)
    intra_dropped: list[str] = field(default_factory=list)  # frame_ids
    cross_aliases: dict[tuple[str, str], str] = field(
        default_factory=dict
    )  # (interval_id, frame_id) -> canonical frame_id
    stats: dict[str, int] = field(default_factory=dict)


def _frame_id(interval_id: str, t: float) -> str:
    return f"{interval_id}__f{int(round(t * 1000)):08d}"


def _frame_id_to_dedup_frame(
    f: ExtractedFrame,
    *,
    interval_id: str,
    source_video_sha256: str = "",
) -> DeduplicatedFrame:
    return DeduplicatedFrame(
        interval_id=interval_id,
        local_index=-1,  # caller assigns
        frame_id=_frame_id(interval_id, f.absolute_timestamp),
        path=f.path,
        timestamp_seconds=round(f.absolute_timestamp, 6),
        relative_timestamp_seconds=round(f.relative_timestamp, 6),
        requested_fps=int(0),  # caller fills
        width=int(f.width),
        height=int(f.height),
        size_bytes=int(f.size_bytes),
        sha256=f.sha256,
        source_video_sha256=source_video_sha256,
    )


def deduplicate_intervals(
    interval_frames: dict[str, list[ExtractedFrame]],
    *,
    time_threshold_seconds: float = 0.005,
    requested_fps_by_interval: dict[str, int] | None = None,
    source_video_sha256: str = "",
) -> DeduplicationReport:
    """Run intra- then cross-interval dedup.

    `interval_frames`: ``interval_id -> [ExtractedFrame]`` already in
    time-sorted order. ``requested_fps_by_interval`` provides the FPS
    label per interval.
    """
    fps_map = requested_fps_by_interval or {}
    report = DeduplicationReport()

    # ---- 1) intra-interval dedup: keep earliest on near-time + same sha ----
    intra_canonicals: list[DeduplicatedFrame] = []
    for interval_id, frames in interval_frames.items():
        kept: list[DeduplicatedFrame] = []
        for f in frames:
            df = _frame_id_to_dedup_frame(
                f, interval_id=interval_id, source_video_sha256=source_video_sha256
            )
            df.requested_fps = int(fps_map.get(interval_id, 0))
            # Compare against last kept
            duplicate = False
            for k in kept:
                if (
                    abs(k.timestamp_seconds - df.timestamp_seconds)
                    <= time_threshold_seconds
                    and k.sha256 == df.sha256
                ):
                    duplicate = True
                    report.intra_dropped.append(df.frame_id)
                    break
            if not duplicate:
                kept.append(df)
        # Re-assign local_index based on kept order
        for i, k in enumerate(kept):
            k.local_index = i
        intra_canonicals.extend(kept)

    # ---- 2) cross-interval dedup: same interval_id+sha+timestamp aliases ----
    intra_canonicals.sort(key=lambda d: (d.timestamp_seconds, d.interval_id, d.sha256))
    final_canonicals: list[DeduplicatedFrame] = []
    for df in intra_canonicals:
        alias_target: str | None = None
        for c in final_canonicals:
            if (
                abs(c.timestamp_seconds - df.timestamp_seconds)
                <= time_threshold_seconds
                and c.sha256 == df.sha256
            ):
                alias_target = c.frame_id
                break
        if alias_target is None:
            final_canonicals.append(df)
        else:
            report.cross_aliases[(df.interval_id, df.frame_id)] = alias_target

    report.canonical_frames = final_canonicals
    report.stats = {
        "input_frames": sum(len(v) for v in interval_frames.values()),
        "after_intra": len(intra_canonicals),
        "after_cross": len(final_canonicals),
        "intra_dropped": len(report.intra_dropped),
        "cross_aliased": len(report.cross_aliases),
    }
    return report


__all__ = [
    "DeduplicatedFrame",
    "DeduplicationReport",
    "deduplicate_intervals",
]