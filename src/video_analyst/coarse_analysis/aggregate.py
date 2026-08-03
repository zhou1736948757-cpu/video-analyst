"""Full-video coarse structure aggregation.

Strategy:
- Take all successful per-batch analyses in time order.
- Each batch becomes (at least) one VideoSection.
- Adjacent batches with very similar summary/subject lists are merged.
- Failures produce explicit VideoGap entries — never silently treated as
  empty content.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.coarse_analysis.batch_runner import coarse_output_paths, ensure_coarse_dirs
from video_analyst.coarse_analysis.models import (
    SectionTimeRange,
    VideoGap,
    VideoSection,
    VideoStructure,
)
from video_analyst.exceptions import CoarseAggregationError
from video_analyst.vision.models import CoarseBatchAnalysis, Observation

_SECTION_MERGE_OVERLAP = 0.0  # batch windows already contiguous
_SIMILARITY_THRESHOLD = 0.55  # word overlap ratio for "same scene"


def _observation_word_set(observations: Iterable[Observation]) -> set[str]:
    words: set[str] = set()
    for o in observations:
        for w in o.description.lower().split():
            if len(w) > 2:
                words.add(w.strip(",.;:!?\"'()[]{}"))
    return words


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def aggregate_structure(
    *,
    analyses: dict[str, CoarseBatchAnalysis],
    failed_batches: dict[str, str],
    video_start: float,
    video_end: float,
    batches_order: list[str],
    batches_meta: dict[str, dict[str, Any]] | None = None,
) -> VideoStructure:
    """Deterministic, LLM-free aggregation."""
    if video_end <= video_start:
        raise CoarseAggregationError(
            f"invalid video duration: {video_end - video_start}"
        )

    failed_order = [bid for bid in batches_order if bid in failed_batches]
    # Sort by start time for failed batch gap computation
    def _batch_start(bid: str) -> float:
        meta = (batches_meta or {}).get(bid) or {}
        return float(meta.get("start", 0.0))

    failed_sorted = sorted(failed_order, key=_batch_start)

    sections: list[VideoSection] = []
    gaps: list[VideoGap] = []
    recurring_subject_counter: Counter[str] = Counter()
    recurring_patterns_counter: Counter[str] = Counter()
    subtitle_visual_notes: list[str] = []
    unresolved: list[str] = []
    summaries: list[str] = []
    important_moments: list[SectionTimeRange] = []

    # Build sections from successful batches in time order
    last_section: VideoSection | None = None
    for bid in batches_order:
        a = analyses.get(bid)
        meta = (batches_meta or {}).get(bid) or {}
        b_start = float(meta.get("start", a.start if a else 0.0))
        b_end = float(meta.get("end", a.end if a else 0.0))
        if a is None:
            # failed batch — emit a gap and skip section building
            gaps.append(
                VideoGap(
                    start=b_start,
                    end=b_end,
                    failed_batch_ids=[bid],
                    reason=failed_batches.get(bid, "unknown"),
                )
            )
            last_section = None
            continue

        # Add subjects / patterns to global counters
        for subj in a.subjects:
            for w in subj.description.lower().split():
                if len(w) > 2:
                    recurring_subject_counter[w.strip(",.;:!?\"'()[]{}")] += 1
        for pat in a.scene_changes + a.camera_observations:
            for w in pat.description.lower().split():
                if len(w) > 3:
                    recurring_patterns_counter[w.strip(",.;:!?\"'()[]{}")] += 1

        # Subtitle-visual notes
        for sv in a.subtitle_visual_relationship:
            if sv.description:
                subtitle_visual_notes.append(
                    f"[{a.batch_id}] {sv.description}"
                )

        # Important moments
        for imp in a.important_moments:
            important_moments.append(
                SectionTimeRange(start=imp.time_range.start, end=imp.time_range.end)
            )

        # Decide whether to merge with previous section
        candidates_words = _observation_word_set(a.subjects + a.scene_changes + a.camera_observations)
        if last_section is not None and candidates_words:
            prev_words = (
                last_section._cached_words  # type: ignore[attr-defined]
                if hasattr(last_section, "_cached_words")
                else None
            )
            if prev_words is None:
                # Reconstruct from current section
                prev_words = candidates_words
            sim = _jaccard(prev_words, candidates_words)
            if sim >= _SIMILARITY_THRESHOLD and last_section.end >= b_start - 1e-3:
                # Merge
                merged_end = max(last_section.end, b_end)
                merged_conf = (last_section.confidence + a.confidence) / 2.0
                merged_section = VideoSection(
                    section_id=last_section.section_id,
                    start=last_section.start,
                    end=merged_end,
                    title=last_section.title,
                    summary=(last_section.summary + " " + (a.summary or "")).strip(),
                    source_batch_ids=list(
                        dict.fromkeys(last_section.source_batch_ids + [bid])
                    ),
                    important_moments=last_section.important_moments,
                    confidence=round(merged_conf, 3),
                )
                # Cache words on merged section for next iteration
                merged_section._cached_words = prev_words | candidates_words  # type: ignore[attr-defined]
                sections.pop()
                sections.append(merged_section)
                last_section = merged_section
                continue

        # New section
        title_seed = (a.summary or a.subjects[0].description if a.subjects else a.batch_id)
        title = _truncate_title(title_seed)
        section = VideoSection(
            section_id=f"section_{len(sections) + 1:04d}",
            start=b_start,
            end=b_end,
            title=title,
            summary=a.summary or "",
            source_batch_ids=[bid],
            important_moments=[
                SectionTimeRange(start=imp.time_range.start, end=imp.time_range.end)
                for imp in a.important_moments
            ],
            confidence=a.confidence,
        )
        section._cached_words = candidates_words  # type: ignore[attr-defined]
        sections.append(section)
        last_section = section
        summaries.append(a.summary)

    # Merge adjacent gaps into contiguous ranges
    merged_gaps: list[VideoGap] = []
    for gap in sorted(gaps, key=lambda g: g.start):
        if merged_gaps and merged_gaps[-1].end >= gap.start - 1e-3:
            prev = merged_gaps[-1]
            merged_gaps[-1] = VideoGap(
                start=prev.start,
                end=max(prev.end, gap.end),
                failed_batch_ids=prev.failed_batch_ids + gap.failed_batch_ids,
                reason="merged_failed_batches",
            )
        else:
            merged_gaps.append(gap)

    overall_summary = _build_overall_summary(sections, summaries, merged_gaps)
    recurring_subjects = [w for w, _ in recurring_subject_counter.most_common(20)]
    recurring_patterns = [w for w, _ in recurring_patterns_counter.most_common(20)]
    unresolved.extend(
        f"Failed batches: {bid} ({failed_batches[bid]})" for bid in failed_sorted
    )

    structure = VideoStructure(
        video_start=video_start,
        video_end=video_end,
        overall_summary=overall_summary,
        sections=sections,
        recurring_subjects=recurring_subjects,
        recurring_visual_patterns=recurring_patterns,
        subtitle_visual_notes=subtitle_visual_notes,
        unresolved_questions=unresolved,
        failed_batches=failed_sorted,
        gaps=merged_gaps,
    )
    return structure


def _truncate_title(seed: str, max_chars: int = 60) -> str:
    seed = (seed or "").strip()
    if len(seed) <= max_chars:
        return seed or "Section"
    return seed[: max_chars - 1] + "…"


def _build_overall_summary(
    sections: list[VideoSection],
    summaries: list[str],
    gaps: list[VideoGap],
) -> str:
    parts: list[str] = []
    if sections:
        parts.append(
            f"Composed of {len(sections)} sections between "
            f"{sections[0].start:.1f}s and {sections[-1].end:.1f}s."
        )
    if gaps:
        gap_total = sum(g.end - g.start for g in gaps)
        parts.append(
            f"{len(gaps)} analysis gap(s) covering {gap_total:.1f}s of content."
        )
    if not parts:
        parts.append("No successful coarse analysis sections were produced.")
    return " ".join(parts)


def save_structure(layout: OutputLayout, structure: VideoStructure) -> Path:
    paths = coarse_output_paths(layout)
    ensure_coarse_dirs(paths)
    cache = CacheStore(layout)
    cache.write_json(paths["structure"], structure.model_dump(mode="json"))
    return paths["structure"]


__all__ = ["aggregate_structure", "save_structure"]