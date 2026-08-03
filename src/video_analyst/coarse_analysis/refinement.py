"""Refinement plan generation from coarse batch outputs.

Inputs:
  - per-batch CoarseBatchAnalysis results
  - config: pad, max interval, max fps, budget ratio

Outputs:
  - RefinementPlan: merged, FPS-decided, prioritized, budget-clipped intervals
  - deferred_intervals: anything that didn't fit the budget, never silently dropped
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from video_analyst.config import CoarseAnalysisConfig
from video_analyst.exceptions import RefinementPlanError
from video_analyst.vision.models import CoarseBatchAnalysis, RefinementRequest

from video_analyst.coarse_analysis.models import (
    RefinementInterval,
    RefinementPlan,
    request_to_interval,
)

_PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3}
_FPS_RANK = {5: 1, 10: 2, 15: 3}
_RANK_FPS = {v: k for k, v in _FPS_RANK.items()}


def _gen_interval_id(idx: int) -> str:
    return f"ref_{idx:04d}"


def _merge_overlapping(
    intervals: list[RefinementInterval],
    *,
    gap_pad_seconds: float,
) -> list[RefinementInterval]:
    """Merge intervals that overlap or are within gap_pad_seconds of each other.

    Strategy:
    - Sort by start
    - For each candidate, if its start <= current.end + gap_pad,
      extend the current interval and combine reasons/questions/etc.
    """
    if not intervals:
        return []

    intervals_sorted = sorted(intervals, key=lambda iv: (iv.start, iv.end))
    merged: list[RefinementInterval] = []
    cur = intervals_sorted[0].model_copy(deep=True)
    for nxt in intervals_sorted[1:]:
        if nxt.start <= cur.end + gap_pad_seconds:
            # Merge
            new_end = max(cur.end, nxt.end)
            new_fps = max(cur.requested_fps, nxt.requested_fps, key=lambda f: _FPS_RANK[f])
            new_priority = max(cur.priority, nxt.priority, key=lambda p: _PRIORITY_RANK[p])
            reasons = list(dict.fromkeys(cur.reasons + nxt.reasons))
            questions = list(dict.fromkeys(cur.questions + nxt.questions))
            sources = list(dict.fromkeys(cur.source_batch_ids + nxt.source_batch_ids))
            evidence = list(dict.fromkeys(cur.evidence_frame_ids + nxt.evidence_frame_ids))
            cur = RefinementInterval(
                interval_id=cur.interval_id,
                start=cur.start,
                end=new_end,
                requested_fps=new_fps,
                priority=new_priority,
                reasons=reasons,
                questions=questions,
                source_batch_ids=sources,
                evidence_frame_ids=evidence,
                estimated_frame_count=max(
                    cur.estimated_frame_count, nxt.estimated_frame_count
                ),
            )
        else:
            merged.append(cur)
            cur = nxt.model_copy(deep=True)
    merged.append(cur)
    return merged


def _clip_to_video(
    intervals: list[RefinementInterval],
    *,
    video_start: float,
    video_end: float,
) -> list[RefinementInterval]:
    """Clip intervals to video bounds and drop zero-length ones."""
    out: list[RefinementInterval] = []
    for iv in intervals:
        s = max(iv.start, video_start)
        e = min(iv.end, video_end)
        if e - s <= 0:
            continue
        out.append(
            iv.model_copy(
                update={
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "estimated_frame_count": max(
                        0, int(round((e - s) * iv.requested_fps))
                    ),
                }
            )
        )
    return out


def _split_long(
    intervals: list[RefinementInterval], *, max_seconds: float
) -> list[RefinementInterval]:
    """Split intervals longer than max_seconds into chunks."""
    out: list[RefinementInterval] = []
    for iv in intervals:
        dur = iv.end - iv.start
        if dur <= max_seconds:
            out.append(iv)
            continue
        cur_start = iv.start
        idx = 0
        while cur_start < iv.end - 1e-3:
            cur_end = min(iv.end, cur_start + max_seconds)
            new_id = f"{iv.interval_id}_{idx:02d}"
            out.append(
                iv.model_copy(
                    update={
                        "interval_id": new_id,
                        "start": round(cur_start, 3),
                        "end": round(cur_end, 3),
                        "estimated_frame_count": max(
                            0, int(round((cur_end - cur_start) * iv.requested_fps))
                        ),
                    }
                )
            )
            cur_start = cur_end
            idx += 1
    return out


def _apply_budget(
    intervals: list[RefinementInterval],
    *,
    budget: float,
) -> tuple[list[RefinementInterval], list[RefinementInterval]]:
    """Greedy pick by priority then by requested_fps, accumulating duration."""
    if budget <= 0:
        return [], list(intervals)

    def sort_key(iv: RefinementInterval) -> tuple[int, int, float]:
        return (
            -_PRIORITY_RANK[iv.priority],
            -_FPS_RANK[iv.requested_fps],
            iv.start,
        )

    sorted_ivs = sorted(intervals, key=sort_key)
    accepted: list[RefinementInterval] = []
    deferred: list[RefinementInterval] = []
    remaining = budget
    for iv in sorted_ivs:
        dur = iv.end - iv.start
        if dur <= remaining + 1e-6:
            accepted.append(iv)
            remaining -= dur
        else:
            deferred.append(iv)
    # Sort accepted/deferred back by time for readability
    accepted.sort(key=lambda iv: iv.start)
    deferred.sort(key=lambda iv: iv.start)
    return accepted, deferred


def build_refinement_plan(
    *,
    batch_analyses: dict[str, CoarseBatchAnalysis],
    video_duration: float,
    video_start: float = 0.0,
    config: CoarseAnalysisConfig | None = None,
) -> RefinementPlan:
    """Build a refinement plan from the per-batch analyses."""
    cfg = config or CoarseAnalysisConfig()
    warnings: list[str] = []

    budget = max(0.0, video_duration * float(cfg.refinement_budget_ratio))
    pad = max(0.0, float(cfg.refinement_pad_seconds))
    max_interval = max(0.5, float(cfg.refinement_max_interval_seconds))

    intervals: list[RefinementInterval] = []
    counter = 0
    for batch_id, analysis in batch_analyses.items():
        for req in analysis.refinement_requests:
            counter += 1
            intervals.append(
                request_to_interval(
                    req,
                    interval_id=_gen_interval_id(counter),
                    source_batch_id=batch_id,
                )
            )

    if not intervals:
        plan = RefinementPlan(
            video_duration=video_duration,
            total_requested_duration=0.0,
            budget_duration=budget,
            intervals=[],
            deferred_intervals=[],
            generation_warnings=warnings,
        )
        return plan

    # 1) pad
    padded = [
        iv.model_copy(
            update={
                "start": round(max(video_start, iv.start - pad), 3),
                "end": round(min(video_duration, iv.end + pad), 3),
            }
        )
        for iv in intervals
    ]
    # 2) merge overlapping / near-overlapping
    merged = _merge_overlapping(padded, gap_pad_seconds=pad)
    # 3) clip to video bounds
    clipped = _clip_to_video(
        merged, video_start=video_start, video_end=video_duration
    )
    # 4) split overly long intervals
    split = _split_long(clipped, max_seconds=max_interval)
    # 5) apply budget
    accepted, deferred = _apply_budget(split, budget=budget)

    total = sum(iv.end - iv.start for iv in accepted)
    if deferred:
        warnings.append(
            f"{len(deferred)} refinement intervals deferred due to budget "
            f"{budget:.1f}s of {video_duration:.1f}s."
        )

    return RefinementPlan(
        video_duration=video_duration,
        total_requested_duration=round(total, 3),
        budget_duration=round(budget, 3),
        intervals=accepted,
        deferred_intervals=deferred,
        generation_warnings=warnings,
    )


def plan_to_dict(plan: RefinementPlan) -> dict[str, Any]:
    return plan.model_dump(mode="json")


__all__ = ["build_refinement_plan", "plan_to_dict"]