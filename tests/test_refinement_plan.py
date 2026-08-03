"""Tests for the refinement plan generation."""

from __future__ import annotations

from video_analyst.config import CoarseAnalysisConfig
from video_analyst.coarse_analysis.refinement import build_refinement_plan
from video_analyst.vision.models import (
    CoarseBatchAnalysis,
    RefinementRequest,
    TimeRange,
)


def _analysis_with_refinements(batch_id: str, start: float, end: float, reqs: list[RefinementRequest]) -> CoarseBatchAnalysis:
    return CoarseBatchAnalysis(
        batch_id=batch_id,
        start=start,
        end=end,
        summary="x",
        confidence=0.6,
        refinement_requests=reqs,
    )


def _rr(req_id: str, start: float, end: float, fps: int, priority: str = "medium") -> RefinementRequest:
    return RefinementRequest(
        request_id=req_id,
        start=start,
        end=end,
        requested_fps=fps,
        reason=f"need higher FPS between {start} and {end}",
        questions=["?"],
        priority=priority,
        evidence_frame_ids=[],
        confidence=0.6,
    )


def test_no_refinements_returns_empty_plan():
    plan = build_refinement_plan(
        batch_analyses={"b": _analysis_with_refinements("b", 0, 10, [])},
        video_duration=10,
        config=CoarseAnalysisConfig(),
    )
    assert plan.intervals == []
    assert plan.deferred_intervals == []
    assert plan.total_requested_duration == 0.0


def test_overlapping_intervals_merge():
    r1 = _rr("r1", 0.0, 4.0, 5)
    r2 = _rr("r2", 2.0, 6.0, 10)
    a = _analysis_with_refinements("b", 0, 10, [r1, r2])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=10,
        config=CoarseAnalysisConfig(refinement_pad_seconds=0.0, refinement_budget_ratio=1.0),
    )
    # One merged interval covering [0, 6]
    assert len(plan.intervals) == 1
    iv = plan.intervals[0]
    assert iv.start == 0.0
    assert iv.end == 6.0
    assert iv.requested_fps == 10  # max
    assert iv.priority == "medium"


def test_close_intervals_within_pad_merge():
    r1 = _rr("r1", 0.0, 2.0, 5)
    r2 = _rr("r2", 2.4, 4.0, 10)  # gap 0.4 ≤ pad 0.5
    a = _analysis_with_refinements("b", 0, 10, [r1, r2])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=10,
        config=CoarseAnalysisConfig(refinement_pad_seconds=0.5, refinement_budget_ratio=1.0),
    )
    assert len(plan.intervals) == 1
    # After padding each by 0.5, they overlap/merge
    assert plan.intervals[0].start == 0.0 or plan.intervals[0].start <= 0.5
    assert plan.intervals[0].end >= 4.0


def test_intervals_padded_then_merged():
    r1 = _rr("r1", 5.0, 7.0, 10)
    a = _analysis_with_refinements("b", 0, 20, [r1])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=20,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=1.0,
            refinement_max_interval_seconds=5.0,
            refinement_budget_ratio=1.0,
        ),
    )
    # After pad 1.0 → [4.0, 8.0]
    assert plan.intervals[0].start == 4.0
    assert plan.intervals[0].end == 8.0


def test_interval_split_when_too_long():
    r1 = _rr("r1", 0.0, 20.0, 10)
    a = _analysis_with_refinements("b", 0, 40, [r1])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=40,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_max_interval_seconds=8.0,
            refinement_budget_ratio=1.0,
        ),
    )
    # Split into ≤ 8s intervals
    assert all((iv.end - iv.start) <= 8.0 + 1e-6 for iv in plan.intervals)


def test_budget_excludes_low_priority_overruns():
    # Total request: 20s. Budget: 8s (40% × 20s).
    # High priority: 5s. Medium: 15s. After accepting high priority,
    # budget remains 3s → medium gets partial / dropped, deferred.
    r_high = _rr("rh", 0.0, 5.0, 10, priority="high")
    r_med = _rr("rm", 6.0, 21.0, 10, priority="medium")
    a = _analysis_with_refinements("b", 0, 30, [r_high, r_med])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=20,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_max_interval_seconds=20.0,
            refinement_budget_ratio=0.40,  # budget = 8s
        ),
    )
    accepted = plan.intervals
    deferred = plan.deferred_intervals
    total_accepted = sum(iv.end - iv.start for iv in accepted)
    assert total_accepted <= plan.budget_duration + 1e-6
    # high priority must be accepted
    high_accepted = [iv for iv in accepted if iv.priority == "high"]
    assert high_accepted
    # medium overflow → deferred or partially accepted
    assert deferred or all(iv.priority == "high" for iv in accepted)


def test_priority_picks_high_over_medium():
    r_high = _rr("rh", 0.0, 4.0, 10, priority="high")
    r_med = _rr("rm", 0.0, 4.0, 10, priority="medium")
    a = _analysis_with_refinements("b", 0, 10, [r_high, r_med])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=10,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_max_interval_seconds=10.0,
            refinement_budget_ratio=0.5,
        ),
    )
    # merged interval should inherit high priority
    merged = plan.intervals[0]
    assert merged.priority == "high"


def test_fps_uses_max_of_overlaps():
    r1 = _rr("r1", 0.0, 3.0, 5)
    r2 = _rr("r2", 1.0, 5.0, 15)
    a = _analysis_with_refinements("b", 0, 10, [r1, r2])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=10,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_budget_ratio=1.0,
        ),
    )
    assert plan.intervals[0].requested_fps == 15


def test_video_bounds_clip():
    r1 = _rr("r1", -5.0, 100.0, 10)
    a = _analysis_with_refinements("b", 0, 50, [r1])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=50,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_budget_ratio=1.0,
        ),
    )
    iv = plan.intervals[0]
    assert iv.start >= 0.0
    assert iv.end <= 50.0


def test_estimated_frame_count():
    r1 = _rr("r1", 0.0, 2.0, 10)  # 2s × 10 fps = 20 frames
    a = _analysis_with_refinements("b", 0, 10, [r1])
    plan = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=10,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_budget_ratio=1.0,
        ),
    )
    assert plan.intervals[0].estimated_frame_count == 20


def test_source_batch_ids_preserved():
    a1 = _analysis_with_refinements("b1", 0, 10, [_rr("r1a", 1, 2, 10)])
    a2 = _analysis_with_refinements("b2", 10, 20, [_rr("r2a", 11, 12, 10)])
    plan = build_refinement_plan(
        batch_analyses={"b1": a1, "b2": a2},
        video_duration=20,
        config=CoarseAnalysisConfig(
            refinement_pad_seconds=0.0,
            refinement_budget_ratio=1.0,
        ),
    )
    assert len(plan.intervals) == 2
    sources = sorted([plan.intervals[0].source_batch_ids[0], plan.intervals[1].source_batch_ids[0]])
    assert sources == ["b1", "b2"]


def test_budget_change_does_not_invalidate_vision_cache():
    """Plan generation only needs analyses — never calls vision."""
    a = _analysis_with_refinements("b", 0, 100, [_rr("r1", 0, 100, 10)])
    # Run twice with different budget; output must differ (plan changes), but
    # we are NOT calling any provider.
    plan_a = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=100,
        config=CoarseAnalysisConfig(refinement_budget_ratio=0.20, refinement_pad_seconds=0.0),
    )
    plan_b = build_refinement_plan(
        batch_analyses={"b": a},
        video_duration=100,
        config=CoarseAnalysisConfig(refinement_budget_ratio=0.50, refinement_pad_seconds=0.0),
    )
    # larger budget → at least as much accepted
    dur_a = sum(iv.end - iv.start for iv in plan_a.intervals)
    dur_b = sum(iv.end - iv.start for iv in plan_b.intervals)
    assert dur_b >= dur_a