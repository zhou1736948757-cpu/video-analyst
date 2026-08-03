"""Tests for fine aggregation (Epic 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.fine.aggregate import aggregate_fine_analysis
from video_analyst.fine.models import (
    FineBatchAnalysis,
    FineEvidenceReference,
    RefinementQuestionAnswer,
    TemporalBoundary,
    TemporalEvent,
)

from tests._epic4_fixtures import has_pillow

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _analysis(
    *,
    fb_id: str = "fine_interval_0001",
    start: float = 5.0,
    end: float = 8.0,
    events: list[TemporalEvent] | None = None,
    questions: list[str] | None = None,
    confidence: float = 0.6,
) -> FineBatchAnalysis:
    return FineBatchAnalysis(
        fine_batch_id=fb_id,
        interval_id=fb_id.replace("fine_", ""),
        start=start,
        end=end,
        summary="summary",
        temporal_events=events or [],
        question_answers=[
            RefinementQuestionAnswer(
                question=q,
                answer=f"answer {q}",
                status="answered",
                evidence=FineEvidenceReference(),
                confidence=0.6,
            )
            for q in (questions or [])
        ],
        confidence=confidence,
    )


def _ev(
    *,
    eid: str = "ev_001",
    category: str = "subject_motion",
    earliest: float = 5.0,
    latest: float = 6.0,
    desc: str = "subject moved",
    conf: float = 0.6,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=eid,
        category=category,  # type: ignore[arg-type]
        boundary=TemporalBoundary(earliest=earliest, latest=latest, precision="bounded"),
        description=desc,
        direct_observations=["obs_1"],
        inference=None,
        evidence=FineEvidenceReference(),
        alternatives=[],
        confidence=conf,
    )


def test_multi_interval_aggregation():
    a1 = _analysis(fb_id="fine_interval_0001", start=5.0, end=8.0, events=[_ev(eid="ev_1")])
    a2 = _analysis(fb_id="fine_interval_0002", start=20.0, end=23.0, events=[_ev(eid="ev_2", earliest=20.0, latest=21.0)])
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1, "fine_interval_0002": a2},
        failed_batches={},
        batches_order=["fine_interval_0001", "fine_interval_0002"],
        batches_meta={
            "fine_interval_0001": {"start": 5.0, "end": 8.0},
            "fine_interval_0002": {"start": 20.0, "end": 23.0},
        },
        video_start=0.0,
        video_end=30.0,
    )
    assert len(agg.temporal_events) == 2
    assert len(agg.analyzed_intervals) == 2
    assert len(agg.analysis_gaps) == 0


def test_failed_interval_forms_gap():
    a1 = _analysis(fb_id="fine_interval_0001", start=5.0, end=8.0)
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1},
        failed_batches={"fine_interval_0002": "error"},
        batches_order=["fine_interval_0001", "fine_interval_0002"],
        batches_meta={
            "fine_interval_0001": {"start": 5.0, "end": 8.0},
            "fine_interval_0002": {"start": 20.0, "end": 23.0},
        },
        video_start=0.0,
        video_end=30.0,
    )
    assert len(agg.analysis_gaps) == 1
    assert agg.analysis_gaps[0].start == 20.0
    assert agg.analysis_gaps[0].end == 23.0


def test_unanalyzed_interval_not_described_as_no_motion():
    """未分析区间不解释为"无动效"。"""
    agg = aggregate_fine_analysis(
        analyses={},
        failed_batches={},
        batches_order=[],
        batches_meta={},
        video_start=0.0,
        video_end=30.0,
    )
    assert agg.temporal_events == []
    # warnings 明确说明"no successful fine analyses"，不当作"无动效"
    assert any("NOT 'no motion'" in w for w in agg.warnings)


def test_question_answers_aggregated():
    a1 = _analysis(fb_id="fine_interval_0001", questions=["q1", "q2"])
    a2 = _analysis(fb_id="fine_interval_0002", start=20.0, end=23.0, questions=["q3"])
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1, "fine_interval_0002": a2},
        failed_batches={},
        batches_order=["fine_interval_0001", "fine_interval_0002"],
        batches_meta={
            "fine_interval_0001": {"start": 5.0, "end": 8.0},
            "fine_interval_0002": {"start": 20.0, "end": 23.0},
        },
        video_start=0.0,
        video_end=30.0,
    )
    assert len(agg.answered_questions) == 3


def test_unresolved_questions_preserved():
    a1 = _analysis(fb_id="fine_interval_0001", questions=["q1"])
    a1 = a1.model_copy(
        update={
            "question_answers": [
                RefinementQuestionAnswer(
                    question="q1",
                    answer="cannot answer",
                    status="unresolved",
                    evidence=FineEvidenceReference(),
                    confidence=0.1,
                )
            ],
            "unresolved_items": ["item1"],
        }
    )
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1},
        failed_batches={},
        batches_order=["fine_interval_0001"],
        batches_meta={"fine_interval_0001": {"start": 5.0, "end": 8.0}},
        video_start=0.0,
        video_end=10.0,
    )
    assert any("q1" in q for q in agg.unresolved_questions)
    assert "item1" in agg.unresolved_questions


def test_summary_classification_correct():
    events = [
        _ev(eid="ev_1", category="subject_motion", desc="subject moved"),
        _ev(eid="ev_2", category="camera_motion", desc="camera panned"),
        _ev(eid="ev_3", category="text_animation", desc="text appeared"),
        _ev(eid="ev_4", category="graphic_animation", desc="progress bar filled"),
        _ev(eid="ev_5", category="transition", desc="hard cut"),
    ]
    a1 = _analysis(events=events)
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1},
        failed_batches={},
        batches_order=["fine_interval_0001"],
        batches_meta={"fine_interval_0001": {"start": 5.0, "end": 8.0}},
        video_start=0.0,
        video_end=10.0,
    )
    assert len(agg.subject_motion_summary) == 1
    assert len(agg.camera_motion_summary) == 1
    assert len(agg.text_animation_summary) == 1
    assert len(agg.graphic_animation_summary) == 1
    assert len(agg.transition_summary) == 1


def test_important_events_high_confidence():
    events = [
        _ev(eid="ev_1", desc="important", conf=0.8),
        _ev(eid="ev_2", desc="low conf", conf=0.3),
    ]
    a1 = _analysis(events=events)
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1},
        failed_batches={},
        batches_order=["fine_interval_0001"],
        batches_meta={"fine_interval_0001": {"start": 5.0, "end": 8.0}},
        video_start=0.0,
        video_end=10.0,
    )
    # 只有 confidence >= 0.6 进 important_events
    assert len(agg.important_events) == 1


def test_invalid_video_duration_raises():
    with pytest.raises(Exception):
        aggregate_fine_analysis(
            analyses={},
            failed_batches={},
            batches_order=[],
            batches_meta={},
            video_start=10.0,
            video_end=5.0,
        )


def test_second_refinement_flag_propagated():
    a1 = _analysis()
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1},
        failed_batches={},
        batches_order=["fine_interval_0001"],
        batches_meta={"fine_interval_0001": {"start": 5.0, "end": 8.0}},
        video_start=0.0,
        video_end=10.0,
        second_refinement_required=True,
    )
    assert agg.second_refinement_required is True


def test_confidence_is_average():
    a1 = _analysis(fb_id="fine_interval_0001", confidence=0.6)
    a2 = _analysis(fb_id="fine_interval_0002", start=20.0, end=23.0, confidence=0.8)
    agg = aggregate_fine_analysis(
        analyses={"fine_interval_0001": a1, "fine_interval_0002": a2},
        failed_batches={},
        batches_order=["fine_interval_0001", "fine_interval_0002"],
        batches_meta={
            "fine_interval_0001": {"start": 5.0, "end": 8.0},
            "fine_interval_0002": {"start": 20.0, "end": 23.0},
        },
        video_start=0.0,
        video_end=30.0,
    )
    assert agg.confidence == pytest.approx(0.7, abs=0.01)
