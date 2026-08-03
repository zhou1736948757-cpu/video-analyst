"""Tests for coarse aggregation (full-video structure)."""

from __future__ import annotations

from pathlib import Path

from video_analyst.coarse_analysis.aggregate import aggregate_structure
from video_analyst.coarse_analysis.models import VideoGap, VideoStructure
from video_analyst.vision.models import (
    CoarseBatchAnalysis,
    EvidenceReference,
    Inference,
    Observation,
    RefinementRequest,
    TimeRange,
)


def _obs(desc: str, category: str = "subjects", start: float = 0.0, end: float = 5.0):
    return Observation(
        category=category,
        description=desc,
        time_range=TimeRange(start=start, end=end),
        evidence=EvidenceReference(frame_ids=["f1"], timestamps=[0.5]),
        confidence=0.7,
    )


def _analysis(batch_id: str, start: float, end: float, subjects: list[str], summary: str = "") -> CoarseBatchAnalysis:
    return CoarseBatchAnalysis(
        batch_id=batch_id,
        start=start,
        end=end,
        summary=summary,
        subjects=[_obs(s) for s in subjects],
        confidence=0.7,
    )


def test_aggregate_produces_sections_per_batch():
    analyses = {
        "batch_0001": _analysis("batch_0001", 0.0, 20.0, ["a person enters"], "first part"),
        "batch_0002": _analysis("batch_0002", 20.0, 40.0, ["dialogue continues"], "second part"),
    }
    s = aggregate_structure(
        analyses=analyses,
        failed_batches={},
        video_start=0.0,
        video_end=40.0,
        batches_order=["batch_0001", "batch_0002"],
        batches_meta={"batch_0001": {"start": 0.0, "end": 20.0}, "batch_0002": {"start": 20.0, "end": 40.0}},
    )
    assert isinstance(s, VideoStructure)
    assert len(s.sections) >= 1
    # overall summary must mention composition
    assert "sections" in s.overall_summary or "section" in s.overall_summary.lower()


def test_failed_batches_emit_gaps_not_empty_sections():
    analyses = {
        "batch_0001": _analysis("batch_0001", 0.0, 20.0, ["ok content"]),
    }
    s = aggregate_structure(
        analyses=analyses,
        failed_batches={"batch_0002": "timeout", "batch_0003": "5xx"},
        video_start=0.0,
        video_end=60.0,
        batches_order=["batch_0001", "batch_0002", "batch_0003"],
        batches_meta={
            "batch_0001": {"start": 0.0, "end": 20.0},
            "batch_0002": {"start": 20.0, "end": 40.0},
            "batch_0003": {"start": 40.0, "end": 60.0},
        },
    )
    # gaps for failed batches must exist
    assert any(g.failed_batch_ids and ("batch_0002" in g.failed_batch_ids or "batch_0003" in g.failed_batch_ids) for g in s.gaps)
    # failed_batches are listed in unresolved_questions
    assert any("batch_0002" in q for q in s.unresolved_questions)
    assert any("batch_0003" in q for q in s.unresolved_questions)


def test_contiguous_failed_batches_merge_into_single_gap():
    analyses = {"batch_0001": _analysis("batch_0001", 0.0, 20.0, ["ok"])}
    s = aggregate_structure(
        analyses=analyses,
        failed_batches={"batch_0002": "e", "batch_0003": "e"},
        video_start=0.0,
        video_end=60.0,
        batches_order=["batch_0001", "batch_0002", "batch_0003"],
        batches_meta={
            "batch_0001": {"start": 0.0, "end": 20.0},
            "batch_0002": {"start": 20.0, "end": 40.0},
            "batch_0003": {"start": 40.0, "end": 60.0},
        },
    )
    # Two contiguous gaps should merge into one
    gap = [g for g in s.gaps if g.start == 20.0 and g.end == 60.0]
    assert len(gap) == 1
    assert set(gap[0].failed_batch_ids) == {"batch_0002", "batch_0003"}


def test_similar_consecutive_sections_merge():
    a1 = _analysis(
        "batch_0001", 0.0, 20.0,
        ["speaker on stage with microphone and audience"],
        "speaking"
    )
    a2 = _analysis(
        "batch_0002", 20.0, 40.0,
        ["speaker on stage with microphone and audience continues"],
        "still speaking"
    )
    s = aggregate_structure(
        analyses={"batch_0001": a1, "batch_0002": a2},
        failed_batches={},
        video_start=0.0,
        video_end=40.0,
        batches_order=["batch_0001", "batch_0002"],
        batches_meta={"batch_0001": {"start": 0.0, "end": 20.0}, "batch_0002": {"start": 20.0, "end": 40.0}},
    )
    assert len(s.sections) == 1
    assert s.sections[0].source_batch_ids == ["batch_0001", "batch_0002"]


def test_recurring_subjects_are_counted():
    a1 = _analysis("batch_0001", 0.0, 20.0, ["Speaker speaks", "Audience reacts"])
    a2 = _analysis("batch_0002", 20.0, 40.0, ["Speaker continues", "Camera pans"])
    s = aggregate_structure(
        analyses={"batch_0001": a1, "batch_0002": a2},
        failed_batches={},
        video_start=0.0,
        video_end=40.0,
        batches_order=["batch_0001", "batch_0002"],
        batches_meta={"batch_0001": {"start": 0.0, "end": 20.0}, "batch_0002": {"start": 20.0, "end": 40.0}},
    )
    # speaker appears in both → should be in recurring_subjects
    assert any("speaker" in s.lower() for s in s.recurring_subjects)


def test_no_sections_for_all_failed():
    analyses: dict = {}
    s = aggregate_structure(
        analyses=analyses,
        failed_batches={"batch_0001": "x", "batch_0002": "x"},
        video_start=0.0,
        video_end=40.0,
        batches_order=["batch_0001", "batch_0002"],
        batches_meta={"batch_0001": {"start": 0.0, "end": 20.0}, "batch_0002": {"start": 20.0, "end": 40.0}},
    )
    assert s.sections == []
    assert any("batch_0001" in g.failed_batch_ids or "batch_0002" in g.failed_batch_ids for g in s.gaps)


def test_gap_object_serializes(tmp_path: Path):
    g = VideoGap(start=10.0, end=20.0, failed_batch_ids=["b"], reason="x")
    d = g.model_dump(mode="json")
    assert d["start"] == 10.0
    assert d["failed_batch_ids"] == ["b"]


def test_missing_batches_meta_is_empty():
    """If batches_meta omits a batch, batch_start is taken from the analysis."""
    a = _analysis("batch_0001", 0.0, 20.0, ["x"], "y")
    a.start, a.end = 0.0, 20.0
    s = aggregate_structure(
        analyses={"batch_0001": a},
        failed_batches={},
        video_start=0.0,
        video_end=20.0,
        batches_order=["batch_0001"],
        batches_meta=None,
    )
    assert len(s.sections) >= 1
    # The section's start/end should still be 0..20.
    assert s.sections[0].start == 0.0
    assert s.sections[0].end == 20.0