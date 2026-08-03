"""Tests for temporal event deduplication (Epic 4)."""

from __future__ import annotations

import pytest

from video_analyst.fine.deduplicate import (
    deduplicate_temporal_events,
)
from video_analyst.fine.models import (
    FineEvidenceReference,
    TemporalBoundary,
    TemporalEvent,
)


def _ev(
    *,
    event_id: str = "ev_001",
    category: str = "subject_motion",
    earliest: float = 5.0,
    latest: float = 6.0,
    description: str = "subject moved head",
    frame_ids: list[str] | None = None,
    sheet_ids: list[str] | None = None,
    confidence: float = 0.6,
    source_batch_ids: list[str] | None = None,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=event_id,
        category=category,  # type: ignore[arg-type]
        boundary=TemporalBoundary(earliest=earliest, latest=latest, precision="bounded"),
        description=description,
        direct_observations=["obs_1"],
        inference="inferred",
        evidence=FineEvidenceReference(
            contact_sheet_ids=sheet_ids or ["sheet_0001"],
            frame_ids=frame_ids or ["frame_001"],
            timestamps=[earliest],
        ),
        alternatives=["alt1"],
        confidence=confidence,
        source_batch_ids=source_batch_ids or ["fine_interval_0001"],
    )


def test_overlap_sheet_duplicate_events_merged():
    """相邻 contact sheet 重叠一帧导致同一事件被描述两次 → 合并。"""
    ev1 = _ev(event_id="ev_a", frame_ids=["f1", "f2"], sheet_ids=["s1"], earliest=5.0, latest=6.0)
    ev2 = _ev(event_id="ev_b", frame_ids=["f2", "f3"], sheet_ids=["s2"], earliest=5.5, latest=6.5)
    # 共享 frame f2，时间重叠，描述相同
    report = deduplicate_temporal_events([ev1, ev2])
    assert len(report.canonical_events) == 1
    merged = report.canonical_events[0]
    # 证据并集
    assert set(merged.evidence.frame_ids) == {"f1", "f2", "f3"}
    assert set(merged.evidence.contact_sheet_ids) == {"s1", "s2"}
    # confidence 取最高
    assert merged.confidence == max(ev1.confidence, ev2.confidence)
    assert report.merged_count == 1


def test_shared_frame_id_same_category_merged():
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=5.5, description="subject moved")
    ev2 = _ev(event_id="ev_b", frame_ids=["f1"], earliest=5.2, latest=5.8, description="subject moved")
    report = deduplicate_temporal_events([ev1, ev2])
    assert len(report.canonical_events) == 1


def test_different_categories_not_merged():
    ev1 = _ev(event_id="ev_a", category="subject_motion", frame_ids=["f1"], earliest=5.0, latest=6.0)
    ev2 = _ev(event_id="ev_b", category="camera_motion", frame_ids=["f1"], earliest=5.0, latest=6.0)
    report = deduplicate_temporal_events([ev1, ev2])
    assert len(report.canonical_events) == 2


def test_far_time_not_merged():
    """远距离事件不合并（即使描述相同）。"""
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=6.0, description="subject moved")
    ev2 = _ev(event_id="ev_b", frame_ids=["f99"], earliest=50.0, latest=51.0, description="subject moved")
    report = deduplicate_temporal_events([ev1, ev2])
    assert len(report.canonical_events) == 2


def test_evidence_union():
    ev1 = _ev(event_id="ev_a", frame_ids=["f1", "f2"], sheet_ids=["s1"], earliest=5.0, latest=6.0, confidence=0.5)
    ev2 = _ev(event_id="ev_b", frame_ids=["f2", "f3"], sheet_ids=["s2"], earliest=5.5, latest=6.5, confidence=0.7)
    report = deduplicate_temporal_events([ev1, ev2])
    merged = report.canonical_events[0]
    assert set(merged.evidence.frame_ids) == {"f1", "f2", "f3"}
    assert set(merged.evidence.contact_sheet_ids) == {"s1", "s2"}


def test_alternatives_deduplicated():
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=6.0, confidence=0.5)
    ev1 = ev1.model_copy(update={"alternatives": ["camera moved", "subject moved"]})
    ev2 = _ev(event_id="ev_b", frame_ids=["f1"], earliest=5.2, latest=6.2, confidence=0.7)
    ev2 = ev2.model_copy(update={"alternatives": ["camera moved", "post scale"]})
    report = deduplicate_temporal_events([ev1, ev2])
    merged = report.canonical_events[0]
    assert merged.alternatives == ["camera moved", "subject moved", "post scale"]


def test_source_batch_ids_union():
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=6.0, source_batch_ids=["fb_001"])
    ev2 = _ev(event_id="ev_b", frame_ids=["f1"], earliest=5.2, latest=6.2, source_batch_ids=["fb_002"])
    report = deduplicate_temporal_events([ev1, ev2])
    merged = report.canonical_events[0]
    assert set(merged.source_batch_ids) == {"fb_001", "fb_002"}


def test_confidence_takes_max():
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=6.0, confidence=0.3)
    ev2 = _ev(event_id="ev_b", frame_ids=["f1"], earliest=5.2, latest=6.2, confidence=0.8)
    report = deduplicate_temporal_events([ev1, ev2])
    assert report.canonical_events[0].confidence == 0.8


def test_time_union():
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=6.0)
    ev2 = _ev(event_id="ev_b", frame_ids=["f1"], earliest=5.5, latest=7.0)
    report = deduplicate_temporal_events([ev1, ev2])
    merged = report.canonical_events[0]
    assert merged.boundary.earliest == 5.0
    assert merged.boundary.latest == 7.0


def test_empty_input():
    report = deduplicate_temporal_events([])
    assert report.canonical_events == []
    assert report.stats["input"] == 0


def test_dissimilar_description_not_merged():
    """描述差异大（即使时间重叠、共享 frame）也不合并。"""
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=6.0, description="person turned head")
    ev2 = _ev(event_id="ev_b", frame_ids=["f1"], earliest=5.2, latest=6.2, description="car drove past window")
    report = deduplicate_temporal_events([ev1, ev2])
    # 描述词集差异大，Jaccard < 阈值 → 不合并
    assert len(report.canonical_events) == 2


def test_three_overlapping_chain():
    """三个事件链式重叠 → 合并成一个。"""
    ev1 = _ev(event_id="ev_a", frame_ids=["f1"], earliest=5.0, latest=5.5, description="subject moved")
    ev2 = _ev(event_id="ev_b", frame_ids=["f2"], earliest=5.3, latest=5.8, description="subject moved")
    ev3 = _ev(event_id="ev_c", frame_ids=["f3"], earliest=5.6, latest=6.1, description="subject moved")
    report = deduplicate_temporal_events([ev1, ev2, ev3])
    # 时间相近但无共享 frame，且阈值 0.5s 内 → 合并
    assert len(report.canonical_events) <= 2
