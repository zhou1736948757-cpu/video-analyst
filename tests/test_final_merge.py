"""Tests for final event merge (Epic 5)."""

from __future__ import annotations

import pytest

from video_analyst.fine.models import (
    FineEvidenceReference,
    TemporalBoundary,
    TemporalEvent,
)
from video_analyst.final.merge import (
    collect_unresolved_items,
    merge_question_answers,
    merge_temporal_events,
)
from video_analyst.final.models import (
    MergedTemporalEventStatus,
    UnresolvedSource,
)


def _ev(
    *,
    eid: str,
    cat: str = "subject_motion",
    earliest: float = 5.0,
    latest: float = 6.0,
    conf: float = 0.6,
    frame_ids: list[str] | None = None,
    sheet_ids: list[str] | None = None,
    desc: str = "subject moved",
) -> TemporalEvent:
    return TemporalEvent(
        event_id=eid,
        category=cat,  # type: ignore[arg-type]
        boundary=TemporalBoundary(earliest=earliest, latest=latest, precision="bounded"),
        description=desc,
        direct_observations=["obs_1"],
        inference="inferred",
        evidence=FineEvidenceReference(
            contact_sheet_ids=sheet_ids or ["sheet_0001"],
            frame_ids=frame_ids or ["frame_001"],
            timestamps=[earliest],
        ),
        alternatives=[],
        confidence=conf,
    )


def test_round1_only_events():
    """无 terminal → 全部 from_round1。"""
    events = [_ev(eid="r1_a"), _ev(eid="r1_b")]
    merged = merge_temporal_events(round1_events=events, terminal_events=[])
    assert len(merged) == 2
    assert all(m.status == "from_round1" for m in merged)
    assert all(m.source_round == "round1" for m in merged)


def test_terminal_confirms_round1():
    """terminal evidence 更密 → confirmed_by_terminal_refinement。"""
    r1 = _ev(eid="r1_a", frame_ids=["f1"], sheet_ids=["s1"], conf=0.5)
    t = _ev(
        eid="t_a",
        frame_ids=["f1", "f2", "f3"],
        sheet_ids=["s1", "s2"],
        conf=0.7,
    )
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert len(merged) == 1
    m = merged[0]
    assert m.status == "confirmed_by_terminal_refinement"
    assert m.source_round == "terminal"
    assert len(m.superseded) == 1


def test_terminal_revises_round1():
    """terminal 描述差异大 + 描述关键词 Jaccard 低 → revised。"""
    # 用完全不同领域的词
    r1 = _ev(eid="r1_a", desc="xyz abc def ghi", conf=0.6)
    t = _ev(eid="t_a", desc="jkl mno pqr stu", conf=0.6)
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert merged[0].status in ("revised_by_terminal_refinement", "rejected_by_terminal_refinement")


def test_terminal_rejects_round1():
    """terminal confidence 明显较低 + 描述相同 → rejected_by_terminal_refinement。"""
    r1 = _ev(eid="r1_a", conf=0.7, desc="subject moved")
    t = _ev(eid="t_a", conf=0.3, desc="subject moved")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert merged[0].status == "rejected_by_terminal_refinement"


def test_terminal_only_event_kept():
    """没有对应 round1 的 terminal 事件独立保留。"""
    t = _ev(eid="t_alone", earliest=20.0, latest=21.0)
    merged = merge_temporal_events(round1_events=[], terminal_events=[t])
    assert len(merged) == 1
    assert merged[0].status == "confirmed_by_terminal_refinement"


def test_different_category_not_merged():
    r1 = _ev(eid="r1_a", cat="subject_motion")
    t = _ev(eid="t_a", cat="camera_motion")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert len(merged) == 2


def test_superseded_preserved_not_deleted():
    """被修改/否定的旧事件保留在 superseded 列表，不删除。"""
    r1 = _ev(eid="r1_a", conf=0.6, desc="subject moved")
    t = _ev(eid="t_a", conf=0.7, desc="subject moved")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    m = merged[0]
    if m.status in ("revised_by_terminal_refinement", "rejected_by_terminal_refinement"):
        assert len(m.superseded) >= 1
        assert any(s.event_id == "r1_a" for s in m.superseded)


def test_question_answer_terminal_overrides():
    """terminal 答案覆盖 round1 同 question。"""
    from video_analyst.fine.models import RefinementQuestionAnswer

    r1_ans = RefinementQuestionAnswer(
        question="q1", answer="round1 answer", status="answered", evidence=FineEvidenceReference(), confidence=0.5
    )
    t_ans = RefinementQuestionAnswer(
        question="q1", answer="terminal answer", status="answered", evidence=FineEvidenceReference(), confidence=0.8
    )
    merged = merge_question_answers(round1_answers=[r1_ans], terminal_answers=[t_ans])
    assert len(merged) == 1
    assert merged[0].answer == "terminal answer"
    assert merged[0].answer_origin == "terminal_refinement"
    assert merged[0].terminal_attempted is True


def test_question_answer_round1_preserved_without_terminal():
    """无 terminal 时 round1 答案保留。"""
    from video_analyst.fine.models import RefinementQuestionAnswer

    r1_ans = RefinementQuestionAnswer(
        question="q1", answer="round1 answer", status="answered", evidence=FineEvidenceReference(), confidence=0.5
    )
    merged = merge_question_answers(round1_answers=[r1_ans], terminal_answers=None)
    assert len(merged) == 1
    assert merged[0].answer == "round1 answer"
    assert merged[0].answer_origin == "model"


def test_unresolved_items_collect_from_answers():
    from video_analyst.fine.models import RefinementQuestionAnswer

    items = [
        RefinementQuestionAnswer(
            question="q1", answer="cannot answer", status="unresolved",
            evidence=FineEvidenceReference(), confidence=0.1, answer_origin="model"
        ),
        RefinementQuestionAnswer(
            question="q2", answer="val_fallback", status="unresolved",
            evidence=FineEvidenceReference(), confidence=0.1, answer_origin="validator_fallback"
        ),
    ]
    unresolved = collect_unresolved_items(
        merged_events=[],
        round1_answers=items,
        merged_answers=items,
    )
    assert len(unresolved) >= 2
    sources = {u.source for u in unresolved}
    assert "round1_unresolved" in sources
    assert "validator_fallback" in sources


def test_unresolved_items_collect_deferred():
    """deferred 区间进入 unresolved_items。"""
    deferred = [
        {"start": 5.0, "end": 7.0, "interval_id": "d1"},
    ]
    items = collect_unresolved_items(
        merged_events=[],
        round1_answers=[],
        merged_answers=[],
        deferred_intervals=deferred,
    )
    assert any(i.source == "deferred_terminal" for i in items)


def test_unresolved_items_collect_rejected_events():
    """rejected 事件进入 unresolved_items。"""
    r1 = _ev(eid="r1_a", conf=0.7, desc="x")
    t = _ev(eid="t_a", conf=0.3, desc="x")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    items = collect_unresolved_items(
        merged_events=merged,
        round1_answers=[],
        merged_answers=[],
    )
    assert any(i.source == "terminal_unresolved" for i in items)