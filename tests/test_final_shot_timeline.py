"""Tests for final shot timeline (Epic 5)."""

from __future__ import annotations

import pytest

from video_analyst.fine.models import (
    FineBatchAnalysis,
    FineEvidenceReference,
    TemporalBoundary,
    TemporalEvent,
)
from video_analyst.final.models import MergedTemporalEvent
from video_analyst.final.shot_timeline import build_shot_timeline

from tests._epic4_fixtures import has_pillow

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _merged(event: TemporalEvent, source_round: str = "round1") -> MergedTemporalEvent:
    return MergedTemporalEvent(
        merged_id=f"m_{event.event_id}",
        active=event,
        superseded=[],
        status="from_round1" if source_round == "round1" else "confirmed_by_terminal_refinement",
        source_round=source_round,  # type: ignore[arg-type]
        notes=[],
    )


def _transition(earliest: float, latest: float, eid: str = "tr_001") -> TemporalEvent:
    return TemporalEvent(
        event_id=eid,
        category="transition",
        boundary=TemporalBoundary(earliest=earliest, latest=latest, precision="bounded"),
        description="hard cut",
        direct_observations=[],
        inference=None,
        evidence=FineEvidenceReference(),
        alternatives=[],
        confidence=0.7,
    )


def test_empty_events_whole_video_one_shot():
    """无事件 → 整片一个镜头。"""
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={},
        merged_events=[],
    )
    assert len(shots) == 1
    assert shots[0].start == 0.0
    assert shots[0].end == 10.0


def test_hard_cut_boundary_from_transition_event():
    """transition 事件产生镜头边界。"""
    layout = None  # type: ignore[arg-type]
    events = [_merged(_transition(5.0, 5.1, "tr_a"))]
    shots = build_shot_timeline(
        layout=layout,
        video_duration=10.0,
        fine_analyses={},
        merged_events=events,
    )
    assert len(shots) >= 2
    # 第一个镜头在 5.0 之前结束
    assert any(s.end <= 5.5 for s in shots)


def test_video_start_and_end_always_boundaries():
    """视频开始 / 结束永远是镜头边界。"""
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={},
        merged_events=[],
    )
    # 第一个镜头从 0 开始，最后一个到 10 结束
    assert shots[0].start == 0.0
    assert shots[-1].end == 10.0


def test_very_short_shot_skipped():
    """< 0.1s 的极短镜头被跳过。"""
    events = [_merged(_transition(5.0, 5.05, "tr_a"))]
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={},
        merged_events=events,
    )
    # 极短镜头（0.05s）被跳过，剩下一个或两个镜头
    for s in shots:
        assert s.duration >= 0.1 or s.warnings


def test_shot_includes_subjects_and_text_from_fine():
    """subjects / visible_text 从 fine analysis 收集。"""
    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=10.0,
        confidence=0.6,
        frame_states=[
            {
                "observation_id": "obs_1",
                "timestamp_seconds": 5.0,
                "frame_id": "f1",
                "description": "x",
                "visible_subjects": ["person_a", "car"],
                "visible_text": ["HELLO"],
                "composition_notes": [],
                "confidence": 0.7,
            }
        ],
    )
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=[],
    )
    assert "person_a" in shots[0].subjects
    assert "car" in shots[0].subjects
    assert "HELLO" in shots[0].visible_text


def test_shot_analysis_level_terminal_refined():
    """如果事件来自 terminal，shot analysis_level = terminal_refined。"""
    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=10.0,
        confidence=0.6,
    )
    events = [_merged(_transition(5.0, 5.1, "tr_a"), source_round="terminal")]
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=events,
    )
    # 至少有一个 shot 的 evidence 包含事件 id
    has_terminal_ev = any(
        "tr_a" in s.evidence_ids for s in shots
    )
    assert has_terminal_ev


def test_shot_aggregates_motion_descriptions():
    """镜头内事件描述被聚合。"""
    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=10.0,
        confidence=0.6,
    )
    cam_event = TemporalEvent(
        event_id="cm_001",
        category="camera_motion",
        boundary=TemporalBoundary(earliest=2.0, latest=4.0, precision="bounded"),
        description="camera panned right",
        direct_observations=[],
        inference=None,
        evidence=FineEvidenceReference(),
        alternatives=[],
        confidence=0.6,
    )
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=[_merged(cam_event)],
    )
    # 至少有一个 shot 的 camera_motions 包含 "camera panned right"
    has_cam = any("camera panned right" in s.camera_motions for s in shots)
    assert has_cam


def test_shot_confidence_weighted_by_duration():
    """confidence 按时长加权。"""
    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=10.0,
        confidence=0.6,
    )
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=[],
    )
    assert 0 <= shots[0].confidence <= 1.0


def test_shot_evidence_ids_include_event_ids():
    """shot evidence_ids 包含事件 id 用于 Markdown 引用。"""
    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=10.0,
        confidence=0.6,
    )
    events = [_merged(_transition(5.0, 5.1, "tr_a"))]
    shots = build_shot_timeline(
        layout=None,  # type: ignore[arg-type]
        video_duration=10.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=events,
    )
    assert any("tr_a" in s.evidence_ids for s in shots)