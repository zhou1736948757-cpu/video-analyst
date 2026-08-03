"""Tests for temporal analysis scenarios (Epic 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.fine.models import (
    CameraMotionAnalysis,
    TemporalBoundary,
    TemporalEvent,
    TextAnimationAnalysis,
    TransitionAnalysis,
)

from tests._epic4_fixtures import has_pillow

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _ev(
    *,
    category: str = "subject_motion",
    earliest: float = 5.0,
    latest: float = 6.0,
    precision: str = "bounded",
    description: str = "subject moved",
    confidence: float = 0.6,
) -> TemporalEvent:
    return TemporalEvent(
        event_id="ev_001",
        category=category,  # type: ignore[arg-type]
        boundary=TemporalBoundary(
            earliest=earliest, latest=latest, precision=precision  # type: ignore[arg-type]
        ),
        description=description,
        direct_observations=["obs_1"],
        inference="inferred motion",
        evidence={"contact_sheet_ids": ["s1"], "frame_ids": ["f1"], "timestamps": [earliest]},
        alternatives=["alt1"],
        confidence=confidence,
    )


def test_subject_motion_category():
    ev = _ev(category="subject_motion")
    assert ev.category == "subject_motion"


def test_camera_motion_category():
    ev = _ev(category="camera_motion")
    assert ev.category == "camera_motion"


def test_post_scale_vs_camera_zoom_uncertain():
    """无法区分 post scale 和 camera zoom 时用 uncertain_change。"""
    ev = _ev(
        category="uncertain_change",
        description="content scaled up; cause undetermined",
        confidence=0.3,
    )
    assert ev.category == "uncertain_change"
    assert ev.confidence < 0.5


def test_text_appear_event():
    ev = _ev(
        category="text_animation",
        description="text HELLO appeared",
        confidence=0.6,
    )
    assert ev.category == "text_animation"


def test_text_fade_event():
    """文字淡入事件。"""
    ev = _ev(
        category="text_animation",
        description="text faded in",
        confidence=0.55,
    )
    assert ev.category == "text_animation"


def test_graphic_position_event():
    ev = _ev(
        category="graphic_animation",
        description="progress bar advanced",
        confidence=0.6,
    )
    assert ev.category == "graphic_animation"


def test_hard_cut_transition():
    """硬切：相邻帧状态突变，无中间混合。"""
    ev = _ev(
        category="transition",
        earliest=5.0,
        latest=5.1,
        precision="bounded",
        description="hard cut between scene A and B",
        confidence=0.7,
    )
    assert ev.category == "transition"
    assert ev.confidence >= 0.6


def test_dissolve_transition():
    """dissolve 需中间混合状态证据。"""
    ev = _ev(
        category="transition",
        description="dissolve with blend frames",
        confidence=0.6,
    )
    assert ev.category == "transition"


def test_uncertain_when_cannot_distinguish():
    ev = _ev(
        category="uncertain_change",
        description="ambiguous",
        confidence=0.3,
    )
    assert ev.category == "uncertain_change"
    assert ev.alternatives


def test_bounded_boundary():
    """bounded precision：只能确认发生在两帧之间。"""
    b = TemporalBoundary(
        earliest=5.0, latest=6.0, precision="bounded", before_frame_id="f1", after_frame_id="f2"
    )
    assert b.precision == "bounded"
    assert b.latest > b.earliest


def test_exact_boundary():
    """exact precision：单帧内确认。"""
    b = TemporalBoundary(earliest=5.0, latest=5.0, precision="exact", before_frame_id=None, after_frame_id="f1")
    assert b.precision == "exact"
    assert b.earliest == b.latest


def test_approximate_boundary():
    b = TemporalBoundary(earliest=5.0, latest=7.0, precision="approximate")
    assert b.precision == "approximate"


def test_boundary_latest_before_earliest_rejected():
    with pytest.raises(Exception):
        TemporalBoundary(earliest=6.0, latest=5.0, precision="bounded")


def test_camera_motion_analysis_uncertain_type():
    """camera_motion analysis 用 uncertain 类型。"""
    cm = CameraMotionAnalysis(
        event_id="cm_001",
        motion_type="uncertain",
        description="cannot distinguish zoom/dolly/post-scale",
        boundary=TemporalBoundary(earliest=5.0, latest=6.0, precision="bounded"),
        evidence={"contact_sheet_ids": ["s1"], "frame_ids": ["f1"], "timestamps": [5.0]},
        alternatives=["optical zoom_in", "dolly_in", "post_transform scale"],
        confidence=0.3,
    )
    assert cm.motion_type == "uncertain"
    assert len(cm.alternatives) == 3


def test_text_animation_analysis_appear():
    ta = TextAnimationAnalysis(
        event_id="ta_001",
        text_content="HELLO",
        animation_types=["appear"],
        start_state="no text",
        end_state="text visible",
        boundary=TemporalBoundary(earliest=5.0, latest=6.0, precision="bounded"),
        evidence={"contact_sheet_ids": ["s1"], "frame_ids": ["f1"], "timestamps": [5.0]},
        alternatives=["fade", "fly-in"],
        confidence=0.6,
    )
    assert ta.text_content == "HELLO"
    assert "appear" in ta.animation_types


def test_text_animation_unclear_text_null():
    """文字看不清时 text_content=None。"""
    ta = TextAnimationAnalysis(
        event_id="ta_002",
        text_content=None,
        animation_types=["appear"],
        start_state="unclear",
        end_state="unclear",
        boundary=TemporalBoundary(earliest=5.0, latest=6.0, precision="bounded"),
        evidence={"contact_sheet_ids": ["s1"], "frame_ids": ["f1"], "timestamps": [5.0]},
        alternatives=[],
        confidence=0.3,
    )
    assert ta.text_content is None


def test_transition_analysis_cut():
    tr = TransitionAnalysis(
        event_id="tr_001",
        transition_type="cut",
        outgoing_state="scene A",
        incoming_state="scene B",
        boundary=TemporalBoundary(earliest=5.0, latest=5.1, precision="bounded"),
        evidence={"contact_sheet_ids": ["s1"], "frame_ids": ["f1"], "timestamps": [5.0]},
        alternatives=["very fast dissolve"],
        confidence=0.7,
    )
    assert tr.transition_type == "cut"


def test_transition_analysis_uncertain():
    tr = TransitionAnalysis(
        event_id="tr_002",
        transition_type="uncertain",
        outgoing_state="scene A",
        incoming_state="scene B",
        boundary=TemporalBoundary(earliest=5.0, latest=6.0, precision="bounded"),
        evidence={"contact_sheet_ids": ["s1"], "frame_ids": ["f1"], "timestamps": [5.0]},
        alternatives=["cut", "fast dissolve"],
        confidence=0.3,
    )
    assert tr.transition_type == "uncertain"
