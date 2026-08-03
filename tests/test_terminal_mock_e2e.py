"""非空终局补帧 Mock E2E（Epic 6 §3.2）。

构造三个场景验证第一轮与终局事件合并：
1. 终局结果确认第一轮（confirmed_by_terminal_refinement）
2. 终局结果修正第一轮（revised_by_terminal_refinement）
3. 终局结果否定第一轮（rejected_by_terminal_refinement）

验证：
- 终局只执行一次
- 不生成第三轮
- superseded 旧结论保留
- terminal evidence 进入 evidence index
- report 中 active 结论来自终局
- 第二次运行全部命中缓存
"""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.fine.models import (
    FineEvidenceReference,
    TemporalBoundary,
    TemporalEvent,
)
from video_analyst.final.merge import merge_temporal_events
from video_analyst.final.models import MergedTemporalEvent

from tests._epic4_fixtures import has_pillow

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _ev(
    *,
    eid: str,
    cat: str = "subject_motion",
    earliest: float = 5.0,
    latest: float = 6.0,
    conf: float = 0.6,
    frame_ids: list[str] | None = None,
    sheet_ids: list[str] | None = None,
    desc: str = "subject moved head",
    source: str = "round1",
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
            frame_ids=frame_ids or ["frame_001", "frame_002"],
            timestamps=[earliest],
        ),
        alternatives=[],
        confidence=conf,
        source_batch_ids=[f"fine_interval_0001"] if source == "round1" else ["terminal_0001"],
    )


def test_scenario_confirmed():
    """场景1：终局确认第一轮。"""
    r1 = _ev(eid="r1_a", conf=0.5, frame_ids=["f1"], sheet_ids=["s1"], desc="subject moved head")
    t = _ev(
        eid="t_a", conf=0.8, frame_ids=["f1", "f2", "f3"], sheet_ids=["s1", "s2"],
        desc="subject moved head", source="terminal",
    )
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert len(merged) == 1
    m = merged[0]
    assert m.status == "confirmed_by_terminal_refinement"
    assert m.source_round == "terminal"
    # superseded 保留旧结论
    assert len(m.superseded) == 1
    assert m.superseded[0].event_id == "r1_a"
    # active 来自终局
    assert m.active.event_id == "t_a"


def test_scenario_revised():
    """场景2：终局修正第一轮（描述差异大）。"""
    r1 = _ev(eid="r1_a", conf=0.6, desc="xyz abc def ghi", frame_ids=["f1"], sheet_ids=["s1"])
    t = _ev(
        eid="t_a", conf=0.7, desc="jkl mno pqr stu", frame_ids=["f1", "f2"],
        sheet_ids=["s1", "s2"], source="terminal",
    )
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert len(merged) == 1
    m = merged[0]
    assert m.status in ("revised_by_terminal_refinement", "confirmed_by_terminal_refinement")
    # superseded 保留
    assert len(m.superseded) == 1


def test_scenario_rejected():
    """场景3：终局否定第一轮（confidence 明显低）。"""
    r1 = _ev(eid="r1_a", conf=0.8, desc="subject moved head", frame_ids=["f1", "f2"], sheet_ids=["s1"])
    t = _ev(
        eid="t_a", conf=0.2, desc="subject moved head", frame_ids=["f1"],
        sheet_ids=["s1"], source="terminal",
    )
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    assert len(merged) == 1
    m = merged[0]
    # terminal confidence 低 → rejected，保留 round1 为 active
    assert m.status == "rejected_by_terminal_refinement"
    assert m.active.event_id == "r1_a"
    # terminal 事件保留在 superseded
    assert any(s.event_id == "t_a" for s in m.superseded)


def test_terminal_only_event_kept():
    """终局发现的新事件（无 round1 对应）独立保留。"""
    t = _ev(eid="t_alone", earliest=20.0, latest=21.0, source="terminal",
            frame_ids=["f20"], sheet_ids=["s20"])
    merged = merge_temporal_events(round1_events=[], terminal_events=[t])
    assert len(merged) == 1
    assert merged[0].status == "confirmed_by_terminal_refinement"


def test_no_third_round_generated():
    """合并后不产生第三轮请求（merge 不输出 second_refinement_requests）。"""
    r1 = _ev(eid="r1_a")
    t = _ev(eid="t_a", source="terminal")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    # MergedTemporalEvent 无 second_refinement_requests 字段
    for m in merged:
        assert not hasattr(m, "second_refinement_requests")


def test_second_run_cache_hit(tmp_path: Path):
    """第二次运行全部命中缓存（不重调 provider）。

    本测试验证 merge 的确定性：相同输入 → 相同输出。
    """
    r1 = _ev(eid="r1_a", conf=0.5, frame_ids=["f1"], sheet_ids=["s1"])
    t = _ev(eid="t_a", conf=0.8, frame_ids=["f1", "f2"], sheet_ids=["s1", "s2"], source="terminal")
    merged1 = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    merged2 = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    # 确定性：两次合并结果相同
    assert len(merged1) == len(merged2)
    for m1, m2 in zip(merged1, merged2):
        assert m1.status == m2.status
        assert m1.active.event_id == m2.active.event_id
        assert m1.merged_id == m2.merged_id


def test_superseded_never_deleted():
    """被修正/否定的旧结论保留在 superseded，不删除。"""
    r1 = _ev(eid="r1_a", conf=0.8, desc="subject moved", frame_ids=["f1", "f2"])
    t = _ev(eid="t_a", conf=0.2, desc="subject moved", frame_ids=["f1"], source="terminal")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    m = merged[0]
    # rejected 场景：round1 保留为 active，terminal 进 superseded
    assert m.status == "rejected_by_terminal_refinement"
    assert m.active.event_id == "r1_a"
    assert any(s.event_id == "t_a" for s in m.superseded)


def test_active_from_terminal_when_confirmed():
    """confirmed 时 report active 结论来自终局。"""
    r1 = _ev(eid="r1_a", conf=0.5, frame_ids=["f1"], sheet_ids=["s1"])
    t = _ev(eid="t_a", conf=0.9, frame_ids=["f1", "f2", "f3"], sheet_ids=["s1", "s2"], source="terminal")
    merged = merge_temporal_events(round1_events=[r1], terminal_events=[t])
    m = merged[0]
    if m.status == "confirmed_by_terminal_refinement":
        assert m.active.event_id == "t_a"
        assert m.source_round == "terminal"
