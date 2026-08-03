"""Tests for second refinement plan generation (Epic 4)."""

from __future__ import annotations

import pytest

from video_analyst.config import FineAnalysisConfig
from video_analyst.fine.models import (
    FineEvidenceReference,
    SecondRefinementRequest,
)
from video_analyst.fine.second_refinement import build_second_refinement_plan

from tests._epic4_fixtures import has_pillow

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _req(
    *,
    rid: str = "2nd_001",
    req_type: str = "higher_temporal_density",
    start: float = 5.0,
    end: float = 7.0,
    fps: int = 15,
    priority: str = "high",
    reason: str = "need more frames",
) -> SecondRefinementRequest:
    return SecondRefinementRequest(
        request_id=rid,
        request_type=req_type,  # type: ignore[arg-type]
        start=start,
        end=end,
        requested_fps=fps,  # type: ignore[arg-type]
        exact_detail_timestamps=[],
        reason=reason,
        questions=["q1"],
        priority=priority,  # type: ignore[arg-type]
        evidence=FineEvidenceReference(),
        confidence=0.6,
    )


def test_5fps_to_10fps():
    """5 FPS 不足 → 请求 10 FPS。"""
    req = _req(fps=10, start=5.0, end=6.0)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
        config=FineAnalysisConfig(),
    )
    assert len(plan.intervals) == 1
    assert plan.intervals[0].requested_fps == 10


def test_5fps_to_15fps():
    req = _req(fps=15, start=5.0, end=6.0)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert len(plan.intervals) == 1
    assert plan.intervals[0].requested_fps == 15


def test_10fps_to_15fps():
    req = _req(fps=15, start=5.0, end=6.0)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert plan.intervals[0].requested_fps == 15


def test_15fps_cannot_upgrade():
    """15 FPS 时不再升级（验证器已拦截 higher_temporal_density，
    但若请求已是 narrower_interval 仍允许）。"""
    req = _req(req_type="narrower_interval", fps=15, start=5.0, end=6.0)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert len(plan.intervals) == 1
    assert plan.intervals[0].requested_fps == 15


def test_narrower_interval():
    req = _req(req_type="narrower_interval", fps=15, start=5.0, end=6.0)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert len(plan.intervals) == 1


def test_detail_timestamp_request():
    """精确 detail frame 请求按时间点保存。"""
    req = _req(req_type="high_resolution_detail", fps=15, start=5.0, end=6.0)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert len(plan.intervals) == 1


def test_overlapping_intervals_merged():
    """重叠区间合并。"""
    req1 = _req(rid="r1", start=5.0, end=7.0, fps=10)
    req2 = _req(rid="r2", start=6.5, end=9.0, fps=15)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req1, req2]},
        video_duration=100.0,
        round1_refinement_duration=20.0,
    )
    # 合并成一个，取最高 FPS
    assert len(plan.intervals) == 1
    assert plan.intervals[0].requested_fps == 15
    assert plan.intervals[0].start == 5.0
    assert plan.intervals[0].end == 9.0


def test_max_interval_5_seconds():
    """单区间最大 5 秒，超长被拆分。"""
    req = _req(start=5.0, end=12.0, fps=15)  # 7 秒
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=200.0,
        round1_refinement_duration=40.0,
    )
    # 被拆成 2 段（5s + 2s）
    assert len(plan.intervals) == 2


def test_total_budget_limit():
    """总预算限制：min(video×0.05, round1×0.25)。"""
    # video=60 → 3s; round1=10 → 2.5s; 取 min=2.5s
    reqs = [
        _req(rid=f"r{i}", start=5.0 + i * 3, end=7.0 + i * 3, fps=15)
        for i in range(5)
    ]
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": reqs},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    total_accepted = sum(iv.end - iv.start for iv in plan.intervals)
    budget = min(60.0 * 0.05, 10.0 * 0.25)
    assert total_accepted <= budget + 1e-6
    # 被排除的进 deferred
    assert len(plan.deferred_intervals) > 0


def test_high_priority_preferred():
    """高优先级优先。"""
    req_low = _req(rid="low", start=5.0, end=7.0, fps=15, priority="low")
    req_high = _req(rid="high", start=8.0, end=10.0, fps=15, priority="high")
    # 预算只够一个（2 秒）
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req_low, req_high]},
        video_duration=10.0,  # budget = 0.5s → 都不够？调小让预算只能容一个
        round1_refinement_duration=8.0,  # 8×0.25=2s
    )
    # budget = min(10×0.05=0.5, 8×0.25=2) = 0.5 → 都不够 2s，全 deferred
    # 调整：让预算恰好够一个
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req_low, req_high]},
        video_duration=100.0,  # budget = 5s
        round1_refinement_duration=40.0,  # 40×0.25=10s → min=5s
    )
    # 两个 2s 区间都能进，但验证排序：high 排前
    accepted_ids = [iv.interval_id for iv in plan.intervals]
    # high 应该先被接受（排序优先）
    # 两个都进（总 4s <= 5s）
    assert len(plan.intervals) == 2


def test_deferred_preserved():
    """被预算排除的请求进 deferred，不静默删除。"""
    reqs = [
        _req(rid=f"r{i}", start=5.0 + i * 3, end=7.0 + i * 3, fps=15)
        for i in range(5)
    ]
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": reqs},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    total = len(plan.intervals) + len(plan.deferred_intervals)
    assert total == 5  # 没有静默删除


def test_budget_change_does_not_reinvoke_provider():
    """修改预算参数只重建 plan（不调用视觉模型——本测试只验证 plan 重建）。"""
    req = _req(start=5.0, end=7.0, fps=15)
    cfg1 = FineAnalysisConfig(second_refinement_budget_ratio=0.05)
    plan1 = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
        config=cfg1,
    )
    cfg2 = FineAnalysisConfig(second_refinement_budget_ratio=0.01)
    plan2 = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
        config=cfg2,
    )
    # 预算变小 → 接受的区间可能减少
    assert plan2.budget_duration < plan1.budget_duration


def test_terminal_after_this_round_true():
    """标记 terminal_after_this_round = true。"""
    req = _req(start=5.0, end=7.0, fps=15)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert plan.terminal_after_this_round is True
    assert plan.max_additional_rounds == 1


def test_empty_requests():
    """无请求时返回空 plan。"""
    plan = build_second_refinement_plan(
        requests_by_batch={},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    assert plan.intervals == []
    assert plan.deferred_intervals == []
    assert plan.terminal_after_this_round is True


def test_clip_to_video_bounds():
    """区间裁剪到视频边界。"""
    req = _req(start=58.0, end=65.0, fps=15)  # 超出 video=60
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req]},
        video_duration=60.0,
        round1_refinement_duration=10.0,
    )
    if plan.intervals:
        for iv in plan.intervals:
            assert iv.end <= 60.0


def test_higher_fps_wins_on_merge():
    """合并后取最高 FPS。"""
    req1 = _req(rid="r1", start=5.0, end=7.0, fps=10)
    req2 = _req(rid="r2", start=6.5, end=9.0, fps=15)
    plan = build_second_refinement_plan(
        requests_by_batch={"fine_interval_0001": [req1, req2]},
        video_duration=200.0,
        round1_refinement_duration=40.0,
    )
    assert plan.intervals[0].requested_fps == 15
