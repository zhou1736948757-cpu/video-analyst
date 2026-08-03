"""第二轮补帧计划生成（Epic 4）。

收集所有合法 SecondRefinementRequest，按预算合并排序，输出
``analysis/fine/second_refinement_plan.json``。

预算规则：
- 第二轮精查总时长 ≤ min(video_duration * budget_ratio, round1_total * round1_ratio)
- 高优先级优先
- 请求范围重叠时合并
- 合并后取最高 FPS
- 只允许 10 或 15 FPS
- 已为 15 FPS 的区间只能缩窄，不能升至更高
- 单区间最大 5 秒
- 精确 detail frame 请求按时间点保存
- 被预算排除的请求进入 deferred
- 不静默删除
- 标记 terminal_after_this_round = true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import FineAnalysisConfig
from video_analyst.exceptions import SecondRefinementPlanError
from video_analyst.fine.models import (
    FINE_SCHEMA_VERSION,
    SecondRefinementInterval,
    SecondRefinementPlan,
    SecondRefinementRequest,
)
from video_analyst.logging.setup import get_logger

log = get_logger("fine.second_refinement")

_PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3}
_FPS_RANK = {10: 1, 15: 2}


def _merge_overlap(
    intervals: list[SecondRefinementInterval],
) -> list[SecondRefinementInterval]:
    """合并重叠或接近的区间（gap ≤ 0.5s）。合并后取最高 FPS / 最高优先级。"""
    if not intervals:
        return []
    gap = 0.5
    sorted_ivs = sorted(intervals, key=lambda iv: (iv.start, iv.end))
    merged: list[SecondRefinementInterval] = []
    cur = sorted_ivs[0].model_copy(deep=True)
    for nxt in sorted_ivs[1:]:
        if nxt.start <= cur.end + gap:
            new_end = max(cur.end, nxt.end)
            new_fps = max(cur.requested_fps, nxt.requested_fps, key=lambda f: _FPS_RANK[f])
            new_priority = max(cur.priority, nxt.priority, key=lambda p: _PRIORITY_RANK[p])
            cur = SecondRefinementInterval(
                interval_id=cur.interval_id,
                start=cur.start,
                end=new_end,
                requested_fps=new_fps,
                priority=new_priority,
                reasons=list(dict.fromkeys(cur.reasons + nxt.reasons)),
                questions=list(dict.fromkeys(cur.questions + nxt.questions)),
                source_batch_ids=list(dict.fromkeys(cur.source_batch_ids + nxt.source_batch_ids)),
                evidence_frame_ids=list(dict.fromkeys(cur.evidence_frame_ids + nxt.evidence_frame_ids)),
                exact_detail_timestamps=sorted(
                    set(cur.exact_detail_timestamps + nxt.exact_detail_timestamps)
                ),
                estimated_frame_count=max(cur.estimated_frame_count, nxt.estimated_frame_count),
            )
        else:
            merged.append(cur)
            cur = nxt.model_copy(deep=True)
    merged.append(cur)
    return merged


def _clip_to_video(
    intervals: list[SecondRefinementInterval],
    *,
    video_start: float,
    video_end: float,
) -> list[SecondRefinementInterval]:
    out: list[SecondRefinementInterval] = []
    for iv in intervals:
        s = max(iv.start, video_start)
        e = min(iv.end, video_end)
        if e - s <= 0:
            continue
        out.append(
            iv.model_copy(
                update={
                    "start": round(s, 6),
                    "end": round(e, 6),
                    "estimated_frame_count": max(0, int(round((e - s) * iv.requested_fps))),
                }
            )
        )
    return out


def _split_long(
    intervals: list[SecondRefinementInterval],
    *,
    max_seconds: float,
) -> list[SecondRefinementInterval]:
    """拆分超过 max_seconds 的区间。"""
    out: list[SecondRefinementInterval] = []
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
                        "start": round(cur_start, 6),
                        "end": round(cur_end, 6),
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
    intervals: list[SecondRefinementInterval],
    *,
    budget: float,
) -> tuple[list[SecondRefinementInterval], list[SecondRefinementInterval]]:
    """按优先级 + FPS 排序，累计到预算为止。剩余进 deferred。"""
    if budget <= 0:
        return [], list(intervals)

    def sort_key(iv: SecondRefinementInterval) -> tuple[int, int, float]:
        return (
            -_PRIORITY_RANK[iv.priority],
            -_FPS_RANK[iv.requested_fps],
            iv.start,
        )

    sorted_ivs = sorted(intervals, key=sort_key)
    accepted: list[SecondRefinementInterval] = []
    deferred: list[SecondRefinementInterval] = []
    remaining = budget
    for iv in sorted_ivs:
        dur = iv.end - iv.start
        if dur <= remaining + 1e-6:
            accepted.append(iv)
            remaining -= dur
        else:
            deferred.append(iv)
    accepted.sort(key=lambda iv: iv.start)
    deferred.sort(key=lambda iv: iv.start)
    return accepted, deferred


def _request_to_interval(
    req: SecondRefinementRequest,
    *,
    idx: int,
) -> SecondRefinementInterval:
    est = max(0, int(round((req.end - req.start) * req.requested_fps)))
    return SecondRefinementInterval(
        interval_id=f"second_ref_{idx:04d}",
        start=round(req.start, 6),
        end=round(req.end, 6),
        requested_fps=req.requested_fps,
        priority=req.priority,
        reasons=[req.reason],
        questions=list(req.questions),
        source_batch_ids=list(req.evidence.frame_ids),  # 复用 evidence frame ids 作为来源标记
        evidence_frame_ids=list(req.evidence.frame_ids),
        exact_detail_timestamps=list(req.exact_detail_timestamps),
        estimated_frame_count=est,
    )


def build_second_refinement_plan(
    *,
    requests_by_batch: dict[str, list[SecondRefinementRequest]],
    video_duration: float,
    round1_refinement_duration: float,
    config: FineAnalysisConfig | None = None,
) -> SecondRefinementPlan:
    """从所有 batch 的 SecondRefinementRequest 生成第二轮补帧计划。"""
    cfg = config or FineAnalysisConfig()
    warnings: list[str] = []

    # 预算：min(video_duration * ratio, round1_total * round1_ratio)
    budget_a = video_duration * float(cfg.second_refinement_budget_ratio)
    budget_b = round1_refinement_duration * float(cfg.second_refinement_round1_ratio)
    budget = min(budget_a, budget_b)

    # 收集所有请求并转成 interval
    intervals: list[SecondRefinementInterval] = []
    counter = 0
    total_requested = 0.0
    for fb_id, reqs in requests_by_batch.items():
        for req in reqs:
            counter += 1
            iv = _request_to_interval(req, idx=counter)
            # 标记来源 batch
            iv = iv.model_copy(
                update={"source_batch_ids": [fb_id] + list(iv.source_batch_ids)}
            )
            intervals.append(iv)
            total_requested += req.end - req.start

    if not intervals:
        return SecondRefinementPlan(
            video_duration=video_duration,
            round1_refinement_duration=round1_refinement_duration,
            total_requested_duration=0.0,
            budget_duration=round(budget, 6),
            max_additional_rounds=1,
            terminal_after_this_round=True,
            intervals=[],
            deferred_intervals=[],
            generation_warnings=warnings,
        )

    # 合并重叠
    merged = _merge_overlap(intervals)
    # 裁剪到视频边界
    clipped = _clip_to_video(merged, video_start=0.0, video_end=video_duration)
    # 拆分超长区间
    split = _split_long(clipped, max_seconds=float(cfg.second_refinement_max_interval_seconds))
    # 应用预算
    accepted, deferred = _apply_budget(split, budget=budget)

    total_accepted = sum(iv.end - iv.start for iv in accepted)
    if deferred:
        warnings.append(
            f"{len(deferred)} second-refinement intervals deferred due to budget "
            f"{budget:.2f}s (video={video_duration:.1f}s, round1={round1_refinement_duration:.1f}s)."
        )

    return SecondRefinementPlan(
        video_duration=video_duration,
        round1_refinement_duration=round1_refinement_duration,
        total_requested_duration=round(total_accepted, 6),
        budget_duration=round(budget, 6),
        max_additional_rounds=1,
        terminal_after_this_round=True,
        intervals=accepted,
        deferred_intervals=deferred,
        generation_warnings=warnings,
    )


def save_second_refinement_plan(
    layout: OutputLayout, plan: SecondRefinementPlan
) -> Path:
    cache = CacheStore(layout)
    cache.write_json(
        layout.fine_second_refinement_plan_json, plan.model_dump(mode="json")
    )
    return layout.fine_second_refinement_plan_json


def load_second_refinement_plan(layout: OutputLayout) -> SecondRefinementPlan | None:
    if not layout.fine_second_refinement_plan_json.is_file():
        return None
    try:
        import json

        data = json.loads(layout.fine_second_refinement_plan_json.read_text(encoding="utf-8"))
        return SecondRefinementPlan.model_validate(data)
    except Exception as exc:
        log.warning("could not load second_refinement_plan.json: %s", exc)
        return None


__all__ = [
    "build_second_refinement_plan",
    "load_second_refinement_plan",
    "save_second_refinement_plan",
]
