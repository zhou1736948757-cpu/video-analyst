"""Round 1 + terminal event merge (Epic 5).

合并规则：
- 优先级：terminal > round1 > coarse
- 时间窗口重叠 + 类别相同 + 证据指向相同变化 → terminal confirmed
- terminal 修正 → 保留旧结论到 superseded
- terminal 否定 → 保留旧结论 + 标记 rejected
- 不允许无条件覆盖：必须 evidence 更密集且 confidence 相当
"""

from __future__ import annotations

from typing import Any

from video_analyst.exceptions import FinalEventMergeError
from video_analyst.fine.models import (
    EventCategory,
    RefinementQuestionAnswer,
    TemporalEvent,
)
from video_analyst.fine.models import AnswerOrigin as _AnswerOrigin
from video_analyst.final.models import (
    MergedTemporalEvent,
    MergedTemporalEventStatus,
    UnresolvedItem,
)


def _window_overlap(a: TemporalEvent, b: TemporalEvent) -> bool:
    return a.boundary.earliest <= b.boundary.latest and b.boundary.earliest <= a.boundary.latest


def _evidence_overlap(a: TemporalEvent, b: TemporalEvent) -> bool:
    a_frames = set(a.evidence.frame_ids)
    b_frames = set(b.evidence.frame_ids)
    if a_frames & b_frames:
        return True
    a_sheets = set(a.evidence.contact_sheet_ids)
    b_sheets = set(b.evidence.contact_sheet_ids)
    if a_sheets & b_sheets:
        return True
    return False


def _evidence_density(ev: TemporalEvent) -> int:
    """事件密度 = frame + sheet 引用总数 + alternatives 数。"""
    return (
        len(ev.evidence.frame_ids)
        + len(ev.evidence.contact_sheet_ids)
        + len(ev.alternatives)
    )


def merge_temporal_events(
    *,
    round1_events: list[TemporalEvent],
    terminal_events: list[TemporalEvent],
) -> list[MergedTemporalEvent]:
    """合并 round1 和 terminal 事件。

    - 每个 round1 事件都被保留（可能被 terminal confirmed/revised/rejected）
    - 没有对应 round1 事件的 terminal 事件独立保留
    """
    if not terminal_events:
        merged = [
            MergedTemporalEvent(
                merged_id=f"m_r1_{i:04d}",
                active=ev,
                superseded=[],
                status="from_round1",
                source_round="round1",
                notes=[],
            )
            for i, ev in enumerate(round1_events)
        ]
        return merged

    matched_r1: set[int] = set()
    out: list[MergedTemporalEvent] = []
    # 先处理 terminal 事件
    for i, t in enumerate(terminal_events):
        # 寻找匹配的 round1 事件：类别相同 + 时间窗口重叠 + 证据有交集
        candidates: list[int] = []
        for j, r1 in enumerate(round1_events):
            if j in matched_r1:
                continue
            if r1.category != t.category:
                continue
            if not _window_overlap(r1, t):
                continue
            if not _evidence_overlap(r1, t):
                continue
            candidates.append(j)
        if candidates:
            # 取证据密度最高的 round1 事件作为基准
            candidates.sort(key=lambda j: _evidence_density(round1_events[j]), reverse=True)
            j = candidates[0]
            matched_r1.add(j)
            r1 = round1_events[j]
            # 文本相似度（Jaccard 词重叠）
            from video_analyst.fine.deduplicate import _jaccard, _word_set

            sim = _jaccard(_word_set(r1.description), _word_set(t.description))
            # 决定 confirmed / revised / rejected
            if _evidence_density(t) >= _evidence_density(r1) * 0.8 and t.confidence + 0.05 >= r1.confidence:
                if sim >= 0.3:
                    # 描述相似 + evidence 更密 → confirmed
                    out.append(
                        MergedTemporalEvent(
                            merged_id=f"m_t_{i:04d}",
                            active=t,
                            superseded=[r1],
                            status="confirmed_by_terminal_refinement",
                            source_round="terminal",
                            notes=[
                                f"confirmed by terminal; round1 evidence_density="
                                f"{_evidence_density(r1)} → terminal={_evidence_density(t)}; sim={sim:.2f}"
                            ],
                        )
                    )
                else:
                    # 描述差异大 + evidence 更密 → revised
                    out.append(
                        MergedTemporalEvent(
                            merged_id=f"m_t_{i:04d}",
                            active=t,
                            superseded=[r1],
                            status="revised_by_terminal_refinement",
                            source_round="terminal",
                            notes=[
                                f"revised by terminal; sim={sim:.2f} < 0.3; "
                                f"round1 conf={r1.confidence:.2f} → terminal={t.confidence:.2f}"
                            ],
                        )
                    )
            elif t.category != r1.category or t.description.strip() != r1.description.strip():
                # 描述差异较大 → revised
                out.append(
                    MergedTemporalEvent(
                        merged_id=f"m_t_{i:04d}",
                        active=t,
                        superseded=[r1],
                        status="revised_by_terminal_refinement",
                        source_round="terminal",
                        notes=[
                            f"revised by terminal; sim={sim:.2f}; "
                            f"round1 conf={r1.confidence:.2f} → terminal={t.confidence:.2f}"
                        ],
                    )
                )
            else:
                # 类似但 terminal confidence 较低 → rejected
                out.append(
                    MergedTemporalEvent(
                        merged_id=f"m_t_{i:04d}",
                        active=r1,
                        superseded=[t],
                        status="rejected_by_terminal_refinement",
                        source_round="round1",
                        notes=[
                            f"terminal refinement rejected; terminal conf={t.confidence:.2f}"
                        ],
                    )
                )
        else:
            # 没有匹配的 round1 事件，terminal 独立保留
            out.append(
                MergedTemporalEvent(
                    merged_id=f"m_t_{i:04d}",
                    active=t,
                    superseded=[],
                    status="confirmed_by_terminal_refinement",
                    source_round="terminal",
                    notes=["newly discovered by terminal"],
                )
            )

    # 未匹配的 round1 事件直接保留
    for j, r1 in enumerate(round1_events):
        if j in matched_r1:
            continue
        out.append(
            MergedTemporalEvent(
                merged_id=f"m_r1_{j:04d}",
                active=r1,
                superseded=[],
                status="from_round1",
                source_round="round1",
                notes=[],
            )
        )

    # 按时间排序
    out.sort(key=lambda m: m.active.boundary.earliest)
    return out


def merge_question_answers(
    *,
    round1_answers: list[RefinementQuestionAnswer],
    terminal_answers: list[RefinementQuestionAnswer] | None = None,
) -> list[RefinementQuestionAnswer]:
    """合并问题答案。

    规则：
    - terminal 答案覆盖 round1 同 question（仅当 terminal 真正回答或更新时）
    - terminal 标 unresolved 但 status="unresolved" + terminal_attempted=true → 保留原 round1 答案，但标记 terminal 尝试过
    - 未在 terminal 出现的问题保留 round1
    """
    terminal_answers = terminal_answers or []
    by_q: dict[str, RefinementQuestionAnswer] = {}
    for ans in round1_answers:
        by_q[ans.question.strip()] = ans

    # 找 terminal 中真正回答/更新的答案
    terminal_by_q: dict[str, RefinementQuestionAnswer] = {}
    for ans in terminal_answers:
        q = ans.question.strip()
        if ans.status in ("answered", "partially_answered") or ans.answer.strip():
            terminal_by_q[q] = ans

    out: list[RefinementQuestionAnswer] = []
    # 先处理 round1 中的所有问题
    for q, ans in by_q.items():
        if q in terminal_by_q:
            t_ans = terminal_by_q[q]
            # terminal 覆盖
            out.append(
                t_ans.model_copy(
                    update={
                        "answer_origin": "terminal_refinement",
                        "terminal_attempted": True,
                    }
                )
            )
        else:
            # round1 答案保留
            out.append(ans)

    # 添加 terminal 新增的问题
    for q, t_ans in terminal_by_q.items():
        if q not in by_q:
            out.append(
                t_ans.model_copy(
                    update={
                        "answer_origin": "terminal_refinement",
                        "terminal_attempted": True,
                    }
                )
            )

    return out


def collect_unresolved_items(
    *,
    merged_events: list[MergedTemporalEvent],
    round1_answers: list[RefinementQuestionAnswer],
    merged_answers: list[RefinementQuestionAnswer],
    deferred_intervals: list | None = None,
    coarse_gaps: list | None = None,
    fine_gaps: list | None = None,
) -> list[UnresolvedItem]:
    """从合并结果中收集未解决项。

    来源：
    - round1 unresolved + validator_fallback
    - 终局仍未解决（status=unresolved + terminal_attempted=true）
    - deferred 区间
    - coarse/fine gap
    """
    deferred_intervals = deferred_intervals or []
    coarse_gaps = coarse_gaps or []
    fine_gaps = fine_gaps or []
    items: list[UnresolvedItem] = []

    # 答案中的未解决项
    for ans in merged_answers:
        if ans.status == "unresolved":
            src = "validator_fallback" if ans.answer_origin == "validator_fallback" else (
                "terminal_unresolved" if ans.terminal_attempted else "round1_unresolved"
            )
            items.append(
                UnresolvedItem(
                    item_id=f"u_q_{len(items):04d}",
                    question_or_issue=ans.question,
                    reason=ans.answer or "insufficient evidence",
                    terminal_attempted=ans.terminal_attempted,
                    source=src,  # type: ignore[arg-type]
                    priority="medium",
                    evidence_ids=[],
                )
            )

    # 合并事件中 rejected 的旧事件
    for m in merged_events:
        if m.status == "rejected_by_terminal_refinement":
            items.append(
                UnresolvedItem(
                    item_id=f"u_e_{len(items):04d}",
                    time_start=m.active.boundary.earliest,
                    time_end=m.active.boundary.latest,
                    question_or_issue=f"event rejected by terminal: {m.active.description}",
                    reason=f"terminal refinement rejected with conf={m.active.confidence:.2f}",
                    terminal_attempted=True,
                    source="terminal_unresolved",
                    priority="low",
                    evidence_ids=[],
                )
            )

    # deferred 区间
    for i, iv in enumerate(deferred_intervals):
        try:
            start = float(getattr(iv, "start", 0.0))
            end = float(getattr(iv, "end", 0.0))
        except Exception:
            start, end = 0.0, 0.0
        items.append(
            UnresolvedItem(
                item_id=f"u_d_{i:04d}",
                time_start=start,
                time_end=end,
                question_or_issue=f"deferred interval excluded by budget",
                reason="round2 budget exceeded",
                terminal_attempted=True,
                source="deferred_terminal",
                priority="low",
                evidence_ids=[],
            )
        )

    # coarse / fine gap
    for i, gap in enumerate(coarse_gaps):
        try:
            start = float(getattr(gap, "start", 0.0))
            end = float(getattr(gap, "end", 0.0))
        except Exception:
            start, end = 0.0, 0.0
        items.append(
            UnresolvedItem(
                item_id=f"u_cg_{i:04d}",
                time_start=start,
                time_end=end,
                question_or_issue="coarse analysis gap",
                reason="batch failed",
                terminal_attempted=False,
                source="coarse_gap",
                priority="low",
                evidence_ids=[],
            )
        )
    for i, gap in enumerate(fine_gaps):
        try:
            start = float(getattr(gap, "start", 0.0))
            end = float(getattr(gap, "end", 0.0))
        except Exception:
            start, end = 0.0, 0.0
        items.append(
            UnresolvedItem(
                item_id=f"u_fg_{i:04d}",
                time_start=start,
                time_end=end,
                question_or_issue="fine analysis gap",
                reason="fine batch failed",
                terminal_attempted=False,
                source="fine_gap",
                priority="low",
                evidence_ids=[],
            )
        )

    return items


__all__ = [
    "collect_unresolved_items",
    "merge_question_answers",
    "merge_temporal_events",
]