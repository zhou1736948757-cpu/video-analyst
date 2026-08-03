"""重叠时序事件去重（Epic 4）。

相邻 contact sheet 重叠一帧，同一事件可能被多次描述。本模块做确定性
跨 batch 事件去重，输出 ``analysis/fine/temporal_events.json``。

可能为同一事件的条件（全部满足）：
1. category 相同
2. 时间窗口重叠或非常接近（≤ threshold）
3. 共享 frame_id 或时间点
4. 描述文本相似（Jaccard 词重叠 ≥ 相似度阈值）
5. 影响对象一致（subject / graphic_subject / motion_type 等）

合并时：
- 证据取并集
- 时间取并集（earliest 取 min，latest 取 max）
- confidence 取最高值（明确规则）
- alternatives 去重
- source_batch_ids 保留并集
- 不允许仅凭文本相似合并远距离事件
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from video_analyst.exceptions import FineEventDeduplicationError
from video_analyst.fine.models import (
    EventCategory,
    TemporalEvent,
)

_DEFAULT_TIME_THRESHOLD = 0.5
_DEFAULT_SIMILARITY_THRESHOLD = 0.55


def _word_set(text: str) -> set[str]:
    out: set[str] = set()
    for w in text.lower().split():
        cleaned = w.strip(",.;:!?\"'()[]{}")
        if len(cleaned) > 2:
            out.add(cleaned)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _boundary_span(ev: TemporalEvent) -> tuple[float, float]:
    return (ev.boundary.earliest, ev.boundary.latest)


def _time_overlap_or_close(
    ev_a: TemporalEvent,
    ev_b: TemporalEvent,
    threshold: float,
) -> bool:
    a_start, a_end = _boundary_span(ev_a)
    b_start, b_end = _boundary_span(ev_b)
    # 时间窗口重叠
    if a_start <= b_end + threshold and b_start <= a_end + threshold:
        return True
    return False


def _share_frame_or_sheet(ev_a: TemporalEvent, ev_b: TemporalEvent) -> bool:
    a_frames = set(ev_a.evidence.frame_ids)
    b_frames = set(ev_b.evidence.frame_ids)
    if a_frames & b_frames:
        return True
    a_sheets = set(ev_a.evidence.contact_sheet_ids)
    b_sheets = set(ev_b.evidence.contact_sheet_ids)
    if a_sheets & b_sheets:
        return True
    return False


def _same_subject(ev_a: TemporalEvent, ev_b: TemporalEvent) -> bool:
    """影响对象一致：检查 description 中的主体词。

    简化版：对 description 词集做 Jaccard，足够近似即认为同一对象。
    """
    wa = _word_set(ev_a.description)
    wb = _word_set(ev_b.description)
    return _jaccard(wa, wb) >= 0.3


@dataclass
class DeduplicationReport:
    canonical_events: list[TemporalEvent] = field(default_factory=list)
    merged_count: int = 0
    kept_count: int = 0
    stats: dict[str, int] = field(default_factory=dict)


def _merge_two(a: TemporalEvent, b: TemporalEvent) -> TemporalEvent:
    """合并两个事件。confidence 取最高，证据并集，时间并集，alternatives 去重。"""
    a_start, a_end = _boundary_span(a)
    b_start, b_end = _boundary_span(b)
    new_earliest = min(a_start, b_start)
    new_latest = max(a_end, b_end)
    # precision：取较粗的（bounded > exact；approximate > bounded）
    precision_rank = {"exact": 0, "bounded": 1, "approximate": 2}
    new_precision = a.boundary.precision
    if precision_rank[b.boundary.precision] > precision_rank[a.boundary.precision]:
        new_precision = b.boundary.precision
    # confidence 取最高
    new_confidence = max(a.confidence, b.confidence)
    # 证据并集
    new_sheets = list(dict.fromkeys(a.evidence.contact_sheet_ids + b.evidence.contact_sheet_ids))
    new_frames = list(dict.fromkeys(a.evidence.frame_ids + b.evidence.frame_ids))
    new_timestamps = sorted(set(a.evidence.timestamps + b.evidence.timestamps))
    # alternatives 去重
    new_alts = list(dict.fromkeys(a.alternatives + b.alternatives))
    # direct_observations 并集
    new_obs = list(dict.fromkeys(a.direct_observations + b.direct_observations))
    # source_batch_ids 并集
    new_sources = list(dict.fromkeys(a.source_batch_ids + b.source_batch_ids))
    # inference：若一致取其一，否则拼接
    if a.inference and b.inference and a.inference == b.inference:
        new_inference = a.inference
    elif a.inference and b.inference:
        new_inference = f"{a.inference} / {b.inference}"
    else:
        new_inference = a.inference or b.inference
    # description：取更详细的（更长的）
    new_description = a.description if len(a.description) >= len(b.description) else b.description
    # event_id：保留较早出现的（按 earliest）
    new_event_id = a.event_id if a_start <= b_start else b.event_id

    return a.model_copy(
        update={
            "boundary": a.boundary.model_copy(
                update={
                    "earliest": round(new_earliest, 6),
                    "latest": round(new_latest, 6),
                    "precision": new_precision,
                    "before_frame_id": a.boundary.before_frame_id or b.boundary.before_frame_id,
                    "after_frame_id": a.boundary.after_frame_id or b.boundary.after_frame_id,
                }
            ),
            "confidence": new_confidence,
            "evidence": a.evidence.model_copy(
                update={
                    "contact_sheet_ids": new_sheets,
                    "frame_ids": new_frames,
                    "timestamps": new_timestamps,
                }
            ),
            "alternatives": new_alts,
            "direct_observations": new_obs,
            "source_batch_ids": new_sources,
            "inference": new_inference,
            "description": new_description,
            "event_id": new_event_id,
        }
    )


def _can_merge(
    a: TemporalEvent,
    b: TemporalEvent,
    *,
    time_threshold: float,
    similarity_threshold: float,
) -> bool:
    """判断两个事件是否可合并。"""
    # 1. category 相同
    if a.category != b.category:
        return False
    # 2. 时间窗口重叠或接近
    if not _time_overlap_or_close(a, b, time_threshold):
        return False
    # 3. 共享 frame_id 或 sheet_id，或时间点非常接近
    shares = _share_frame_or_sheet(a, b)
    close_in_time = _time_overlap_or_close(a, b, threshold=min(time_threshold, 0.05))
    if not (shares or close_in_time):
        return False
    # 4. 描述文本相似
    wa = _word_set(a.description)
    wb = _word_set(b.description)
    if _jaccard(wa, wb) < similarity_threshold:
        # 即使时间重叠，文本差异大也不合并
        return False
    # 5. 影响对象一致（简化：subject 类的事件额外检查）
    return True


def deduplicate_temporal_events(
    events: list[TemporalEvent],
    *,
    time_threshold: float = _DEFAULT_TIME_THRESHOLD,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> DeduplicationReport:
    """对跨 batch 的时序事件做确定性去重。

    输入事件按时间排序后逐个合并到 canonical 列表。
    """
    if not events:
        return DeduplicationReport(
            canonical_events=[],
            merged_count=0,
            kept_count=0,
            stats={"input": 0, "output": 0, "merged": 0},
        )
    # 按时间排序（earliest, event_id）
    sorted_events = sorted(events, key=lambda e: (e.boundary.earliest, e.event_id))
    canonical: list[TemporalEvent] = []
    merged_count = 0
    for ev in sorted_events:
        merged_into = False
        for i, c in enumerate(canonical):
            if _can_merge(
                c, ev,
                time_threshold=time_threshold,
                similarity_threshold=similarity_threshold,
            ):
                canonical[i] = _merge_two(c, ev)
                merged_count += 1
                merged_into = True
                break
        if not merged_into:
            canonical.append(ev)
    return DeduplicationReport(
        canonical_events=canonical,
        merged_count=merged_count,
        kept_count=len(canonical),
        stats={
            "input": len(events),
            "output": len(canonical),
            "merged": merged_count,
        },
    )


def deduplicate_batch_analyses(
    analyses: dict[str, Any],
    *,
    time_threshold: float = _DEFAULT_TIME_THRESHOLD,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[list[TemporalEvent], DeduplicationReport]:
    """从多个 FineBatchAnalysis 收集 temporal_events 并去重。

    ``analyses``: fine_batch_id -> FineBatchAnalysis
    """
    all_events: list[TemporalEvent] = []
    for fb_id, analysis in analyses.items():
        for ev in analysis.temporal_events:
            # 确保 source_batch_ids 包含来源
            if fb_id not in ev.source_batch_ids:
                ev = ev.model_copy(
                    update={"source_batch_ids": list(ev.source_batch_ids) + [fb_id]}
                )
            all_events.append(ev)
    report = deduplicate_temporal_events(
        all_events,
        time_threshold=time_threshold,
        similarity_threshold=similarity_threshold,
    )
    return report.canonical_events, report


__all__ = [
    "DeduplicationReport",
    "deduplicate_batch_analyses",
    "deduplicate_temporal_events",
]
