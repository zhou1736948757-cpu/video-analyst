"""全片精细分析聚合（Epic 4）。

第一版优先使用确定性聚合，不额外调用总结模型。

规则：
- 按 batch 时间顺序汇总
- failed batch 对应范围标记为 analysis_gaps
- 不能把未分析区间描述为"没有动效"
- question_answers 聚合
- unresolved_questions 保留
- summary 分类：subject_motion / camera_motion / text_animation /
  graphic_animation / transition
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.exceptions import FineAggregationError
from video_analyst.fine.deduplicate import deduplicate_batch_analyses
from video_analyst.fine.models import (
    FINE_SCHEMA_VERSION,
    FineBatchAnalysis,
    FineVideoAnalysis,
    RefinementQuestionAnswer,
    TemporalEvent,
    TimeRange,
)
from video_analyst.logging.setup import get_logger

log = get_logger("fine.aggregate")


def _classify_events(
    events: list[TemporalEvent],
) -> dict[str, list[str]]:
    """把事件按类别汇总成人类可读摘要。"""
    out: dict[str, list[str]] = {
        "subject_motion": [],
        "camera_motion": [],
        "text_animation": [],
        "graphic_animation": [],
        "transition": [],
    }
    for ev in events:
        cat = ev.category
        line = (
            f"[{ev.boundary.earliest:.3f}-{ev.boundary.latest:.3f}s] "
            f"{ev.description} (confidence={ev.confidence:.2f})"
        )
        if cat == "subject_motion":
            out["subject_motion"].append(line)
        elif cat == "camera_motion":
            out["camera_motion"].append(line)
        elif cat == "text_animation":
            out["text_animation"].append(line)
        elif cat == "graphic_animation":
            out["graphic_animation"].append(line)
        elif cat == "transition":
            out["transition"].append(line)
        # appearance_change / layout_change / uncertain_change 不进分类摘要
    return out


def _build_analyzed_intervals(
    analyses: dict[str, FineBatchAnalysis],
    batches_order: list[str],
) -> list[TimeRange]:
    out: list[TimeRange] = []
    for fb_id in batches_order:
        a = analyses.get(fb_id)
        if a is None:
            continue
        out.append(TimeRange(start=round(a.start, 6), end=round(a.end, 6)))
    return out


def _build_analysis_gaps(
    failed_batch_ids: list[str],
    batches_meta: dict[str, dict[str, Any]] | None,
) -> list[TimeRange]:
    """failed batch 对应范围标记为 gap。"""
    meta = batches_meta or {}
    gaps: list[TimeRange] = []
    for fb_id in failed_batch_ids:
        m = meta.get(fb_id) or {}
        start = float(m.get("start", 0.0))
        end = float(m.get("end", 0.0))
        if end > start:
            gaps.append(TimeRange(start=round(start, 6), end=round(end, 6)))
    return gaps


def _collect_question_answers(
    analyses: dict[str, FineBatchAnalysis],
) -> tuple[list[RefinementQuestionAnswer], list[str]]:
    """聚合问题回答，收集未解决问题。"""
    answered: list[RefinementQuestionAnswer] = []
    unresolved: list[str] = []
    seen_questions: set[str] = set()
    for fb_id, a in analyses.items():
        for qa in a.question_answers:
            if qa.question in seen_questions:
                # 同一问题可能跨 batch 出现；保留第一个有效回答
                continue
            seen_questions.add(qa.question)
            answered.append(qa)
            if qa.status == "unresolved":
                unresolved.append(qa.question)
        for item in a.unresolved_items:
            if item.strip() and item not in unresolved:
                unresolved.append(item)
    return answered, unresolved


def _collect_important_events(
    events: list[TemporalEvent],
) -> list[str]:
    """挑出 confidence ≥ 0.6 的事件作为重要事件。"""
    out: list[str] = []
    for ev in sorted(events, key=lambda e: e.boundary.earliest):
        if ev.confidence >= 0.6:
            out.append(
                f"[{ev.boundary.earliest:.3f}-{ev.boundary.latest:.3f}s] "
                f"{ev.category}: {ev.description}"
            )
    return out


def aggregate_fine_analysis(
    *,
    analyses: dict[str, FineBatchAnalysis],
    failed_batches: dict[str, str],
    batches_order: list[str],
    batches_meta: dict[str, dict[str, Any]] | None = None,
    video_start: float,
    video_end: float,
    second_refinement_required: bool = False,
    second_refinement_plan_path: Path | None = None,
) -> FineVideoAnalysis:
    """确定性聚合全片精细分析。"""
    if video_end <= video_start:
        raise FineAggregationError(
            f"invalid video duration: {video_end - video_start}"
        )

    # 去重时序事件
    canonical_events, dedup_report = deduplicate_batch_analyses(analyses)
    log.info(
        "fine aggregation: %d input events -> %d canonical (merged %d)",
        dedup_report.stats.get("input", 0),
        dedup_report.stats.get("output", 0),
        dedup_report.stats.get("merged", 0),
    )

    # 分类摘要
    classified = _classify_events(canonical_events)

    # 分析区间 / 缺口
    analyzed = _build_analyzed_intervals(analyses, batches_order)
    gaps = _build_analysis_gaps(list(failed_batches.keys()), batches_meta)

    # 问题回答聚合
    answered, unresolved = _collect_question_answers(analyses)

    # 重要事件
    important = _collect_important_events(canonical_events)

    # 整体 confidence：所有 batch 的均值
    confidences = [a.confidence for a in analyses.values()]
    overall_conf = sum(confidences) / len(confidences) if confidences else 0.3

    warnings: list[str] = []
    if failed_batches:
        warnings.append(
            f"{len(failed_batches)} fine batch(es) failed; analysis_gaps recorded."
        )
    if not analyses:
        warnings.append(
            "no successful fine analyses; temporal_events empty (NOT 'no motion')."
        )

    return FineVideoAnalysis(
        schema_version=FINE_SCHEMA_VERSION,
        video_start=round(video_start, 6),
        video_end=round(video_end, 6),
        analyzed_intervals=analyzed,
        analysis_gaps=gaps,
        temporal_events=canonical_events,
        subject_motion_summary=classified["subject_motion"],
        camera_motion_summary=classified["camera_motion"],
        text_animation_summary=classified["text_animation"],
        graphic_animation_summary=classified["graphic_animation"],
        transition_summary=classified["transition"],
        answered_questions=answered,
        unresolved_questions=unresolved,
        important_events=important,
        second_refinement_required=second_refinement_required,
        second_refinement_plan_path=second_refinement_plan_path,
        confidence=round(overall_conf, 3),
        warnings=warnings,
    )


def save_fine_video_analysis(
    layout: OutputLayout, analysis: FineVideoAnalysis
) -> Path:
    cache = CacheStore(layout)
    cache.write_json(
        layout.fine_video_analysis_json, analysis.model_dump(mode="json")
    )
    return layout.fine_video_analysis_json


def save_temporal_events(
    layout: OutputLayout, events: list[TemporalEvent]
) -> Path:
    """保存去重后的时序事件到 temporal_events.json。"""
    cache = CacheStore(layout)
    payload = {
        "schema_version": FINE_SCHEMA_VERSION,
        "events": [e.model_dump(mode="json") for e in events],
    }
    cache.write_json(layout.fine_temporal_events_json, payload)
    return layout.fine_temporal_events_json


__all__ = [
    "aggregate_fine_analysis",
    "save_fine_video_analysis",
    "save_temporal_events",
]
