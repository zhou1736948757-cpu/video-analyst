"""Section-level aggregation (Epic 5).

聚合 video_structure.json 的 sections + ShotAnalysis + merged events。

约束：
- 不得只用自然语言摘要字符串进行聚合
- 所有结论都有 evidence_ids
- 不连续镜头不误合并
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.exceptions import SectionAggregationError
from video_analyst.final.models import (
    MergedTemporalEvent,
    SectionAnalysis,
    ShotAnalysis,
)


def _shot_ids_in_range(
    shots: list[ShotAnalysis],
    start: float,
    end: float,
) -> list[str]:
    """返回时间窗口内的镜头 id 列表（按时间顺序）。"""
    return [
        s.shot_id for s in shots
        if s.start < end and s.end > start
    ]


def build_sections(
    *,
    layout: OutputLayout,
    shots: list[ShotAnalysis],
    merged_events: list[MergedTemporalEvent],
) -> list[SectionAnalysis]:
    """基于 video_structure.json 的 sections 构建段落级聚合。"""
    sections_raw: list[dict[str, Any]] = []
    if layout.coarse_structure_json.is_file():
        try:
            data = CacheStore(layout).read_json(layout.coarse_structure_json)
            sections_raw = list(data.get("sections") or [])
        except Exception:
            pass

    if not sections_raw:
        # 无结构 → 把整片当作一段
        return [
            SectionAnalysis(
                section_id="section_0001",
                start=shots[0].start if shots else 0.0,
                end=shots[-1].end if shots else 0.0,
                title="Section 1",
                summary="",
                shot_ids=[s.shot_id for s in shots],
                evidence_ids=[eid for s in shots for eid in s.evidence_ids],
                confidence=0.5,
            )
        ]

    out: list[SectionAnalysis] = []
    for i, sec in enumerate(sections_raw):
        try:
            start = float(sec["start"])
            end = float(sec["end"])
        except Exception:
            continue
        shot_ids = _shot_ids_in_range(shots, start, end)
        # 段内事件
        section_events = [
            m for m in merged_events
            if m.active.boundary.earliest < end and m.active.boundary.latest > start
        ]
        # 汇总 patterns（从 shot 文本聚合，仅结构化来源）
        main_subjects: list[str] = []
        visual_patterns: list[str] = []
        text_patterns: list[str] = []
        motion_patterns: list[str] = []
        transition_patterns: list[str] = []
        important_events: list[str] = []
        section_evidence: list[str] = []

        for s in shots:
            if s.shot_id not in shot_ids:
                continue
            for sub in s.subjects:
                if sub and sub not in main_subjects:
                    main_subjects.append(sub)
            for cm in s.camera_motions:
                if cm and cm not in visual_patterns:
                    visual_patterns.append(cm)
            for tm in s.text_animations:
                if tm and tm not in text_patterns:
                    text_patterns.append(tm)
            for sm in s.subject_motions:
                if sm and sm not in motion_patterns:
                    motion_patterns.append(sm)
            for tn in s.transitions_in + s.transitions_out:
                if tn and tn not in transition_patterns:
                    transition_patterns.append(tn)
            for eid in s.evidence_ids:
                if eid not in section_evidence:
                    section_evidence.append(eid)

        for m in section_events:
            important_events.append(f"[{m.active.boundary.earliest:.3f}-{m.active.boundary.latest:.3f}s] {m.active.description}")

        # confidence: shots 平均（按时长加权）
        shot_objs = [s for s in shots if s.shot_id in shot_ids]
        if shot_objs:
            total_dur = sum(s.duration for s in shot_objs)
            if total_dur > 0:
                confidence = sum(s.confidence * s.duration for s in shot_objs) / total_dur
            else:
                confidence = sum(s.confidence for s in shot_objs) / len(shot_objs)
        else:
            confidence = 0.5

        warnings: list[str] = []
        if not shot_objs:
            warnings.append("no shots inside this section")
        if not section_events:
            warnings.append("no temporal events inside this section")

        out.append(
            SectionAnalysis(
                section_id=sec.get("section_id", f"section_{i + 1:04d}"),
                start=round(start, 6),
                end=round(end, 6),
                title=sec.get("title", f"Section {i + 1}"),
                summary=sec.get("summary", ""),
                shot_ids=shot_ids,
                main_subjects=main_subjects,
                visual_patterns=visual_patterns,
                text_patterns=text_patterns,
                motion_patterns=motion_patterns,
                transition_patterns=transition_patterns,
                important_events=important_events,
                evidence_ids=section_evidence,
                confidence=round(confidence, 3),
                warnings=warnings,
            )
        )

    return out


def save_sections(layout: OutputLayout, sections: list[SectionAnalysis]) -> None:
    cache = CacheStore(layout)
    cache.write_json(
        layout.final_analysis_dir / "sections.json",
        {"sections": [s.model_dump(mode="json") for s in sections]},
    )


__all__ = ["build_sections", "save_sections"]