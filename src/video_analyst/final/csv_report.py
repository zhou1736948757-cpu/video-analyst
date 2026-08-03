"""CSV report writers (Epic 5).

- shot_table.csv
- event_table.csv
- unresolved_items.csv

所有 CSV 从统一 FinalVideoAnalysis 生成，禁止各自独立重新推理。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.exceptions import ReportSerializationError
from video_analyst.final.models import (
    FinalVideoAnalysis,
    MergedTemporalEvent,
    UnresolvedItem,
)


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _join(values: list[str], sep: str = "|") -> str:
    """稳定编码多值字段。"""
    return sep.join(v.replace(sep, " ") for v in values if v)


def write_shot_table_csv(
    layout: OutputLayout,
    analysis: FinalVideoAnalysis,
) -> Path:
    path = layout.final_shot_table_csv
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "shot_id", "start", "end", "duration",
                "section_id",
                "visual_summary",
                "subjects",
                "camera_motions",
                "subject_motions",
                "text_animations",
                "graphic_animations",
                "transition_in",
                "transition_out",
                "narrative_function",
                "confidence",
                "analysis_level",
                "evidence_ids",
                "warnings",
            ])
            for shot in analysis.shots:
                writer.writerow([
                    shot.shot_id,
                    f"{shot.start:.3f}",
                    f"{shot.end:.3f}",
                    f"{shot.duration:.3f}",
                    shot.section_id or "",
                    shot.visual_summary.replace("\n", " ")[:500],
                    _join(shot.subjects),
                    _join(shot.camera_motions),
                    _join(shot.subject_motions),
                    _join(shot.text_animations),
                    _join(shot.graphic_animations),
                    _join(shot.transitions_in),
                    _join(shot.transitions_out),
                    _join(shot.narrative_function),
                    f"{shot.confidence:.3f}",
                    shot.analysis_level,
                    _join(shot.evidence_ids),
                    _join(shot.warnings),
                ])
    except Exception as exc:
        raise ReportSerializationError(f"failed to write shot_table.csv: {exc}") from exc
    return path


def write_event_table_csv(
    layout: OutputLayout,
    merged_events: list[MergedTemporalEvent],
) -> Path:
    path = layout.final_event_table_csv
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_id",
                "category",
                "start_earliest",
                "start_latest",
                "precision",
                "description",
                "alternatives",
                "confidence",
                "source_round",
                "evidence_ids",
                "status",
            ])
            for m in merged_events:
                ev = m.active
                writer.writerow([
                    m.merged_id,
                    ev.category,
                    f"{ev.boundary.earliest:.3f}",
                    f"{ev.boundary.latest:.3f}",
                    ev.boundary.precision,
                    ev.description.replace("\n", " ")[:500],
                    _join(ev.alternatives),
                    f"{ev.confidence:.3f}",
                    m.source_round,
                    _join(list(ev.evidence.frame_ids) + list(ev.evidence.contact_sheet_ids)),
                    m.status,
                ])
    except Exception as exc:
        raise ReportSerializationError(f"failed to write event_table.csv: {exc}") from exc
    return path


def write_unresolved_table_csv(
    layout: OutputLayout,
    items: list[UnresolvedItem],
) -> Path:
    path = layout.final_unresolved_table_csv
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "item_id",
                "time_start",
                "time_end",
                "question_or_issue",
                "reason",
                "terminal_attempted",
                "source",
                "priority",
                "evidence_ids",
            ])
            for it in items:
                writer.writerow([
                    it.item_id,
                    f"{it.time_start:.3f}" if it.time_start is not None else "",
                    f"{it.time_end:.3f}" if it.time_end is not None else "",
                    it.question_or_issue.replace("\n", " ")[:500],
                    it.reason.replace("\n", " ")[:500],
                    "yes" if it.terminal_attempted else "no",
                    it.source,
                    it.priority,
                    _join(it.evidence_ids),
                ])
    except Exception as exc:
        raise ReportSerializationError(f"failed to write unresolved_items.csv: {exc}") from exc
    return path


def write_csv_reports(
    layout: OutputLayout,
    *,
    analysis: FinalVideoAnalysis,
    merged_events: list[MergedTemporalEvent],
) -> dict[str, Path]:
    paths = {
        "shot_table": write_shot_table_csv(layout, analysis),
        "event_table": write_event_table_csv(layout, merged_events),
        "unresolved_table": write_unresolved_table_csv(layout, analysis.unresolved_items),
    }
    return paths


__all__ = [
    "write_csv_reports",
    "write_event_table_csv",
    "write_shot_table_csv",
    "write_unresolved_table_csv",
]