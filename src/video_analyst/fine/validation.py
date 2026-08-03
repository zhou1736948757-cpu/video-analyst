"""输入输出契约校验（Epic 4 fine analysis）。"""

from __future__ import annotations

import json
from pathlib import Path

from video_analyst.exceptions import (
    FineBatchAnalysisError,
    FineVisionResponseValidationError,
)
from video_analyst.fine.models import (
    FineAnalysisManifest,
    FineBatchAnalysis,
    FineVisionRequest,
)


def load_fine_batches_index(path: Path) -> dict:
    """读取并最小校验 fine_batches.json。"""
    if not path.is_file():
        raise FineBatchAnalysisError(
            f"fine_batches.json missing at {path}; run Epic 3 first"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FineBatchAnalysisError(f"fine_batches.json unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise FineBatchAnalysisError("fine_batches.json must be a JSON object")
    batches = data.get("batches")
    if not isinstance(batches, list):
        raise FineBatchAnalysisError(
            "fine_batches.json must contain a 'batches' list"
        )
    return data


def assert_fine_analysis_within_interval(
    analysis: FineBatchAnalysis,
    *,
    interval_start: float,
    interval_end: float,
    tol: float = 1e-3,
) -> None:
    """跨阶段不变量：所有时间边界必须在 interval 内。"""
    for ev in analysis.temporal_events:
        if ev.boundary.earliest < interval_start - tol:
            raise FineVisionResponseValidationError(
                f"event {ev.event_id} earliest {ev.boundary.earliest} < interval start {interval_start}"
            )
        if ev.boundary.latest > interval_end + tol:
            raise FineVisionResponseValidationError(
                f"event {ev.event_id} latest {ev.boundary.latest} > interval end {interval_end}"
            )
    for sm in analysis.subject_motions:
        if sm.boundary.earliest < interval_start - tol:
            raise FineVisionResponseValidationError(
                f"subject_motion {sm.event_id} earliest < interval start"
            )
    for cm in analysis.camera_motions:
        if cm.boundary.latest > interval_end + tol:
            raise FineVisionResponseValidationError(
                f"camera_motion {cm.event_id} latest > interval end"
            )


def assert_question_coverage(
    analysis: FineBatchAnalysis,
    *,
    refinement_questions: list[str],
) -> None:
    """确保每个 refinement_question 都有回答。"""
    answered = {qa.question.strip() for qa in analysis.question_answers}
    for q in refinement_questions:
        if q.strip() not in answered:
            raise FineVisionResponseValidationError(
                f"refinement question not answered: {q[:60]}"
            )


def assert_evidence_exists(
    analysis: FineBatchAnalysis,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
) -> None:
    """确保所有证据引用真实存在（验证器已清理，此处 belt-and-suspenders）。"""
    for ev in analysis.temporal_events:
        for fid in ev.evidence.frame_ids:
            if fid not in known_frame_ids:
                raise FineVisionResponseValidationError(
                    f"event {ev.event_id} references unknown frame_id {fid}"
                )
        for sid in ev.evidence.contact_sheet_ids:
            if sid not in known_sheet_ids:
                raise FineVisionResponseValidationError(
                    f"event {ev.event_id} references unknown sheet_id {sid}"
                )


def load_fine_manifest_or_raise(layout) -> FineAnalysisManifest:
    if not layout.fine_manifest_json.is_file():
        raise FineBatchAnalysisError(
            f"fine manifest missing at {layout.fine_manifest_json}"
        )
    try:
        data = json.loads(layout.fine_manifest_json.read_text(encoding="utf-8"))
        return FineAnalysisManifest.model_validate(data)
    except Exception as exc:
        raise FineBatchAnalysisError(f"fine manifest invalid: {exc}") from exc


__all__ = [
    "assert_evidence_exists",
    "assert_fine_analysis_within_interval",
    "assert_question_coverage",
    "load_fine_batches_index",
    "load_fine_manifest_or_raise",
]
