"""Input/output contract validation for coarse analysis.

These checks are intentionally separate from response_parser.py (which
operates on raw model output). This module verifies cross-stage contracts:
visual_batches.json shape, completed batch files, manifest, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from video_analyst.exceptions import (
    CoarseAggregationError,
    RefinementPlanError,
    VisionBatchAnalysisError,
)
from video_analyst.vision.models import CoarseBatchAnalysis


def load_visual_batches(path: Path) -> dict[str, Any]:
    """Read and minimally validate visual_batches.json.

    Returns the full dict so the batch runner can iterate. Raises
    VisionBatchAnalysisError if the file is missing or invalid.
    """
    if not path.is_file():
        raise VisionBatchAnalysisError(
            f"visual_batches.json missing at {path}; run Phase 1 first"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisionBatchAnalysisError(
            f"visual_batches.json unreadable: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise VisionBatchAnalysisError(
            "visual_batches.json must be a JSON object"
        )
    batches = data.get("batches")
    if not isinstance(batches, list) or not batches:
        raise VisionBatchAnalysisError(
            "visual_batches.json must contain non-empty 'batches' list"
        )
    return data


def load_batch_analysis(path: Path) -> CoarseBatchAnalysis:
    if not path.is_file():
        raise VisionBatchAnalysisError(f"batch analysis missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CoarseBatchAnalysis.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise VisionBatchAnalysisError(
            f"batch analysis unreadable or invalid: {exc}"
        ) from exc


def assert_analysis_within_batch(
    analysis: CoarseBatchAnalysis, batch_start: float, batch_end: float
) -> None:
    """Cross-stage invariant: refinement requests / observations must lie
    inside the batch window. ResponseParser already enforces this per call;
    this is a belt-and-suspenders check after re-loading from disk."""
    tol = 1e-3
    for rr in analysis.refinement_requests:
        if rr.start < batch_start - tol or rr.end > batch_end + tol:
            raise VisionBatchAnalysisError(
                f"refinement_request out of batch: "
                f"{rr.start}..{rr.end} outside {batch_start}..{batch_end}"
            )


def assert_video_structure_invariants(structure_path: Path) -> None:
    if not structure_path.is_file():
        raise CoarseAggregationError(
            f"video_structure.json missing at {structure_path}"
        )


def assert_refinement_plan_invariants(plan_path: Path) -> None:
    if not plan_path.is_file():
        raise RefinementPlanError(
            f"refinement_plan.json missing at {plan_path}"
        )


__all__ = [
    "assert_analysis_within_batch",
    "assert_refinement_plan_invariants",
    "assert_video_structure_invariants",
    "load_batch_analysis",
    "load_visual_batches",
]