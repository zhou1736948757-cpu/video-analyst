"""Cross-stage contract validation for Epic 3 outputs."""

from __future__ import annotations

import json
from pathlib import Path

from video_analyst.cache.store import OutputLayout
from video_analyst.exceptions import (
    Epic3PipelineError,
    FineBatchBuildError,
    RefinementFrameMismatchError,
)
from video_analyst.refinement.models import (
    Epic3Manifest,
    FineBatchesIndex,
    IntervalArtifact,
    RefinementFrame,
)


def assert_interval_artifact_on_disk(layout: OutputLayout, interval_id: str) -> Path:
    path = layout.refinement_intervals_dir / f"{interval_id}.json"
    if not path.is_file():
        raise Epic3PipelineError(
            f"interval artifact missing for {interval_id} at {path}"
        )
    return path


def assert_fine_batches_index_on_disk(layout: OutputLayout) -> Path:
    path = layout.refinement_fine_batches_json
    if not path.is_file():
        raise FineBatchBuildError(f"fine_batches.json missing at {path}")
    return path


def validate_interval_artifact(
    *,
    interval_id: str,
    start: float,
    end: float,
    frames: list[RefinementFrame],
) -> None:
    """Verify a persisted IntervalArtifact's invariants."""
    tol = 1e-3
    last_ts = -1.0
    for fr in frames:
        if fr.timestamp_seconds < start - tol:
            raise RefinementFrameMismatchError(
                f"frame {fr.frame_id} timestamp {fr.timestamp_seconds} "
                f"< interval start {start}"
            )
        if fr.timestamp_seconds > end + tol:
            raise RefinementFrameMismatchError(
                f"frame {fr.frame_id} timestamp {fr.timestamp_seconds} "
                f"> interval end {end}"
            )
        if fr.timestamp_seconds + tol < last_ts:
            raise RefinementFrameMismatchError(
                f"frame {fr.frame_id} timestamp {fr.timestamp_seconds} "
                f"decreased after {last_ts}"
            )
        last_ts = max(last_ts, fr.timestamp_seconds)
        if fr.local_index < 0:
            raise RefinementFrameMismatchError(
                f"frame {fr.frame_id} has negative local_index"
            )


def load_manifest_or_raise(layout: OutputLayout) -> Epic3Manifest:
    if not layout.refinement_manifest_json.is_file():
        raise Epic3PipelineError(
            f"Epic3 manifest missing at {layout.refinement_manifest_json}"
        )
    try:
        data = json.loads(layout.refinement_manifest_json.read_text(encoding="utf-8"))
        return Epic3Manifest.model_validate(data)
    except Exception as exc:
        raise Epic3PipelineError(f"Epic3 manifest invalid: {exc}") from exc


def load_fine_batches_index_or_raise(layout: OutputLayout) -> FineBatchesIndex:
    path = layout.refinement_fine_batches_json
    if not path.is_file():
        raise FineBatchBuildError(f"fine_batches.json missing at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FineBatchesIndex.model_validate(data)
    except Exception as exc:
        raise FineBatchBuildError(f"fine_batches.json invalid: {exc}") from exc


def load_interval_artifact_or_raise(
    layout: OutputLayout, interval_id: str
) -> IntervalArtifact:
    path = layout.refinement_intervals_dir / f"{interval_id}.json"
    if not path.is_file():
        raise Epic3PipelineError(f"interval artifact missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return IntervalArtifact.model_validate(data)
    except Exception as exc:
        raise Epic3PipelineError(
            f"interval artifact invalid ({interval_id}): {exc}"
        ) from exc


__all__ = [
    "assert_fine_batches_index_on_disk",
    "assert_interval_artifact_on_disk",
    "load_fine_batches_index_or_raise",
    "load_interval_artifact_or_raise",
    "load_manifest_or_raise",
    "validate_interval_artifact",
]