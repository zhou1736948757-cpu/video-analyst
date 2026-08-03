"""Epic 2 pipeline orchestration.

Wires together:
  visual_batches.json + active subtitles + user_request
  → batch runner (provider per batch)
  → aggregation → video_structure.json
  → refinement plan → refinement_plan.json
  → manifest update + cache state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.coarse_analysis.aggregate import (
    aggregate_structure,
    save_structure,
)
from video_analyst.coarse_analysis.batch_runner import (
    coarse_output_paths,
    ensure_coarse_dirs,
    run_coarse_analysis,
    save_manifest,
)
from video_analyst.coarse_analysis.models import CoarseAnalysisResult
from video_analyst.coarse_analysis.refinement import build_refinement_plan
from video_analyst.logging.setup import get_logger
from video_analyst.models import PipelineStage
from video_analyst.vision.registry import create_vision_provider

log = get_logger("pipeline.epic2")


def _video_duration_from_batches(layout: OutputLayout) -> float:
    data = json.loads(layout.phase2_batches_json.read_text(encoding="utf-8"))
    batches = data.get("batches") or []
    if not batches:
        return 0.0
    end = max(float(b.get("end", 0)) for b in batches)
    start = min(float(b.get("start", 0)) for b in batches)
    return max(0.0, end - start)


def _batches_meta(layout: OutputLayout) -> tuple[list[str], dict[str, dict[str, Any]]]:
    data = json.loads(layout.phase2_batches_json.read_text(encoding="utf-8"))
    batches = data.get("batches") or []
    order = [str(b["batch_id"]) for b in batches]
    meta = {
        str(b["batch_id"]): {
            "start": float(b.get("start", 0)),
            "end": float(b.get("end", 0)),
        }
        for b in batches
    }
    return order, meta


def _update_pipeline_state(layout: OutputLayout, stages: list[PipelineStage]) -> None:
    cache = CacheStore(layout)
    state = cache.load_state(Path(state_video_path(layout)) if False else _resolve_state_video(layout))
    for s in stages:
        state.mark(s)
    cache.save_state(state)


def _resolve_state_video(layout: OutputLayout) -> Path:
    """Find the source video path from the saved pipeline_state (if any)."""
    cache = CacheStore(layout)
    state_path = layout.pipeline_state_json
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            vp = data.get("video_path")
            if vp:
                return Path(vp)
        except Exception:
            pass
    # Fall back to a sentinel — we don't need the real video path for vision
    # cache logic, but PipelineState requires something.
    return Path("video.mp4")


def state_video_path(layout: OutputLayout) -> Path:  # pragma: no cover - helper
    return _resolve_state_video(layout)


def run_epic2(
    output_dir: Path,
    *,
    config: AppConfig | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    user_request: str = "",
    refresh_vision: bool = False,
    refresh_batches: list[str] | None = None,
    allow_partial: bool | None = None,
    no_aggregate_model: bool = True,
    refinement_budget_ratio: float | None = None,
    analysis_dimensions: list[str] | None = None,
) -> CoarseAnalysisResult:
    """End-to-end Epic 2: batch runner → aggregation → refinement plan."""
    cfg = config or DEFAULT_CONFIG
    provider_name = provider_name or cfg.vision.provider
    if model is not None:
        cfg.vision.model = model
    if refinement_budget_ratio is not None:
        cfg.coarse_analysis.refinement_budget_ratio = float(refinement_budget_ratio)
    if allow_partial is not None:
        cfg.coarse_analysis.allow_partial = allow_partial

    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)
    paths = coarse_output_paths(layout)
    ensure_coarse_dirs(paths)

    # Build the provider via the registry.
    provider = create_vision_provider(provider_name, cfg.vision)

    # 1) Run all batches (with caching + retries)
    log.info(
        "Epic 2 start: provider=%s model=%s refresh=%s refresh_batches=%s",
        provider.provider_name,
        provider.model_name,
        refresh_vision,
        refresh_batches or [],
    )
    result = run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=provider,
        user_request=user_request,
        refresh_vision=refresh_vision,
        refresh_batches=refresh_batches,
        video_id=output_dir.name,
        allow_partial=cfg.coarse_analysis.allow_partial,
    )

    # 2) Aggregate
    order, meta = _batches_meta(layout)
    duration = _video_duration_from_batches(layout)
    video_start = float(meta[order[0]]["start"]) if order else 0.0
    video_end = float(meta[order[-1]]["end"]) if order else duration

    structure = aggregate_structure(
        analyses=result.analyses,
        failed_batches=result.failed_batches,
        video_start=video_start,
        video_end=video_end,
        batches_order=order,
        batches_meta=meta,
    )
    structure_path = save_structure(layout, structure)
    result.structure_path = str(structure_path.relative_to(layout.root))

    # 3) Refinement plan
    plan = build_refinement_plan(
        batch_analyses=result.analyses,
        video_duration=duration,
        video_start=video_start,
        config=cfg.coarse_analysis,
    )
    cache = CacheStore(layout)
    cache.write_json(paths["refinement_plan"], plan.model_dump(mode="json"))
    result.refinement_plan_path = str(paths["refinement_plan"].relative_to(layout.root))

    # 4) Save manifest (final state)
    save_manifest(layout, result.manifest)

    # 5) Mark pipeline stages
    try:
        _update_pipeline_state(
            layout,
            [
                PipelineStage.VISION_REQUEST_BUILD,
                PipelineStage.VISION_BATCH_ANALYSIS,
                PipelineStage.COARSE_AGGREGATION,
                PipelineStage.REFINEMENT_PLAN,
            ],
        )
    except Exception as exc:  # pragma: no cover - non-fatal
        log.warning("Could not update pipeline state: %s", exc)

    log.info(
        "Epic 2 done: total=%d success=%d failed=%d",
        result.total,
        result.succeeded,
        result.failed,
    )
    return result


__all__ = ["run_epic2"]