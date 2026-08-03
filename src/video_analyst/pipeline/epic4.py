"""Epic 4 orchestration — fine temporal vision analysis.

End-to-end flow:

    fine_visual_batches.json (Epic 3)
    → build FineVisionRequest per batch
    → call FineVisionProvider
    → validate + parse response
    → persist per-batch analysis + raw response
    → deduplicate cross-batch temporal events
    → aggregate full-video fine analysis
    → generate second_refinement_plan.json (if needed)
    → update pipeline_state.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig
from video_analyst.exceptions import (
    Epic4PipelineError,
    FineBatchAnalysisError,
    FineVisionProviderUnavailableError,
)
from video_analyst.fine.batch_runner import run_fine_analysis
from video_analyst.fine.models import FineAnalysisResult
from video_analyst.fine.registry import create_fine_vision_provider
from video_analyst.logging.setup import get_logger
from video_analyst.models import PipelineStage

log = get_logger("pipeline.epic4")


def _update_pipeline_state(layout: OutputLayout) -> None:
    """记录 Epic 4 完成到 pipeline_state.json。"""
    cache = CacheStore(layout)
    # 用 layout.root 推断 video_path
    state_path = layout.pipeline_state_json
    video_path = Path("video.mp4")
    if state_path.is_file():
        try:
            import json

            data = json.loads(state_path.read_text(encoding="utf-8"))
            vp = data.get("video_path")
            if vp:
                video_path = Path(vp)
        except Exception:
            pass
    try:
        state = cache.load_state(video_path)
    except Exception:
        return
    state.mark(PipelineStage.FINE_BATCH_BUILD)
    cache.save_state(state)


def run_epic4(
    video_path: Path,
    output_dir: Path,
    config: AppConfig,
    *,
    user_request: str = "",
    analysis_dimensions: list[str] | None = None,
    refresh: bool = False,
    refresh_batches: list[str] | None = None,
    allow_partial: bool | None = None,
    build_second_plan: bool = True,
) -> FineAnalysisResult:
    """Top-level Epic 4 entry point.

    Reads Epic 3's ``fine_visual_batches.json``, runs fine temporal vision
    analysis per batch via the configured provider, aggregates the
    full-video analysis, and optionally generates a second refinement plan.
    """
    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)

    # 前置：必须有 fine_batches.json（Epic 3 产物）
    if not layout.refinement_fine_batches_json.is_file():
        raise Epic4PipelineError(
            f"fine_visual_batches.json missing at "
            f"{layout.refinement_fine_batches_json}; run Epic 3 first"
        )

    # 构造 provider（复用 Epic 2 的 VisionConfig）
    provider = create_fine_vision_provider(
        provider=config.vision.provider,
        config=config.vision,
    )

    log.info(
        "Epic 4 start: video=%s output=%s provider=%s model=%s",
        video_path,
        output_dir,
        provider.provider_name,
        provider.model_name,
    )

    try:
        result = run_fine_analysis(
            layout=layout,
            config=config,
            provider=provider,
            user_request=user_request,
            analysis_dimensions=analysis_dimensions,
            video_id=output_dir.name,
            refresh=refresh,
            refresh_batches=refresh_batches,
            allow_partial=allow_partial,
            build_second_plan=build_second_plan,
        )
    except FineVisionProviderUnavailableError as exc:
        raise Epic4PipelineError(f"fine vision provider unavailable: {exc}") from exc
    except FineBatchAnalysisError as exc:
        raise Epic4PipelineError(f"fine batch analysis failed: {exc}") from exc

    # 更新 pipeline_state
    try:
        _update_pipeline_state(layout)
    except Exception as exc:
        log.warning("could not update pipeline_state: %s", exc)

    log.info(
        "Epic 4 done: status=%s batches=%d succeeded=%d cached=%d failed=%d events=%d",
        result.status,
        result.total_batches,
        result.succeeded_batches,
        result.cached_batches,
        result.failed_batches,
        result.total_events,
    )
    return result


__all__ = ["run_epic4"]
