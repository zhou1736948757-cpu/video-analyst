"""Coarse analysis batch runner.

Responsibilities:
- Iterate over visual batches in time order.
- For each batch, build a VisionBatchRequest, call the provider,
  parse + validate the response, persist raw + analysis.
- Cache hit → reuse previous result.
- On retryable error → retry up to max_retries.
- On unrecoverable error → mark batch failed, continue.
- Persist a manifest that records every batch state.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import (
    AppConfig,
    CoarseAnalysisConfig,
    VisionConfig,
)
from video_analyst.exceptions import (
    VisionBatchAnalysisError,
    VisionProviderUnavailableError,
    VisionRateLimitError,
    VisionRequestError,
    VisionResponseParseError,
    VisionResponseValidationError,
    VisionTimeoutError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.models import PipelineStage
from video_analyst.util.hashing import sha256_file, sha256_json
from video_analyst.vision.base import VisionProvider
from video_analyst.vision.models import (
    CoarseBatchAnalysis,
    PreviousBatchContext,
    VisionBatchRequest,
    VisionBatchSelection,
)
from video_analyst.vision.prompts import PROMPT_VERSION
from video_analyst.vision.request_builder import build_request_for_batch

from video_analyst.coarse_analysis.models import (
    BatchRunRecord,
    CoarseAnalysisManifest,
    CoarseAnalysisResult,
)

log = get_logger("coarse_analysis.batch_runner")


# ---- Filesystem helpers ----


def coarse_output_paths(layout: OutputLayout) -> dict[str, Path]:
    base = layout.root / "analysis" / "coarse"
    return {
        "root": base,
        "batches_dir": base / "batches",
        "raw_dir": base / "raw",
        "manifest": base / "manifest.json",
        "structure": base / "video_structure.json",
        "refinement_plan": base / "refinement_plan.json",
    }


def ensure_coarse_dirs(paths: dict[str, Path]) -> None:
    for key in ("root", "batches_dir", "raw_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _safe_filename(batch_id: str) -> str:
    return f"{batch_id}.json"


def _raw_filename(batch_id: str) -> str:
    return f"{batch_id}_response.json"


# ---- Cache key construction ----


def _frame_hash_for_cache(frames: Iterable) -> str:
    """Aggregate content-aware hash for a batch's frame set.

    Includes (frame_id, relative_path, size_bytes, sha256) for each frame so:
    - identical-size images with different contents invalidate the cache;
    - mtime-only changes do NOT invalidate (we don't read mtime here).

    Falls back to (frame_id, path, 0, "") if the file is missing.
    """
    items = []
    for f in frames:
        p = Path(f.path)
        try:
            st = p.stat()
            size = int(st.st_size)
            sha = sha256_file(p)
        except (FileNotFoundError, OSError):
            size = 0
            sha = ""
        items.append((f.frame_id, str(p), size, sha))
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_json(payload)


def _previous_context_signature(prev_ctx) -> str:
    """Stable hash of a PreviousBatchContext (when present).

    Only the five documented fields are included; we DO NOT hash the full
    previous batch response to keep the key stable across serialization
    differences.
    """
    if prev_ctx is None:
        return ""
    payload = {
        "previous_batch_id": prev_ctx.previous_batch_id,
        "previous_summary": prev_ctx.previous_summary,
        "last_known_subjects": list(prev_ctx.last_known_subjects),
        "last_known_scene": prev_ctx.last_known_scene,
        "unresolved_questions": list(prev_ctx.unresolved_questions),
    }
    return sha256_json(payload)


def _build_cache_key(
    *,
    batch_id: str,
    request: VisionBatchRequest,
    selection: VisionBatchSelection,
    active_subtitle_path: Path | None,
    active_subtitle_signature: str | None,
    user_request: str,
    provider_name: str,
    model: str,
    prompt_version: str,
    schema_version: str,
) -> str:
    """Cache key for a single batch.

    Per spec the key MUST include: batch JSON content, image file hash,
    active subtitle hash, user_request, provider, model, prompt_version,
    schema_version.

    previous_context is included via `_previous_context_signature` so a
    change in the previous batch's summary forces downstream batches to
    invalidate (technical-debt fix from Epic 2 → Epic 3).
    """
    payload = {
        "batch_id": batch_id,
        "frame_ids": [f.frame_id for f in request.frames],
        "frame_signature": _frame_hash_for_cache(request.frames),
        "selection": selection.model_dump(mode="json"),
        "subtitle_signature": active_subtitle_signature
        or (sha256_file(active_subtitle_path) if active_subtitle_path else ""),
        "subtitle_metadata": request.subtitle_metadata.model_dump(mode="json"),
        "subtitles": [s.model_dump(mode="json") for s in request.subtitles],
        "user_request": user_request,
        "analysis_dimensions": list(request.analysis_dimensions),
        "previous_context": _previous_context_signature(request.previous_context),
        "provider": provider_name,
        "model": model,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
    }
    return sha256_json(payload)


# ---- Manifest helpers ----


def load_manifest(layout: OutputLayout) -> CoarseAnalysisManifest | None:
    paths = coarse_output_paths(layout)
    if not paths["manifest"].is_file():
        return None
    try:
        data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        return CoarseAnalysisManifest.model_validate(data)
    except Exception:
        return None


def save_manifest(layout: OutputLayout, manifest: CoarseAnalysisManifest) -> None:
    paths = coarse_output_paths(layout)
    ensure_coarse_dirs(paths)
    cache = CacheStore(layout)
    cache.write_json(paths["manifest"], manifest.model_dump(mode="json"))


# ---- Batch execution primitives ----


def _persist_analysis(
    layout: OutputLayout,
    paths: dict[str, Path],
    batch_id: str,
    analysis: CoarseBatchAnalysis,
    raw_text: str,
    cache_key: str,
) -> tuple[str, str]:
    """Atomically save raw response and analysis. Returns (analysis_relpath,
    raw_relpath) — both relative to output root."""
    cache = CacheStore(layout)
    analysis_path = paths["batches_dir"] / _safe_filename(batch_id)
    raw_path = paths["raw_dir"] / _raw_filename(batch_id)
    raw_rel = str(raw_path.relative_to(layout.root))
    analysis_rel = str(analysis_path.relative_to(layout.root))

    raw_payload = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "cache_key": cache_key,
        "raw_text": raw_text,
        "provider_response": _try_parse(raw_text),
    }
    cache.write_json(raw_path, raw_payload)
    cache.write_json(analysis_path, analysis.model_dump(mode="json"))
    return analysis_rel, raw_rel


def _try_parse(raw_text: str) -> Any:
    from video_analyst.vision.response_parser import parse_response_payload

    try:
        return parse_response_payload(raw_text)
    except Exception:
        return None


def _is_cached(
    manifest: CoarseAnalysisManifest | None,
    batch_id: str,
    cache_key: str,
) -> tuple[bool, BatchRunRecord | None]:
    if manifest is None:
        return False, None
    for rec in manifest.batches:
        if rec.batch_id == batch_id and rec.status == "ok" and rec.cache_key == cache_key:
            return True, rec
    return False, None


def _previous_context_for(
    manifest: CoarseAnalysisManifest | None,
    analyses: dict[str, CoarseBatchAnalysis],
    batch_index: int,
) -> PreviousBatchContext | None:
    if batch_index == 0 or manifest is None:
        return None
    prev_batch_id = manifest.batches[batch_index - 1].batch_id
    prev = analyses.get(prev_batch_id)
    if prev is None:
        return None
    last_subjects = [o.description for o in prev.subjects[:3] if o.description]
    last_scene = prev.scene_changes[0].description if prev.scene_changes else None
    return PreviousBatchContext(
        previous_batch_id=prev_batch_id,
        previous_summary=prev.summary,
        last_known_subjects=last_subjects,
        last_known_scene=last_scene,
        unresolved_questions=[],
    )


# ---- Per-batch runner ----


def _analyze_with_provider(
    provider: VisionProvider,
    request: VisionBatchRequest,
) -> tuple[CoarseBatchAnalysis, str]:
    """Call the provider and parse. Returns (analysis, raw_text).

    The provider is responsible for converting its raw response into the
    CoarseBatchAnalysis structure (with the help of the response_parser).
    Epic 6: 经过统一 TelemetryRecorder 记录。
    """
    from video_analyst.telemetry.recorder import get_global_recorder

    recorder = get_global_recorder()
    with recorder.record_call(
        provider=provider.provider_name,
        model=provider.model_name,
        stage="coarse_vision",
        batch_id=request.batch_id,
    ) as rec:
        rec.input_image_count = len(request.frames)
        # Mock provider 不发送网络请求
        if provider.provider_name == "mock":
            rec.network_request_sent = False
            rec.response_received = True
        else:
            rec.network_request_sent = True
        try:
            analysis = provider.analyze_batch(request)
            rec.response_received = True
            rec.success = True
        except Exception as exc:
            rec.response_received = False
            rec.success = False
            rec.error_type = type(exc).__name__
            raise
    raw_snapshot = json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False)
    return analysis, raw_snapshot


def _execute_one_batch(
    layout: OutputLayout,
    paths: dict[str, Path],
    batch_index: int,
    user_request: str,
    video_id: str,
    provider: VisionProvider,
    vision_cfg: VisionConfig,
    coarse_cfg: CoarseAnalysisConfig,
    config_fingerprint: str,
    *,
    previous_analyses: dict[str, CoarseBatchAnalysis],
    manifest: CoarseAnalysisManifest | None,
    allow_repair: bool = True,
    extra_force_refresh: bool = False,
) -> tuple[BatchRunRecord, CoarseBatchAnalysis | None]:
    """Run a single batch end-to-end with retry. Returns the (record, analysis)
    pair — analysis is None on unrecoverable failure."""

    request, selection = build_request_for_batch(
        batch_index,
        layout,
        user_request=user_request,
        video_id=video_id,
        max_frames=vision_cfg.batch_max_frames,
        analysis_dimensions=None,
    )
    # Attach previous-batch context (if any) so cache key reflects it.
    prev_ctx = _previous_context_for(manifest, previous_analyses, batch_index)
    if prev_ctx is not None:
        request = request.model_copy(update={"previous_context": prev_ctx})

    active_subs_signature = (
        _active_subtitle_signature(layout) if layout.subtitles_json.is_file() else ""
    )

    cache_key = _build_cache_key(
        batch_id=request.batch_id,
        request=request,
        selection=selection,
        active_subtitle_path=layout.subtitles_json,
        active_subtitle_signature=active_subs_signature,
        user_request=user_request,
        provider_name=provider.provider_name,
        model=provider.model_name,
        prompt_version=vision_cfg.prompt_version or PROMPT_VERSION,
        schema_version=request.schema_version,
    )

    if not extra_force_refresh:
        hit, existing = _is_cached(manifest, request.batch_id, cache_key)
        if hit and existing is not None:
            log.info(
                "cache hit: batch=%s key=%s (skipping provider call)",
                request.batch_id,
                cache_key[:10],
            )
            cached = _load_cached_analysis(layout, paths, request.batch_id)
            return existing, cached

    log.info(
        "running batch=%s frames=%d (selected=%d omitted=%d)",
        request.batch_id,
        len(request.frames),
        selection.selected,
        len(selection.omitted_frame_ids),
    )

    last_exc: Exception | None = None
    for attempt in range(max(1, int(vision_cfg.max_retries)) + 1):
        try:
            analysis, raw_text = _analyze_with_provider(provider, request)
            analysis_rel, raw_rel = _persist_analysis(
                layout,
                paths,
                request.batch_id,
                analysis,
                raw_text,
                cache_key,
            )
            return (
                BatchRunRecord(
                    batch_id=request.batch_id,
                    provider=provider.provider_name,
                    model=provider.model_name,
                    prompt_version=vision_cfg.prompt_version or PROMPT_VERSION,
                    status="ok",
                    attempts=attempt + 1,
                    raw_response_path=raw_rel,
                    analysis_path=analysis_rel,
                    selected_frames=selection.selected,
                    omitted_frames=len(selection.omitted_frame_ids),
                    cache_key=cache_key,
                ),
                analysis,
            )
        except (
            VisionTimeoutError,
            VisionRateLimitError,
            VisionRequestError,
        ) as exc:
            last_exc = exc
            # retry on 429 / 5xx / timeout / network
            if attempt >= int(vision_cfg.max_retries):
                break
            log.warning(
                "batch=%s retryable error attempt=%d: %s",
                request.batch_id,
                attempt,
                exc,
            )
            continue
        except VisionResponseParseError as exc:
            last_exc = exc
            if allow_repair and provider.supports_repair_request():
                try:
                    log.info(
                        "batch=%s attempting repair after parse error", request.batch_id
                    )
                    repair_payload = provider.build_repair_payload(
                        getattr(exc, "raw_text", "") or "", str(exc)
                    )
                    if repair_payload is not None:
                        # The provider SHOULD raise or return a CoarseBatchAnalysis
                        # via its standard analyze_batch pathway in real
                        # implementations. We re-ask the provider with the
                        # repair context as part of the request. The Mock
                        # provider short-circuits via parse_repair_response.
                        repaired = provider.parse_repair_response(
                            json.dumps(repair_payload.get("messages", [])[-1])
                            if repair_payload.get("messages")
                            else "{}"
                        )
                        if isinstance(repaired, dict):
                            from video_analyst.vision.response_parser import (
                                validate_batch_response,
                            )

                            analysis, _w = validate_batch_response(
                                repaired,
                                request=request,
                                video_start=request.batch_start,
                                video_end=request.batch_end,
                            )
                            analysis_rel, raw_rel = _persist_analysis(
                                layout,
                                paths,
                                request.batch_id,
                                analysis,
                                json.dumps(repaired, ensure_ascii=False),
                                cache_key,
                            )
                            return (
                                BatchRunRecord(
                                    batch_id=request.batch_id,
                                    provider=provider.provider_name,
                                    model=provider.model_name,
                                    prompt_version=vision_cfg.prompt_version
                                    or PROMPT_VERSION,
                                    status="ok",
                                    attempts=attempt + 1,
                                    raw_response_path=raw_rel,
                                    analysis_path=analysis_rel,
                                    selected_frames=selection.selected,
                                    omitted_frames=len(selection.omitted_frame_ids),
                                    cache_key=cache_key,
                                ),
                                analysis,
                            )
                except Exception as repair_exc:
                    log.warning(
                        "batch=%s repair failed: %s",
                        request.batch_id,
                        repair_exc,
                    )
            # one-shot repair, no more retries for parse errors
            break
        except VisionResponseValidationError as exc:
            last_exc = exc
            if allow_repair and attempt < int(vision_cfg.max_retries):
                # single repair attempt by re-asking the provider
                log.info(
                    "batch=%s attempting repair after validation error", request.batch_id
                )
                try:
                    repair_payload = provider.build_repair_payload("", str(exc))
                    if repair_payload is not None:
                        repaired = provider.parse_repair_response(
                            json.dumps(repair_payload.get("messages", [])[-1])
                            if repair_payload.get("messages")
                            else "{}"
                        )
                        if isinstance(repaired, dict):
                            from video_analyst.vision.response_parser import (
                                validate_batch_response,
                            )

                            analysis, _w = validate_batch_response(
                                repaired,
                                request=request,
                                video_start=request.batch_start,
                                video_end=request.batch_end,
                            )
                            analysis_rel, raw_rel = _persist_analysis(
                                layout,
                                paths,
                                request.batch_id,
                                analysis,
                                json.dumps(repaired, ensure_ascii=False),
                                cache_key,
                            )
                            return (
                                BatchRunRecord(
                                    batch_id=request.batch_id,
                                    provider=provider.provider_name,
                                    model=provider.model_name,
                                    prompt_version=vision_cfg.prompt_version
                                    or PROMPT_VERSION,
                                    status="ok",
                                    attempts=attempt + 1,
                                    raw_response_path=raw_rel,
                                    analysis_path=analysis_rel,
                                    selected_frames=selection.selected,
                                    omitted_frames=len(selection.omitted_frame_ids),
                                    cache_key=cache_key,
                                ),
                                analysis,
                            )
                except Exception as repair_exc:
                    log.warning(
                        "batch=%s repair failed: %s",
                        request.batch_id,
                        repair_exc,
                    )
            break

    # Unrecoverable
    err_msg = (
        f"{type(last_exc).__name__}: {last_exc}"
        if last_exc is not None
        else "unknown error"
    )
    log.error("batch=%s failed permanently: %s", request.batch_id, err_msg)
    return (
        BatchRunRecord(
            batch_id=request.batch_id,
            provider=provider.provider_name,
            model=provider.model_name,
            prompt_version=vision_cfg.prompt_version or PROMPT_VERSION,
            status="failed",
            attempts=max(1, int(vision_cfg.max_retries)) + 1,
            error=err_msg,
            selected_frames=selection.selected,
            omitted_frames=len(selection.omitted_frame_ids),
            cache_key=cache_key,
        ),
        None,
    )


def _active_subtitle_signature(layout: OutputLayout) -> str:
    """A signature that changes when active subtitle content changes."""
    cache = CacheStore(layout)
    parts: list[str] = []
    if layout.subtitles_json.is_file():
        try:
            data = json.loads(layout.subtitles_json.read_text(encoding="utf-8"))
            # Use segments text+times so any active subtitle change is detected.
            segs = data.get("segments") or []
            slim = [
                {"i": s.get("index"), "s": s.get("start"), "e": s.get("end"), "t": s.get("text")}
                for s in segs
            ]
            parts.append(sha256_json({"origin": data.get("origin"), "segs": slim}))
        except Exception:
            parts.append(sha256_file(layout.subtitles_json))
    if layout.review_json.is_file():
        try:
            data = json.loads(layout.review_json.read_text(encoding="utf-8"))
            parts.append(
                sha256_json(
                    {
                        "active": data.get("active"),
                        "reviewed": data.get("reviewed"),
                        "needs_review": data.get("needs_review"),
                    }
                )
            )
        except Exception:
            parts.append(sha256_file(layout.review_json))
    return sha256_json(parts)


def _load_cached_analysis(
    layout: OutputLayout, paths: dict[str, Path], batch_id: str
) -> CoarseBatchAnalysis | None:
    p = paths["batches_dir"] / _safe_filename(batch_id)
    if not p.is_file():
        return None
    try:
        return CoarseBatchAnalysis.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


# ---- Top-level orchestrator ----


def run_coarse_analysis(
    *,
    layout: OutputLayout,
    config: AppConfig,
    provider: VisionProvider,
    user_request: str = "",
    refresh_vision: bool = False,
    refresh_batches: list[str] | None = None,
    video_id: str = "video",
    allow_partial: bool | None = None,
) -> CoarseAnalysisResult:
    """Execute coarse analysis end-to-end over all visual batches."""
    paths = coarse_output_paths(layout)
    ensure_coarse_dirs(paths)

    vision_cfg = config.vision
    coarse_cfg = config.coarse_analysis
    if allow_partial is not None:
        coarse_cfg.allow_partial = allow_partial

    batches_data = json.loads(layout.phase2_batches_json.read_text(encoding="utf-8"))
    batches = batches_data.get("batches") or []
    if not batches:
        raise VisionBatchAnalysisError("no batches in visual_batches.json")

    # Optionally clear stale outputs
    if refresh_vision and refresh_batches is None:
        for d in (paths["batches_dir"], paths["raw_dir"]):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        if paths["manifest"].is_file():
            paths["manifest"].unlink()
        if paths["structure"].is_file():
            paths["structure"].unlink()
        if paths["refinement_plan"].is_file():
            paths["refinement_plan"].unlink()
        ensure_coarse_dirs(paths)
        manifest: CoarseAnalysisManifest | None = None
    else:
        manifest = load_manifest(layout)

    config_fingerprint = sha256_json(config.fingerprint())

    # Reload any prior successful analyses (for cache + previous-context)
    previous_analyses: dict[str, CoarseBatchAnalysis] = {}
    if manifest is not None and not refresh_vision:
        for rec in manifest.batches:
            cached = _load_cached_analysis(layout, paths, rec.batch_id)
            if cached is not None and rec.status == "ok":
                previous_analyses[rec.batch_id] = cached

    analyses: dict[str, CoarseBatchAnalysis] = dict(previous_analyses)
    failed: dict[str, str] = {}
    new_records: list[BatchRunRecord] = []

    if manifest is None:
        manifest = CoarseAnalysisManifest(
            video_id=video_id,
            provider=provider.provider_name,
            model=provider.model_name,
            prompt_version=vision_cfg.prompt_version or PROMPT_VERSION,
            user_request=user_request,
            analysis_dimensions=[],
            allow_partial=coarse_cfg.allow_partial,
            config_fingerprint=config_fingerprint,
        )
    else:
        manifest.video_id = video_id
        manifest.provider = provider.provider_name
        manifest.model = provider.model_name
        manifest.prompt_version = vision_cfg.prompt_version or PROMPT_VERSION
        manifest.user_request = user_request
        manifest.allow_partial = coarse_cfg.allow_partial
        manifest.config_fingerprint = config_fingerprint

    target_refresh = set(refresh_batches or [])

    for i, batch in enumerate(batches):
        bid = str(batch["batch_id"])
        if refresh_vision and target_refresh and bid not in target_refresh:
            # already in manifest as ok; keep it
            continue
        if refresh_vision and target_refresh and bid in target_refresh:
            # Drop any prior records for this batch
            manifest.batches = [r for r in manifest.batches if r.batch_id != bid]
            previous_analyses.pop(bid, None)
            analyses.pop(bid, None)
            # remove disk artifacts
            for p in (
                paths["batches_dir"] / _safe_filename(bid),
                paths["raw_dir"] / _raw_filename(bid),
            ):
                if p.is_file():
                    p.unlink()

        record, analysis = _execute_one_batch(
            layout=layout,
            paths=paths,
            batch_index=i,
            user_request=user_request,
            video_id=video_id,
            provider=provider,
            vision_cfg=vision_cfg,
            coarse_cfg=coarse_cfg,
            config_fingerprint=config_fingerprint,
            previous_analyses=previous_analyses,
            manifest=manifest,
            allow_repair=True,
            extra_force_refresh=bool(target_refresh and bid in target_refresh),
        )
        new_records.append(record)
        # Replace prior record (if any) in manifest for this batch_id, then
        # RE-SORT by batch_id so iteration order is stable. This matters
        # because `_previous_context_for` indexes by `manifest.batches[i-1]`,
        # and a previous run that re-recorded `batch_0002` at the tail
        # would otherwise flip the order on disk.
        manifest.batches = sorted(
            [r for r in manifest.batches if r.batch_id != bid] + [record],
            key=lambda r: r.batch_id,
        )
        if analysis is not None:
            analyses[bid] = analysis
        else:
            failed[bid] = record.error or "unknown"

        manifest.recompute_counts()
        save_manifest(layout, manifest)
        # Update previous_analyses as we go so subsequent batches see fresh context
        if analysis is not None:
            previous_analyses[bid] = analysis

    result = CoarseAnalysisResult(
        video_id=video_id,
        output_dir=str(layout.root),
        manifest=manifest,
        analyses=analyses,
        failed_batches=failed,
        allow_partial=coarse_cfg.allow_partial,
    )

    if not coarse_cfg.allow_partial and result.failed > 0:
        # If no partial success allowed and at least one batch failed, raise.
        raise VisionBatchAnalysisError(
            f"coarse analysis failed for {result.failed}/{result.total} batches: "
            + "; ".join(f"{k}={v}" for k, v in list(failed.items())[:3])
        )
    return result


__all__ = [
    "coarse_output_paths",
    "ensure_coarse_dirs",
    "load_manifest",
    "run_coarse_analysis",
    "save_manifest",
]