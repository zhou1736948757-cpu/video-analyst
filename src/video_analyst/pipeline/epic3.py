"""Epic 3 orchestration — fine-frame extraction + presentation + fine batches.

End-to-end flow:

    refinement_plan.json
    → load + validate (plan_loader)
    → for each interval:
        → extract frames (extractor)
        → dedupe (deduplicate)
        → contact sheets (contact_sheet)
        → detail frames (detail_frames)
        → build fine batch + persist (batch_builder)
    → write Epic3 manifest (manifest)
    → write fine_batches.json (batch_builder)
    → update pipeline_state.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig
from video_analyst.env.detect import FFmpegEnvironment, ensure_ready_for_media
from video_analyst.exceptions import (
    Epic3PipelineError,
    RefinementFrameExtractionError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.batch_builder import (
    build_fine_batch,
    build_fine_batches_index,
    persist_fine_batch,
    persist_fine_batches_index,
)
from video_analyst.refinement.contact_sheet import build_contact_sheets
from video_analyst.refinement.deduplicate import (
    DeduplicatedFrame,
    deduplicate_intervals,
)
from video_analyst.refinement.detail_frames import (
    select_detail_frames,
    select_detail_frame_specs,
)
from video_analyst.refinement.detail_extractor import (
    extract_detail_frame,
    to_detail_frame_reference,
)
from video_analyst.refinement.extractor import (
    ExtractedFrame,
    extract_interval_frames,
    remove_extracted_interval,
)
from video_analyst.refinement.manifest import (
    build_extraction_signature,
    build_presentation_signature,
    load_interval_artifact,
    load_manifest,
    persist_interval_artifact,
    save_manifest,
)
from video_analyst.refinement.models import (
    EPIC3_SCHEMA_VERSION,
    Epic3Manifest,
    Epic3Result,
    IntervalArtifact,
    IntervalManifestEntry,
)
from video_analyst.refinement.plan_loader import load_refinement_plan
from video_analyst.refinement.timestamp_parser import compute_actual_fps
from video_analyst.refinement.validation import validate_interval_artifact
from video_analyst.util.hashing import sha256_file, sha256_json

log = get_logger("pipeline.epic3")


def _source_video_sha(video_path: Path) -> str:
    try:
        return sha256_file(video_path, max_bytes=8 * 1024 * 1024)
    except FileNotFoundError:
        return ""


def _state_video_path(layout: OutputLayout) -> Path:
    if layout.pipeline_state_json.is_file():
        try:
            import json

            data = json.loads(layout.pipeline_state_json.read_text(encoding="utf-8"))
            vp = data.get("video_path")
            if vp:
                return Path(vp)
        except Exception:
            pass
    return Path("video.mp4")


def _update_pipeline_state(
    layout: OutputLayout,
    stages: list,
) -> None:
    cache = CacheStore(layout)
    state = cache.load_state(_state_video_path(layout))
    for s in stages:
        state.mark(s)
    state.refinement_manifest_path = str(
        layout.refinement_manifest_json.relative_to(layout.root)
    )
    state.fine_batches_path = str(
        layout.refinement_fine_batches_json.relative_to(layout.root)
    )
    cache.save_state(state)


def _frame_dicts_from_extracted(
    ef_list: list[ExtractedFrame],
) -> list[dict]:
    """Convert ExtractedFrame objects to the simple dicts used by
    contact-sheet and detail-frame code paths."""
    out: list[dict] = []
    for ef in ef_list:
        out.append(
            {
                "frame_id": f"{ef.path.stem}__{int(round(ef.absolute_timestamp * 1000)):08d}",
                "timestamp_seconds": round(ef.absolute_timestamp, 6),
                "path": ef.path,
                "requested_fps": 0,  # filled by caller
            }
        )
    return out


def _build_dedup_frames(
    interval_id: str,
    extracted: list[ExtractedFrame],
    requested_fps: int,
    source_video_sha: str,
) -> list[DeduplicatedFrame]:
    out: list[DeduplicatedFrame] = []
    for i, ef in enumerate(extracted):
        out.append(
            DeduplicatedFrame(
                interval_id=interval_id,
                local_index=i,
                frame_id=f"{interval_id}__f{int(round(ef.absolute_timestamp * 1000)):08d}",
                path=ef.path,
                timestamp_seconds=round(ef.absolute_timestamp, 6),
                relative_timestamp_seconds=round(ef.relative_timestamp, 6),
                requested_fps=int(requested_fps),
                width=int(ef.width),
                height=int(ef.height),
                size_bytes=int(ef.size_bytes),
                sha256=ef.sha256,
                source_video_sha256=source_video_sha,
            )
        )
    # Sort by timestamp ascending and re-number local_index
    out.sort(key=lambda d: (d.timestamp_seconds, d.frame_id))
    for i, d in enumerate(out):
        d.local_index = i
    return out


def _interval_frames_dir(layout: OutputLayout, interval_id: str) -> Path:
    return layout.refinement_frames_dir / interval_id


def _interval_sheets_dir(layout: OutputLayout, interval_id: str) -> Path:
    return layout.refinement_contact_sheets_dir / interval_id


def _interval_detail_frames_dir(layout: OutputLayout, interval_id: str) -> Path:
    return layout.refinement_detail_frames_dir / interval_id


def _extract_detail_frames_for_interval(
    *,
    layout: OutputLayout,
    ffmpeg_env: FFmpegEnvironment,
    source_video: Path,
    canonical_frames: list[DeduplicatedFrame],
    interval_id: str,
    interval_start: float,
    interval_end: float,
    cfg: Any,
) -> list:
    """选择 detail frame 规格并从源视频单独抽取高分辨率版本。

    Epic 4 遗留修复：不再复用 1600 缩放后的普通 refinement frame。
    """
    from video_analyst.refinement.models import DetailFrameReference

    specs = select_detail_frame_specs(
        canonical_frames,
        interval_id=interval_id,
        interval_start=interval_start,
        interval_end=interval_end,
        cfg=cfg,
    )
    out_dir = _interval_detail_frames_dir(layout, interval_id)
    refs: list[DetailFrameReference] = []
    for spec in specs:
        try:
            extracted = extract_detail_frame(
                ffmpeg_env=ffmpeg_env,
                source_video=source_video,
                frame_id=spec.frame_id,
                timestamp_seconds=spec.timestamp_seconds,
                out_dir=out_dir,
                cfg=cfg,
                selection_reason=spec.selection_reason,
            )
        except Exception as exc:
            log.warning(
                "detail frame %s extraction failed (falling back to low-res): %s",
                spec.frame_id,
                exc,
            )
            # 回退到普通 refinement frame（仍保留 frame_id/timestamp）
            match = next(
                (d for d in canonical_frames if d.frame_id == spec.frame_id), None
            )
            if match is None:
                continue
            refs.append(
                DetailFrameReference(
                    frame_id=spec.frame_id,
                    timestamp_seconds=spec.timestamp_seconds,
                    path=match.path,  # type: ignore[arg-type]
                    selection_reason=spec.selection_reason,
                )
            )
            continue
        refs.append(
            to_detail_frame_reference(extracted, selection_reason=spec.selection_reason)
        )
    return refs


def _process_one_interval(
    *,
    layout: OutputLayout,
    interval: Any,
    ffmpeg_env: FFmpegEnvironment,
    source_video: Path,
    source_video_sha: str,
    config: AppConfig,
    user_request: str,
    refresh: bool,
    allow_partial: bool,
    pipeline_extraction_sig: str,
    pipeline_presentation_sig: str,
) -> tuple[IntervalManifestEntry, IntervalArtifact | None, list[DeduplicatedFrame]]:
    """Execute one interval end-to-end. Returns (entry, artifact_or_None,
    canonical_frames) — artifact is None when the interval failed and no
    fine batch can be built."""
    interval_id = interval.interval_id
    cfg = config.refinement
    entry = IntervalManifestEntry(interval_id=interval_id, status="failed")

    frames_dir = _interval_frames_dir(layout, interval_id)
    sheets_dir = _interval_sheets_dir(layout, interval_id)

    if refresh:
        remove_extracted_interval(frames_dir)
        if sheets_dir.exists():
            from shutil import rmtree

            rmtree(sheets_dir, ignore_errors=True)
        # also remove the interval artifact / fine batch
        cache = CacheStore(layout)
        for p in (
            layout.refinement_intervals_dir / f"{interval_id}.json",
            layout.refinement_batches_dir / f"fine_{interval_id}.json",
        ):
            if p.is_file():
                p.unlink()

    extraction_sig = build_extraction_signature(
        interval=interval,
        source_video_sha=source_video_sha,
        ffmpeg_version=ffmpeg_env.ffmpeg_version,
        cfg=cfg,
    )

    # ---- Cache check (presentation signature includes subtitles etc.,
    # so we still rebuild sheets/batches if those change).
    cached_artifact = None if refresh else load_interval_artifact(layout, interval_id)
    if cached_artifact is not None and cached_artifact.extraction_signature == extraction_sig:
        # Frame set unchanged → reuse frames and contact sheets.
        entry.status = "cached"
        entry.attempts = 0
        entry.cache_key = extraction_sig
        entry.frame_count = len(cached_artifact.frames)
        entry.contact_sheet_count = len(cached_artifact.contact_sheets)
        entry.detail_frame_count = len(cached_artifact.detail_frames)
        entry.extraction_signature = extraction_sig
        entry.presentation_signature = cached_artifact.presentation_signature
        # Convert cached frames → DeduplicatedFrame (drop detail flag, regenerate via batch builder)
        dedup_frames: list[DeduplicatedFrame] = []
        for fr in cached_artifact.frames:
            dedup_frames.append(
                DeduplicatedFrame(
                    interval_id=interval_id,
                    local_index=fr.local_index,
                    frame_id=fr.frame_id,
                    path=fr.path,
                    timestamp_seconds=fr.timestamp_seconds,
                    relative_timestamp_seconds=fr.relative_timestamp_seconds,
                    requested_fps=int(fr.requested_fps),
                    width=fr.width,
                    height=fr.height,
                    size_bytes=fr.size_bytes,
                    sha256=fr.sha256,
                    source_video_sha256=fr.source_video_sha256,
                )
            )
        # If presentation signature changed (e.g. subtitles updated), we still
        # rebuild contact sheets and fine batch. We use the cached frames
        # directly (they don't change unless extraction sig changes).
        if cached_artifact.presentation_signature != pipeline_presentation_sig:
            _rebuild_presentation(
                layout=layout,
                interval=interval,
                canonical_frames=dedup_frames,
                cached_artifact=cached_artifact,
                cfg=cfg,
                user_request=user_request,
                pipeline_presentation_sig=pipeline_presentation_sig,
                extraction_sig=extraction_sig,
                ffmpeg_env=ffmpeg_env,
                source_video=source_video,
            )
        return entry, cached_artifact, dedup_frames

    # ---- Real extraction
    try:
        extracted = extract_interval_frames(
            ffmpeg_env=ffmpeg_env,
            source_video=source_video,
            interval=interval,
            out_dir=frames_dir,
            cfg=cfg,
        )
    except RefinementFrameExtractionError as exc:
        log.error("interval %s extraction failed: %s", interval_id, exc)
        entry.error = str(exc)
        return entry, None, []

    entry.attempts = 1
    entry.extraction_signature = extraction_sig
    entry.cache_key = extraction_sig

    # ---- Dedup (intra + cross)
    interval_fps_map = {interval_id: int(interval.requested_fps)}
    dedup_report = deduplicate_intervals(
        {interval_id: extracted},
        time_threshold_seconds=cfg.dedup_time_threshold_seconds,
        requested_fps_by_interval=interval_fps_map,
        source_video_sha256=source_video_sha,
    )
    canonical = dedup_report.canonical_frames
    entry.frame_count = len(canonical)

    # ---- Build presentation signature AFTER frames are known
    presentation_sig = build_presentation_signature(
        frames=canonical,
        cfg=cfg,
        user_request=user_request,
        refinement_questions=list(interval.questions),
        layout=layout,
    )
    entry.presentation_signature = presentation_sig

    # ---- Contact sheets (optional)
    contact_sheets = []
    if cfg.sheets_enabled and not cfg.no_contact_sheets:
        sheet_frames = _frame_dicts_from_extracted(extracted)
        for d in sheet_frames:
            d["requested_fps"] = int(interval.requested_fps)
            d["frame_id"] = (
                f"{interval_id}__f{int(round(float(d['timestamp_seconds']) * 1000)):08d}"
            )
        contact_sheets = build_contact_sheets(
            interval_id=interval_id,
            frames=sheet_frames,
            out_dir=sheets_dir,
            cfg=cfg,
        )
    entry.contact_sheet_count = len(contact_sheets)

    # ---- Detail frames (真高分辨率，从源视频单独抽取)
    detail_frames = _extract_detail_frames_for_interval(
        layout=layout,
        ffmpeg_env=ffmpeg_env,
        source_video=source_video,
        canonical_frames=canonical,
        interval_id=interval_id,
        interval_start=interval.start,
        interval_end=interval.end,
        cfg=cfg,
    )
    entry.detail_frame_count = len(detail_frames)

    # ---- Fine batch
    fine = build_fine_batch(
        interval=interval,
        frames=canonical,
        contact_sheets=contact_sheets,
        detail_frames=detail_frames,
        layout=layout,
        cfg=cfg,
        user_request=user_request,
        refinement_questions=list(interval.questions),
        coarse_analysis_paths=_coarse_analysis_paths(layout),
    )
    persist_fine_batch(layout, fine)

    # ---- Persist interval artifact
    artifact = IntervalArtifact(
        interval_id=interval_id,
        status="ok",
        interval=interval,
        frames=[
            fr for fr in fine.frames  # includes is_detail flag set by builder
        ],
        contact_sheets=contact_sheets,
        detail_frames=detail_frames,
        extraction_signature=extraction_sig,
        presentation_signature=presentation_sig,
    )
    validate_interval_artifact(
        interval_id=interval_id,
        start=interval.start,
        end=interval.end,
        frames=artifact.frames,
    )
    persist_interval_artifact(layout, artifact)

    entry.status = "ok"
    return entry, artifact, canonical


def _rebuild_presentation(
    *,
    layout: OutputLayout,
    interval: Any,
    canonical_frames: list[DeduplicatedFrame],
    cached_artifact: IntervalArtifact,
    cfg: Any,
    user_request: str,
    pipeline_presentation_sig: str,
    extraction_sig: str,
    ffmpeg_env: FFmpegEnvironment,
    source_video: Path,
) -> None:
    """Rebuild contact sheets + fine batch using cached frames."""
    interval_id = interval.interval_id
    sheets_dir = _interval_sheets_dir(layout, interval_id)

    contact_sheets = []
    if cfg.sheets_enabled and not cfg.no_contact_sheets:
        sheet_frames: list[dict] = []
        for d in canonical_frames:
            sheet_frames.append(
                {
                    "frame_id": d.frame_id,
                    "timestamp_seconds": float(d.timestamp_seconds),
                    "path": d.path,
                    "requested_fps": int(interval.requested_fps),
                }
            )
        from shutil import rmtree

        if sheets_dir.exists():
            rmtree(sheets_dir, ignore_errors=True)
        contact_sheets = build_contact_sheets(
            interval_id=interval_id,
            frames=sheet_frames,
            out_dir=sheets_dir,
            cfg=cfg,
        )
    detail_frames = _extract_detail_frames_for_interval(
        layout=layout,
        ffmpeg_env=ffmpeg_env,
        source_video=source_video,
        canonical_frames=canonical_frames,
        interval_id=interval_id,
        interval_start=interval.start,
        interval_end=interval.end,
        cfg=cfg,
    )

    fine = build_fine_batch(
        interval=interval,
        frames=canonical_frames,
        contact_sheets=contact_sheets,
        detail_frames=detail_frames,
        layout=layout,
        cfg=cfg,
        user_request=user_request,
        refinement_questions=list(interval.questions),
        coarse_analysis_paths=_coarse_analysis_paths(layout),
    )
    persist_fine_batch(layout, fine)
    new_artifact = IntervalArtifact(
        interval_id=interval_id,
        status="ok",
        interval=interval,
        frames=fine.frames,
        contact_sheets=contact_sheets,
        detail_frames=detail_frames,
        extraction_signature=extraction_sig,
        presentation_signature=pipeline_presentation_sig,
    )
    validate_interval_artifact(
        interval_id=interval_id,
        start=interval.start,
        end=interval.end,
        frames=new_artifact.frames,
    )
    persist_interval_artifact(layout, new_artifact)


def _coarse_analysis_paths(layout: OutputLayout) -> dict[str, str]:
    out: dict[str, str] = {}
    if layout.coarse_manifest_json.is_file():
        out["manifest"] = str(layout.coarse_manifest_json.relative_to(layout.root))
    if layout.coarse_structure_json.is_file():
        out["video_structure"] = str(
            layout.coarse_structure_json.relative_to(layout.root)
        )
    if layout.coarse_refinement_plan_json.is_file():
        out["refinement_plan"] = str(
            layout.coarse_refinement_plan_json.relative_to(layout.root)
        )
    return out


def run_epic3(
    video_path: Path,
    output_dir: Path,
    config: AppConfig,
    *,
    user_request: str = "",
    refresh: bool = False,
    refresh_intervals: list[str] | None = None,
    allow_partial: bool | None = None,
) -> Epic3Result:
    """Top-level Epic 3 entry point."""
    cfg = config
    if allow_partial is not None:
        cfg.refinement.allow_partial = bool(allow_partial)

    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)
    # Refresh whole Epic 3 tree (preserve Epic 2 outputs)
    if refresh and not refresh_intervals:
        CacheStore(layout).clear_epic3_tree()
        layout = OutputLayout.create(output_dir)

    # Detect FFmpeg
    env: FFmpegEnvironment = ensure_ready_for_media(
        input_path=video_path, output_dir=output_dir
    )

    # Validate refinement plan
    intervals, plan = load_refinement_plan(layout)
    video_sha = _source_video_sha(video_path)
    video_duration = float(plan.video_duration or 0.0)

    # Load existing manifest or create new
    manifest = load_manifest(layout)
    if manifest is None:
        manifest = Epic3Manifest(
            video_id=output_dir.name,
            source_video_sha256=video_sha,
            video_duration=video_duration,
            source_video_path=str(video_path),
            ffmpeg_version=env.ffmpeg_version,
            config_fingerprint=sha256_json(cfg.fingerprint()),
            plan_signature=sha256_json(plan.model_dump(mode="json")),
        )
    else:
        # Update possibly-changed run-level fields
        manifest = manifest.model_copy(
            update={
                "source_video_sha256": video_sha,
                "video_duration": video_duration,
                "ffmpeg_version": env.ffmpeg_version,
                "config_fingerprint": sha256_json(cfg.fingerprint()),
                "plan_signature": sha256_json(plan.model_dump(mode="json")),
                "updated_at": manifest.updated_at,
            }
        )

    # Pre-emptively remove the listed refresh_intervals so they're rebuilt
    cache = CacheStore(layout)
    if refresh_intervals:
        for iid in refresh_intervals:
            cache.clear_epic3_interval(iid)
        # Also drop from manifest so re-process is clean
        manifest.intervals = [
            e for e in manifest.intervals if e.interval_id not in refresh_intervals
        ]
        manifest.recompute_counts()

    # Handle empty plan
    if not intervals:
        manifest.recompute_counts()
        save_manifest(layout, manifest)
        index = build_fine_batches_index([], video_id=output_dir.name)
        persist_fine_batches_index(layout, index)
        try:
            _update_pipeline_state(
                layout,
                [
                    # No specific Epic 3 stages run; still flag the manifest exists
                ],
            )
        except Exception:
            pass
        log.info("Epic 3 done: no intervals to process")
        return Epic3Result(
            schema_version=EPIC3_SCHEMA_VERSION,
            status="completed_no_intervals",
            total_intervals=0,
            succeeded_intervals=0,
            failed_intervals=0,
            cached_intervals=0,
            total_frames=0,
            total_contact_sheets=0,
            fine_batches_path=layout.refinement_fine_batches_json,
            manifest_path=layout.refinement_manifest_json,
        )

    total_frames = 0
    total_sheets = 0
    succeeded = 0
    failed = 0
    cached = 0
    failed_ids: list[str] = []
    all_fine_batches = []

    for interval in intervals:
        log.info(
            "Epic 3 interval=%s [%.3f..%.3f] fps=%d priority=%s",
            interval.interval_id,
            interval.start,
            interval.end,
            interval.requested_fps,
            interval.priority,
        )
        try:
            entry, _artifact, _canonical = _process_one_interval(
                layout=layout,
                interval=interval,
                ffmpeg_env=env,
                source_video=video_path,
                source_video_sha=video_sha,
                config=cfg,
                user_request=user_request,
                refresh=refresh,
                allow_partial=cfg.refinement.allow_partial,
                pipeline_extraction_sig="",
                pipeline_presentation_sig="",
            )
        except Exception as exc:
            log.error("interval %s crashed: %s", interval.interval_id, exc)
            entry = IntervalManifestEntry(
                interval_id=interval.interval_id,
                status="failed",
                error=str(exc),
            )
            if not cfg.refinement.allow_partial:
                raise Epic3PipelineError(
                    f"interval {interval.interval_id} failed and partial not allowed"
                ) from exc

        # Persist entry into manifest immediately
        new_intervals = [
            e for e in manifest.intervals if e.interval_id != entry.interval_id
        ]
        new_intervals.append(entry)
        manifest.intervals = sorted(new_intervals, key=lambda e: e.interval_id)
        manifest.recompute_counts()
        save_manifest(layout, manifest)

        if entry.status in ("ok", "cached"):
            total_frames += entry.frame_count
            total_sheets += entry.contact_sheet_count
            if entry.status == "ok":
                succeeded += 1
            else:
                cached += 1
        elif entry.status == "failed":
            failed += 1
            failed_ids.append(entry.interval_id)
            if not cfg.refinement.allow_partial:
                raise Epic3PipelineError(
                    f"interval {interval.interval_id} failed and partial not allowed"
                )

    # Build the fine-batches index from the per-interval fine batch files
    # that were persisted during the loop. This is the single source of
    # truth — we never trust an in-memory list, we always re-read from disk
    # so a partial run still produces a consistent index.
    from video_analyst.refinement.validation import load_interval_artifact_or_raise

    fine_batches = []
    for interval in intervals:
        fb_path = layout.refinement_batches_dir / f"fine_{interval.interval_id}.json"
        if not fb_path.is_file():
            continue
        try:
            from video_analyst.refinement.models import FineVisualBatch

            fine_batches.append(
                FineVisualBatch.model_validate(
                    json.loads(fb_path.read_text(encoding="utf-8"))
                )
            )
        except Exception as exc:
            log.warning("could not load fine batch %s: %s", fb_path, exc)

    index = build_fine_batches_index(fine_batches, video_id=output_dir.name)
    all_fine_batches = list(index.batches)
    persist_fine_batches_index(layout, index)

    # Update pipeline state
    try:
        _update_pipeline_state(
            layout,
            [
                # Stages were implicit per interval; we don't have dedicated
                # stage names for every sub-step.
            ],
        )
    except Exception:
        pass

    # Status
    if failed == 0 and succeeded + cached == len(intervals):
        status = "completed"
    elif succeeded + cached > 0 and failed > 0 and cfg.refinement.allow_partial:
        status = "partial"
    elif cfg.refinement.allow_partial and succeeded + cached == 0:
        # nothing succeeded but partial allowed
        status = "partial"
    else:
        status = "failed"

    log.info(
        "Epic 3 done: intervals=%d succeeded=%d cached=%d failed=%d frames=%d sheets=%d",
        len(intervals),
        succeeded,
        cached,
        failed,
        total_frames,
        total_sheets,
    )

    return Epic3Result(
        schema_version=EPIC3_SCHEMA_VERSION,
        status=status,  # type: ignore[arg-type]
        total_intervals=len(intervals),
        succeeded_intervals=succeeded,
        failed_intervals=failed,
        cached_intervals=cached,
        total_frames=total_frames,
        total_contact_sheets=total_sheets,
        fine_batches_path=layout.refinement_fine_batches_json,
        manifest_path=layout.refinement_manifest_json,
        failed_interval_ids=failed_ids,
        warnings=[],
    )


__all__ = ["run_epic3"]