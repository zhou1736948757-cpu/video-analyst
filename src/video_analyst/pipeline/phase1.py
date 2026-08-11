"""Phase 1 pipeline: env → probe → audio → ASR → LLM calibrate → Phase 2 prep."""

from __future__ import annotations

from pathlib import Path

from video_analyst.asr.pipeline import transcribe_audio
from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.env.detect import FFmpegEnvironment, ensure_ready_for_media
from video_analyst.llm.calibrate import calibrate_subtitles
from video_analyst.llm.review import save_review_package
from video_analyst.logging.setup import get_logger, setup_logging
from video_analyst.media.audio import extract_audio
from video_analyst.media.probe import probe_media
from video_analyst.models import (
    ASRMeta,
    PipelineStage,
    SourceHashes,
    SubtitleDocument,
    SubtitleReviewPackage,
    SubtitleVersion,
)
from video_analyst.phase2.prepare import prepare_phase2
from video_analyst.subtitle.convert import raw_asr_to_document
from video_analyst.subtitle.srt import write_srt
from video_analyst.util.hashing import sha256_file, sha256_json

log = get_logger("pipeline")


def run_phase1(
    video_path: Path,
    output_dir: Path,
    *,
    language: str | None = None,
    resume: bool = True,
    refresh_asr: bool = False,
    multilingual: bool = True,
    llm_calibrate: bool = True,
    run_phase2_prep: bool = True,
    user_request: str = "",
    ffmpeg_env: FFmpegEnvironment | None = None,
    config: AppConfig | None = None,
    pinned_timestamps: list[float] | None = None,
) -> SubtitleDocument:
    """
    Phase 1 (+ optional Phase 2 prep):

    env → probe → audio → chunked ASR hierarchy → merge → LLM calibrate/validate
    → activate subtitles → (optional) 1fps frames + visual batches contracts
    """
    cfg = (config or DEFAULT_CONFIG).with_cli_overrides(
        language=language,
        multilingual=multilingual,
        llm_calibrate=llm_calibrate,
    )

    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    layout = OutputLayout.create(output_dir)
    setup_logging(layout.run_log)
    cache = CacheStore(layout)

    log.info("Pipeline start: video=%s output=%s", video_path, output_dir)

    env = ffmpeg_env or ensure_ready_for_media(
        input_path=video_path,
        output_dir=output_dir,
    )
    log.info("Environment OK: ffmpeg=%s ffprobe=%s", env.ffmpeg_path, env.ffprobe_path)

    if resume:
        state = cache.load_state(video_path)
    else:
        from video_analyst.models import PipelineState

        state = PipelineState(
            video_path=str(video_path),
            output_dir=str(layout.root),
        )

    if refresh_asr and resume:
        from video_analyst.models import PipelineStage as PS

        keep = {PS.ENV_CHECK, PS.MEDIA_PROBE, PS.AUDIO_EXTRACT}
        state.completed_stages = [s for s in state.completed_stages if s in keep]
        state.asr_raw_path = None
        state.asr_manifest_path = None
        state.subtitles_json_path = None
        state.subtitles_srt_path = None
        state.review_json_path = None
        state.active_subtitle_version = None
        state.phase2_batches_path = None
        state.engine = None
        state.last_error = None
        cache.clear_asr_tree()
        # also clear phase2 subtitle-derived; keep frames if present
        cache.clear_phase2_subtitle_derived()
        log.info("refresh_asr: cleared ASR/subtitle/LLM + phase2 mapping; media/audio kept")

    state.mark(PipelineStage.ENV_CHECK)
    state.config_fingerprint = sha256_json(cfg.fingerprint())
    cache.save_state(state)

    # --- media probe ---
    if resume and state.has(PipelineStage.MEDIA_PROBE) and layout.media_info_json.is_file():
        log.info("Resume: media probe from cache")
        media_info_data = cache.read_json(layout.media_info_json)
    else:
        media_info = probe_media(video_path, env)
        media_info_data = media_info.model_dump(mode="json")
        cache.write_json(layout.media_info_json, media_info_data)
        state.media_info_path = str(layout.media_info_json)
        state.mark(PipelineStage.MEDIA_PROBE)
        cache.save_state(state)

    video_duration = None
    try:
        video_duration = float(media_info_data.get("duration") or 0) or None
    except (TypeError, ValueError):
        video_duration = None

    # --- audio extract ---
    if resume and state.has(PipelineStage.AUDIO_EXTRACT) and layout.audio_mp3.is_file():
        audio_path = layout.audio_mp3
        log.info("Resume: extracted audio from cache")
    else:
        audio_path = extract_audio(video_path, layout.audio_mp3, env)
        state.audio_path = str(audio_path)
        state.mark(PipelineStage.AUDIO_EXTRACT)
        cache.save_state(state)

    # --- ASR hierarchy ---
    asr_doc: SubtitleDocument | None = None
    raw: dict
    engine = None

    asr_ready = (
        resume
        and layout.asr_manifest_json.is_file()
        and layout.subtitles_asr_json.is_file()
        and (
            state.has(PipelineStage.ASR_TRANSCRIPTION)
            or state.has(PipelineStage.ASR)
            or state.has(PipelineStage.SUBTITLE_RENDER_ASR)
        )
    )

    if asr_ready:
        log.info("Resume: ASR subtitles from cache")
        asr_doc = SubtitleDocument.model_validate(cache.read_json(layout.subtitles_asr_json))
        raw = (
            cache.read_json(layout.asr_raw_json)
            if layout.asr_raw_json.is_file()
            else {"segments": []}
        )
        from video_analyst.models import ASREngine

        try:
            engine = ASREngine(asr_doc.engine) if asr_doc.engine else ASREngine.GROQ_WHISPER_LARGE_V3_TURBO
        except Exception:
            engine = ASREngine.GROQ_WHISPER_LARGE_V3_TURBO
    else:
        try:
            raw, engine, asr_doc = transcribe_audio(
                audio_path,
                language=cfg.asr.language,
                multilingual=cfg.asr.multilingual,
                ffmpeg_env=env,
                layout=layout,
                cache=cache,
                config=cfg,
                source_video=str(video_path),
                video_duration=video_duration,
            )
        except Exception as exc:
            state.last_error = str(exc)
            cache.save_state(state)
            raise

        if asr_doc is None:
            asr_doc = raw_asr_to_document(
                raw,
                source_video=video_path,
                audio_path=audio_path,
                engine=engine,
            )
            asr_doc.origin = "asr"

        # render ASR outputs
        cache.write_json(layout.subtitles_asr_json, asr_doc.model_dump(mode="json"))
        write_srt(asr_doc.segments, layout.subtitles_asr_srt)
        if not layout.subtitles_asr_segments_json.is_file():
            # ensure low-level segments file exists even for odd paths
            cache.write_json(
                layout.subtitles_asr_segments_json,
                {
                    "schema_version": "1.0",
                    "source_video": str(video_path),
                    "segments": [
                        {
                            "id": f"raw_seg_{s.index:04d}",
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                            "language": s.language,
                        }
                        for s in asr_doc.segments
                    ],
                    "note": "fallback segments from merged ASR",
                },
            )

        state.engine = engine
        state.asr_raw_path = str(layout.asr_raw_json)
        state.asr_manifest_path = str(layout.asr_manifest_json)
        for st in (
            PipelineStage.AUDIO_CHUNKING,
            PipelineStage.ASR_TRANSCRIPTION,
            PipelineStage.ASR_TIME_OFFSET,
            PipelineStage.ASR_OVERLAP_DEDUPE,
            PipelineStage.ASR_LANGUAGE_MERGE,
            PipelineStage.SUBTITLE_RENDER_ASR,
            PipelineStage.ASR,
            PipelineStage.SUBTITLES,
        ):
            state.mark(st)
        cache.save_state(state)

    assert asr_doc is not None

    # --- LLM calibrate + validate ---
    if (
        resume
        and state.has(PipelineStage.LLM_CALIBRATE)
        and layout.review_json.is_file()
        and layout.subtitles_json.is_file()
    ):
        log.info("Resume: LLM calibration / active subtitles from cache")
        from video_analyst.llm.review import load_review

        package = load_review(layout)
    else:
        hashes = SourceHashes(
            video=sha256_file(video_path, max_bytes=2 * 1024 * 1024),
            audio=sha256_file(audio_path, max_bytes=2 * 1024 * 1024),
            asr_subtitles=sha256_json(asr_doc.model_dump(mode="json")),
            config=state.config_fingerprint,
        )
        asr_meta = ASRMeta(
            provider="groq",
            model=cfg.asr.cloud_model,
            multilingual=cfg.asr.multilingual,
            prompt_version=cfg.asr.prompt_version,
        )
        if cfg.subtitle_calibration.enabled and llm_calibrate:
            package = calibrate_subtitles(
                asr_doc,
                config=cfg,
                source_hashes=hashes,
                asr_meta=asr_meta,
            )
        else:
            package = SubtitleReviewPackage(
                source_video=asr_doc.source_video,
                active=SubtitleVersion.ASR,
                reviewed=False,
                asr=asr_doc,
                llm=None,
                diffs=[],
                asr_meta=asr_meta,
                source_hashes=hashes,
                llm_error="LLM calibration disabled",
            )
            package.llm_meta.enabled = False
            package.llm_meta.status = "skipped"
            package.llm_meta.error = "disabled"

        save_review_package(package, layout, cache)
        cache.write_json(
            layout.llm_response_json,
            {
                "model": package.llm_meta.model,
                "status": package.llm_meta.status,
                "error": package.llm_meta.error,
                "summary": package.summary.model_dump(mode="json"),
                "active": package.active.value,
            },
        )
        state.review_json_path = str(layout.review_json)
        state.subtitles_json_path = str(layout.subtitles_json)
        state.subtitles_srt_path = str(layout.subtitles_srt)
        state.active_subtitle_version = package.active
        state.mark(PipelineStage.LLM_CALIBRATE)
        state.mark(PipelineStage.LLM_VALIDATE)
        state.mark(PipelineStage.SUBTITLE_ACTIVATE)
        cache.save_state(state)

    active = package.active_document()
    log.info(
        "Phase 1 complete: segments=%d active=%s needs_review=%s",
        len(active.segments),
        package.active.value,
        package.needs_review,
    )

    # --- Phase 2 prep (no visual API) ---
    if run_phase2_prep and cfg.phase2.enabled:
        try:
            if (
                resume
                and state.has(PipelineStage.PHASE2_VISUAL_BATCHES)
                and layout.phase2_batches_json.is_file()
                and not pinned_timestamps
            ):
                log.info("Resume: phase2 batches from cache")
            else:
                result = prepare_phase2(
                    video_path,
                    layout,
                    env,
                    cfg=cfg.phase2,
                    user_request=user_request,
                    resume=resume,
                    pinned_timestamps=pinned_timestamps,
                )
                state.phase2_batches_path = result.batches_path
                state.mark(PipelineStage.PHASE2_FIXED_FRAMES)
                state.mark(PipelineStage.PHASE2_SCENE_FRAMES)
                state.mark(PipelineStage.PHASE2_SUBTITLE_MAPPING)
                state.mark(PipelineStage.PHASE2_VISUAL_BATCHES)
                cache.save_state(state)
                log.info(
                    "Phase 2 prep: fixed=%d scene=%d batches=%d",
                    result.fixed_frame_count,
                    result.scene_frame_count,
                    result.batch_count,
                )
        except Exception as exc:
            log.warning("Phase 2 prep failed (non-fatal for Phase 1): %s", exc)
            state.last_error = f"phase2_prep: {exc}"
            cache.save_state(state)

    state.mark(PipelineStage.DONE)
    cache.save_state(state)
    return active
