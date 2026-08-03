"""Chunked multilingual ASR with persistent hierarchy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.asr.groq_whisper import (
    DEFAULT_MULTILINGUAL_PROMPT,
    normalize_provider_response,
    transcribe_with_groq,
)
from video_analyst.asr.merge import (
    build_asr_document,
    dedupe_overlap_segments,
    merge_same_language_segments,
    offset_raw_segments,
)
from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.env.detect import FFmpegEnvironment, detect_ffmpeg_environment
from video_analyst.exceptions import ASRTranscriptionError, GroqASRError
from video_analyst.logging.setup import get_logger
from video_analyst.media.chunk_plan import plan_audio_chunks
from video_analyst.media.silence import extract_audio_segment, _probe_duration
from video_analyst.models import (
    ASRChunkRecord,
    ASREngine,
    ASRManifest,
    SubtitleDocument,
)

log = get_logger("asr.chunked")


def transcribe_chunked_groq(
    audio_path: Path,
    *,
    env: FFmpegEnvironment | None = None,
    language: str | None = None,
    prompt: str | None = DEFAULT_MULTILINGUAL_PROMPT,
    layout: OutputLayout | None = None,
    cache: CacheStore | None = None,
    config: AppConfig | None = None,
    source_video: str | None = None,
    video_duration: float | None = None,
    min_chunks_for_split: int = 1,
    max_chunk_seconds: float | None = None,
) -> tuple[dict[str, Any], ASREngine, SubtitleDocument | None]:
    """
    Multilingual ASR with optional persistent chunk hierarchy.

    Returns (merged_raw_dict, engine, asr_document_or_none).
    When layout is provided, writes asr/chunks, raw responses, manifest,
    subtitles_asr_segments.json.
    """
    cfg = config or DEFAULT_CONFIG
    ffmpeg_env = env or detect_ffmpeg_environment()
    audio_path = audio_path.expanduser().resolve()
    duration = _probe_duration(audio_path, ffmpeg_env)
    if video_duration:
        duration = min(duration, video_duration)

    # Single-language lock → one shot
    if language or not cfg.asr.multilingual:
        raw_payload, engine = transcribe_with_groq(
            audio_path,
            language=language,
            prompt=prompt if cfg.asr.multilingual else prompt,
        )
        payload = normalize_provider_response(raw_payload)
        doc = None
        if layout is not None:
            doc = _persist_single_shot(
                payload,
                layout=layout,
                cache=cache,
                audio_path=audio_path,
                source_video=source_video or str(audio_path),
                duration=duration,
                language=language,
                engine=engine,
                cfg=cfg,
            )
        return payload, engine, doc

    chunk_cfg = cfg.asr.chunking
    if max_chunk_seconds is not None:
        from dataclasses import replace

        chunk_cfg = replace(chunk_cfg, maximum_seconds=max_chunk_seconds)

    plans = plan_audio_chunks(
        audio_path,
        ffmpeg_env,
        duration=duration,
        cfg=chunk_cfg,
        video_duration=video_duration,
    )

    if len(plans) < min_chunks_for_split:
        raw_payload, engine = transcribe_with_groq(audio_path, language=None, prompt=prompt)
        payload = normalize_provider_response(raw_payload)
        doc = None
        if layout is not None:
            doc = _persist_single_shot(
                payload,
                layout=layout,
                cache=cache,
                audio_path=audio_path,
                source_video=source_video or str(audio_path),
                duration=duration,
                language=None,
                engine=engine,
                cfg=cfg,
            )
        return payload, engine, doc

    log.info("Chunked multilingual ASR: %d chunks", len(plans))
    chunk_records: list[ASRChunkRecord] = []
    chunks_for_merge: list[dict[str, Any]] = []
    store = cache or (CacheStore(layout) if layout else None)

    for plan in plans:
        if layout is not None:
            audio_chunk_path = layout.audio_chunks_dir / f"{plan.chunk_id}.mp3"
            raw_path = layout.asr_groq_raw_dir / f"{plan.chunk_id}.json"
            chunk_json_path = layout.asr_chunks_dir / f"{plan.chunk_id}.json"
        else:
            audio_chunk_path = Path("/tmp") / f"{plan.chunk_id}.mp3"
            raw_path = None
            chunk_json_path = None

        extract_audio_segment(
            audio_path,
            audio_chunk_path,
            ffmpeg_env,
            start=plan.absolute_start,
            end=plan.absolute_end,
        )
        if audio_chunk_path.stat().st_size < 800:
            log.info("Skip tiny chunk %s", plan.chunk_id)
            continue

        try:
            raw_payload, _ = transcribe_with_groq(
                audio_chunk_path,
                language=None,
                prompt=prompt,
            )
            payload = normalize_provider_response(raw_payload)
        except GroqASRError as exc:
            log.warning("Chunk %s failed: %s", plan.chunk_id, exc)
            rec = ASRChunkRecord(
                chunk_id=plan.chunk_id,
                audio_path=str(audio_chunk_path),
                absolute_start=plan.absolute_start,
                absolute_end=plan.absolute_end,
                overlap_before=plan.overlap_before,
                overlap_after=plan.overlap_after,
                error=str(exc),
            )
            chunk_records.append(rec)
            continue

        lang = payload.get("language")
        if raw_path is not None and store is not None:
            store.write_json(raw_path, payload)

        rec = ASRChunkRecord(
            chunk_id=plan.chunk_id,
            audio_path=str(audio_chunk_path),
            absolute_start=plan.absolute_start,
            absolute_end=plan.absolute_end,
            overlap_before=plan.overlap_before,
            overlap_after=plan.overlap_after,
            detected_language=str(lang) if lang else None,
            is_code_switched=False,
            provider="groq",
            model=str(payload.get("_model") or cfg.asr.cloud_model),
            raw_response_path=str(raw_path) if raw_path else None,
            segments=list(payload.get("segments") or []),
            text=str(payload.get("text") or "").strip(),
        )
        chunk_records.append(rec)
        if chunk_json_path is not None and store is not None:
            store.write_json(chunk_json_path, rec.model_dump(mode="json"))

        chunks_for_merge.append(
            {
                "chunk_id": plan.chunk_id,
                "absolute_start": plan.absolute_start,
                "absolute_end": plan.absolute_end,
                "detected_language": rec.detected_language,
                "segments": rec.segments,
                "text": rec.text,
            }
        )
        log.info(
            "Chunk %s OK lang=%s text=%r",
            plan.chunk_id,
            rec.detected_language,
            rec.text[:60],
        )

    if not chunks_for_merge:
        raise ASRTranscriptionError("Chunked ASR produced no successful chunks")

    # Build hierarchy artifacts
    raw_segments = offset_raw_segments(chunks_for_merge)
    deduped = dedupe_overlap_segments(raw_segments)
    merged_segments = merge_same_language_segments(deduped, cfg=cfg.asr.merge)

    languages: list[str] = []
    for r in chunk_records:
        if r.detected_language and r.detected_language not in languages:
            languages.append(r.detected_language)

    engine = ASREngine.GROQ_WHISPER_LARGE_V3_TURBO
    doc = build_asr_document(
        merged_segments,
        source_video=source_video or str(audio_path),
        audio_path=str(audio_path),
        duration=duration,
        engine=engine,
        languages=languages,
    )

    merged_raw: dict[str, Any] = {
        "task": "transcribe",
        "language": doc.language,
        "duration": duration,
        "text": " ".join(s.text for s in doc.segments),
        "segments": [
            {
                "id": s.index - 1,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "language": s.language,
                "source_segment_ids": s.source_segment_ids,
                "merge_reason": s.merge_reason,
            }
            for s in doc.segments
        ],
        "_engine": engine.value,
        "_model": cfg.asr.cloud_model,
        "_chunked": True,
        "_chunk_count": len(chunk_records),
        "_languages_detected": languages,
        "_raw_segment_count": len(raw_segments),
    }

    if layout is not None and store is not None:
        manifest = ASRManifest(
            source_video=source_video or str(audio_path),
            audio_path=str(audio_path),
            duration=duration,
            multilingual=True,
            language_lock=None,
            provider="groq",
            model=cfg.asr.cloud_model,
            prompt_version=cfg.asr.prompt_version,
            chunks=chunk_records,
        )
        store.write_json(layout.asr_manifest_json, manifest.model_dump(mode="json"))
        store.write_json(
            layout.subtitles_asr_segments_json,
            {
                "schema_version": "1.0",
                "source_video": source_video or str(audio_path),
                "segments": deduped,
                "note": "absolute timestamps; before same-language merge",
            },
        )
        store.write_json(layout.asr_raw_json, merged_raw)
        store.write_json(layout.groq_response_json, merged_raw)

    log.info(
        "Chunked ASR merged: raw=%d deduped=%d final=%d langs=%s",
        len(raw_segments),
        len(deduped),
        len(doc.segments),
        languages,
    )
    return merged_raw, engine, doc


def _persist_single_shot(
    payload: dict[str, Any],
    *,
    layout: OutputLayout,
    cache: CacheStore | None,
    audio_path: Path,
    source_video: str,
    duration: float,
    language: str | None,
    engine: ASREngine,
    cfg: AppConfig,
) -> SubtitleDocument:
    store = cache or CacheStore(layout)
    # Normalize provider response before any cache write so every downstream
    # artifact sees validated start/end/text and provider markers.
    safe = normalize_provider_response(payload)
    rec = ASRChunkRecord(
        chunk_id="chunk_0001",
        audio_path=str(audio_path),
        absolute_start=0.0,
        absolute_end=duration,
        detected_language=str(safe.get("language") or language or "") or None,
        provider="groq",
        model=str(safe.get("_model") or cfg.asr.cloud_model),
        raw_response_path=str(layout.asr_groq_raw_dir / "chunk_0001.json"),
        segments=list(safe.get("segments") or []),
        text=str(safe.get("text") or "").strip(),
    )
    store.write_json(layout.asr_groq_raw_dir / "chunk_0001.json", safe)
    store.write_json(layout.asr_chunks_dir / "chunk_0001.json", rec.model_dump(mode="json"))

    chunks = [
        {
            "chunk_id": "chunk_0001",
            "absolute_start": 0.0,
            "absolute_end": duration,
            "detected_language": rec.detected_language,
            "segments": rec.segments,
            "text": rec.text,
        }
    ]
    raw_segments = offset_raw_segments(chunks)
    store.write_json(
        layout.subtitles_asr_segments_json,
        {
            "schema_version": "1.0",
            "source_video": source_video,
            "segments": raw_segments,
            "note": "single-shot ASR; absolute timestamps",
        },
    )
    merged_segments = merge_same_language_segments(raw_segments, cfg=cfg.asr.merge)
    doc = build_asr_document(
        merged_segments,
        source_video=source_video,
        audio_path=str(audio_path),
        duration=duration,
        engine=engine,
        languages=[rec.detected_language] if rec.detected_language else None,
    )
    manifest = ASRManifest(
        source_video=source_video,
        audio_path=str(audio_path),
        duration=duration,
        multilingual=language is None,
        language_lock=language,
        provider="groq",
        model=cfg.asr.cloud_model,
        prompt_version=cfg.asr.prompt_version,
        chunks=[rec],
    )
    store.write_json(layout.asr_manifest_json, manifest.model_dump(mode="json"))
    store.write_json(layout.asr_raw_json, safe)
    return doc
