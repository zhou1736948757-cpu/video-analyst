"""ASR orchestration: Groq primary with local Whisper fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from video_analyst.asr.chunked import transcribe_chunked_groq
from video_analyst.asr.groq_whisper import DEFAULT_MULTILINGUAL_PROMPT, groq_available
from video_analyst.asr.local_whisper import (
    local_whisper_installed,
    transcribe_with_local_whisper,
)
from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import ASRError, ASRTranscriptionError, GroqASRError, LocalWhisperError
from video_analyst.logging.setup import get_logger
from video_analyst.models import ASREngine, SubtitleDocument

log = get_logger("asr.pipeline")

_FORCE_LOCAL_ENV = "VIDEO_ANALYST_FORCE_LOCAL_WHISPER"


def transcribe_audio(
    audio_path: Path,
    *,
    language: str | None = None,
    multilingual: bool = True,
    prompt: str | None = DEFAULT_MULTILINGUAL_PROMPT,
    ffmpeg_env: FFmpegEnvironment | None = None,
    layout: OutputLayout | None = None,
    cache: CacheStore | None = None,
    config: AppConfig | None = None,
    source_video: str | None = None,
    video_duration: float | None = None,
    force_local: bool = False,
) -> tuple[dict[str, Any], ASREngine, SubtitleDocument | None]:
    cfg = (config or DEFAULT_CONFIG).with_cli_overrides(
        language=language,
        multilingual=multilingual,
    )

    lang = cfg.asr.language

    # Resolve effective routing mode
    effective_local = (
        force_local
        or os.environ.get(_FORCE_LOCAL_ENV, "").strip() in ("1", "true", "yes")
        or cfg.asr.provider == "local"
    )

    # Local Whisper path (offline / forced local)
    if effective_local:
        if not local_whisper_installed():
            raise ASRError(
                "Local Whisper is not installed. "
                "Install with: pip install 'video-analyst[local-whisper]'"
            )
        log.info("Routing ASR to local Whisper (offline mode)")
        try:
            payload, engine = transcribe_with_local_whisper(
                audio_path,
                language=lang,
            )
            return payload, engine, None
        except LocalWhisperError as exc:
            raise ASRError(f"Local Whisper failed: {exc}") from exc

    # Groq cloud path (default primary)
    if not groq_available():
        # Fall back to local if available
        if local_whisper_installed():
            log.info("Groq unavailable; falling back to local Whisper")
            try:
                payload, engine = transcribe_with_local_whisper(
                    audio_path,
                    language=lang,
                )
                return payload, engine, None
            except LocalWhisperError as exc:
                raise ASRError(
                    "ASR unavailable: Groq not configured and local Whisper failed."
                ) from exc
        raise ASRError(
            "ASR requires Groq cloud API. "
            "Set GROQ_API_KEY in your .env file."
        )

    try:
        return transcribe_chunked_groq(
            audio_path,
            env=ffmpeg_env,
            language=lang,
            prompt=prompt,
            layout=layout,
            cache=cache,
            config=cfg,
            source_video=source_video,
            video_duration=video_duration,
        )
    except (GroqASRError, ASRTranscriptionError) as exc:
        raise ASRError(f"Groq ASR failed: {exc}") from exc
