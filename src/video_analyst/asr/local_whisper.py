"""Local Whisper Small fallback (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.exceptions import LocalWhisperError
from video_analyst.logging.setup import get_logger
from video_analyst.models import ASREngine

log = get_logger("asr.local")

LOCAL_MODEL = "small"


def local_whisper_installed() -> bool:
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def transcribe_with_local_whisper(
    audio_path: Path,
    *,
    language: str | None = None,
    allow_download: bool = False,
) -> tuple[dict[str, Any], ASREngine]:
    if not local_whisper_installed():
        raise LocalWhisperError(
            "Local Whisper is not installed. "
            "Install with: pip install 'video-analyst[local-whisper]'"
        )

    import whisper  # type: ignore

    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise LocalWhisperError(f"Audio file not found: {path}")

    # openai-whisper downloads models into ~/.cache/whisper on first use.
    # Refuse silent download unless the caller explicitly allows it.
    cache_dir = Path.home() / ".cache" / "whisper"
    model_candidates = list(cache_dir.glob(f"{LOCAL_MODEL}.pt")) if cache_dir.is_dir() else []
    if not model_candidates and not allow_download:
        raise LocalWhisperError(
            f"Local Whisper model '{LOCAL_MODEL}' is not present under {cache_dir}. "
            "Re-run with --allow-whisper-download to download it, "
            "or install/configure GROQ_API_KEY for cloud ASR."
        )

    log.info(
        "Loading local Whisper model '%s'%s",
        LOCAL_MODEL,
        " (download allowed)" if allow_download else "",
    )
    try:
        model = whisper.load_model(LOCAL_MODEL)
        result = model.transcribe(
            str(path),
            language=language,
            verbose=False,
        )
    except Exception as exc:
        raise LocalWhisperError(f"Local Whisper failed: {exc}") from exc

    segments = []
    for i, seg in enumerate(result.get("segments") or []):
        segments.append(
            {
                "id": int(seg.get("id", i)),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
            }
        )

    payload: dict[str, Any] = {
        "text": result.get("text", ""),
        "language": result.get("language"),
        "duration": _estimate_duration(segments),
        "segments": segments,
        "task": result.get("task", "transcribe"),
        "_engine": ASREngine.LOCAL_WHISPER_SMALL.value,
        "_model": LOCAL_MODEL,
    }
    log.info(
        "Local Whisper OK: segments=%d language=%s",
        len(segments),
        payload.get("language"),
    )
    return payload, ASREngine.LOCAL_WHISPER_SMALL


def _estimate_duration(segments: list[dict[str, Any]]) -> float | None:
    if not segments:
        return None
    try:
        return float(max(s.get("end", 0.0) for s in segments))
    except (TypeError, ValueError):
        return None
