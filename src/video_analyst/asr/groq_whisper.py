"""Groq Whisper Large V3 Turbo transcription client."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx

from video_analyst.exceptions import GroqASRError
from video_analyst.logging.setup import get_logger
from video_analyst.models import ASREngine

log = get_logger("asr.groq")

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
# Multilingual / code-switch hint. Whisper often locks onto one language without this.
DEFAULT_MULTILINGUAL_PROMPT = (
    "按音频中实际使用的语言忠实转写。"
    "保留中文、日语、英语及其他语言的原始语言。"
    "不要把整段翻译成单一语言。"
    "保留人名、品牌名、产品名和英文术语。"
)
# Groq free-tier practical limit is often ~25MB; keep a safety margin.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def normalize_provider_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize Groq Whisper response to guaranteed contract before caching.

    Every segment receives valid numeric start/end (never None), stripped text,
    and a re-indexed id.  Drops empty segments.  Adds _engine / _model markers
    if the raw response lacks them (e.g. older payloads or synthetic test data).
    """
    result = dict(payload)

    # Ensure provider markers
    if "_engine" not in result:
        result["_engine"] = ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value
    if "_model" not in result:
        result["_model"] = str(result.get("model", GROQ_MODEL))

    # Ensure segments list exists
    raw_segs: list[dict[str, Any]] = list(result.get("segments") or [])
    if not raw_segs:
        text = str(result.get("text", "")).strip()
        if text:
            dur = float(result.get("duration") or 0.0)
            raw_segs = [{"id": 0, "start": 0.0, "end": dur, "text": text}]

    # Normalize each segment
    normalized: list[dict[str, Any]] = []
    for seg in raw_segs:
        s = dict(seg)
        s["id"] = int(s.get("id", 0))
        s["start"] = max(float(s.get("start") or 0.0), 0.0)
        raw_end = float(s.get("end") or s.get("start") or 0.0)
        s["end"] = max(raw_end, s["start"])
        s["text"] = str(s.get("text", "")).strip()
        if not s["text"]:
            continue  # drop empty segments
        normalized.append(s)

    # Re-index
    for i, seg in enumerate(normalized):
        seg["id"] = i

    result["segments"] = normalized
    return result


def groq_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def transcribe_with_groq(
    audio_path: Path,
    *,
    language: str | None = None,
    prompt: str | None = DEFAULT_MULTILINGUAL_PROMPT,
    model: str = GROQ_MODEL,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], ASREngine]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise GroqASRError("GROQ_API_KEY is not configured")

    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise GroqASRError(f"Audio file not found: {path}")

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise GroqASRError(
            f"Audio file too large for Groq upload ({size} bytes > {MAX_UPLOAD_BYTES}). "
            "Re-extract with a lower bitrate or use local Whisper."
        )
    if size < 100:
        raise GroqASRError(f"Audio file too small: {path} ({size} bytes)")

    log.info("Calling Groq Whisper (%s) for %s", model, path.name)

    headers = {"Authorization": f"Bearer {api_key}"}
    data: dict[str, str] = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": str(temperature),
    }
    # Only lock language when the caller explicitly requests it.
    # Auto-detect (omit language) is required for mixed JP/ZH/EN speech.
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    try:
        with path.open("rb") as fh:
            files = {"file": (path.name, fh, "audio/mpeg")}
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    GROQ_TRANSCRIPTIONS_URL,
                    headers=headers,
                    data=data,
                    files=files,
                )
    except httpx.HTTPError as exc:
        raise GroqASRError(f"Groq HTTP error: {exc}") from exc

    if response.status_code >= 400:
        # Do not include response headers (may echo auth); body is usually safe.
        body = response.text[:800]
        raise GroqASRError(f"Groq API error {response.status_code}: {body}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise GroqASRError("Groq returned non-JSON response") from exc

    if not isinstance(payload, dict):
        raise GroqASRError("Unexpected Groq response shape")

    # Normalize segments if missing
    if "segments" not in payload:
        text = str(payload.get("text", "")).strip()
        payload["segments"] = (
            [{"id": 0, "start": 0.0, "end": float(payload.get("duration") or 0.0), "text": text}]
            if text
            else []
        )

    payload["_engine"] = ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value
    payload["_model"] = model
    log.info(
        "Groq transcription OK: segments=%d language=%s text_preview=%r",
        len(payload.get("segments") or []),
        payload.get("language"),
        str(payload.get("text", ""))[:80],
    )
    return payload, ASREngine.GROQ_WHISPER_LARGE_V3_TURBO


def normalize_lang_key(lang: str | None) -> str:
    """Map Whisper language labels to a coarse family key for merging."""
    if not lang:
        return "unknown"
    s = str(lang).strip().lower()
    if s in {"ja", "jp", "jpn", "japanese", "日本語"}:
        return "japanese"
    if s in {"zh", "zh-cn", "zh-tw", "cn", "chinese", "mandarin", "中文", "汉语"}:
        return "chinese"
    if s in {"en", "eng", "english", "english (us)", "english (uk)"}:
        return "english"
    return s


def _join_segment_text(left: str, right: str, lang_key: str) -> str:
    """Join two pieces of the same language without awkward double spaces."""
    a = left.rstrip()
    b = right.lstrip()
    if not a:
        return b
    if not b:
        return a
    # CJK: usually no space between fragments of one sentence
    if lang_key in {"japanese", "chinese", "korean"}:
        # avoid duplicating punctuation when both sides have 。．.
        if a[-1] in "。．.!?！？" and b[0] in "。．.!?！？":
            return a + b[1:]
        return a + b
    # Latin scripts: single space unless left already ends with space/punct joining
    if a[-1].isspace():
        return a + b
    if a[-1] in "-/'":
        return a + b
    return a + " " + b


def merge_same_language_segments(
    segments: list[dict[str, Any]],
    *,
    max_gap: float = 1.8,
) -> list[dict[str, Any]]:
    """
    Collapse adjacent same-language pieces that were split only by a short pause.

    Silence-based chunking often cuts one spoken sentence into 2+ ranges when the
    speaker breathes mid-sentence. After each range has its language detected,
    re-join consecutive same-language segments if the gap is small.
    """
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lang = seg.get("language")
        key = normalize_lang_key(lang if isinstance(lang, str) else None)
        start = float(seg["start"])
        end = float(seg["end"])
        if end < start:
            end = start

        if not merged:
            merged.append(
                {
                    "id": 0,
                    "start": start,
                    "end": end,
                    "text": text,
                    "language": lang,
                    "_lang_key": key,
                }
            )
            continue

        prev = merged[-1]
        gap = start - float(prev["end"])
        same_lang = prev.get("_lang_key") == key and key != "unknown"
        if same_lang and gap <= max_gap:
            prev["end"] = max(float(prev["end"]), end)
            prev["text"] = _join_segment_text(
                str(prev["text"]),
                text,
                key,
            )
            # keep first language label
            continue

        merged.append(
            {
                "id": len(merged),
                "start": start,
                "end": end,
                "text": text,
                "language": lang,
                "_lang_key": key,
            }
        )

    for i, seg in enumerate(merged):
        seg["id"] = i
        seg.pop("_lang_key", None)
    return merged


def merge_chunk_payloads(
    chunks: list[tuple[float, dict[str, Any]]],
    *,
    total_duration: float | None = None,
    same_lang_max_gap: float = 1.8,
) -> dict[str, Any]:
    """
    Merge per-chunk verbose_json payloads.

    chunks: list of (time_offset_seconds, payload)

    After offsetting timestamps, consecutive segments with the same detected
    language and only a short pause between them are joined into one subtitle
    line (so mid-sentence breaths do not become fake new sentences).
    """
    segments: list[dict[str, Any]] = []
    languages: list[str] = []
    max_end = 0.0

    for offset, payload in chunks:
        lang = payload.get("language")
        if lang:
            languages.append(str(lang))

        raw_segs = payload.get("segments") or []
        chunk_text = str(payload.get("text", "")).strip()
        if not raw_segs and chunk_text:
            dur = float(payload.get("duration") or 0.0)
            raw_segs = [{"id": 0, "start": 0.0, "end": dur, "text": chunk_text}]

        for seg in raw_segs:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0)) + offset
            end = float(seg.get("end", seg.get("start", 0.0))) + offset
            if end < start:
                end = start
            max_end = max(max_end, end)
            segments.append(
                {
                    "id": len(segments),
                    "start": start,
                    "end": end,
                    "text": text,
                    "language": lang,
                }
            )

    # de-dupe nearly identical adjacent segments (chunk pad overlap)
    deduped: list[dict[str, Any]] = []
    for seg in segments:
        if deduped:
            prev = deduped[-1]
            if (
                seg["text"].strip() == prev["text"].strip()
                and abs(seg["start"] - prev["start"]) < 0.6
            ):
                prev["end"] = max(prev["end"], seg["end"])
                continue
        deduped.append(seg)

    # Re-join same-language fragments split only by short pauses
    cleaned = merge_same_language_segments(deduped, max_gap=same_lang_max_gap)

    unique_langs: list[str] = []
    for lang in languages:
        if lang not in unique_langs:
            unique_langs.append(lang)
    # also collect from merged segment labels (order of speech)
    for seg in cleaned:
        lang = seg.get("language")
        if lang and lang not in unique_langs:
            unique_langs.append(str(lang))

    duration = total_duration if total_duration is not None else max_end
    full_text = " ".join(str(s["text"]).strip() for s in cleaned if s.get("text")).strip()
    return {
        "task": "transcribe",
        "language": "+".join(unique_langs) if unique_langs else None,
        "duration": duration,
        "text": full_text,
        "segments": cleaned,
        "_engine": ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value,
        "_model": GROQ_MODEL,
        "_chunked": True,
        "_chunk_count": len(chunks),
        "_languages_detected": unique_langs,
        "_same_lang_merged": True,
    }
