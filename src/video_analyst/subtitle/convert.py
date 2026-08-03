"""Convert ASR raw payloads into SubtitleDocument."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.models import ASREngine, SubtitleDocument, SubtitleSegment


def raw_asr_to_document(
    raw: dict[str, Any],
    *,
    source_video: Path,
    audio_path: Path | None,
    engine: ASREngine,
) -> SubtitleDocument:
    language = raw.get("language")
    duration_raw = raw.get("duration")
    duration: float | None
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None

    segments: list[SubtitleSegment] = []
    for i, item in enumerate(raw.get("segments") or [], start=1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end < start:
            end = start
        segments.append(
            SubtitleSegment(
                index=i,
                start=start,
                end=end,
                text=text,
                language=(
                    str(item.get("language"))
                    if item.get("language") is not None
                    else None
                ),
            )
        )

    # Stable 1-based indices after filtering empty text
    for i, seg in enumerate(segments, start=1):
        seg.index = i

    return SubtitleDocument(
        source_video=str(source_video.resolve()),
        audio_path=str(audio_path.resolve()) if audio_path else None,
        language=language,
        engine=engine,
        duration=duration,
        segments=segments,
        metadata={
            "text": raw.get("text"),
            "task": raw.get("task"),
        },
        origin="asr",
    )
