"""SRT subtitle formatting."""

from __future__ import annotations

from pathlib import Path

from video_analyst.models import SubtitleSegment


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis_total = int(round(seconds * 1000))
    hours, rem = divmod(millis_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[SubtitleSegment]) -> str:
    blocks: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        idx = seg.index if seg.index > 0 else len(blocks) + 1
        start = _format_timestamp(seg.start)
        end = _format_timestamp(seg.end)
        blocks.append(f"{idx}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(segments: list[SubtitleSegment], path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(segments_to_srt(segments), encoding="utf-8")
    return path
