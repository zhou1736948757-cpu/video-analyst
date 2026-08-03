"""Time-offset, overlap dedupe, and same-language merge for ASR segments."""

from __future__ import annotations

import re
from typing import Any

from video_analyst.config import MergeConfig
from video_analyst.exceptions import ASRMergeError
from video_analyst.logging.setup import get_logger
from video_analyst.models import SubtitleDocument, SubtitleSegment, ASREngine

log = get_logger("asr.merge")

_SENTENCE_END = re.compile(r"[。．.!?！？…]+$")
_SENTENCE_START = re.compile(r"^[「『(\[【]")


def normalize_lang_key(lang: str | None) -> str:
    if not lang:
        return "unknown"
    s = str(lang).strip().lower()
    mapping = {
        "ja": "japanese",
        "jp": "japanese",
        "jpn": "japanese",
        "japanese": "japanese",
        "日本語": "japanese",
        "zh": "chinese",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
        "cn": "chinese",
        "chinese": "chinese",
        "mandarin": "chinese",
        "中文": "chinese",
        "en": "english",
        "eng": "english",
        "english": "english",
    }
    return mapping.get(s, s)


def _join_text(left: str, right: str, lang_key: str) -> str:
    a = left.rstrip()
    b = right.lstrip()
    if not a:
        return b
    if not b:
        return a
    # drop exact overlap suffix/prefix
    max_ov = min(len(a), len(b), 40)
    for k in range(max_ov, 3, -1):
        if a.endswith(b[:k]):
            b = b[k:]
            break
    if not b:
        return a
    if lang_key in {"japanese", "chinese", "korean"}:
        if a[-1] in "。．.!?！？" and b[0] in "。．.!?！？":
            return a + b[1:]
        return a + b
    if a[-1].isspace() or a[-1] in "-/'":
        return a + b
    return a + " " + b


def offset_raw_segments(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Flatten chunk payloads into absolute-time raw segments (pre-merge).

    Each item gains id like raw_seg_0001 and absolute start/end.
    """
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        offset = float(chunk.get("absolute_start") or 0.0)
        lang = chunk.get("detected_language") or chunk.get("language")
        chunk_id = chunk.get("chunk_id", "chunk")
        segs = chunk.get("segments") or []
        text_fallback = str(chunk.get("text") or "").strip()
        if not segs and text_fallback:
            segs = [
                {
                    "start": 0.0,
                    "end": float(chunk.get("absolute_end", offset) or offset) - offset,
                    "text": text_fallback,
                }
            ]
        for j, seg in enumerate(segs):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0)) + offset
            end = float(seg.get("end", seg.get("start", 0.0))) + offset
            if end < start:
                end = start
            rid = f"raw_seg_{len(out)+1:04d}"
            out.append(
                {
                    "id": rid,
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end,
                    "text": text,
                    "language": lang or seg.get("language"),
                    "source_index_in_chunk": j,
                }
            )
    return out


def dedupe_overlap_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove near-duplicate text from overlapping chunk windows."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    out: list[dict[str, Any]] = []
    for seg in ordered:
        if out:
            prev = out[-1]
            same_text = seg["text"].strip() == prev["text"].strip()
            close = abs(float(seg["start"]) - float(prev["start"])) < 0.75
            # high text similarity with heavy time overlap
            overlap = min(float(prev["end"]), float(seg["end"])) - max(
                float(prev["start"]), float(seg["start"])
            )
            if same_text and (close or overlap > 0.3):
                prev["end"] = max(float(prev["end"]), float(seg["end"]))
                continue
            # prefix/suffix containment with overlap
            if overlap > 0.4:
                a, b = prev["text"].strip(), seg["text"].strip()
                if a in b or b in a:
                    if len(b) > len(a):
                        prev["text"] = b
                    prev["end"] = max(float(prev["end"]), float(seg["end"]))
                    continue
        out.append(dict(seg))
    return out


def merge_same_language_segments(
    segments: list[dict[str, Any]],
    *,
    cfg: MergeConfig | None = None,
) -> list[SubtitleSegment]:
    """
    Merge adjacent same-language fragments using gap + punctuation rules.
    """
    cfg = cfg or MergeConfig()
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
        sid = str(seg.get("id") or f"raw_{len(merged)}")

        if not merged:
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "language": lang,
                    "_lang_key": key,
                    "source_segment_ids": [sid],
                    "merge_reason": None,
                }
            )
            continue

        prev = merged[-1]
        gap = start - float(prev["end"])
        same = prev.get("_lang_key") == key and key != "unknown"
        reason = _should_merge(prev, text, gap=gap, same_lang=same, cfg=cfg)
        if reason:
            prev["end"] = max(float(prev["end"]), end)
            prev["text"] = _join_text(str(prev["text"]), text, key)
            prev["source_segment_ids"].append(sid)
            prev["merge_reason"] = reason
            continue

        merged.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "language": lang,
                "_lang_key": key,
                "source_segment_ids": [sid],
                "merge_reason": None,
            }
        )

    result: list[SubtitleSegment] = []
    for i, m in enumerate(merged, start=1):
        result.append(
            SubtitleSegment(
                index=i,
                start=float(m["start"]),
                end=float(m["end"]),
                text=str(m["text"]).strip(),
                language=m.get("language"),
                is_code_switched=False,
                source_segment_ids=list(m.get("source_segment_ids") or []),
                source="asr",
                merge_reason=m.get("merge_reason"),
            )
        )
    return result


def _should_merge(
    prev: dict[str, Any],
    next_text: str,
    *,
    gap: float,
    same_lang: bool,
    cfg: MergeConfig,
) -> str | None:
    if not same_lang:
        return None
    if gap > cfg.conditional_merge_gap_seconds:
        return None

    prev_text = str(prev.get("text") or "")
    merged_dur = (float(prev["end"]) + max(gap, 0) + 0.01) - float(prev["start"])
    # rough: if we merge, end will extend
    prospective_end = float(prev["end"])  # updated by caller; check length limits on text
    if len(prev_text) + len(next_text) > cfg.maximum_text_length:
        return None

    if _SENTENCE_END.search(prev_text.strip()) and gap > cfg.auto_merge_gap_seconds:
        return None

    if gap <= cfg.auto_merge_gap_seconds:
        # still respect max duration after merge (approx using gap)
        if float(prev["end"]) - float(prev["start"]) + gap + 1.0 > cfg.maximum_merged_duration:
            return None
        return "same_language_short_gap"

    # conditional band 0.8–1.8s
    if _SENTENCE_END.search(prev_text.strip()):
        return None
    if _SENTENCE_START.search(next_text.strip()):
        return None
    if float(prev["end"]) - float(prev["start"]) > cfg.maximum_merged_duration * 0.85:
        return None
    return "same_language_conditional_gap"


def build_asr_document(
    segments: list[SubtitleSegment],
    *,
    source_video: str,
    audio_path: str | None,
    duration: float | None,
    engine: ASREngine | str | None,
    languages: list[str] | None = None,
) -> SubtitleDocument:
    if not segments:
        raise ASRMergeError("No ASR segments after merge")
    # renumber
    for i, s in enumerate(segments, start=1):
        s.index = i
        s.source = "asr"
    lang = "+".join(languages) if languages else None
    if not lang:
        seen: list[str] = []
        for s in segments:
            if s.language and s.language not in seen:
                seen.append(s.language)
        lang = "+".join(seen) if seen else None
    doc = SubtitleDocument(
        source_video=source_video,
        audio_path=audio_path,
        language=lang,
        engine=engine,
        duration=duration,
        segments=segments,
        origin="asr",
        metadata={"merged": True},
    )
    doc.validate_timeline(duration)
    return doc
