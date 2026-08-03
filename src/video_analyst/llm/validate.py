"""Automatic validation of LLM subtitle calibrations."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from video_analyst.config import CalibrationValidationConfig
from video_analyst.logging.setup import get_logger
from video_analyst.models import (
    CalibrationStatus,
    SegmentDiff,
    SubtitleDocument,
    SubtitleSegment,
)

log = get_logger("llm.validate")

_FILLERS = ("嗯", "啊", "呃", "就是", "然后", "那个", "um", "uh", "like", "you know")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JP_RE = re.compile(r"[\u3040-\u30ff]")


def char_change_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio


def looks_like_translation(src: str, dst: str, language: str | None) -> bool:
    """Heuristic: major script family flipped implies translation."""
    if not src or not dst or src.strip() == dst.strip():
        return False
    lang = (language or "").lower()
    src_cjk = bool(_CJK_RE.search(src) or _JP_RE.search(src))
    dst_cjk = bool(_CJK_RE.search(dst) or _JP_RE.search(dst))
    src_latin = bool(_LATIN_WORD_RE.search(src))
    dst_latin = bool(_LATIN_WORD_RE.search(dst))
    # Japanese/Chinese body became pure Latin paragraph
    if src_cjk and not dst_cjk and dst_latin and len(dst) > 8:
        return True
    # English became mostly CJK
    if "en" in lang and src_latin and not src_cjk and dst_cjk and not dst_latin:
        return True
    return False


def protect_sensitive_change(src: str, dst: str) -> list[str]:
    reasons: list[str] = []
    if _NUMBER_RE.findall(src) != _NUMBER_RE.findall(dst):
        reasons.append("number_change")
    src_latin = set(m.group(0).lower() for m in _LATIN_WORD_RE.finditer(src))
    dst_latin = set(m.group(0).lower() for m in _LATIN_WORD_RE.finditer(dst))
    # brand/model-like tokens removed or added
    if src_latin - dst_latin or dst_latin - src_latin:
        # only flag if short token set differs significantly
        if abs(len(src_latin) - len(dst_latin)) >= 1 and (src_latin or dst_latin):
            # ignore trivial case changes
            if {x.lower() for x in src_latin} != {x.lower() for x in dst_latin}:
                reasons.append("latin_token_change")
    return reasons


def filler_stripped(src: str, dst: str) -> bool:
    for f in _FILLERS:
        if f in src and f not in dst:
            return True
    return False


def validate_structure(
    asr: SubtitleDocument,
    llm_segments: list[dict],
) -> str | None:
    """Return error message if structure invalid, else None."""
    if len(llm_segments) != len(asr.segments):
        return f"segment count mismatch: asr={len(asr.segments)} llm={len(llm_segments)}"
    by_idx = {}
    for item in llm_segments:
        try:
            by_idx[int(item["index"])] = item
        except Exception:
            return "invalid index in LLM output"
    for seg in asr.segments:
        if seg.index not in by_idx:
            return f"missing index {seg.index} in LLM output"
        # timestamps must not be provided differently; if present must match
        item = by_idx[seg.index]
        if "start" in item and abs(float(item["start"]) - seg.start) > 1e-3:
            return f"start changed at index {seg.index}"
        if "end" in item and abs(float(item["end"]) - seg.end) > 1e-3:
            return f"end changed at index {seg.index}"
    return None


def validate_and_build_llm_document(
    asr: SubtitleDocument,
    llm_items: list[dict],
    *,
    cfg: CalibrationValidationConfig | None = None,
    model_name: str | None = None,
) -> tuple[SubtitleDocument | None, list[SegmentDiff], bool, str | None]:
    """
    Validate LLM outputs entry-wise.

    Returns (llm_doc_or_none, diffs, overall_ok_for_active_llm, error).
    overall_ok_for_active_llm False → active should be asr (or hybrid with care).
    """
    cfg = cfg or CalibrationValidationConfig()
    struct_err = validate_structure(asr, llm_items)
    if struct_err:
        log.warning("LLM structure validation failed: %s", struct_err)
        diffs = [
            SegmentDiff(
                index=s.index,
                start=s.start,
                end=s.end,
                language=s.language,
                asr_text=s.text,
                llm_text=s.text,
                active_text=s.text,
                status=CalibrationStatus.REJECTED,
                risk_reasons=["structure_error"],
                changed=False,
            )
            for s in asr.segments
        ]
        return None, diffs, False, struct_err

    by_idx = {int(i["index"]): i for i in llm_items}
    diffs: list[SegmentDiff] = []
    out_segments: list[SubtitleSegment] = []
    high_risk = 0

    for seg in asr.segments:
        item = by_idx[seg.index]
        llm_text = str(item.get("text", seg.text)).strip() or seg.text
        note = str(item["note"]) if item.get("note") else None
        ratio = char_change_ratio(seg.text, llm_text)
        risks: list[str] = []
        status = CalibrationStatus.UNCHANGED
        active = seg.text

        if llm_text == seg.text:
            status = CalibrationStatus.UNCHANGED
            active = seg.text
        else:
            if cfg.reject_translation and looks_like_translation(
                seg.text, llm_text, seg.language
            ):
                risks.append("translation")
                status = CalibrationStatus.REJECTED
                active = seg.text
                high_risk += 1
            else:
                if cfg.protect_numbers or cfg.protect_named_entities:
                    risks.extend(protect_sensitive_change(seg.text, llm_text))
                if cfg.preserve_fillers and filler_stripped(seg.text, llm_text):
                    risks.append("filler_stripped")
                # length blow-up
                if len(llm_text) > max(40, int(len(seg.text) * 2.2)):
                    risks.append("length_blowup")
                if ratio > cfg.review_change_ratio or "length_blowup" in risks:
                    status = CalibrationStatus.REJECTED
                    active = seg.text
                    high_risk += 1
                elif ratio > cfg.accept_change_ratio or risks:
                    status = CalibrationStatus.NEEDS_REVIEW
                    # still use LLM text for needs_review per mixed policy,
                    # but mark needs_review
                    active = llm_text
                else:
                    status = CalibrationStatus.ACCEPTED
                    active = llm_text

        diffs.append(
            SegmentDiff(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                language=seg.language,
                asr_text=seg.text,
                llm_text=llm_text,
                active_text=active,
                change_ratio=round(ratio, 4),
                status=status,
                risk_reasons=risks,
                note=note,
                changed=(llm_text != seg.text),
            )
        )
        out_segments.append(
            SubtitleSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=active,
                language=seg.language,
                is_code_switched=seg.is_code_switched,
                source_segment_ids=list(seg.source_segment_ids),
                source="llm",
                merge_reason=seg.merge_reason,
                calibration_status=status.value,
            )
        )

    # overall decision
    n = max(1, len(diffs))
    rejected_frac = sum(1 for d in diffs if d.status == CalibrationStatus.REJECTED) / n
    overall_ok = rejected_frac <= cfg.max_high_risk_fraction

    llm_doc = SubtitleDocument(
        source_video=asr.source_video,
        audio_path=asr.audio_path,
        language=asr.language,
        engine=asr.engine,
        duration=asr.duration,
        segments=out_segments,
        origin="llm",
        metadata={
            "calibrated_from": "asr",
            "llm_model": model_name,
            "overall_ok": overall_ok,
            "rejected_fraction": rejected_frac,
        },
    )
    if not overall_ok:
        log.warning(
            "LLM calibration high-risk fraction %.2f exceeds threshold; prefer ASR active",
            rejected_frac,
        )
    return llm_doc, diffs, overall_ok, None
