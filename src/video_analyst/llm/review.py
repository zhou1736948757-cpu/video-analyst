"""Load/save dual-version subtitle review packages and apply active choice."""

from __future__ import annotations

from pathlib import Path

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.exceptions import SubtitleActivationError, SubtitleReviewError
from video_analyst.logging.setup import get_logger
from video_analyst.models import (
    PipelineStage,
    SubtitleReviewPackage,
    SubtitleVersion,
    utc_now,
)
from video_analyst.subtitle.srt import write_srt

log = get_logger("llm.review")

PHASE2_SUBTITLE_DEPENDENT = {
    PipelineStage.PHASE2_SUBTITLE_MAPPING,
    PipelineStage.PHASE2_VISUAL_BATCHES,
}


def save_review_package(
    package: SubtitleReviewPackage,
    layout: OutputLayout,
    cache: CacheStore | None = None,
) -> SubtitleReviewPackage:
    store = cache or CacheStore(layout)
    package.updated_at = utc_now()
    package.recompute_summary()

    asr = package.asr.model_copy(deep=True)
    asr.origin = "asr"
    store.write_json(layout.subtitles_asr_json, asr.model_dump(mode="json"))
    write_srt(asr.segments, layout.subtitles_asr_srt)

    if package.llm is not None:
        llm = package.llm.model_copy(deep=True)
        llm.origin = "llm"
        store.write_json(layout.subtitles_llm_json, llm.model_dump(mode="json"))
        write_srt(llm.segments, layout.subtitles_llm_srt)

    # review.json — export with asr/llm keys in diffs for readability
    review_data = package.model_dump(mode="json", by_alias=True)
    # normalize diffs keys for the contract
    for d in review_data.get("diffs") or []:
        if "asr_text" in d and "asr" not in d:
            d["asr"] = d.get("asr_text")
        if "llm_text" in d and "llm" not in d:
            d["llm"] = d.get("llm_text")
    # nested asr/llm meta aliases expected by contract
    review_data["asr_info"] = review_data.get("asr_meta")
    review_data["llm"] = review_data.get("llm_meta")
    # keep documents under different keys to avoid clash with meta
    review_data["asr_document"] = asr.model_dump(mode="json")
    if package.llm is not None:
        review_data["llm_document"] = package.llm.model_dump(mode="json")
    # primary fields per contract
    review_data["asr"] = {
        "provider": package.asr_meta.provider,
        "model": package.asr_meta.model,
        "multilingual": package.asr_meta.multilingual,
        "prompt_version": package.asr_meta.prompt_version,
    }
    review_data["llm"] = {
        "enabled": package.llm_meta.enabled,
        "provider": package.llm_meta.provider,
        "model": package.llm_meta.model,
        "prompt_version": package.llm_meta.prompt_version,
        "status": package.llm_meta.status,
        "error": package.llm_meta.error,
    }
    store.write_json(layout.review_json, review_data)

    active_doc = package.active_document().model_copy(deep=True)
    active_doc.origin = "active"
    store.write_json(layout.subtitles_json, active_doc.model_dump(mode="json"))
    write_srt(active_doc.segments, layout.subtitles_srt)

    log.info(
        "Saved dual subtitles: active=%s needs_review=%s",
        package.active.value,
        package.needs_review,
    )
    return package


def load_review(layout: OutputLayout) -> SubtitleReviewPackage:
    store = CacheStore(layout)
    if not layout.review_json.is_file():
        raise SubtitleReviewError(f"review.json missing: {layout.review_json}")
    data = store.read_json(layout.review_json)

    # Prefer embedded documents when present (new format)
    if "asr_document" in data:
        payload = {
            "schema_version": data.get("schema_version", "1.0"),
            "source_video": data.get("source_video"),
            "active": data.get("active", "asr"),
            "reviewed": data.get("reviewed", False),
            "needs_review": data.get("needs_review", False),
            "asr": data["asr_document"],
            "llm": data.get("llm_document"),
            "diffs": _normalize_diffs(data.get("diffs") or []),
            "asr_meta": data.get("asr_info") or data.get("asr_meta") or data.get("asr"),
            "llm_meta": data.get("llm_meta")
            if isinstance(data.get("llm_meta"), dict)
            and "status" in (data.get("llm_meta") or {})
            else data.get("llm"),
            "source_hashes": data.get("source_hashes") or {},
            "summary": data.get("summary") or {},
            "llm_model": (data.get("llm") or {}).get("model")
            if isinstance(data.get("llm"), dict)
            else data.get("llm_model"),
            "llm_error": (data.get("llm") or {}).get("error")
            if isinstance(data.get("llm"), dict)
            else data.get("llm_error"),
        }
        return SubtitleReviewPackage.model_validate(payload)

    # Legacy format: asr/llm are full SubtitleDocument
    return SubtitleReviewPackage.model_validate(data)


def _normalize_diffs(diffs: list[dict]) -> list[dict]:
    out = []
    for d in diffs:
        item = dict(d)
        if "asr" in item and "asr_text" not in item:
            item["asr_text"] = item["asr"]
        if "llm" in item and "llm_text" not in item:
            item["llm_text"] = item["llm"]
        out.append(item)
    return out


def apply_active_version(
    layout: OutputLayout,
    version: SubtitleVersion,
    *,
    reviewed: bool = True,
) -> SubtitleReviewPackage:
    package = load_review(layout)
    if version == SubtitleVersion.LLM and package.llm is None:
        raise SubtitleActivationError("LLM version not available in this review package")

    package.active = version
    package.reviewed = reviewed
    package.updated_at = utc_now()
    save_review_package(package, layout)

    # Invalidate Phase 2 subtitle-derived caches only (keep frames)
    store = CacheStore(layout)
    store.clear_phase2_subtitle_derived()
    try:
        state = store.load_state(Path(package.source_video))
        state.invalidate_from(PHASE2_SUBTITLE_DEPENDENT)
        # also drop DONE so phase2 can re-run mapping
        state.completed_stages = [
            s
            for s in state.completed_stages
            if s not in PHASE2_SUBTITLE_DEPENDENT and s != PipelineStage.DONE
        ]
        store.save_state(state)
    except Exception:
        # state may not match video path; still cleared files
        pass

    log.info("Active subtitle version set to %s (reviewed=%s)", version.value, reviewed)
    return package


def format_review_table(package: SubtitleReviewPackage) -> str:
    lines = [
        f"source: {package.source_video}",
        f"active: {package.active.value}",
        f"reviewed: {package.reviewed}",
        f"needs_review: {package.needs_review}",
        f"llm_model: {package.llm_meta.model if package.llm_meta else package.llm_model}",
        f"llm_status: {package.llm_meta.status if package.llm_meta else '-'}",
        f"llm_error: {package.llm_meta.error or package.llm_error or '-'}",
        f"summary: {package.summary.model_dump()}",
        "",
        f"{'#':<4} {'st':<12} {'ratio':<6} {'time':<16} {'ASR':<36} {'LLM'}",
        "-" * 110,
    ]
    for d in package.diffs:
        asr = (d.asr_text[:34] + "…") if len(d.asr_text) > 35 else d.asr_text
        llm = (d.llm_text[:34] + "…") if len(d.llm_text) > 35 else d.llm_text
        t = f"{d.start:.1f}-{d.end:.1f}"
        lines.append(
            f"{d.index:<4} {d.status.value:<12} {d.change_ratio:<6.2f} {t:<16} {asr:<36} {llm}"
        )
        if d.risk_reasons:
            lines.append(f"     risk: {', '.join(d.risk_reasons)}")
        if d.note and d.changed:
            lines.append(f"     note: {d.note}")
    return "\n".join(lines)
