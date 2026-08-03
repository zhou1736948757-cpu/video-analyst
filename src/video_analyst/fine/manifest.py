"""Fine analysis manifest, cache key construction, and batch persistence (Epic 4).

缓存键必须包含：
    Fine Batch JSON hash
    Contact Sheet SHA-256
    Detail Frame SHA-256
    active 字幕 hash
    review.json hash
    user_request hash
    refinement_questions hash
    coarse_context hash
    provider
    model
    fine prompt version
    fine schema version

缓存依赖：
    时序拼图变化      → 对应 Fine Batch 及下游失效
    detail frame 变化 → 对应 Fine Batch 及下游失效
    active 字幕变化   → Fine 分析及下游失效（不重新抽帧）
    用户要求变化      → Fine 分析及下游失效
    粗分析上下文变化  → 对应 Fine Batch 及下游失效
    Provider / 模型变化 → Fine 分析及下游失效
    Fine Prompt 变化  → Fine 分析及下游失效
    只修改第二轮预算  → 只重建 second_refinement_plan（不重新调用视觉模型）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import FineAnalysisConfig, VisionConfig
from video_analyst.fine.models import (
    FINE_SCHEMA_VERSION,
    FineAnalysisManifest,
    FineBatchAnalysis,
    FineBatchRunRecord,
    FineVisionRequest,
)
from video_analyst.fine.prompts import PROMPT_VERSION
from video_analyst.fine.request_builder import RequestBuildReport
from video_analyst.logging.setup import get_logger
from video_analyst.util.hashing import sha256_file, sha256_json

log = get_logger("fine.manifest")


def _read_subtitle_hash(layout: OutputLayout) -> str:
    """active subtitles.json + review.json 的哈希。"""
    parts: list[str] = []
    for p in (layout.subtitles_json, layout.review_json):
        if p.is_file():
            try:
                parts.append(sha256_file(p))
            except Exception:
                parts.append(f"missing:{p}")
    return sha256_json(parts)


def _read_coarse_context_hash(layout: OutputLayout) -> str:
    """coarse manifest + video_structure + refinement_plan 的哈希。"""
    parts: list[str] = []
    for p in (
        layout.coarse_manifest_json,
        layout.coarse_structure_json,
        layout.coarse_refinement_plan_json,
    ):
        if p.is_file():
            try:
                parts.append(sha256_file(p))
            except Exception:
                parts.append(f"missing:{p}")
    return sha256_json(parts)


def build_fine_cache_key(
    *,
    request: FineVisionRequest,
    report: RequestBuildReport,
    layout: OutputLayout,
    provider_name: str,
    model: str,
    cfg: FineAnalysisConfig,
    vision_cfg: VisionConfig,
) -> str:
    """构造单个 fine batch 的缓存键。"""
    payload = {
        "fine_batch_id": request.fine_batch_id,
        "interval": request.interval.model_dump(mode="json"),
        "image_signature": report.image_signature,
        "contact_sheet_ids": report.contact_sheet_ids,
        "detail_frame_ids": report.detail_frame_ids,
        "representative_frame_ids": report.representative_frame_ids,
        "subtitles": [s.model_dump(mode="json") for s in request.subtitles],
        "subtitle_metadata": request.subtitle_metadata.model_dump(mode="json"),
        "subtitle_hash": _read_subtitle_hash(layout),
        "coarse_context": request.coarse_context.model_dump(mode="json"),
        "coarse_context_hash": _read_coarse_context_hash(layout),
        "user_request": request.user_request,
        "refinement_questions": list(request.refinement_questions),
        "analysis_dimensions": list(request.analysis_dimensions),
        "provider": provider_name,
        "model": model,
        "prompt_version": cfg.prompt_version or PROMPT_VERSION,
        "schema_version": cfg.fine_schema_version or FINE_SCHEMA_VERSION,
        "vision_config": {
            "request_timeout": vision_cfg.request_timeout,
            "max_retries": vision_cfg.max_retries,
        },
    }
    return sha256_json(payload)


# ---- Manifest load / save ----


def load_fine_manifest(layout: OutputLayout) -> FineAnalysisManifest | None:
    if not layout.fine_manifest_json.is_file():
        return None
    try:
        data = json.loads(layout.fine_manifest_json.read_text(encoding="utf-8"))
        return FineAnalysisManifest.model_validate(data)
    except Exception as exc:
        log.warning("could not load fine manifest: %s", exc)
        return None


def save_fine_manifest(layout: OutputLayout, manifest: FineAnalysisManifest) -> None:
    cache = CacheStore(layout)
    cache.write_json(layout.fine_manifest_json, manifest.model_dump(mode="json"))


# ---- Batch analysis persistence ----


def persist_fine_batch_analysis(
    layout: OutputLayout,
    analysis: FineBatchAnalysis,
) -> Path:
    """原子写入单个 fine batch 分析结果。"""
    cache = CacheStore(layout)
    path = layout.fine_batches_out_dir / f"{analysis.fine_batch_id}.json"
    cache.write_json(path, analysis.model_dump(mode="json"))
    return path


def load_fine_batch_analysis(
    layout: OutputLayout, fine_batch_id: str
) -> FineBatchAnalysis | None:
    path = layout.fine_batches_out_dir / f"{fine_batch_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FineBatchAnalysis.model_validate(data)
    except Exception as exc:
        log.warning("could not load fine batch %s: %s", fine_batch_id, exc)
        return None


def persist_fine_raw_response(
    layout: OutputLayout,
    fine_batch_id: str,
    cache_key: str,
    raw_text: str,
) -> Path:
    """持久化脱敏后的原始响应。"""
    cache = CacheStore(layout)
    path = layout.fine_raw_dir / f"{fine_batch_id}_response.json"
    payload = {
        "schema_version": FINE_SCHEMA_VERSION,
        "fine_batch_id": fine_batch_id,
        "cache_key": cache_key,
        "raw_text": raw_text,
    }
    cache.write_json(path, payload)
    return path


def upsert_fine_batch_record(
    manifest: FineAnalysisManifest,
    record: FineBatchRunRecord,
) -> FineAnalysisManifest:
    """按 fine_batch_id 替换或插入 manifest 记录，并排序保持稳定。"""
    new_batches = [
        b for b in manifest.batches if b.fine_batch_id != record.fine_batch_id
    ]
    new_batches.append(record)
    new_batches.sort(key=lambda b: b.fine_batch_id)
    return manifest.model_copy(update={"batches": new_batches})


__all__ = [
    "build_fine_cache_key",
    "load_fine_batch_analysis",
    "load_fine_manifest",
    "persist_fine_batch_analysis",
    "persist_fine_raw_response",
    "save_fine_manifest",
    "upsert_fine_batch_record",
]
