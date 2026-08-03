"""Build FineVisionRequest from a FineVisualBatch (Epic 3 output).

读 Epic 3 的 FineVisualBatch，转成 Epic 4 Provider 接受的 FineVisionRequest。

图片发送策略：
- 默认发送全部 contact sheets
- 加最多 max_detail_frames_per_request 张高分辨率 detail frame
- 若没有 contact sheets，从精细帧按时间顺序选代表帧（记录省略）
- 不静默截断；超过上限时记录 omitted

字幕只读 active subtitles.json + review.json（Epic 3 batch 已携带）。
coarse_context 从 batch.coarse_context + 粗分析 manifest/structure 读取。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.config import FineAnalysisConfig
from video_analyst.fine.models import (
    FineContactSheetInput,
    FineCoarseContext,
    FineDetailFrameInput,
    FineVisionRequest,
)
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.models import FineVisualBatch
from video_analyst.util.hashing import sha256_file
from video_analyst.vision.models import SubtitleContext, SubtitleMetadata

log = get_logger("fine.request_builder")


@dataclass
class RequestBuildReport:
    """Request 构建报告（用于缓存键与测试）。"""

    fine_batch_id: str
    contact_sheet_ids: list[str] = field(default_factory=list)
    detail_frame_ids: list[str] = field(default_factory=list)
    representative_frame_ids: list[str] = field(default_factory=list)
    omitted_detail_frame_ids: list[str] = field(default_factory=list)
    omitted_representative_frame_ids: list[str] = field(default_factory=list)
    used_fallback_representatives: bool = False
    # 用于缓存键的图片签名（contact sheet sha + detail frame sha）
    image_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fine_batch_id": self.fine_batch_id,
            "contact_sheet_ids": list(self.contact_sheet_ids),
            "detail_frame_ids": list(self.detail_frame_ids),
            "representative_frame_ids": list(self.representative_frame_ids),
            "omitted_detail_frame_ids": list(self.omitted_detail_frame_ids),
            "omitted_representative_frame_ids": list(
                self.omitted_representative_frame_ids
            ),
            "used_fallback_representatives": self.used_fallback_representatives,
            "image_signature": self.image_signature,
        }


def _safe_sha256(path: Path) -> str:
    try:
        return sha256_file(path) if path.is_file() else ""
    except Exception:
        return ""


def _load_coarse_summaries(layout: OutputLayout) -> list[str]:
    """从 coarse video_structure.json 读 section summaries（参考用）。"""
    if not layout.coarse_structure_json.is_file():
        return []
    try:
        data = json.loads(layout.coarse_structure_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for sec in data.get("sections") or []:
        s = sec.get("summary")
        if isinstance(s, str) and s.strip():
            out.append(s.strip())
    return out


def _load_coarse_refinement_reasons(layout: OutputLayout) -> list[str]:
    """从 coarse refinement_plan.json 读 interval.reasons（参考用）。"""
    if not layout.coarse_refinement_plan_json.is_file():
        return []
    try:
        data = json.loads(layout.coarse_refinement_plan_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for iv in data.get("intervals") or []:
        for r in iv.get("reasons") or []:
            if isinstance(r, str) and r.strip():
                out.append(r.strip())
    return out


def _load_coarse_unresolved(layout: OutputLayout) -> list[str]:
    if not layout.coarse_structure_json.is_file():
        return []
    try:
        data = json.loads(layout.coarse_structure_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for q in data.get("unresolved_questions") or []:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out


def build_fine_request(
    batch: FineVisualBatch,
    *,
    layout: OutputLayout,
    cfg: FineAnalysisConfig,
    user_request: str,
    analysis_dimensions: list[str] | None = None,
    video_id: str = "video",
) -> tuple[FineVisionRequest, RequestBuildReport]:
    """从 FineVisualBatch 构建 FineVisionRequest。"""
    report = RequestBuildReport(fine_batch_id=batch.fine_batch_id)

    # ---- Contact sheets（按时间顺序；Epic 3 已按 sheet_0001/0002 排序）
    contact_sheets: list[FineContactSheetInput] = []
    cs_shas: list[str] = []
    for sheet in batch.contact_sheets:
        sha = sheet.sha256 or _safe_sha256(sheet.path)
        cs_shas.append(sha)
        contact_sheets.append(
            FineContactSheetInput(
                sheet_id=sheet.sheet_id,
                path=sheet.path,
                frame_ids=list(sheet.frame_ids),
                timestamps=list(sheet.timestamps),
                reading_order="row_major",
                overlap_with_previous=list(sheet.overlap_with_previous),
                sha256=sha,
            )
        )
        report.contact_sheet_ids.append(sheet.sheet_id)

    # ---- Detail frames（最多 max_detail_frames_per_request 张）
    max_detail = max(0, int(cfg.max_detail_frames_per_request))
    detail_inputs: list[FineDetailFrameInput] = []
    df_shas: list[str] = []
    for df in batch.detail_frames[:max_detail]:
        sha = getattr(df, "sha256", "") or _safe_sha256(df.path)
        df_shas.append(sha)
        detail_inputs.append(
            FineDetailFrameInput(
                frame_id=df.frame_id,
                timestamp_seconds=float(df.timestamp_seconds),
                path=df.path,
                width=int(getattr(df, "output_width", 0) or 0),
                height=int(getattr(df, "output_height", 0) or 0),
                resolution_mode=str(getattr(df, "resolution_mode", "source_original")),
                selection_reason=str(getattr(df, "selection_reason", "")),
                sha256=sha,
            )
        )
        report.detail_frame_ids.append(df.frame_id)
    # 记录被省略的 detail frame
    for df in batch.detail_frames[max_detail:]:
        report.omitted_detail_frame_ids.append(df.frame_id)

    # ---- 若没有 contact sheets，从精细帧按时间顺序选代表帧
    if not contact_sheets and batch.frames:
        rep_count = max(1, int(cfg.representative_frame_count))
        # frames 已按时间排序
        chosen = batch.frames[:rep_count]
        for fr in chosen:
            sha = fr.sha256 or _safe_sha256(fr.path)
            df_shas.append(sha)
            detail_inputs.append(
                FineDetailFrameInput(
                    frame_id=fr.frame_id,
                    timestamp_seconds=float(fr.timestamp_seconds),
                    path=fr.path,
                    width=int(fr.width),
                    height=int(fr.height),
                    resolution_mode="refinement_frame",
                    selection_reason="representative_fallback",
                    sha256=sha,
                )
            )
            report.representative_frame_ids.append(fr.frame_id)
        for fr in batch.frames[rep_count:]:
            report.omitted_representative_frame_ids.append(fr.frame_id)
        report.used_fallback_representatives = True
        log.info(
            "fine batch %s: no contact sheets, using %d representative frames "
            "(omitted %d)",
            batch.fine_batch_id,
            len(chosen),
            len(report.omitted_representative_frame_ids),
        )

    # ---- 字幕（batch 已携带 active 字幕 + metadata）
    subtitles = [
        SubtitleContext(
            index=s.index,
            start=s.start,
            end=s.end,
            text=s.text,
            language=s.language,
        )
        for s in batch.subtitles
    ]
    subtitle_metadata = SubtitleMetadata(
        active_source=batch.subtitle_metadata.active_source,
        reviewed=batch.subtitle_metadata.reviewed,
        needs_review=batch.subtitle_metadata.needs_review,
        asr_model=batch.subtitle_metadata.asr_model,
        calibration_model=batch.subtitle_metadata.calibration_model,
    )

    # ---- coarse context
    coarse_ctx_dict = batch.coarse_context or {}
    coarse_context = FineCoarseContext(
        source_batch_ids=list(coarse_ctx_dict.get("source_batch_ids") or []),
        evidence_frame_ids=list(coarse_ctx_dict.get("evidence_frame_ids") or []),
        coarse_summaries=_load_coarse_summaries(layout),
        coarse_refinement_reasons=_load_coarse_refinement_reasons(layout),
        unresolved_questions=_load_coarse_unresolved(layout),
    )

    # ---- 图片签名（contact sheet sha + detail frame sha）用于缓存键
    image_sig_payload = {
        "contact_sheet_shas": cs_shas,
        "detail_frame_shas": df_shas,
    }
    from video_analyst.util.hashing import sha256_json

    report.image_signature = sha256_json(image_sig_payload)

    # ---- analysis dimensions
    dims = list(analysis_dimensions or cfg.default_analysis_dimensions)

    request = FineVisionRequest(
        schema_version=cfg.fine_schema_version,
        fine_batch_id=batch.fine_batch_id,
        video_id=video_id,
        interval=batch.interval,
        contact_sheets=contact_sheets,
        detail_frames=detail_inputs,
        subtitles=subtitles,
        subtitle_metadata=subtitle_metadata,
        user_request=user_request or batch.user_request or "",
        refinement_questions=list(batch.refinement_questions),
        coarse_context=coarse_context,
        analysis_dimensions=dims,
    )
    return request, report


def build_all_fine_requests(
    layout: OutputLayout,
    *,
    cfg: FineAnalysisConfig,
    user_request: str,
    analysis_dimensions: list[str] | None = None,
    video_id: str = "video",
) -> list[tuple[FineVisionRequest, RequestBuildReport]]:
    """读 fine_batches.json，按时间顺序为每个 batch 构建请求。"""
    if not layout.refinement_fine_batches_json.is_file():
        log.warning(
            "fine_batches.json missing at %s; run Epic 3 first",
            layout.refinement_fine_batches_json,
        )
        return []
    try:
        data = json.loads(layout.refinement_fine_batches_json.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("could not read fine_batches.json: %s", exc)
        return []
    batches = data.get("batches") or []
    # 按 interval.start 排序
    batches_sorted = sorted(
        batches, key=lambda b: (b.get("interval", {}).get("start", 0.0), b.get("fine_batch_id", ""))
    )
    out: list[tuple[FineVisionRequest, RequestBuildReport]] = []
    for b in batches_sorted:
        try:
            batch = FineVisualBatch.model_validate(b)
        except Exception as exc:
            log.warning("skipping invalid fine batch: %s", exc)
            continue
        req, report = build_fine_request(
            batch,
            layout=layout,
            cfg=cfg,
            user_request=user_request,
            analysis_dimensions=analysis_dimensions,
            video_id=video_id,
        )
        out.append((req, report))
    return out


__all__ = [
    "RequestBuildReport",
    "build_all_fine_requests",
    "build_fine_request",
]
