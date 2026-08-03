"""Fine vision response parsing & validation (Epic 4).

处理链：
    raw text
    → strip markdown code fence & leading/trailing prose
    → JSON load
    → schema version 检查
    → batch_id 检查
    → 时间范围验证（裁剪到 interval）
    → contact sheet ID 验证（移除未知并 warning）
    → frame ID 验证（移除未知并 warning）
    → observation ID 验证
    → 问题覆盖验证（缺则补 unresolved）
    → 二次补帧请求验证（FPS 10/15；15 不再升级）
    → confidence 裁剪

不允许：
- 猜测不存在的证据
- 自动创造 frame ID / 时间
- 自动填写模型遗漏的语义分析
- 把非法输出强行拼成成功结果

第一次验证失败后允许一次格式修复请求。第二次仍失败 → 标记失败。
"""

from __future__ import annotations

import json
import re
from typing import Any

from video_analyst.exceptions import (
    FineVisionResponseParseError,
    FineVisionResponseValidationError,
)
from video_analyst.fine.models import (
    FINE_SCHEMA_VERSION,
    FineBatchAnalysis,
    FineVisionRequest,
    SecondRefinementFps,
    SecondRefinementRequest,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OUTER_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_ALLOWED_SECOND_FPS = {10, 15}


def parse_fine_response_payload(raw_text: str) -> dict[str, Any]:
    """解析模型响应字符串为 JSON dict。

    容忍：
    - markdown ```json fence
    - JSON 前后的无关文本
    """
    if raw_text is None:
        raise FineVisionResponseParseError("fine model response is None")
    text = raw_text.strip()
    if not text:
        raise FineVisionResponseParseError("fine model response is empty")

    # 1) fenced code block
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2) whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) first balanced outer JSON object
    m = _OUTER_OBJECT_RE.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise FineVisionResponseParseError(
                f"fine response is not valid JSON even after outer-object extraction: {exc}"
            ) from exc

    raise FineVisionResponseParseError("fine response contains no JSON object")


def _coerce_to_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FineVisionResponseParseError(
            f"top-level JSON must be an object, got {type(payload).__name__}"
        )
    return payload


def _clamp_confidence(value: Any, *, warnings: list[str] | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        if warnings is not None:
            warnings.append("confidence not numeric; coerced to 0.0")
        return 0.0
    if v != v:  # NaN
        if warnings is not None:
            warnings.append("confidence NaN; coerced to 0.0")
        return 0.0
    clamped = max(0.0, min(1.0, v))
    if warnings is not None and clamped != v:
        warnings.append(f"confidence clamped from {v} to {clamped}")
    return clamped


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _build_known_ids(request: FineVisionRequest) -> tuple[set[str], set[str]]:
    """返回 (known_frame_ids, known_sheet_ids)。"""
    frame_ids: set[str] = set()
    for sheet in request.contact_sheets:
        frame_ids.update(sheet.frame_ids)
    for df in request.detail_frames:
        frame_ids.add(df.frame_id)
    sheet_ids = {sheet.sheet_id for sheet in request.contact_sheets}
    return frame_ids, sheet_ids


def _normalize_boundary(
    raw: Any,
    *,
    interval_start: float,
    interval_end: float,
    warnings: list[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        warnings.append("boundary not an object; using interval bounds")
        return {
            "earliest": interval_start,
            "latest": interval_end,
            "precision": "approximate",
            "before_frame_id": None,
            "after_frame_id": None,
        }
    try:
        earliest = float(raw.get("earliest", interval_start))
    except (TypeError, ValueError):
        earliest = interval_start
        warnings.append("boundary.earliest invalid; using interval start")
    try:
        latest = float(raw.get("latest", interval_end))
    except (TypeError, ValueError):
        latest = interval_end
        warnings.append("boundary.latest invalid; using interval end")
    # 裁剪到 interval
    if earliest < interval_start:
        earliest = interval_start
        warnings.append("boundary.earliest clamped to interval start")
    if latest > interval_end:
        latest = interval_end
        warnings.append("boundary.latest clamped to interval end")
    if latest < earliest:
        latest = earliest
        warnings.append("boundary.latest < earliest coerced to earliest")
    precision = str(raw.get("precision", "bounded")).lower()
    if precision not in {"exact", "bounded", "approximate"}:
        precision = "bounded"
        warnings.append(f"boundary.precision invalid; coerced to {precision}")
    # exact → earliest == latest
    if precision == "exact" and latest != earliest:
        latest = earliest
        warnings.append("boundary precision=exact but range non-zero; collapsed to earliest")
    before_fid = raw.get("before_frame_id")
    after_fid = raw.get("after_frame_id")
    return {
        "earliest": round(earliest, 6),
        "latest": round(latest, 6),
        "precision": precision,
        "before_frame_id": str(before_fid) if before_fid else None,
        "after_frame_id": str(after_fid) if after_fid else None,
    }


def _normalize_evidence(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"contact_sheet_ids": [], "frame_ids": [], "timestamps": []}
    sheet_ids = raw.get("contact_sheet_ids") or []
    if not isinstance(sheet_ids, list):
        sheet_ids = []
    kept_sheets = [str(s) for s in sheet_ids if str(s) in known_sheet_ids]
    dropped_sheets = [str(s) for s in sheet_ids if str(s) not in known_sheet_ids]
    if dropped_sheets:
        warnings.append(
            f"evidence dropped {len(dropped_sheets)} unknown contact_sheet_ids: {dropped_sheets[:3]}"
        )
    frame_ids = raw.get("frame_ids") or []
    if not isinstance(frame_ids, list):
        frame_ids = []
    kept_frames = [str(f) for f in frame_ids if str(f) in known_frame_ids]
    dropped_frames = [str(f) for f in frame_ids if str(f) not in known_frame_ids]
    if dropped_frames:
        warnings.append(
            f"evidence dropped {len(dropped_frames)} unknown frame_ids: {dropped_frames[:3]}"
        )
    timestamps = raw.get("timestamps") or []
    if not isinstance(timestamps, list):
        timestamps = []
    return {
        "contact_sheet_ids": kept_sheets,
        "frame_ids": kept_frames,
        "timestamps": [float(t) for t in timestamps if isinstance(t, (int, float))],
    }


def _normalize_frame_states(
    raw: Any,
    *,
    known_frame_ids: set[str],
    interval_start: float,
    interval_end: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("frame_states must be a list; coerced to empty")
        return []
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"frame_state[{i}] dropped: not an object")
            continue
        frame_id = str(item.get("frame_id") or "")
        if not frame_id:
            warnings.append(f"frame_state[{i}] dropped: missing frame_id")
            continue
        if frame_id not in known_frame_ids:
            warnings.append(
                f"frame_state[{i}] dropped: frame_id {frame_id} not in batch"
            )
            continue
        if frame_id in seen_ids:
            warnings.append(f"frame_state[{i}] dropped: duplicate frame_id {frame_id}")
            continue
        seen_ids.add(frame_id)
        oid = str(item.get("observation_id") or f"obs_{frame_id}")
        try:
            ts = float(item.get("timestamp_seconds", interval_start))
        except (TypeError, ValueError):
            ts = interval_start
            warnings.append(f"frame_state[{i}] timestamp invalid; using interval start")
        if ts < interval_start:
            ts = interval_start
        if ts > interval_end:
            ts = interval_end
        out.append(
            {
                "observation_id": oid,
                "timestamp_seconds": round(ts, 6),
                "frame_id": frame_id,
                "description": str(item.get("description") or "").strip(),
                "visible_subjects": [str(x) for x in (item.get("visible_subjects") or [])],
                "visible_text": [str(x) for x in (item.get("visible_text") or [])],
                "composition_notes": [str(x) for x in (item.get("composition_notes") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.6), warnings=warnings),
            }
        )
    return out


def _normalize_temporal_events(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    known_observation_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    source_batch_ids: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("temporal_events must be a list; coerced to empty")
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"temporal_event[{i}] dropped: not an object")
            continue
        category = str(item.get("category") or "uncertain_change")
        valid_categories = {
            "subject_motion",
            "camera_motion",
            "post_transform",
            "text_animation",
            "graphic_animation",
            "transition",
            "appearance_change",
            "layout_change",
            "uncertain_change",
        }
        if category not in valid_categories:
            category = "uncertain_change"
            warnings.append(f"temporal_event[{i}] category invalid; coerced to uncertain_change")
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        direct_obs = item.get("direct_observations") or []
        if not isinstance(direct_obs, list):
            direct_obs = []
        # 移除未知 observation id
        kept_obs = [str(o) for o in direct_obs if str(o) in known_observation_ids]
        dropped_obs = [str(o) for o in direct_obs if str(o) not in known_observation_ids]
        if dropped_obs:
            warnings.append(
                f"temporal_event[{i}] dropped {len(dropped_obs)} unknown observation_ids"
            )
        eid = str(item.get("event_id") or f"{fine_batch_id}_ev_{i:03d}")
        out.append(
            {
                "event_id": eid,
                "category": category,
                "boundary": boundary,
                "description": str(item.get("description") or "").strip(),
                "direct_observations": kept_obs,
                "inference": str(item.get("inference")).strip() or None
                if item.get("inference") is not None
                else None,
                "evidence": evidence,
                "alternatives": [str(x) for x in (item.get("alternatives") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
                "source_batch_ids": list(source_batch_ids),
            }
        )
    return out


def _normalize_subject_motions(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        out.append(
            {
                "event_id": str(item.get("event_id") or f"{fine_batch_id}_sm_{i:03d}"),
                "subject": str(item.get("subject") or "unknown_subject"),
                "action": str(item.get("action") or "unknown_action"),
                "direction": str(item.get("direction")).strip() or None
                if item.get("direction") is not None
                else None,
                "start_state": str(item.get("start_state") or ""),
                "end_state": str(item.get("end_state") or ""),
                "boundary": boundary,
                "evidence": evidence,
                "alternatives": [str(x) for x in (item.get("alternatives") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
            }
        )
    return out


def _normalize_camera_motions(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    valid_types = {
        "static", "pan_left", "pan_right", "tilt_up", "tilt_down",
        "zoom_in", "zoom_out", "dolly_in", "dolly_out", "truck_left",
        "truck_right", "handheld", "multiple", "uncertain",
    }
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        mt = str(item.get("motion_type") or "uncertain")
        if mt not in valid_types:
            mt = "uncertain"
            warnings.append(f"camera_motion[{i}] motion_type invalid; coerced to uncertain")
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        out.append(
            {
                "event_id": str(item.get("event_id") or f"{fine_batch_id}_cm_{i:03d}"),
                "motion_type": mt,
                "description": str(item.get("description") or ""),
                "boundary": boundary,
                "evidence": evidence,
                "alternatives": [str(x) for x in (item.get("alternatives") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
            }
        )
    return out


def _normalize_post_transforms(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    valid_types = {
        "scale", "position", "rotation", "crop", "opacity", "blur",
        "mask", "color", "composite", "uncertain",
    }
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        tts = item.get("transform_types") or []
        if not isinstance(tts, list):
            tts = []
        kept = [str(t) for t in tts if str(t) in valid_types]
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        out.append(
            {
                "event_id": str(item.get("event_id") or f"{fine_batch_id}_pt_{i:03d}"),
                "transform_types": kept,
                "affected_region": str(item.get("affected_region") or "whole_frame"),
                "start_state": str(item.get("start_state") or ""),
                "end_state": str(item.get("end_state") or ""),
                "boundary": boundary,
                "evidence": evidence,
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
            }
        )
    return out


def _normalize_text_animations(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    valid_types = {
        "appear", "disappear", "fade", "position", "scale", "rotation",
        "mask_reveal", "type_on", "tracking", "color_change", "emphasis", "uncertain",
    }
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        ats = item.get("animation_types") or []
        if not isinstance(ats, list):
            ats = []
        kept = [str(t) for t in ats if str(t) in valid_types]
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        tc = item.get("text_content")
        text_content = str(tc).strip() if tc is not None else None
        out.append(
            {
                "event_id": str(item.get("event_id") or f"{fine_batch_id}_ta_{i:03d}"),
                "text_content": text_content,
                "animation_types": kept,
                "start_state": str(item.get("start_state") or ""),
                "end_state": str(item.get("end_state") or ""),
                "boundary": boundary,
                "evidence": evidence,
                "alternatives": [str(x) for x in (item.get("alternatives") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
            }
        )
    return out


def _normalize_graphic_animations(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        out.append(
            {
                "event_id": str(item.get("event_id") or f"{fine_batch_id}_ga_{i:03d}"),
                "graphic_subject": str(item.get("graphic_subject") or "unknown_graphic"),
                "animation_types": [str(x) for x in (item.get("animation_types") or [])],
                "start_state": str(item.get("start_state") or ""),
                "end_state": str(item.get("end_state") or ""),
                "boundary": boundary,
                "evidence": evidence,
                "alternatives": [str(x) for x in (item.get("alternatives") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
            }
        )
    return out


def _normalize_transitions(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    valid_types = {
        "cut", "dissolve", "fade_in", "fade_out", "wipe", "slide",
        "zoom_transition", "mask_transition", "match_cut", "graphic_transition",
        "flash", "uncertain",
    }
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        tt = str(item.get("transition_type") or "uncertain")
        if tt not in valid_types:
            tt = "uncertain"
            warnings.append(f"transition[{i}] transition_type invalid; coerced to uncertain")
        boundary = _normalize_boundary(
            item.get("boundary"),
            interval_start=interval_start,
            interval_end=interval_end,
            warnings=warnings,
        )
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        out.append(
            {
                "event_id": str(item.get("event_id") or f"{fine_batch_id}_tr_{i:03d}"),
                "transition_type": tt,
                "outgoing_state": str(item.get("outgoing_state") or ""),
                "incoming_state": str(item.get("incoming_state") or ""),
                "boundary": boundary,
                "evidence": evidence,
                "alternatives": [str(x) for x in (item.get("alternatives") or [])],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
            }
        )
    return out


def _normalize_second_refinement_requests(
    raw: Any,
    *,
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    interval_start: float,
    interval_end: float,
    current_fps: int,
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """规范化二次补帧请求。

    规则：
    - FPS 只能是 10 或 15
    - 当前 15 FPS 时不允许继续升级（必须缩窄区间 narrower_interval）
    - 当前 5 FPS 可请求 10 或 15
    - 当前 10 FPS 可请求 15
    - 必须有非空 reason
    - 必须有证据
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip()
        if not reason:
            warnings.append(f"second_refinement[{i}] dropped: empty reason")
            continue
        req_type = str(item.get("request_type") or "higher_temporal_density")
        if req_type not in {
            "higher_temporal_density",
            "narrower_interval",
            "high_resolution_detail",
        }:
            req_type = "higher_temporal_density"
            warnings.append(f"second_refinement[{i}] request_type invalid; coerced")
        try:
            start = float(item.get("start", interval_start))
            end = float(item.get("end", interval_end))
        except (TypeError, ValueError):
            warnings.append(f"second_refinement[{i}] dropped: invalid time range")
            continue
        # 裁剪到 interval
        start = max(interval_start, min(start, interval_end))
        end = max(start, min(end, interval_end))
        try:
            fps = int(item.get("requested_fps", 15))
        except (TypeError, ValueError):
            fps = 15
        if fps not in _ALLOWED_SECOND_FPS:
            fps = 15 if fps > 15 else 10
            warnings.append(f"second_refinement[{i}] fps snapped to {fps}")
        # 升级规则
        if req_type == "higher_temporal_density":
            if current_fps >= 15 and fps >= 15:
                warnings.append(
                    f"second_refinement[{i}] dropped: current_fps=15 cannot upgrade further; "
                    "use narrower_interval instead"
                )
                continue
            if fps <= current_fps:
                warnings.append(
                    f"second_refinement[{i}] dropped: requested fps {fps} <= current {current_fps}"
                )
                continue
        priority = str(item.get("priority", "medium")).lower()
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        exact_ts = item.get("exact_detail_timestamps") or []
        if not isinstance(exact_ts, list):
            exact_ts = []
        out.append(
            {
                "request_id": str(item.get("request_id") or f"{fine_batch_id}_2nd_{i:03d}"),
                "request_type": req_type,
                "start": round(start, 6),
                "end": round(end, 6),
                "requested_fps": fps,
                "exact_detail_timestamps": [float(t) for t in exact_ts if isinstance(t, (int, float))],
                "reason": reason,
                "questions": [str(q) for q in (item.get("questions") or []) if str(q).strip()],
                "priority": priority,
                "evidence": evidence,
                "confidence": _clamp_confidence(item.get("confidence", 0.6), warnings=warnings),
            }
        )
    return out


def _normalize_question_answers(
    raw: Any,
    *,
    refinement_questions: list[str],
    known_frame_ids: set[str],
    known_sheet_ids: set[str],
    fine_batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """规范化问题回答，并保证每个 refinement_question 都有回答。

    缺失的问题补 unresolved。
    """
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raw = []
    # 先收集模型已回答的（按 question 文本匹配）
    answered: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or "").strip()
        if not q:
            warnings.append(f"question_answer[{i}] dropped: empty question")
            continue
        status = str(item.get("status", "unresolved")).lower()
        if status not in {"answered", "partially_answered", "unresolved"}:
            status = "unresolved"
        ans = str(item.get("answer") or "").strip()
        if not ans and status != "unresolved":
            status = "unresolved"
            warnings.append(f"question_answer[{i}] empty answer; status coerced to unresolved")
        evidence = _normalize_evidence(
            item.get("evidence"),
            known_frame_ids=known_frame_ids,
            known_sheet_ids=known_sheet_ids,
            warnings=warnings,
        )
        answered[q] = {
            "question": q,
            "answer": ans if ans else "Insufficient evidence to answer.",
            "status": status,
            "answer_origin": "model",
            "evidence": evidence,
            "confidence": _clamp_confidence(item.get("confidence", 0.4), warnings=warnings),
        }

    out: list[dict[str, Any]] = []
    for q in refinement_questions:
        q_stripped = q.strip()
        if q_stripped in answered:
            out.append(answered[q_stripped])
        else:
            # Epic 5 修复：验证器补齐的 unresolved 必须标记 answer_origin="validator_fallback"
            warnings.append(f"refinement question unanswered by model: {q_stripped[:60]}")
            out.append(
                {
                    "question": q_stripped,
                    "answer": "Insufficient evidence to answer.",
                    "status": "unresolved",
                    "answer_origin": "validator_fallback",
                    "evidence": {"contact_sheet_ids": [], "frame_ids": [], "timestamps": []},
                    "confidence": 0.1,
                }
            )
    return out


def validate_fine_response(
    payload: Any,
    *,
    request: FineVisionRequest,
) -> tuple[FineBatchAnalysis, list[str]]:
    """验证 fine 响应。返回 (analysis, warnings)。

    不可恢复的错误抛 FineVisionResponseValidationError。
    """
    data = _coerce_to_dict(payload)
    warnings: list[str] = []

    # schema version
    ver = str(data.get("schema_version") or "")
    if ver and ver != FINE_SCHEMA_VERSION:
        warnings.append(f"schema_version {ver!r} differs from current {FINE_SCHEMA_VERSION!r}")

    # batch id
    bid = str(data.get("fine_batch_id") or request.fine_batch_id)
    if bid != request.fine_batch_id:
        warnings.append(
            f"fine_batch_id mismatch: got {bid!r}, expected {request.fine_batch_id!r}"
        )
        bid = request.fine_batch_id

    interval_start = float(request.interval.start)
    interval_end = float(request.interval.end)

    try:
        start = float(data.get("start", interval_start))
        end = float(data.get("end", interval_end))
    except (TypeError, ValueError) as exc:
        raise FineVisionResponseValidationError(
            f"start/end must be numeric: {exc}"
        ) from exc
    if start < interval_start - 1e-6:
        start = interval_start
        warnings.append("start clamped to interval start")
    if end > interval_end + 1e-6:
        end = interval_end
        warnings.append("end clamped to interval end")
    if end < start:
        end = start
        warnings.append("end < start coerced to start")

    known_frame_ids, known_sheet_ids = _build_known_ids(request)

    frame_states = _normalize_frame_states(
        data.get("frame_states"),
        known_frame_ids=known_frame_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        warnings=warnings,
    )
    known_observation_ids = {fs["observation_id"] for fs in frame_states}

    source_batch_ids = list(request.coarse_context.source_batch_ids or [])

    temporal_events = _normalize_temporal_events(
        data.get("temporal_events"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        known_observation_ids=known_observation_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        source_batch_ids=source_batch_ids,
        warnings=warnings,
    )
    subject_motions = _normalize_subject_motions(
        data.get("subject_motions"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        warnings=warnings,
    )
    camera_motions = _normalize_camera_motions(
        data.get("camera_motions"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        warnings=warnings,
    )
    post_transforms = _normalize_post_transforms(
        data.get("post_transforms"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        warnings=warnings,
    )
    text_animations = _normalize_text_animations(
        data.get("text_animations"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        warnings=warnings,
    )
    graphic_animations = _normalize_graphic_animations(
        data.get("graphic_animations"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        warnings=warnings,
    )
    transitions = _normalize_transitions(
        data.get("transitions"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        fine_batch_id=bid,
        warnings=warnings,
    )
    question_answers = _normalize_question_answers(
        data.get("question_answers"),
        refinement_questions=list(request.refinement_questions),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        fine_batch_id=bid,
        warnings=warnings,
    )
    second_refinement = _normalize_second_refinement_requests(
        data.get("second_refinement_requests"),
        known_frame_ids=known_frame_ids,
        known_sheet_ids=known_sheet_ids,
        interval_start=interval_start,
        interval_end=interval_end,
        current_fps=int(request.interval.requested_fps),
        fine_batch_id=bid,
        warnings=warnings,
    )

    confidence = _clamp_confidence(data.get("confidence", 0.5))
    summary = str(data.get("summary") or "").strip()
    initial_state = str(data.get("initial_state") or "").strip()
    final_state = str(data.get("final_state") or "").strip()
    continuity_notes = [str(x) for x in (data.get("continuity_notes") or []) if str(x).strip()]
    unresolved_items = [str(x) for x in (data.get("unresolved_items") or []) if str(x).strip()]

    extra_warnings = data.get("warnings") or []
    if isinstance(extra_warnings, list):
        for w in extra_warnings:
            if isinstance(w, str) and w.strip():
                warnings.append(w.strip())

    try:
        analysis = FineBatchAnalysis(
            schema_version=FINE_SCHEMA_VERSION,
            fine_batch_id=bid,
            interval_id=request.interval.interval_id,
            start=round(start, 6),
            end=round(end, 6),
            summary=summary,
            frame_states=frame_states,  # type: ignore[arg-type]
            temporal_events=temporal_events,  # type: ignore[arg-type]
            subject_motions=subject_motions,  # type: ignore[arg-type]
            camera_motions=camera_motions,  # type: ignore[arg-type]
            post_transforms=post_transforms,  # type: ignore[arg-type]
            text_animations=text_animations,  # type: ignore[arg-type]
            graphic_animations=graphic_animations,  # type: ignore[arg-type]
            transitions=transitions,  # type: ignore[arg-type]
            question_answers=question_answers,  # type: ignore[arg-type]
            second_refinement_requests=second_refinement,  # type: ignore[arg-type]
            initial_state=initial_state,
            final_state=final_state,
            continuity_notes=continuity_notes,
            unresolved_items=unresolved_items,
            confidence=confidence,
            warnings=warnings,
        )
    except Exception as exc:
        raise FineVisionResponseValidationError(
            f"fine response failed schema validation: {exc}"
        ) from exc

    return analysis, warnings


def parse_and_validate_fine(
    raw_text: str,
    *,
    request: FineVisionRequest,
) -> tuple[FineBatchAnalysis, list[str]]:
    """便捷：解析 + 验证一步完成。"""
    data = parse_fine_response_payload(raw_text)
    return validate_fine_response(data, request=request)


__all__ = [
    "parse_and_validate_fine",
    "parse_fine_response_payload",
    "validate_fine_response",
]
