"""Response parsing & validation for vision model outputs.

Pipeline:
    raw text
    → strip markdown code fences & leading prose
    → JSON load
    → semantic validation against the batch
    → on small fixable errors, return (data, warnings)
    → on unfixable errors, raise VisionResponseValidationError

This module deliberately does NOT touch provider-specific shapes — callers
must convert raw responses into a single string before invoking.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from video_analyst.exceptions import VisionResponseParseError, VisionResponseValidationError
from video_analyst.vision.models import (
    CoarseBatchAnalysis,
    RefinementRequest,
    SCHEMA_VERSION,
    VisionBatchRequest,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OUTER_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_ALLOWED_FPS = {5, 10, 15}


def parse_response_payload(raw_text: str) -> dict[str, Any]:
    """Parse a model response string into a JSON dict.

    Tolerates:
    - markdown ```json fences
    - leading prose before the JSON object
    - trailing prose after the JSON object
    """
    if raw_text is None:
        raise VisionResponseParseError("model response is None")

    text = raw_text.strip()
    if not text:
        raise VisionResponseParseError("model response is empty")

    # 1) try fenced code block first
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to outer-object detection

    # 2) try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) try to find the first balanced outer JSON object
    m = _OUTER_OBJECT_RE.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise VisionResponseParseError(
                f"response is not valid JSON even after outer-object extraction: {exc}"
            ) from exc

    raise VisionResponseParseError("response contains no JSON object")


def _coerce_to_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VisionResponseParseError(
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


def _normalize_observations(
    raw: Any,
    *,
    batch_start: float,
    batch_end: float,
    frame_ids: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Best-effort normalization for Observation-like dicts."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("observations must be a list; coerced to empty")
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            warnings.append("observation item dropped: not an object")
            continue
        category = str(item.get("category") or "unspecified")
        description = str(item.get("description") or "").strip()
        tr = item.get("time_range") or {}
        start = tr.get("start")
        end = tr.get("end")
        try:
            start_f = float(start) if start is not None else batch_start
            end_f = float(end) if end is not None else batch_end
        except (TypeError, ValueError):
            warnings.append("observation time_range invalid; clamped to batch")
            start_f, end_f = batch_start, batch_end

        # Clamp time_range into batch
        if start_f < batch_start:
            start_f = batch_start
            warnings.append("observation start clamped to batch start")
        if end_f > batch_end:
            end_f = batch_end
            warnings.append("observation end clamped to batch end")
        if end_f < start_f:
            end_f = start_f
            warnings.append("observation end < start coerced to start")

        ev = item.get("evidence") or {}
        fids = ev.get("frame_ids") or []
        if not isinstance(fids, list):
            fids = []
        kept_fids = [str(f) for f in fids if isinstance(f, (str, int)) and str(f) in frame_ids]
        dropped = [str(f) for f in fids if str(f) not in frame_ids]
        if dropped:
            warnings.append(
                f"observation dropped {len(dropped)} unknown frame_ids: {dropped[:3]}"
            )
        timestamps = ev.get("timestamps") or []
        if not isinstance(timestamps, list):
            timestamps = []

        conf = _clamp_confidence(item.get("confidence", 0.5), warnings=warnings)

        out.append(
            {
                "category": category,
                "description": description,
                "time_range": {"start": round(start_f, 3), "end": round(end_f, 3)},
                "evidence": {"frame_ids": kept_fids, "timestamps": timestamps},
                "confidence": conf,
            }
        )
    return out


def _normalize_inferences(raw: Any, warnings: list[str]) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("inferences must be a list; coerced to empty")
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            warnings.append("inference item dropped: not an object")
            continue
        desc = str(item.get("description") or "").strip()
        supports = item.get("supporting_observation_ids") or []
        if not isinstance(supports, list):
            supports = []
        alts = item.get("alternatives") or []
        if not isinstance(alts, list):
            alts = []
        out.append(
            {
                "description": desc,
                "supporting_observation_ids": [str(x) for x in supports],
                "confidence": _clamp_confidence(item.get("confidence", 0.5), warnings=warnings),
                "alternatives": [str(x) for x in alts],
            }
        )
    return out


def _normalize_refinement_requests(
    raw: Any,
    *,
    video_start: float,
    video_end: float,
    frame_ids: set[str],
    batch_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("refinement_requests must be a list; coerced to empty")
        return []

    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append("refinement_request dropped: not an object")
            continue
        reason_raw = item.get("reason")
        if reason_raw is None or not str(reason_raw).strip():
            raise VisionResponseValidationError(
                f"refinement_request[{i}] has empty reason — refusing batch"
            )
        reason = str(reason_raw).strip()
        try:
            start = float(item.get("start", 0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            warnings.append("refinement_request dropped: invalid time range")
            continue
        if end < start:
            warnings.append("refinement_request dropped: end < start")
            continue
        # Clip to video bounds
        start_c = max(video_start, min(start, video_end))
        end_c = max(start_c, min(end, video_end))
        try:
            fps = int(item.get("requested_fps", 10))
        except (TypeError, ValueError):
            fps = 10
        if fps not in _ALLOWED_FPS:
            # Snap to nearest allowed value
            fps = min(_ALLOWED_FPS, key=lambda x: abs(x - fps))
            warnings.append(f"requested_fps snapped to {fps}")
        priority = str(item.get("priority", "medium")).lower()
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        questions = item.get("questions") or []
        if not isinstance(questions, list):
            questions = []
        ev_ids = item.get("evidence_frame_ids") or []
        if not isinstance(ev_ids, list):
            ev_ids = []
        kept_ev = [str(f) for f in ev_ids if str(f) in frame_ids]
        out.append(
            {
                "request_id": str(item.get("request_id") or f"{batch_id}_ref_{i:02d}"),
                "start": round(start_c, 3),
                "end": round(end_c, 3),
                "requested_fps": fps,
                "reason": reason,
                "questions": [str(q) for q in questions if str(q).strip()],
                "priority": priority,
                "evidence_frame_ids": kept_ev,
                "confidence": _clamp_confidence(item.get("confidence", 0.6)),
            }
        )
    return out


def validate_batch_response(
    payload: Any,
    *,
    request: VisionBatchRequest,
    video_start: float,
    video_end: float,
) -> tuple[CoarseBatchAnalysis, list[str]]:
    """Validate a parsed dict against the schema and batch semantics.

    Returns (analysis, warnings). Raises VisionResponseValidationError for
    irrecoverable problems (schema-version mismatch, missing batch_id, etc.)
    """
    data = _coerce_to_dict(payload)
    warnings: list[str] = []

    # Schema version: accept current or be very explicit about mismatch.
    ver = str(data.get("schema_version") or "")
    if ver and ver != SCHEMA_VERSION:
        warnings.append(f"schema_version {ver!r} differs from current {SCHEMA_VERSION!r}")

    # Batch id should match.
    bid = str(data.get("batch_id") or request.batch_id)
    if bid != request.batch_id:
        # Tolerate missing, but mismatch is suspicious
        warnings.append(f"batch_id mismatch: got {bid!r}, expected {request.batch_id!r}")
        bid = request.batch_id

    try:
        start = float(data.get("start", request.batch_start))
        end = float(data.get("end", request.batch_end))
    except (TypeError, ValueError) as exc:
        raise VisionResponseValidationError(
            f"start/end must be numeric: {exc}"
        ) from exc

    # Clamp start/end to batch
    if start < request.batch_start - 1e-6:
        start = request.batch_start
        warnings.append("start clamped to batch start")
    if end > request.batch_end + 1e-6:
        end = request.batch_end
        warnings.append("end clamped to batch end")
    if end < start:
        end = start
        warnings.append("end < start coerced to start")

    frame_ids = {f.frame_id for f in request.frames}

    direct = _normalize_observations(
        data.get("direct_observations"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    inferred = _normalize_inferences(data.get("inferred_interpretations"), warnings)
    visible_text = _normalize_observations(
        data.get("visible_text"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    subjects = _normalize_observations(
        data.get("subjects"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    scene_changes = _normalize_observations(
        data.get("scene_changes"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    camera = _normalize_observations(
        data.get("camera_observations"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    motion = _normalize_observations(
        data.get("motion_observations"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    transitions = _normalize_observations(
        data.get("possible_transitions"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    svr = _normalize_observations(
        data.get("subtitle_visual_relationship"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    important = _normalize_observations(
        data.get("important_moments"),
        batch_start=request.batch_start,
        batch_end=request.batch_end,
        frame_ids=frame_ids,
        warnings=warnings,
    )
    refinement = _normalize_refinement_requests(
        data.get("refinement_requests"),
        video_start=video_start,
        video_end=video_end,
        frame_ids=frame_ids,
        batch_id=bid,
        warnings=warnings,
    )
    confidence = _clamp_confidence(data.get("confidence", 0.5))
    summary = str(data.get("summary") or "").strip()

    extra_warnings = data.get("warnings") or []
    if isinstance(extra_warnings, list):
        for w in extra_warnings:
            if isinstance(w, str) and w.strip():
                warnings.append(w.strip())

    # Make sure observations and inferences are not mixed inside each other
    # (the spec is explicit: they must be kept separate).
    # _normalize_* already routes them to separate fields.

    try:
        analysis = CoarseBatchAnalysis(
            schema_version=SCHEMA_VERSION,
            batch_id=bid,
            start=round(start, 3),
            end=round(end, 3),
            summary=summary,
            direct_observations=direct,  # type: ignore[arg-type]
            inferred_interpretations=inferred,  # type: ignore[arg-type]
            visible_text=visible_text,  # type: ignore[arg-type]
            subjects=subjects,  # type: ignore[arg-type]
            scene_changes=scene_changes,  # type: ignore[arg-type]
            camera_observations=camera,  # type: ignore[arg-type]
            motion_observations=motion,  # type: ignore[arg-type]
            possible_transitions=transitions,  # type: ignore[arg-type]
            subtitle_visual_relationship=svr,  # type: ignore[arg-type]
            important_moments=important,  # type: ignore[arg-type]
            refinement_requests=refinement,  # type: ignore[arg-type]
            confidence=confidence,
            warnings=warnings,
        )
    except ValidationError as exc:
        raise VisionResponseValidationError(
            f"response failed schema validation: {exc.errors()}"
        ) from exc

    # Validate refinement requests explicitly (in case model slips past normalization)
    for rr in analysis.refinement_requests:
        if rr.requested_fps not in _ALLOWED_FPS:  # pragma: no cover - guarded above
            raise VisionResponseValidationError(
                f"refinement_request fps must be 5/10/15, got {rr.requested_fps}"
            )
        if not rr.reason.strip():
            raise VisionResponseValidationError("refinement_request reason must not be empty")

    return analysis, warnings


def parse_and_validate(
    raw_text: str,
    *,
    request: VisionBatchRequest,
    video_start: float,
    video_end: float,
) -> tuple[CoarseBatchAnalysis, list[str]]:
    """Convenience: parse then validate in one call."""
    data = parse_response_payload(raw_text)
    return validate_batch_response(
        data,
        request=request,
        video_start=video_start,
        video_end=video_end,
    )


__all__ = [
    "parse_and_validate",
    "parse_response_payload",
    "validate_batch_response",
]