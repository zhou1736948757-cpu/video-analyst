"""Prompt construction for vision coarse analysis.

A single source of truth so both Mock provider tests and real providers stay
aligned with the same instructions to the model.
"""

from __future__ import annotations

import json
from typing import Any

from video_analyst.vision.models import (
    SCHEMA_VERSION,
    SubtitleMetadata,
    VisionBatchRequest,
)

PROMPT_VERSION = "coarse-vision-v1"

# Default analysis dimensions asked of the model if the user doesn't override.
DEFAULT_ANALYSIS_DIMENSIONS: tuple[str, ...] = (
    "structure",
    "subjects",
    "camera",
    "motion",
    "text",
    "transition",
    "refinement_needs",
)


def build_system_prompt() -> str:
    """System-level instructions shared across every batch."""
    return (
        "You are the visual-analysis component of a Video Analyst pipeline. "
        "Your job is to produce structured JSON observations about short "
        "video batches sampled at 1 FPS.\n"
        "Strict rules you MUST follow:\n"
        "1. Output STRICT JSON (no markdown, no commentary outside JSON).\n"
        "2. Top-level keys MUST match the requested schema exactly.\n"
        "3. Every Observation or RefinementRequest MUST include time ranges "
        "and evidence frame_ids taken from the provided frames only.\n"
        "4. NEVER fabricate animation. If two adjacent 1 FPS frames show "
        "different states, describe the difference as a state change; do NOT "
        "infer motion paths, easing, or timing that you cannot verify.\n"
        "5. Distinguish DIRECT OBSERVATIONS (things you can see in the frames) "
        "from INFERRED INTERPRETATIONS (things you guess). "
        "Place each in the correct field. Never mix them.\n"
        "6. Subtitles are SUPPORTING CONTEXT only — they are not absolute "
        "truth. If subtitles conflict with what you see, describe what you "
        "see and note the conflict in `subtitle_visual_relationship`.\n"
        "7. When a time range cannot be reliably characterized at 1 FPS "
        "(e.g. fast text animations, motion blur, transitions), emit a "
        "`refinement_request` and explain WHY higher FPS is needed.\n"
        "8. If you cannot see anything meaningful (e.g. all frames blank), "
        "still emit the schema with empty arrays and explain in "
        "`warnings`. Do not invent content.\n"
        "9. `confidence` MUST be in [0.0, 1.0]. Lower it when evidence is thin.\n"
        "10. `requested_fps` MUST be one of 5, 10, 15.\n"
    )


def build_user_prompt(
    request: VisionBatchRequest,
    omitted_frame_ids: list[str] | None = None,
) -> str:
    """User-level prompt that frames the specific batch."""
    meta = request.subtitle_metadata
    meta_block = _subtitle_metadata_block(meta)
    prev_block = _previous_context_block(request)
    frames_block = _frames_block(request.frames)
    dimensions = (
        list(request.analysis_dimensions)
        if request.analysis_dimensions
        else list(DEFAULT_ANALYSIS_DIMENSIONS)
    )
    omitted_note = ""
    if omitted_frame_ids:
        omitted_note = (
            "\nNote: "
            f"{len(omitted_frame_ids)} frame(s) were capped from this batch to "
            "respect the per-batch frame budget; they are listed below for "
            "context but the model MUST rely on `selected_frame_ids`:\n"
            + ", ".join(omitted_frame_ids)
            + "\n"
        )

    payload: dict[str, Any] = {
        "task": (
            "Analyze the visual content of this short batch. Produce JSON "
            "matching the schema below. Each frame is a single JPG captured "
            "at 1 FPS (or a scene-change protective frame, marked). The "
            "subtitles below are SUPPORTING CONTEXT ONLY and MUST NOT be "
            "treated as ground truth."
        ),
        "schema": {
            "schema_version": "1.0",
            "batch_id": request.batch_id,
            "start": request.batch_start,
            "end": request.batch_end,
            "summary": "string",
            "direct_observations": "Observation[]",
            "inferred_interpretations": "Inference[]",
            "visible_text": "Observation[]",
            "subjects": "Observation[]",
            "scene_changes": "Observation[]",
            "camera_observations": "Observation[]",
            "motion_observations": "Observation[]",
            "possible_transitions": "Observation[]",
            "subtitle_visual_relationship": "Observation[]",
            "important_moments": "Observation[]",
            "refinement_requests": "RefinementRequest[]",
            "confidence": "float in [0,1]",
            "warnings": "string[]",
        },
        "batch_info": {
            "batch_id": request.batch_id,
            "video_id": request.video_id,
            "start": request.batch_start,
            "end": request.batch_end,
            "frame_count": len(request.frames),
        },
        "user_request": request.user_request or "(none)",
        "analysis_dimensions": dimensions,
        "subtitle_metadata": meta_block,
        "subtitles": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "language": s.language,
            }
            for s in request.subtitles
        ],
        "frames": frames_block,
        "previous_context": prev_block,
        "rules": [
            "Output STRICT JSON that matches the schema above.",
            "Never include text outside the JSON object.",
            "frame_ids MUST reference one of: " + ", ".join(f.frame_id for f in request.frames),
            "Every time_range MUST lie inside this batch ("
            f"start={request.batch_start}, end={request.batch_end}).",
            "Never fabricate motion. State differences are observations; "
            "the animation between them is unknown at 1 FPS.",
            "If in doubt, prefer a lower confidence or an empty list.",
        ],
        "reminder": (
            "Subtitles are only context. If a subtitle disagrees with what "
            "you can see in the frames, describe the visual and add a "
            "note in `subtitle_visual_relationship`."
        ),
    }
    if omitted_note:
        payload["omitted_frame_ids"] = omitted_frame_ids
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_repair_user_prompt(raw_text: str, reason: str) -> str:
    """Single deterministic repair prompt after a recoverable parse failure."""
    return json.dumps(
        {
            "task": (
                "Your previous response could not be parsed. "
                "Return STRICT JSON (only the JSON object, no markdown, "
                "no commentary) that fixes the issue described below. "
                "Do NOT change the semantic content."
            ),
            "previous_reason": reason,
            "previous_text": raw_text[:8000],
            "expected_schema": "see original request",
        },
        ensure_ascii=False,
        indent=2,
    )


# ---- Helpers ----


def _subtitle_metadata_block(meta: SubtitleMetadata) -> dict[str, Any]:
    return {
        "active_source": meta.active_source,
        "reviewed": meta.reviewed,
        "needs_review": meta.needs_review,
        "asr_model": meta.asr_model,
        "calibration_model": meta.calibration_model,
        "treat_subtitles_as_supporting_context": True,
        "subtitle_is_not_absolute_ground_truth": True,
    }


def _previous_context_block(request: VisionBatchRequest) -> dict[str, Any] | None:
    pc = request.previous_context
    if pc is None:
        return None
    return {
        "previous_batch_id": pc.previous_batch_id,
        "previous_summary": pc.previous_summary,
        "last_known_subjects": list(pc.last_known_subjects),
        "last_known_scene": pc.last_known_scene,
        "unresolved_questions": list(pc.unresolved_questions),
    }


def _frames_block(frames) -> list[dict[str, Any]]:
    out = []
    for f in frames:
        out.append(
            {
                "frame_id": f.frame_id,
                "timestamp_seconds": f.timestamp_seconds,
                "timecode": f.timecode,
                "sources": list(f.sources),
                "scene_score": f.scene_score,
                # Use the path string only; the actual image bytes are attached
                # by the provider when calling the API.
                "filename": f.path.name,
            }
        )
    return out


__all__ = [
    "DEFAULT_ANALYSIS_DIMENSIONS",
    "PROMPT_VERSION",
    "build_repair_user_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "SCHEMA_VERSION",  # re-export so callers can build a single import
]
