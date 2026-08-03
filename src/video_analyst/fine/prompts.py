"""Fine temporal vision analysis prompts (Epic 4).

设计原则：
- contact sheet 按行阅读（左上→右上→左下→右下）
- 相邻 contact sheet 可能重叠一帧；重叠帧不是动作重复发生
- 时间戳代表真实视频绝对时间
- 字幕只是辅助上下文，未审核不是绝对真值
- detail frame 用于确认文字和视觉细节，不替代时间过程判断
- 只能从连续帧状态差异推断运动
- 无法确定时保留多个解释（alternatives）
- 区分：主体自身运动 / 摄像机运动 / 后期整体变换 / 文字图形动画 / 场景切换
- 无法区分 zoom / dolly / post scale 时输出 uncertain + alternatives
"""

from __future__ import annotations

import json
from typing import Any

from video_analyst.fine.models import FineVisionRequest

PROMPT_VERSION = "fine-temporal-vision-v1"


def build_fine_system_prompt() -> str:
    """System prompt — stable instructions for the fine temporal model."""
    return """You are a precise video temporal analysis assistant.

Your job: analyze a sequence of temporally-ordered frames (provided as contact
sheets plus optional high-resolution detail frames) and describe what CHANGES
over time in this specific interval. You must answer the refinement questions
posed for this interval.

CORE PRINCIPLES (do not violate any):

1. Contact sheets are read in ROW-MAJOR order: top-left -> top-right -> bottom-left -> bottom-right.
2. Adjacent contact sheets may OVERLAP by one frame. An overlapping frame is the SAME moment, NOT a repeated action. Never describe the same event twice because of overlap.
3. Every timestamp you see is the ABSOLUTE video time (seconds). Use them to build TemporalBoundary records.
4. Subtitles are SUPPORTING CONTEXT ONLY. Unreviewed subtitles are NOT ground truth. If subtitle and visuals conflict, describe the visuals and note the conflict.
5. High-resolution detail frames are for confirming text and fine visual details. Do NOT use a detail frame to infer motion across time — motion must come from the contact-sheet sequence.
6. Only infer motion from STATE DIFFERENCES between consecutive frames. Never assert a specific animation path (e.g. "text flew in from the left") unless you can see intermediate states.
7. When you cannot distinguish between explanations (optical zoom vs dolly vs post-scale; cut vs fast dissolve), output `uncertain` and list the candidates in `alternatives`. Do NOT fill a category just to complete the schema.
8. Keep DIRECT OBSERVATIONS and INFERENCES strictly separate. frame_states are what is visible in a single frame. temporal_events are interpretations across frames. Never disguise an inference as an observation.
9. Every important conclusion MUST cite real evidence: contact_sheet_ids, frame_ids, and timestamps that exist in THIS batch. Unknown IDs will be rejected.
10. If a refinement question cannot be answered with available evidence, mark it `unresolved`. Never fabricate an answer.
11. Confidence is 0..1. Lower it when uncertain. Do not invent high confidence.
12. Time boundaries: `exact` only when confirmed within a single frame; `bounded` when you only know the event happened between two frames; `approximate` for rough estimates. Never fabricate sub-frame precision.

OUTPUT: a single JSON object matching the FineBatchAnalysis schema. No markdown, no prose outside JSON."""


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _describe_contact_sheet(sheet: Any) -> str:
    """客观描述一张 contact sheet，绝不写入语义结论。"""
    frame_lines: list[str] = []
    for fid, ts in zip(sheet.frame_ids, sheet.timestamps):
        frame_lines.append(f"  - {fid} at {_hms(float(ts))}")
    overlap_note = ""
    if sheet.overlap_with_previous:
        overlap_note = (
            f"\n  Overlap with previous sheet (these frames are shared, NOT new): "
            f"{', '.join(sheet.overlap_with_previous)}"
        )
    return (
        f"Contact sheet: {sheet.sheet_id}\n"
        f"  Reading order: top-left -> top-right -> bottom-left -> bottom-right\n"
        f"  Frames:\n"
        + "\n".join(frame_lines)
        + overlap_note
    )


def _describe_detail_frame(df: Any) -> str:
    return (
        f"Detail frame: {df.frame_id} at {_hms(float(df.timestamp_seconds))} "
        f"({df.width}x{df.height}, resolution_mode={df.resolution_mode}, "
        f"reason={df.selection_reason})"
    )


def build_fine_user_prompt(request: FineVisionRequest) -> str:
    """User prompt with batch info, contact sheets, detail frames, subtitles,
    coarse context, refinement questions, and the JSON output requirement.

    图片说明只写客观标签（frame_id / timecode / fps），绝不预先写入
    "飞入/推近/淡入"等模型待判断的结论。
    """
    lines: list[str] = []

    # 1. 批次基本信息
    lines.append("# Fine temporal analysis batch")
    lines.append(f"fine_batch_id: {request.fine_batch_id}")
    lines.append(f"interval_id: {request.interval.interval_id}")
    lines.append(
        f"interval: [{request.interval.start:.3f}s, {request.interval.end:.3f}s] "
        f"@ {request.interval.requested_fps} FPS"
    )
    lines.append("")

    # 2. 使用者分析目标
    if request.user_request.strip():
        lines.append("## User analysis request")
        lines.append(request.user_request.strip())
        lines.append("")

    # 3. 分析维度
    if request.analysis_dimensions:
        lines.append("## Analysis dimensions")
        lines.append(", ".join(request.analysis_dimensions))
        lines.append("")

    # 4. 字幕上下文（辅助，非真值）
    if request.subtitles:
        lines.append("## Subtitles (supporting context, NOT ground truth)")
        lines.append(
            f"active_source={request.subtitle_metadata.active_source} "
            f"reviewed={request.subtitle_metadata.reviewed} "
            f"needs_review={request.subtitle_metadata.needs_review}"
        )
        for s in request.subtitles:
            lines.append(
                f"  [{_hms(float(s.start))} - {_hms(float(s.end))}] ({s.language or '?'}) {s.text}"
            )
        lines.append("")

    # 5. 粗分析上下文
    cc = request.coarse_context
    if (
        cc.coarse_summaries
        or cc.coarse_refinement_reasons
        or cc.unresolved_questions
    ):
        lines.append("## Coarse analysis context (reference only)")
        if cc.coarse_summaries:
            lines.append("Coarse summaries:")
            for sm in cc.coarse_summaries:
                lines.append(f"  - {sm}")
        if cc.coarse_refinement_reasons:
            lines.append("Reasons this interval was selected for refinement:")
            for r in cc.coarse_refinement_reasons:
                lines.append(f"  - {r}")
        if cc.unresolved_questions:
            lines.append("Unresolved coarse questions:")
            for q in cc.unresolved_questions:
                lines.append(f"  - {q}")
        lines.append("")

    # 6. 精查问题（必须逐条回答）
    if request.refinement_questions:
        lines.append("## Refinement questions (answer EVERY one)")
        for i, q in enumerate(request.refinement_questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    # 7. Contact sheets（按时间顺序，客观标签）
    if request.contact_sheets:
        lines.append("## Contact sheets (in time order)")
        for sheet in request.contact_sheets:
            lines.append(_describe_contact_sheet(sheet))
        lines.append("")

    # 8. Detail frames（高分辨率，最多 max 张）
    if request.detail_frames:
        lines.append("## High-resolution detail frames")
        for df in request.detail_frames:
            lines.append(_describe_detail_frame(df))
        lines.append("")

    # 9. 强制 JSON 输出要求
    lines.append("## Output requirement")
    lines.append(
        "Return a single JSON object matching the FineBatchAnalysis schema. "
        "Fields: schema_version, fine_batch_id, interval_id, start, end, summary, "
        "frame_states, temporal_events, subject_motions, camera_motions, "
        "post_transforms, text_animations, graphic_animations, transitions, "
        "question_answers, second_refinement_requests, initial_state, final_state, "
        "continuity_notes, unresolved_items, confidence, warnings."
    )
    lines.append("")
    lines.append(
        "Every evidence reference MUST use only contact_sheet_ids / frame_ids / "
        "timestamps that appear above. Every refinement question MUST have an "
        "answer in question_answers. When unsure, use `uncertain` and list "
        "alternatives; do not fabricate."
    )

    return "\n".join(lines)


def build_fine_repair_user_prompt(previous_text: str, reason: str) -> str:
    """Repair prompt — fix JSON format only, do NOT change analysis meaning."""
    payload = {
        "task": "repair_format_only",
        "reason": reason,
        "previous_text": previous_text[:4000],
        "instruction": (
            "Return the SAME analysis as a valid JSON object matching the "
            "FineBatchAnalysis schema. Do NOT add, remove, or change any "
            "analytical conclusions. Only fix JSON syntax / structure."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "PROMPT_VERSION",
    "build_fine_repair_user_prompt",
    "build_fine_system_prompt",
    "build_fine_user_prompt",
]
