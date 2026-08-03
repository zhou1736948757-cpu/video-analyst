"""Stage 3b — terminal refinement frame extraction (local only).

Reads analysis/fine/second_refinement_plan.json and extracts frames
into analysis/terminal_refinement/. Does NOT call vision models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.env.detect import ensure_ready_for_media
from video_analyst.exceptions import (
    TerminalRefinementPlanValidationError,
    VideoAnalystError,
)
from video_analyst.fine.terminal import validate_terminal_refinement_plan
from video_analyst.logging.setup import get_logger, setup_logging
from video_analyst.refinement.models import RefinementSource
from video_analyst.refinement.round import prepare_refinement_round
from video_analyst.util.hashing import sha256_file
from video_analyst.workflow.state import save_workflow_state, detect_workflow_status

log = get_logger("workflow.terminal_prepare")


def run_terminal_prepare(
    video_path: Path,
    output_dir: Path,
    *,
    config: AppConfig | None = None,
    user_request: str = "",
) -> dict[str, Any]:
    """Extract terminal refinement frames from second_refinement_plan.json."""
    cfg = config or DEFAULT_CONFIG
    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)
    setup_logging(layout.run_log)

    plan_path = layout.fine_second_refinement_plan_json
    if not plan_path.is_file():
        raise VideoAnalystError(
            f"second_refinement_plan.json missing at {plan_path}"
        )

    try:
        plan = validate_terminal_refinement_plan(plan_path)
    except TerminalRefinementPlanValidationError:
        # Allow not_required / empty plans without terminal flags
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

    intervals_raw = plan.get("intervals") or []
    if not intervals_raw:
        state = detect_workflow_status(output_dir)
        state.set_status(
            "ready_to_finalize",
            message="No terminal intervals; terminal not required",
        )
        save_workflow_state(output_dir, state)
        return {
            "status": "not_required",
            "message": "No terminal intervals in second_refinement_plan.json",
            "intervals": 0,
        }

    env = ensure_ready_for_media(input_path=video_path, output_dir=output_dir)
    try:
        source_sha = sha256_file(video_path, max_bytes=8 * 1024 * 1024)
    except Exception:
        source_sha = ""

    intervals: list[RefinementSource] = []
    questions: dict[str, list[str]] = {}
    for iv in intervals_raw:
        interval_id = str(iv.get("interval_id") or f"terminal_{len(intervals)+1:04d}")
        fps = int(iv.get("requested_fps") or iv.get("fps") or 15)
        if fps not in (5, 10, 15):
            fps = 15
        pri = str(iv.get("priority") or "high")
        if pri not in ("low", "medium", "high"):
            pri = "high"
        reasons = list(iv.get("reasons") or [])
        if not reasons and iv.get("reason"):
            reasons = [str(iv["reason"])]
        qs = list(iv.get("questions") or iv.get("refinement_questions") or [])
        intervals.append(
            RefinementSource(
                interval_id=interval_id,
                start=float(iv["start"]),
                end=float(iv["end"]),
                requested_fps=fps,  # type: ignore[arg-type]
                priority=pri,  # type: ignore[arg-type]
                reasons=reasons,
                questions=qs,
                source_batch_ids=list(iv.get("source_batch_ids") or []),
                evidence_frame_ids=list(iv.get("evidence_frame_ids") or []),
            )
        )
        questions[interval_id] = qs

    result = prepare_refinement_round(
        video_path=video_path,
        output_dir=output_dir,
        intervals=intervals,
        round_name="terminal",
        config=cfg.refinement,
        ffmpeg_env=env,
        source_video_sha=source_sha,
        questions_by_interval=questions,
        user_request=user_request,
    )

    state = detect_workflow_status(output_dir)
    state.set_status(
        "waiting_for_terminal_agent_analysis",
        message="Terminal preparation complete",
    )
    save_workflow_state(output_dir, state)

    return {
        "status": "waiting_for_terminal_agent_analysis",
        "intervals_processed": result.intervals_processed,
        "intervals_failed": result.intervals_failed,
        "total_frames": result.total_frames,
        "manifest_path": str(result.manifest_path),
        "fine_batches_path": str(result.fine_batches_path),
        "message": (
            "Refinement preparation complete.\n"
            "The current Agent must now inspect terminal contact sheets "
            "and write analysis/terminal_fine/."
        ),
    }


__all__ = ["run_terminal_prepare"]
