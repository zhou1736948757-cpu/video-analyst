"""`video-analyst run` — advance local stages only; pause for agent vision.

Never calls mock or external vision APIs to fill analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.logging.setup import get_logger, setup_logging
from video_analyst.pipeline.epic3 import run_epic3
from video_analyst.pipeline.video_precheck import validate_video_input
from video_analyst.refinement.round import prepare_refinement_round
from video_analyst.workflow.finalize import run_finalize
from video_analyst.workflow.prepare import run_prepare
from video_analyst.workflow.state import (
    detect_workflow_status,
    format_status_report,
    save_workflow_state,
)
from video_analyst.workflow.validate_agent import (
    validate_coarse,
    validate_fine,
    validate_terminal,
)

log = get_logger("workflow.run")


def run_until_agent_needed(
    video_path: Path,
    output_dir: Path,
    *,
    user_request: str = "",
    config: AppConfig | None = None,
    language: str | None = None,
    resume: bool = True,
    allow_partial: bool = True,
) -> dict[str, Any]:
    """Execute local stages until an agent vision step is required."""
    cfg = config or DEFAULT_CONFIG
    resolved = validate_video_input(video_path)
    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)
    setup_logging(layout.run_log)

    state = detect_workflow_status(output_dir)
    actions: list[str] = []

    # Stage 1
    if state.status in ("not_started", "preparing", "failed"):
        # If coarse inputs missing, prepare
        inputs_manifest = layout.coarse_dir / "inputs" / "manifest.json"
        if not inputs_manifest.is_file():
            prep = run_prepare(
                resolved,
                output_dir,
                language=language,
                resume=resume,
                user_request=user_request,
                config=cfg,
            )
            actions.append("prepare")
            return {
                "status": prep["status"],
                "actions": actions,
                "message": prep["message"],
                "report": format_status_report(detect_workflow_status(output_dir)),
            }

    state = detect_workflow_status(output_dir)

    if state.status == "waiting_for_coarse_agent_analysis":
        # Check if agent already wrote analysis
        v = validate_coarse(output_dir)
        if v.ok:
            actions.append("validate-coarse")
            state = detect_workflow_status(output_dir)
        else:
            return {
                "status": "waiting_for_coarse_agent_analysis",
                "actions": actions,
                "message": (
                    "STATUS: waiting_for_coarse_agent_analysis\n"
                    "The current Agent must inspect coarse images under "
                    "analysis/coarse/inputs/ and write coarse analysis JSON.\n"
                    "Scripts will not invent visual analysis."
                ),
                "validation_errors": v.errors,
                "report": format_status_report(state),
            }

    state = detect_workflow_status(output_dir)

    # Stage 3 refine-prepare
    if state.status == "coarse_analysis_complete":
        run_epic3(
            video_path=resolved,
            output_dir=output_dir,
            config=cfg,
            user_request=user_request,
            allow_partial=allow_partial,
        )
        actions.append("refine-prepare")
        state = detect_workflow_status(output_dir)
        state.set_status(
            "waiting_for_fine_agent_analysis",
            message="Refinement preparation complete",
        )
        save_workflow_state(output_dir, state)
        return {
            "status": "waiting_for_fine_agent_analysis",
            "actions": actions,
            "message": (
                "Refinement preparation complete.\n"
                "The current Agent must now inspect fine contact sheets and detail frames."
            ),
            "report": format_status_report(detect_workflow_status(output_dir)),
        }

    state = detect_workflow_status(output_dir)

    if state.status == "waiting_for_fine_agent_analysis":
        v = validate_fine(output_dir)
        if v.ok:
            actions.append("validate-fine")
            state = detect_workflow_status(output_dir)
        else:
            return {
                "status": "waiting_for_fine_agent_analysis",
                "actions": actions,
                "message": (
                    "STATUS: waiting_for_fine_agent_analysis\n"
                    "The current Agent must inspect fine frames and write fine analysis JSON."
                ),
                "validation_errors": v.errors,
                "report": format_status_report(state),
            }

    state = detect_workflow_status(output_dir)

    # Terminal prepare if needed
    if state.status == "fine_analysis_complete":
        second = layout.fine_second_refinement_plan_json
        needs_terminal = False
        if second.is_file():
            import json

            try:
                plan = json.loads(second.read_text(encoding="utf-8"))
                needs_terminal = bool(plan.get("intervals"))
            except Exception:
                needs_terminal = False
        if needs_terminal:
            from video_analyst.workflow.terminal_prepare import run_terminal_prepare

            try:
                tp = run_terminal_prepare(
                    video_path=resolved,
                    output_dir=output_dir,
                    config=cfg,
                    user_request=user_request,
                )
            except Exception as exc:
                log.warning("terminal prepare failed: %s", exc)
                return {
                    "status": "fine_analysis_complete",
                    "actions": actions,
                    "message": (
                        f"Terminal required. Run: "
                        f"video-analyst terminal-prepare {resolved} -o {output_dir}\n"
                        f"({exc})"
                    ),
                    "report": format_status_report(state),
                }
            actions.append("terminal-prepare")
            return {
                "status": tp.get("status", "waiting_for_terminal_agent_analysis"),
                "actions": actions,
                "message": tp.get("message", ""),
                "report": format_status_report(detect_workflow_status(output_dir)),
            }
        else:
            state.set_status("ready_to_finalize")
            save_workflow_state(output_dir, state)

    state = detect_workflow_status(output_dir)

    if state.status == "waiting_for_terminal_agent_analysis":
        v = validate_terminal(output_dir)
        if not v.ok:
            return {
                "status": "waiting_for_terminal_agent_analysis",
                "actions": actions,
                "message": "STATUS: waiting_for_terminal_agent_analysis",
                "validation_errors": v.errors,
                "report": format_status_report(state),
            }
        actions.append("validate-terminal")
        state = detect_workflow_status(output_dir)

    state = detect_workflow_status(output_dir)

    if state.status in ("ready_to_finalize", "terminal_analysis_complete"):
        result = run_finalize(
            resolved,
            output_dir,
            user_request=user_request,
            config=cfg,
            allow_partial=allow_partial,
        )
        actions.append("finalize")
        return {
            "status": "completed",
            "actions": actions,
            "message": f"Finalize complete (report status={result.status})",
            "report_json": str(result.report_json_path) if result.report_json_path else None,
            "report_md": str(result.report_md_path) if result.report_md_path else None,
            "report": format_status_report(detect_workflow_status(output_dir)),
        }

    if state.status == "completed":
        return {
            "status": "completed",
            "actions": actions,
            "message": "Pipeline already completed.",
            "report": format_status_report(state),
        }

    return {
        "status": state.status,
        "actions": actions,
        "message": f"STATUS: {state.status}",
        "report": format_status_report(state),
    }


__all__ = ["run_until_agent_needed"]
