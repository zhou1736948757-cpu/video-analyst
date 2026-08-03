"""Five-stage pipeline workflow state machine.

Stages:
  prepare → coarse-agent-analysis → refine-prepare
  → fine-agent-analysis → (optional terminal) → finalize
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from video_analyst.models import utc_now

PipelineWorkflowStatus = Literal[
    "not_started",
    "preparing",
    "waiting_for_coarse_agent_analysis",
    "coarse_analysis_complete",
    "preparing_refinement",
    "waiting_for_fine_agent_analysis",
    "fine_analysis_complete",
    "waiting_for_terminal_agent_analysis",
    "terminal_analysis_complete",
    "ready_to_finalize",
    "completed",
    "failed",
]

WORKFLOW_STATE_FILENAME = "workflow_state.json"
WORKFLOW_SCHEMA_VERSION = "1.0"


class WorkflowState(BaseModel):
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    status: PipelineWorkflowStatus = "not_started"
    video_path: str | None = None
    output_dir: str | None = None
    vision_execution_mode: Literal["legacy", "in_session_agent"] = "in_session_agent"
    real_vision_analysis_used: bool = True
    external_api_used: bool = False
    api_key_used: bool = False
    last_error: str | None = None
    messages: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)
    # File presence hints (filled by detect)
    present_files: list[str] = Field(default_factory=list)
    missing_files: list[str] = Field(default_factory=list)
    agent_next_images: list[str] = Field(default_factory=list)
    suggested_command: str | None = None

    def set_status(
        self,
        status: PipelineWorkflowStatus,
        *,
        message: str | None = None,
    ) -> None:
        self.status = status
        if message:
            self.messages.append(message)
        self.updated_at = utc_now()


def workflow_state_path(output_dir: Path) -> Path:
    return output_dir.expanduser().resolve() / "cache" / WORKFLOW_STATE_FILENAME


def load_workflow_state(output_dir: Path) -> WorkflowState:
    path = workflow_state_path(output_dir)
    if not path.is_file():
        return WorkflowState(output_dir=str(output_dir.expanduser().resolve()))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorkflowState.model_validate(data)
    except Exception:
        return WorkflowState(
            output_dir=str(output_dir.expanduser().resolve()),
            status="failed",
            last_error="workflow_state.json unreadable",
        )


def save_workflow_state(output_dir: Path, state: WorkflowState) -> Path:
    path = workflow_state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.output_dir = str(output_dir.expanduser().resolve())
    state.updated_at = utc_now()
    # Always stamp in-session provenance for new runs
    if state.vision_execution_mode != "legacy":
        state.vision_execution_mode = "in_session_agent"
        state.external_api_used = False
        state.api_key_used = False
        state.real_vision_analysis_used = True
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _exists(path: Path) -> bool:
    return path.is_file() if path.suffix else (
        path.is_dir() and any(path.iterdir())
    )


def detect_workflow_status(output_dir: Path) -> WorkflowState:
    """Infer status from artifacts on disk (authoritative over stale state)."""
    root = output_dir.expanduser().resolve()
    state = load_workflow_state(root)
    state.output_dir = str(root)

    present: list[str] = []
    missing: list[str] = []
    next_images: list[str] = []
    suggested: str | None = None
    status: PipelineWorkflowStatus = "not_started"

    # Paths
    media_info = root / "media_info.json"
    metadata = root / "metadata" / "media_info.json"
    subtitles = root / "subtitles" / "subtitles.json"
    phase2_batches = root / "phase2" / "visual_batches.json"
    coarse_inputs_manifest = root / "analysis" / "coarse" / "inputs" / "manifest.json"
    coarse_inputs_frames = root / "analysis" / "coarse" / "inputs" / "frames"
    coarse_inputs_sheets = root / "analysis" / "coarse" / "inputs" / "contact_sheets"
    coarse_batches_dir = root / "analysis" / "coarse" / "batches"
    structure = root / "analysis" / "coarse" / "video_structure.json"
    refinement_plan = root / "analysis" / "coarse" / "refinement_plan.json"
    refine_manifest = root / "analysis" / "refinement" / "manifest.json"
    fine_batches_index = (
        root / "analysis" / "refinement" / "visual_inputs" / "fine_batches.json"
    )
    fine_contact = root / "analysis" / "refinement" / "contact_sheets"
    fine_detail = root / "analysis" / "refinement" / "detail_frames"
    fine_out_batches = root / "analysis" / "fine" / "batches"
    fine_analysis = root / "analysis" / "fine" / "fine_video_analysis.json"
    second_plan = root / "analysis" / "fine" / "second_refinement_plan.json"
    terminal_manifest = root / "analysis" / "terminal_refinement" / "manifest.json"
    terminal_fine = root / "analysis" / "terminal_fine"
    report_json = root / "report" / "report.json"

    def mark(path: Path, label: str | None = None) -> bool:
        key = label or _rel(root, path)
        if path.is_file() or (path.is_dir() and any(path.rglob("*"))):
            present.append(key)
            return True
        missing.append(key)
        return False

    has_media = mark(media_info) or mark(metadata)
    has_subs = mark(subtitles)
    has_phase2 = mark(phase2_batches)
    has_coarse_inputs = mark(coarse_inputs_manifest)
    has_structure = mark(structure)
    has_plan = mark(refinement_plan)
    coarse_batch_files = (
        list(coarse_batches_dir.glob("batch_*.json"))
        if coarse_batches_dir.is_dir()
        else []
    )
    if coarse_batch_files:
        present.append(_rel(root, coarse_batches_dir))
    else:
        missing.append(_rel(root, coarse_batches_dir))

    has_refine = mark(refine_manifest) and mark(fine_batches_index)
    fine_batch_files = (
        list(fine_out_batches.glob("*.json")) if fine_out_batches.is_dir() else []
    )
    if fine_batch_files:
        present.append(_rel(root, fine_out_batches))
    else:
        missing.append(_rel(root, fine_out_batches))
    has_fine_analysis = mark(fine_analysis)
    has_second = second_plan.is_file()
    if has_second:
        present.append(_rel(root, second_plan))
    has_terminal_prep = mark(terminal_manifest)
    terminal_done = (
        terminal_fine.is_dir()
        and (
            (terminal_fine / "terminal_fine_analysis.json").is_file()
            or any((terminal_fine / "batches").glob("*.json"))
            if (terminal_fine / "batches").is_dir()
            else (terminal_fine / "terminal_fine_analysis.json").is_file()
        )
    )
    if terminal_done:
        present.append(_rel(root, terminal_fine))
    has_report = mark(report_json)

    # Decide status (later stages win)
    if has_report:
        status = "completed"
        suggested = None
    elif has_fine_analysis or fine_batch_files:
        # Check terminal need
        terminal_required = False
        if has_second:
            try:
                plan_data = json.loads(second_plan.read_text(encoding="utf-8"))
                intervals = plan_data.get("intervals") or []
                terminal_required = bool(intervals)
            except Exception:
                terminal_required = False
        if terminal_required and not terminal_done:
            if has_terminal_prep:
                status = "waiting_for_terminal_agent_analysis"
                next_images.append(_rel(root, root / "analysis" / "terminal_refinement" / "contact_sheets"))
                next_images.append(_rel(root, root / "analysis" / "terminal_refinement" / "detail_frames"))
                suggested = (
                    f"# Inspect terminal frames, write analysis/terminal_fine/, then:\n"
                    f"video-analyst validate-terminal {root}\n"
                    f"video-analyst finalize VIDEO -o {root}"
                )
            else:
                status = "fine_analysis_complete"
                suggested = f"video-analyst terminal-prepare VIDEO -o {root}"
        else:
            status = "ready_to_finalize"
            suggested = f"video-analyst finalize VIDEO -o {root}"
    elif has_refine:
        status = "waiting_for_fine_agent_analysis"
        if fine_contact.is_dir():
            next_images.append(_rel(root, fine_contact))
        if fine_detail.is_dir():
            next_images.append(_rel(root, fine_detail))
        next_images.append(_rel(root, fine_batches_index))
        suggested = (
            f"# Inspect fine contact sheets + detail frames, write analysis/fine/, then:\n"
            f"video-analyst validate-fine {root}"
        )
    elif has_structure and has_plan and coarse_batch_files:
        status = "coarse_analysis_complete"
        suggested = f"video-analyst refine-prepare VIDEO -o {root}"
    elif has_coarse_inputs:
        status = "waiting_for_coarse_agent_analysis"
        if coarse_inputs_frames.is_dir():
            next_images.append(_rel(root, coarse_inputs_frames))
        if coarse_inputs_sheets.is_dir():
            next_images.append(_rel(root, coarse_inputs_sheets))
        next_images.append(_rel(root, coarse_inputs_manifest))
        suggested = (
            f"# Inspect coarse frames + contact sheets, write coarse analysis, then:\n"
            f"video-analyst validate-coarse {root}"
        )
    elif has_media or has_subs or has_phase2:
        status = "preparing"
        suggested = f"video-analyst prepare VIDEO -o {root}"
    else:
        status = "not_started"
        suggested = f"video-analyst prepare VIDEO -o {root}"

    # Prefer persisted failed status if nothing progressed
    if state.status == "failed" and status == "not_started":
        status = "failed"

    state.status = status
    state.present_files = present
    state.missing_files = missing
    state.agent_next_images = next_images
    state.suggested_command = suggested
    state.vision_execution_mode = "in_session_agent"
    state.external_api_used = False
    state.api_key_used = False
    state.real_vision_analysis_used = True
    return state


def format_status_report(state: WorkflowState) -> str:
    lines = [
        f"STATUS: {state.status}",
        f"vision_execution_mode: {state.vision_execution_mode}",
        f"real_vision_analysis_used: {state.real_vision_analysis_used}",
        f"external_api_used: {state.external_api_used}",
        f"api_key_used: {state.api_key_used}",
        "",
        "Present files:",
    ]
    if state.present_files:
        for p in state.present_files[:40]:
            lines.append(f"  + {p}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Missing files:")
    if state.missing_files:
        for p in state.missing_files[:40]:
            lines.append(f"  - {p}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Agent should inspect next:")
    if state.agent_next_images:
        for p in state.agent_next_images:
            lines.append(f"  * {p}")
    else:
        lines.append("  (none — local stage or completed)")
    lines.append("")
    if state.suggested_command:
        lines.append("Suggested next:")
        lines.append(state.suggested_command)
    return "\n".join(lines)


__all__ = [
    "PipelineWorkflowStatus",
    "WORKFLOW_SCHEMA_VERSION",
    "WorkflowState",
    "detect_workflow_status",
    "format_status_report",
    "load_workflow_state",
    "save_workflow_state",
    "workflow_state_path",
]
