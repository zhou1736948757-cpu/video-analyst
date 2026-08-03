"""Validate agent-written coarse / fine / terminal analysis JSON.

These commands do NOT call vision models. They only:
  - Pydantic schema validation
  - batch_id / frame_id / sheet_id / timestamp checks
  - evidence existence
  - refinement question coverage
  - atomic write confirmation (files fully readable)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from video_analyst.cache.store import OutputLayout
from video_analyst.coarse_analysis.models import RefinementPlan, VideoStructure
from video_analyst.fine.models import FineBatchAnalysis, FineVideoAnalysis
from video_analyst.vision.models import CoarseBatchAnalysis
from video_analyst.workflow.state import (
    WorkflowState,
    detect_workflow_status,
    save_workflow_state,
)


@dataclass
class ValidationResult:
    ok: bool
    stage: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty file: {path}")
    return json.loads(text)


def _collect_coarse_known_ids(layout: OutputLayout) -> tuple[set[str], set[str], set[str]]:
    """Return (batch_ids, frame_ids, sheet_ids) from coarse inputs."""
    batch_ids: set[str] = set()
    frame_ids: set[str] = set()
    sheet_ids: set[str] = set()
    manifest_path = layout.coarse_dir / "inputs" / "manifest.json"
    if not manifest_path.is_file():
        # fall back to phase2
        if layout.phase2_batches_json.is_file():
            data = _load_json(layout.phase2_batches_json)
            for b in data.get("batches") or []:
                batch_ids.add(str(b["batch_id"]))
                for i, fr in enumerate(b.get("frames") or []):
                    fid = fr.get("frame_id") or f"{b['batch_id']}_f{i:04d}"
                    frame_ids.add(str(fid))
        return batch_ids, frame_ids, sheet_ids

    data = _load_json(manifest_path)
    for b in data.get("batches") or []:
        batch_ids.add(str(b["batch_id"]))
    for fr in data.get("frames") or []:
        frame_ids.add(str(fr["frame_id"]))
    for sh in data.get("contact_sheets") or []:
        sheet_ids.add(str(sh.get("sheet_id") or sh.get("contact_sheet_path") or ""))
    return batch_ids, frame_ids, sheet_ids


def validate_coarse(output_dir: Path) -> ValidationResult:
    layout = OutputLayout.create(output_dir)
    result = ValidationResult(ok=True, stage="coarse")
    root = layout.root

    known_batches, known_frames, _known_sheets = _collect_coarse_known_ids(layout)
    if not known_batches:
        result.add_error(
            "No coarse input batches found. Run: video-analyst prepare VIDEO -o OUTPUT"
        )
        return result

    batches_dir = layout.coarse_batches_dir
    if not batches_dir.is_dir():
        result.add_error(f"Missing directory: {batches_dir}")
        return result

    batch_files = sorted(batches_dir.glob("batch_*.json"))
    if not batch_files:
        result.add_error(
            f"No batch_*.json under {batches_dir}. "
            "Agent must write CoarseBatchAnalysis JSON per batch."
        )
        return result

    analyses: dict[str, CoarseBatchAnalysis] = {}
    for path in batch_files:
        rel = str(path.relative_to(root))
        result.checked_files.append(rel)
        try:
            data = _load_json(path)
            analysis = CoarseBatchAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            result.add_error(f"{rel}: invalid CoarseBatchAnalysis — {exc}")
            continue

        if analysis.batch_id not in known_batches:
            result.add_error(
                f"{rel}: unknown batch_id {analysis.batch_id!r} "
                f"(known: {sorted(known_batches)[:5]}…)"
            )

        # Evidence frame IDs must exist when provided
        for obs_list_name in (
            "direct_observations",
            "visible_text",
            "subjects",
            "scene_changes",
            "camera_observations",
            "motion_observations",
            "possible_transitions",
            "important_moments",
        ):
            for obs in getattr(analysis, obs_list_name) or []:
                for fid in obs.evidence.frame_ids:
                    if known_frames and fid not in known_frames:
                        result.add_error(
                            f"{rel}: {obs_list_name} references unknown frame_id {fid!r}"
                        )
                for ts in obs.evidence.timestamps:
                    if ts < 0:
                        result.add_error(f"{rel}: negative timestamp {ts}")

        for rr in analysis.refinement_requests:
            for fid in rr.evidence_frame_ids:
                if known_frames and fid not in known_frames:
                    result.add_error(
                        f"{rel}: refinement_request references unknown frame_id {fid!r}"
                    )
            if rr.start < analysis.start - 1e-3 or rr.end > analysis.end + 1e-3:
                result.add_warning(
                    f"{rel}: refinement_request {rr.start}..{rr.end} "
                    f"outside batch {analysis.start}..{analysis.end}"
                )

        analyses[analysis.batch_id] = analysis

    # Required aggregate files
    structure_path = layout.coarse_structure_json
    plan_path = layout.coarse_refinement_plan_json
    if not structure_path.is_file():
        result.add_error(
            f"Missing {structure_path.relative_to(root)} — "
            "Agent must write VideoStructure JSON"
        )
    else:
        result.checked_files.append(str(structure_path.relative_to(root)))
        try:
            VideoStructure.model_validate(_load_json(structure_path))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            result.add_error(f"video_structure.json invalid: {exc}")

    if not plan_path.is_file():
        result.add_error(
            f"Missing {plan_path.relative_to(root)} — "
            "Agent must write RefinementPlan JSON"
        )
    else:
        result.checked_files.append(str(plan_path.relative_to(root)))
        try:
            plan = RefinementPlan.model_validate(_load_json(plan_path))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            result.add_error(f"refinement_plan.json invalid: {exc}")
            plan = None
        if plan is not None:
            for iv in plan.intervals:
                if iv.requested_fps not in (5, 10, 15):
                    result.add_error(
                        f"interval {iv.interval_id}: fps must be 5/10/15, got {iv.requested_fps}"
                    )
                if iv.end < iv.start:
                    result.add_error(f"interval {iv.interval_id}: end < start")
                for bid in iv.source_batch_ids:
                    if known_batches and bid not in known_batches:
                        result.add_error(
                            f"interval {iv.interval_id}: unknown source_batch_id {bid!r}"
                        )
                for fid in iv.evidence_frame_ids:
                    if known_frames and fid not in known_frames:
                        result.add_error(
                            f"interval {iv.interval_id}: unknown evidence_frame_id {fid!r}"
                        )
            # Mechanical full-video 15 FPS is discouraged
            if plan.intervals and plan.video_duration > 0:
                full_15 = all(
                    iv.requested_fps == 15
                    and abs(iv.end - iv.start - plan.video_duration) < 1.0
                    for iv in plan.intervals
                )
                if full_15 and len(plan.intervals) == 1:
                    result.add_error(
                        "Refinement plan uses full-video 15 FPS — not allowed. "
                        "Select short targeted intervals."
                    )

    # Coverage: prefer all known batches present
    missing_batches = known_batches - set(analyses.keys())
    if missing_batches:
        result.add_warning(
            f"Missing analysis for batches: {sorted(missing_batches)[:10]}"
        )

    result.details = {
        "batches_validated": len(analyses),
        "known_batches": len(known_batches),
        "known_frames": len(known_frames),
    }

    if result.ok:
        state = detect_workflow_status(root)
        if state.status in (
            "waiting_for_coarse_agent_analysis",
            "preparing",
            "not_started",
        ):
            state.set_status(
                "coarse_analysis_complete",
                message="validate-coarse passed",
            )
            save_workflow_state(root, state)

    return result


def _collect_fine_known_ids(
    layout: OutputLayout,
) -> tuple[set[str], set[str], set[str], dict[str, list[str]]]:
    """Return (fine_batch_ids, frame_ids, sheet_ids, questions_by_batch)."""
    batch_ids: set[str] = set()
    frame_ids: set[str] = set()
    sheet_ids: set[str] = set()
    questions: dict[str, list[str]] = {}

    index_path = layout.refinement_fine_batches_json
    if not index_path.is_file():
        return batch_ids, frame_ids, sheet_ids, questions

    data = _load_json(index_path)
    for b in data.get("batches") or []:
        fb_id = str(b.get("fine_batch_id") or b.get("batch_id") or "")
        if not fb_id:
            continue
        batch_ids.add(fb_id)
        questions[fb_id] = list(b.get("refinement_questions") or b.get("questions") or [])
        for fr in b.get("frames") or []:
            if isinstance(fr, dict) and fr.get("frame_id"):
                frame_ids.add(str(fr["frame_id"]))
            elif hasattr(fr, "frame_id"):
                frame_ids.add(str(fr.frame_id))
        for sh in b.get("contact_sheets") or []:
            if isinstance(sh, dict):
                sid = sh.get("sheet_id") or sh.get("contact_sheet_id")
                if sid:
                    sheet_ids.add(str(sid))
                for fid in sh.get("frame_ids") or []:
                    frame_ids.add(str(fid))
        for df in b.get("detail_frames") or []:
            if isinstance(df, dict) and df.get("frame_id"):
                frame_ids.add(str(df["frame_id"]))

    # Also scan frames dirs
    for p in layout.refinement_frames_dir.rglob("*.jpg"):
        frame_ids.add(p.stem)
    for p in layout.refinement_contact_sheets_dir.rglob("*.jpg"):
        sheet_ids.add(p.stem)
    return batch_ids, frame_ids, sheet_ids, questions


def validate_fine(output_dir: Path) -> ValidationResult:
    layout = OutputLayout.create(output_dir)
    result = ValidationResult(ok=True, stage="fine")
    root = layout.root

    known_batches, known_frames, known_sheets, questions = _collect_fine_known_ids(layout)
    if not known_batches:
        result.add_error(
            "No fine_batches.json. Run: video-analyst refine-prepare VIDEO -o OUTPUT"
        )
        return result

    batches_dir = layout.fine_batches_out_dir
    batch_files = sorted(batches_dir.glob("*.json")) if batches_dir.is_dir() else []
    if not batch_files:
        result.add_error(
            f"No fine batch analysis JSON under {batches_dir}. "
            "Agent must write FineBatchAnalysis per fine batch."
        )
        return result

    analyses: dict[str, FineBatchAnalysis] = {}
    for path in batch_files:
        rel = str(path.relative_to(root))
        result.checked_files.append(rel)
        try:
            data = _load_json(path)
            analysis = FineBatchAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            result.add_error(f"{rel}: invalid FineBatchAnalysis — {exc}")
            continue

        if analysis.fine_batch_id not in known_batches:
            result.add_error(f"{rel}: unknown fine_batch_id {analysis.fine_batch_id!r}")

        # Question coverage
        expected_q = questions.get(analysis.fine_batch_id) or []
        answered = {qa.question.strip() for qa in analysis.question_answers}
        for q in expected_q:
            if q.strip() and q.strip() not in answered:
                result.add_error(
                    f"{rel}: refinement question not answered: {q[:80]}"
                )

        # Evidence
        for ev in analysis.temporal_events:
            if not ev.evidence.frame_ids and not ev.evidence.contact_sheet_ids:
                result.add_error(
                    f"{rel}: event {ev.event_id} has no evidence frame/sheet IDs"
                )
            for fid in ev.evidence.frame_ids:
                if known_frames and fid not in known_frames:
                    result.add_error(
                        f"{rel}: event {ev.event_id} unknown frame_id {fid!r}"
                    )
            for sid in ev.evidence.contact_sheet_ids:
                if known_sheets and sid not in known_sheets:
                    result.add_error(
                        f"{rel}: event {ev.event_id} unknown sheet_id {sid!r}"
                    )
            for ts in ev.evidence.timestamps:
                if ts < 0:
                    result.add_error(f"{rel}: negative timestamp {ts}")

        analyses[analysis.fine_batch_id] = analysis

    # Aggregate files
    for name, path, model_cls in (
        (
            "fine_video_analysis.json",
            layout.fine_video_analysis_json,
            FineVideoAnalysis,
        ),
    ):
        if not path.is_file():
            result.add_error(
                f"Missing analysis/fine/{name} — Agent must write aggregate fine analysis"
            )
        else:
            result.checked_files.append(str(path.relative_to(root)))
            try:
                model_cls.model_validate(_load_json(path))
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                result.add_error(f"{name} invalid: {exc}")

    # temporal_events.json optional but recommended
    te_path = layout.fine_temporal_events_json
    if te_path.is_file():
        result.checked_files.append(str(te_path.relative_to(root)))
        try:
            te = _load_json(te_path)
            if not isinstance(te, (list, dict)):
                result.add_error("temporal_events.json must be list or object")
        except (json.JSONDecodeError, ValueError) as exc:
            result.add_error(f"temporal_events.json invalid: {exc}")
    else:
        result.add_warning("temporal_events.json missing (recommended)")

    # second_refinement_plan — may be empty / not_required
    sr_path = layout.fine_second_refinement_plan_json
    if not sr_path.is_file():
        # Write a default not_required stub is NOT done here — agent should write it
        result.add_warning(
            "second_refinement_plan.json missing; "
            "Agent should write it with terminal_refinement_status=not_required "
            "or a single terminal plan"
        )
    else:
        result.checked_files.append(str(sr_path.relative_to(root)))
        try:
            plan = _load_json(sr_path)
            if not isinstance(plan, dict):
                result.add_error("second_refinement_plan.json must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            result.add_error(f"second_refinement_plan.json invalid: {exc}")

    result.details = {
        "batches_validated": len(analyses),
        "known_batches": len(known_batches),
    }

    if result.ok:
        state = detect_workflow_status(root)
        # Re-detect will pick fine_analysis_complete / ready_to_finalize
        save_workflow_state(root, state)

    return result


def validate_terminal(output_dir: Path) -> ValidationResult:
    layout = OutputLayout.create(output_dir)
    result = ValidationResult(ok=True, stage="terminal")
    root = layout.root

    terminal_dir = layout.terminal_fine_dir
    if not terminal_dir.is_dir():
        result.add_error(f"Missing {terminal_dir}")
        return result

    analysis_path = layout.terminal_fine_analysis_json
    batches_dir = layout.terminal_fine_batches_out_dir
    batch_files = list(batches_dir.glob("*.json")) if batches_dir.is_dir() else []

    if not analysis_path.is_file() and not batch_files:
        result.add_error(
            "No terminal analysis found. Agent must write analysis/terminal_fine/"
        )
        return result

    if analysis_path.is_file():
        result.checked_files.append(str(analysis_path.relative_to(root)))
        try:
            _load_json(analysis_path)
        except (json.JSONDecodeError, ValueError) as exc:
            result.add_error(f"terminal_fine_analysis.json invalid: {exc}")

    for path in batch_files:
        rel = str(path.relative_to(root))
        result.checked_files.append(rel)
        try:
            data = _load_json(path)
            FineBatchAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            result.add_error(f"{rel}: invalid FineBatchAnalysis — {exc}")

    if result.ok:
        state = detect_workflow_status(root)
        state.set_status(
            "terminal_analysis_complete"
            if state.status != "completed"
            else "completed",
            message="validate-terminal passed",
        )
        if state.status == "terminal_analysis_complete":
            state.set_status("ready_to_finalize")
        save_workflow_state(root, state)

    return result


def format_validation_result(result: ValidationResult) -> str:
    lines = [
        f"validate-{result.stage}: {'OK' if result.ok else 'FAILED'}",
        f"checked_files: {len(result.checked_files)}",
    ]
    if result.details:
        for k, v in result.details.items():
            lines.append(f"  {k}: {v}")
    if result.errors:
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  ! {e}")
    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  ~ {w}")
    return "\n".join(lines)


__all__ = [
    "ValidationResult",
    "format_validation_result",
    "validate_coarse",
    "validate_fine",
    "validate_terminal",
]
