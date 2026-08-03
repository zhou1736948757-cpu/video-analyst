"""Tests for the sole product workflow: in-session agent + local scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_analyst.coarse_analysis.models import RefinementInterval, RefinementPlan, VideoStructure
from video_analyst.fine.models import (
    FineBatchAnalysis,
    FineEvidenceReference,
    FineVideoAnalysis,
    RefinementQuestionAnswer,
    TemporalBoundary,
    TemporalEvent,
)
from video_analyst.vision.models import CoarseBatchAnalysis, EvidenceReference, Observation, TimeRange
from video_analyst.workflow.provenance import AgentSessionProvenance, default_in_session_flags
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


def test_default_in_session_flags():
    flags = default_in_session_flags()
    assert flags["vision_execution_mode"] == "in_session_agent"
    assert flags["real_vision_analysis_used"] is True
    assert flags["external_api_used"] is False
    assert flags["api_key_used"] is False
    assert flags["token_status"] == "unavailable"
    assert flags["estimated_cost"] is None
    assert flags["pricing_source"] == "in_session_agent"


def test_agent_session_provenance_defaults():
    p = AgentSessionProvenance()
    assert p.execution_mode == "in_session_agent"
    assert p.external_api_used is False
    assert p.api_key_used is False
    assert p.automated_visual_analysis is False
    assert p.estimated_cost is None
    assert p.token_status == "unavailable"


def test_state_not_started(tmp_path: Path):
    state = detect_workflow_status(tmp_path)
    assert state.status == "not_started"
    assert state.vision_execution_mode == "in_session_agent"
    assert "prepare" in (state.suggested_command or "")


def test_state_waiting_for_coarse(tmp_path: Path):
    root = tmp_path
    (root / "analysis" / "coarse" / "inputs").mkdir(parents=True)
    (root / "analysis" / "coarse" / "inputs" / "manifest.json").write_text(
        json.dumps({"video_id": "v", "batches": [], "frames": []}),
        encoding="utf-8",
    )
    (root / "subtitles").mkdir(parents=True)
    (root / "subtitles" / "subtitles.json").write_text("{}", encoding="utf-8")
    state = detect_workflow_status(root)
    assert state.status == "waiting_for_coarse_agent_analysis"
    report = format_status_report(state)
    assert "waiting_for_coarse_agent_analysis" in report


def _write_coarse_inputs(root: Path, batch_id: str = "batch_0001") -> None:
    inputs = root / "analysis" / "coarse" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frames = [
        {
            "frame_id": f"{batch_id}_f0000",
            "batch_id": batch_id,
            "timestamp": 0.0,
            "image_path": f"analysis/coarse/inputs/frames/{batch_id}_f0000.jpg",
            "sha256": "a" * 64,
        }
    ]
    manifest = {
        "video_id": "v",
        "duration": 10.0,
        "batches": [{"batch_id": batch_id, "start": 0.0, "end": 10.0}],
        "frames": frames,
        "contact_sheets": [],
    }
    (inputs / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "phase2").mkdir(exist_ok=True)
    (root / "phase2" / "visual_batches.json").write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_id": batch_id,
                        "start": 0.0,
                        "end": 10.0,
                        "frames": [{"path": "x", "time": 0.0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_validate_coarse_missing(tmp_path: Path):
    _write_coarse_inputs(tmp_path)
    result = validate_coarse(tmp_path)
    assert not result.ok
    assert any("batch" in e.lower() for e in result.errors)


def test_validate_coarse_illegal_frame_id(tmp_path: Path):
    _write_coarse_inputs(tmp_path)
    batches = tmp_path / "analysis" / "coarse" / "batches"
    batches.mkdir(parents=True)
    analysis = CoarseBatchAnalysis(
        batch_id="batch_0001",
        start=0.0,
        end=10.0,
        summary="test",
        direct_observations=[
            Observation(
                category="subject",
                description="person",
                time_range=TimeRange(start=0.0, end=1.0),
                evidence=EvidenceReference(frame_ids=["not_a_real_frame"], timestamps=[0.0]),
                confidence=0.5,
            )
        ],
        confidence=0.5,
    )
    (batches / "batch_0001.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )
    structure = VideoStructure(
        video_start=0.0,
        video_end=10.0,
        overall_summary="test",
    )
    (tmp_path / "analysis" / "coarse" / "video_structure.json").write_text(
        structure.model_dump_json(indent=2), encoding="utf-8"
    )
    plan = RefinementPlan(
        video_duration=10.0,
        total_requested_duration=2.0,
        budget_duration=2.0,
        intervals=[
            RefinementInterval(
                interval_id="interval_0001",
                start=0.0,
                end=2.0,
                requested_fps=5,
                priority="medium",
                reasons=["motion"],
                questions=["what moves?"],
                source_batch_ids=["batch_0001"],
                evidence_frame_ids=["batch_0001_f0000"],
            )
        ],
    )
    (tmp_path / "analysis" / "coarse" / "refinement_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    result = validate_coarse(tmp_path)
    assert not result.ok
    assert any("unknown frame_id" in e for e in result.errors)


def test_validate_coarse_ok(tmp_path: Path):
    _write_coarse_inputs(tmp_path)
    batches = tmp_path / "analysis" / "coarse" / "batches"
    batches.mkdir(parents=True)
    analysis = CoarseBatchAnalysis(
        batch_id="batch_0001",
        start=0.0,
        end=10.0,
        summary="talking head",
        direct_observations=[
            Observation(
                category="subject",
                description="speaker",
                time_range=TimeRange(start=0.0, end=2.0),
                evidence=EvidenceReference(
                    frame_ids=["batch_0001_f0000"], timestamps=[0.0]
                ),
                confidence=0.8,
            )
        ],
        confidence=0.7,
    )
    (batches / "batch_0001.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )
    structure = VideoStructure(
        video_start=0.0, video_end=10.0, overall_summary="intro"
    )
    (tmp_path / "analysis" / "coarse" / "video_structure.json").write_text(
        structure.model_dump_json(indent=2), encoding="utf-8"
    )
    plan = RefinementPlan(
        video_duration=10.0,
        total_requested_duration=2.0,
        budget_duration=2.0,
        intervals=[
            RefinementInterval(
                interval_id="interval_0001",
                start=0.0,
                end=2.0,
                requested_fps=10,
                priority="high",
                reasons=["text animation"],
                questions=["how does title enter?"],
                source_batch_ids=["batch_0001"],
                evidence_frame_ids=["batch_0001_f0000"],
            )
        ],
    )
    (tmp_path / "analysis" / "coarse" / "refinement_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    result = validate_coarse(tmp_path)
    assert result.ok, result.errors
    state = detect_workflow_status(tmp_path)
    assert state.status in (
        "coarse_analysis_complete",
        "waiting_for_fine_agent_analysis",
        "preparing_refinement",
    )


def test_validate_fine_missing_question(tmp_path: Path):
    root = tmp_path
    ref = root / "analysis" / "refinement" / "visual_inputs"
    ref.mkdir(parents=True)
    (ref / "fine_batches.json").write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "fine_batch_id": "fine_interval_0001",
                        "interval_id": "interval_0001",
                        "refinement_questions": ["How does the title animate?"],
                        "frames": [{"frame_id": "f1"}],
                        "contact_sheets": [{"sheet_id": "s1", "frame_ids": ["f1"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = root / "analysis" / "fine" / "batches"
    out.mkdir(parents=True)
    analysis = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=2.0,
        confidence=0.5,
        question_answers=[],  # missing required question
        temporal_events=[
            TemporalEvent(
                event_id="e1",
                category="text_animation",
                description="title appears",
                boundary=TemporalBoundary(earliest=0.0, latest=0.5, precision="bounded"),
                evidence=FineEvidenceReference(
                    frame_ids=["f1"], contact_sheet_ids=["s1"], timestamps=[0.0]
                ),
                confidence=0.6,
            )
        ],
    )
    (out / "fine_interval_0001.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )
    (root / "analysis" / "fine" / "fine_video_analysis.json").write_text(
        FineVideoAnalysis().model_dump_json(indent=2), encoding="utf-8"
    )
    result = validate_fine(root)
    assert not result.ok
    assert any("question not answered" in e for e in result.errors)


def test_validate_fine_ok_with_not_required_terminal(tmp_path: Path):
    root = tmp_path
    ref = root / "analysis" / "refinement" / "visual_inputs"
    ref.mkdir(parents=True)
    (ref / "fine_batches.json").write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "fine_batch_id": "fine_interval_0001",
                        "interval_id": "interval_0001",
                        "refinement_questions": ["How does the title animate?"],
                        "frames": [{"frame_id": "f1"}],
                        "contact_sheets": [{"sheet_id": "s1", "frame_ids": ["f1"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "analysis" / "refinement" / "manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    out = root / "analysis" / "fine" / "batches"
    out.mkdir(parents=True)
    analysis = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=2.0,
        confidence=0.7,
        question_answers=[
            RefinementQuestionAnswer(
                question="How does the title animate?",
                answer="Fades in from opacity 0 to 1",
                status="answered",
                confidence=0.7,
                evidence=FineEvidenceReference(
                    frame_ids=["f1"], contact_sheet_ids=["s1"], timestamps=[0.0]
                ),
            )
        ],
        temporal_events=[
            TemporalEvent(
                event_id="e1",
                category="text_animation",
                description="title fade-in",
                boundary=TemporalBoundary(earliest=0.0, latest=0.5, precision="bounded"),
                evidence=FineEvidenceReference(
                    frame_ids=["f1"], contact_sheet_ids=["s1"], timestamps=[0.0]
                ),
                confidence=0.7,
            )
        ],
    )
    (out / "fine_interval_0001.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )
    (root / "analysis" / "fine" / "fine_video_analysis.json").write_text(
        FineVideoAnalysis(
            video_start=0.0,
            video_end=2.0,
            temporal_events=analysis.temporal_events,
            answered_questions=analysis.question_answers,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (root / "analysis" / "fine" / "temporal_events.json").write_text(
        json.dumps([analysis.temporal_events[0].model_dump(mode="json")]),
        encoding="utf-8",
    )
    (root / "analysis" / "fine" / "second_refinement_plan.json").write_text(
        json.dumps(
            {
                "intervals": [],
                "terminal_refinement_status": "not_required",
                "max_additional_rounds": 1,
                "terminal_after_this_round": True,
            }
        ),
        encoding="utf-8",
    )
    result = validate_fine(root)
    assert result.ok, result.errors


def test_config_default_not_mock():
    from video_analyst.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.vision.provider == "in_session_agent"
    assert DEFAULT_CONFIG.vision.provider != "mock"


def test_cli_rejects_provider_check():
    from typer.testing import CliRunner
    from video_analyst.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["provider-check"])
    assert result.exit_code == 2
    assert "removed" in result.output.lower() or "in_session_agent" in result.output


def test_cli_rejects_coarse_analyze_provider_path():
    from typer.testing import CliRunner
    from video_analyst.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["coarse-analyze"])
    assert result.exit_code == 2
    assert "intentionally removed" in result.output.lower() or "removed" in result.output.lower()


def test_finalize_report_provenance_fields(tmp_path: Path):
    """Final report models default to in_session_agent, not mock/API."""
    from video_analyst.final.models import FinalVideoAnalysis, ReportRuntimeStatus

    rs = ReportRuntimeStatus()
    assert rs.provider_mode == "in_session_agent"
    assert rs.vision_execution_mode == "in_session_agent"
    assert rs.external_api_used is False
    assert rs.api_key_used is False
    analysis = FinalVideoAnalysis()
    assert analysis.provider_mode == "in_session_agent"
    assert analysis.real_vision_api_used is False
    assert analysis.real_vision_analysis_used is True
    assert analysis.external_api_used is False
    assert analysis.api_key_used is False
    assert analysis.token_status == "unavailable"
    assert analysis.estimated_cost is None


def test_run_summary_in_session(tmp_path: Path):
    from video_analyst.cache.store import OutputLayout
    from video_analyst.final.run_summary import build_run_summary

    layout = OutputLayout.create(tmp_path)
    summary = build_run_summary(
        layout=layout,
        provider_mode="in_session_agent",
        real_api_used=False,
        terminal_refinement_required=False,
        terminal_refinement_executed=False,
        terminal_refinement_status="not_required",
        status="final",
        coarse_batches_count=1,
        fine_intervals_count=1,
        terminal_intervals_count=0,
        failed_ranges=[],
        deferred_ranges=[],
        unresolved_count=0,
    )
    assert summary["vision_execution_mode"] == "in_session_agent"
    assert summary["external_api_used"] is False
    assert summary["api_key_used"] is False
    assert summary["real_api_used"] is False
    assert summary["estimated_cost"] is None
    assert summary["token_status"] == "unavailable"
