"""Stage 5 — deterministic finalize (no vision inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.final.orchestrator import Epic5Result, run_epic5
from video_analyst.logging.setup import get_logger, setup_logging
from video_analyst.workflow.provenance import (
    AgentSessionProvenance,
    default_in_session_flags,
)
from video_analyst.workflow.state import save_workflow_state, detect_workflow_status

log = get_logger("workflow.finalize")


def run_finalize(
    video_path: Path,
    output_dir: Path,
    *,
    user_request: str = "",
    config: AppConfig | None = None,
    allow_partial: bool = True,
    no_markdown: bool = False,
    no_csv: bool = False,
    no_evidence_copy: bool = False,
    agent_provenance: AgentSessionProvenance | None = None,
) -> Epic5Result:
    """Aggregate agent-written coarse/fine/terminal into final report."""
    cfg = config or DEFAULT_CONFIG
    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)
    setup_logging(layout.run_log)

    result = run_epic5(
        video_path=video_path,
        output_dir=output_dir,
        config=cfg,
        user_request=user_request,
        allow_partial=allow_partial,
        no_markdown=no_markdown,
        no_csv=no_csv,
        no_evidence_copy=no_evidence_copy,
        # Never auto-run terminal via mock/external vision provider.
        # Agent writes terminal analysis; finalize only aggregates.
        execute_terminal=False,
        allow_provisional=True,
    )

    # Stamp in-session provenance onto report artifacts
    flags = default_in_session_flags()
    prov = agent_provenance or AgentSessionProvenance()
    prov.mark_completed()

    # Patch report.json / run_summary if present
    _stamp_provenance(layout, flags, prov)

    state = detect_workflow_status(output_dir)
    if result.analysis is not None and layout.final_report_json.is_file():
        state.set_status("completed", message="finalize complete")
    save_workflow_state(output_dir, state)
    return result


def _stamp_provenance(
    layout: OutputLayout,
    flags: dict[str, Any],
    prov: AgentSessionProvenance,
) -> None:
    import json

    for path in (layout.final_report_json, layout.final_run_summary_json):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data.update(flags)
        data["agent_session_provenance"] = prov.model_dump(mode="json")
        # Remove misleading mock/real_api fields when present
        data["provider_mode"] = "in_session_agent"
        data["real_vision_api_used"] = False
        data["real_vision_analysis_used"] = True
        data["external_api_used"] = False
        data["api_key_used"] = False
        if "runtime_status" in data and isinstance(data["runtime_status"], dict):
            data["runtime_status"]["provider_mode"] = "in_session_agent"
            data["runtime_status"]["vision_execution_mode"] = "in_session_agent"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)


__all__ = ["run_finalize"]
