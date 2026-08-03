"""Provenance for in-session agent visual analysis.

Vision analysis is performed by the current conversation Agent inspecting
frames — never by external vision APIs or mock providers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from video_analyst.models import utc_now

VisionExecutionMode = Literal["legacy", "in_session_agent"]


class AgentSessionProvenance(BaseModel):
    """Records how visual analysis was produced in this run."""

    execution_mode: Literal["in_session_agent"] = "in_session_agent"
    agent_name: str | None = None
    agent_session_id: str | None = None
    automated_visual_analysis: bool = False
    external_api_used: bool = False
    api_key_used: bool = False
    coarse_images_inspected: int = 0
    fine_images_inspected: int = 0
    terminal_images_inspected: int = 0
    analysis_started_at: str = Field(
        default_factory=lambda: utc_now().isoformat()
    )
    analysis_completed_at: str | None = None
    # Token/cost are unavailable for in-session agent vision
    token_status: Literal["unavailable"] = "unavailable"
    estimated_cost: float | None = None
    pricing_source: Literal["in_session_agent"] = "in_session_agent"
    real_vision_analysis_used: bool = True

    def mark_completed(self) -> None:
        self.analysis_completed_at = utc_now().isoformat()


def default_in_session_flags() -> dict:
    """Flags that every in-session run must record."""
    return {
        "vision_execution_mode": "in_session_agent",
        "real_vision_analysis_used": True,
        "external_api_used": False,
        "api_key_used": False,
        "token_status": "unavailable",
        "estimated_cost": None,
        "pricing_source": "in_session_agent",
    }


__all__ = [
    "AgentSessionProvenance",
    "VisionExecutionMode",
    "default_in_session_flags",
]
