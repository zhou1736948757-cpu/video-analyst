"""Pydantic models for vision batch requests and coarse analysis responses.

These models are the canonical contract between any VisionProvider and the
rest of the pipeline. Providers convert raw responses into this shape and
the rest of the system MUST NOT depend on provider-specific JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"


# ---- Request models ----


class VisionFrame(BaseModel):
    """A single frame included in a vision batch request."""

    frame_id: str
    timestamp_seconds: float
    timecode: str
    path: Path
    sources: list[str] = Field(default_factory=list)  # e.g. ["fixed_fps", "scene"]
    scene_score: float | None = None

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


class SubtitleContext(BaseModel):
    """A subtitle segment provided as context to the vision model."""

    index: int
    start: float
    end: float
    text: str
    language: str | None = None


class SubtitleMetadata(BaseModel):
    """Metadata about the active subtitle source."""

    active_source: Literal["asr", "llm"] = "asr"
    reviewed: bool = False
    needs_review: bool = False
    asr_model: str | None = None
    calibration_model: str | None = None


class PreviousBatchContext(BaseModel):
    """Compressed context from the previous batch (not full images/responses)."""

    previous_batch_id: str
    previous_summary: str
    last_known_subjects: list[str] = Field(default_factory=list)
    last_known_scene: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)


class VisionBatchRequest(BaseModel):
    """Unified request sent to a VisionProvider for one batch."""

    schema_version: str = SCHEMA_VERSION
    batch_id: str
    video_id: str
    batch_start: float
    batch_end: float
    frames: list[VisionFrame] = Field(default_factory=list)
    subtitles: list[SubtitleContext] = Field(default_factory=list)
    subtitle_metadata: SubtitleMetadata = Field(default_factory=SubtitleMetadata)
    user_request: str = ""
    analysis_dimensions: list[str] = Field(default_factory=list)
    previous_context: PreviousBatchContext | None = None

    @model_validator(mode="after")
    def _check_range(self) -> VisionBatchRequest:
        if self.batch_end < self.batch_start:
            raise ValueError("batch_end < batch_start")
        return self


# ---- Response models ----


class TimeRange(BaseModel):
    start: float
    end: float

    @model_validator(mode="after")
    def _check_range(self) -> TimeRange:
        if self.end < self.start:
            raise ValueError("end < start")
        return self


class EvidenceReference(BaseModel):
    frame_ids: list[str] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)


class Observation(BaseModel):
    """A direct observation from the visual content."""

    category: str
    description: str
    time_range: TimeRange
    evidence: EvidenceReference = Field(default_factory=EvidenceReference)
    confidence: float = Field(ge=0.0, le=1.0)


class Inference(BaseModel):
    """An interpretation inferred from observations (not directly visible)."""

    description: str
    supporting_observation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list)


class RefinementRequest(BaseModel):
    """A request for higher-FPS sampling in a specific time range."""

    request_id: str
    start: float
    end: float
    requested_fps: Literal[5, 10, 15]
    reason: str
    questions: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high"] = "medium"
    evidence_frame_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> RefinementRequest:
        if self.end < self.start:
            raise ValueError("end < start")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        return self


class CoarseBatchAnalysis(BaseModel):
    """Structured output from vision model for one batch."""

    schema_version: str = SCHEMA_VERSION
    batch_id: str
    start: float
    end: float
    summary: str = ""
    direct_observations: list[Observation] = Field(default_factory=list)
    inferred_interpretations: list[Inference] = Field(default_factory=list)
    visible_text: list[Observation] = Field(default_factory=list)
    subjects: list[Observation] = Field(default_factory=list)
    scene_changes: list[Observation] = Field(default_factory=list)
    camera_observations: list[Observation] = Field(default_factory=list)
    motion_observations: list[Observation] = Field(default_factory=list)
    possible_transitions: list[Observation] = Field(default_factory=list)
    subtitle_visual_relationship: list[Observation] = Field(default_factory=list)
    important_moments: list[Observation] = Field(default_factory=list)
    refinement_requests: list[RefinementRequest] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> CoarseBatchAnalysis:
        if self.end < self.start:
            raise ValueError("end < start")
        return self


# ---- Output-style descriptors used by RequestBuilder (no external I/O) ----


class VisionBatchSelection(BaseModel):
    """Internal record of which frames were selected for a batch after capping.

    Visible as part of the saved batch_*.json so the user can see what was
    sent to the model vs. what was available.
    """

    batch_id: str
    total_available: int
    selected: int
    omitted_frame_ids: list[str] = Field(default_factory=list)
    preserved_scene_frame_ids: list[str] = Field(default_factory=list)


# Re-export public models
__all__ = [
    "SCHEMA_VERSION",
    "CoarseBatchAnalysis",
    "EvidenceReference",
    "Inference",
    "Observation",
    "PreviousBatchContext",
    "RefinementRequest",
    "SubtitleContext",
    "SubtitleMetadata",
    "TimeRange",
    "VisionBatchRequest",
    "VisionBatchSelection",
    "VisionFrame",
]
