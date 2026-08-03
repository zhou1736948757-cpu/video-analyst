"""Data models for Epic 4 — fine temporal vision analysis.

All public models are Pydantic v2 with strict validation. Schema versions
are explicit so downstream stages (aggregation, second refinement plan,
final report) can refuse to consume unknown contracts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from video_analyst.models import utc_now
from video_analyst.refinement.models import RefinementSource
from video_analyst.vision.models import SubtitleContext, SubtitleMetadata

FINE_SCHEMA_VERSION = "1.0"

# Re-exported literal aliases for convenience
EventCategory = Literal[
    "subject_motion",
    "camera_motion",
    "post_transform",
    "text_animation",
    "graphic_animation",
    "transition",
    "appearance_change",
    "layout_change",
    "uncertain_change",
]

BoundaryPrecision = Literal["exact", "bounded", "approximate"]

PriorityLevel = Literal["low", "medium", "high"]

SecondRefinementFps = Literal[10, 15]


# ---- Time boundary ----


class TemporalBoundary(BaseModel):
    """Time boundary of an event.

    - ``exact``: confirmed within a single frame.
    - ``bounded``: only know it happened between two frames.
    - ``approximate``: rough estimate only.
    """

    earliest: float = Field(ge=0.0)
    latest: float = Field(ge=0.0)
    precision: BoundaryPrecision = "bounded"
    before_frame_id: str | None = None
    after_frame_id: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "TemporalBoundary":
        if self.latest < self.earliest:
            raise ValueError(
                f"latest ({self.latest}) must be >= earliest ({self.earliest})"
            )
        # exact → earliest == latest (single instant); bounded/approximate
        # allow a range.
        return self


# ---- Evidence ----


class FineEvidenceReference(BaseModel):
    """Evidence reference for a fine-analysis conclusion."""

    contact_sheet_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)


# ---- Frame state observation (direct observation only) ----


class FrameStateObservation(BaseModel):
    """A direct observation of a single frame's state.

    Only what is visible in that frame — NEVER motion paths or transitions.
    """

    observation_id: str
    timestamp_seconds: float = Field(ge=0.0)
    frame_id: str
    description: str
    visible_subjects: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)
    composition_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Temporal event (the unified event record) ----


class TemporalEvent(BaseModel):
    """A temporal event spanning one or more frames.

    ``direct_observations`` references observation IDs; ``inference`` is the
    model's interpretation (may be null). The two MUST stay separate.
    """

    event_id: str
    category: EventCategory
    boundary: TemporalBoundary
    description: str
    direct_observations: list[str] = Field(default_factory=list)
    inference: str | None = None
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_batch_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_inference_separation(self) -> "TemporalEvent":
        # inference is optional but must not be disguised as a direct
        # observation. We can't structurally enforce this, but we reject
        # empty-description events that would otherwise pad the schema.
        if not self.description.strip():
            raise ValueError("event description must not be empty")
        return self


# ---- Subject motion ----


class SubjectMotionAnalysis(BaseModel):
    event_id: str
    subject: str
    action: str
    direction: str | None = None
    start_state: str
    end_state: str
    boundary: TemporalBoundary
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Camera motion ----


CameraMotionType = Literal[
    "static",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "zoom_in",
    "zoom_out",
    "dolly_in",
    "dolly_out",
    "truck_left",
    "truck_right",
    "handheld",
    "multiple",
    "uncertain",
]


class CameraMotionAnalysis(BaseModel):
    event_id: str
    motion_type: CameraMotionType
    description: str
    boundary: TemporalBoundary
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Post transform ----


PostTransformType = Literal[
    "scale",
    "position",
    "rotation",
    "crop",
    "opacity",
    "blur",
    "mask",
    "color",
    "composite",
    "uncertain",
]


class PostTransformAnalysis(BaseModel):
    event_id: str
    transform_types: list[PostTransformType] = Field(default_factory=list)
    affected_region: str = "whole_frame"
    start_state: str = ""
    end_state: str = ""
    boundary: TemporalBoundary
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Text animation ----


TextAnimationType = Literal[
    "appear",
    "disappear",
    "fade",
    "position",
    "scale",
    "rotation",
    "mask_reveal",
    "type_on",
    "tracking",
    "color_change",
    "emphasis",
    "uncertain",
]


class TextAnimationAnalysis(BaseModel):
    event_id: str
    text_content: str | None = None
    animation_types: list[TextAnimationType] = Field(default_factory=list)
    start_state: str = ""
    end_state: str = ""
    boundary: TemporalBoundary
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Graphic animation ----


class GraphicAnimationAnalysis(BaseModel):
    event_id: str
    graphic_subject: str
    animation_types: list[str] = Field(default_factory=list)
    start_state: str = ""
    end_state: str = ""
    boundary: TemporalBoundary
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Transition ----


TransitionType = Literal[
    "cut",
    "dissolve",
    "fade_in",
    "fade_out",
    "wipe",
    "slide",
    "zoom_transition",
    "mask_transition",
    "match_cut",
    "graphic_transition",
    "flash",
    "uncertain",
]


class TransitionAnalysis(BaseModel):
    event_id: str
    transition_type: TransitionType
    outgoing_state: str = ""
    incoming_state: str = ""
    boundary: TemporalBoundary
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---- Refinement question answer ----

AnswerOrigin = Literal[
    "model",
    "validator_fallback",
    "terminal_refinement",
]


class RefinementQuestionAnswer(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "partially_answered", "unresolved"] = "unresolved"
    # Epic 5 修复：区分答案来源（模型主动 / 验证器补齐 / 终局补帧解决）
    answer_origin: AnswerOrigin = "model"
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    confidence: float = Field(ge=0.0, le=1.0)
    # 终局补帧是否尝试过（当 answer_origin=terminal_refinement 或仍未解决时为 True）
    terminal_attempted: bool = False

    @model_validator(mode="after")
    def _check(self) -> "RefinementQuestionAnswer":
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.answer.strip() and self.status != "unresolved":
            raise ValueError(
                "answer must not be empty when status is not unresolved"
            )
        if self.status == "unresolved" and self.answer_origin == "model":
            # 模型主动判断为 unresolved
            pass
        if self.answer_origin == "validator_fallback" and self.status != "unresolved":
            raise ValueError(
                "validator_fallback must imply status=unresolved"
            )
        return self


# ---- Second refinement request ----


SecondRefinementRequestType = Literal[
    "higher_temporal_density",
    "narrower_interval",
    "high_resolution_detail",
]


class SecondRefinementRequest(BaseModel):
    request_id: str
    request_type: SecondRefinementRequestType
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    requested_fps: SecondRefinementFps
    exact_detail_timestamps: list[float] = Field(default_factory=list)
    reason: str
    questions: list[str] = Field(default_factory=list)
    priority: PriorityLevel = "medium"
    evidence: FineEvidenceReference = Field(default_factory=FineEvidenceReference)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> "SecondRefinementRequest":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        return self


# ---- Fine batch analysis output ----


class FineBatchAnalysis(BaseModel):
    """Structured fine analysis output for one FineVisualBatch."""

    schema_version: str = FINE_SCHEMA_VERSION
    fine_batch_id: str
    interval_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    summary: str = ""

    frame_states: list[FrameStateObservation] = Field(default_factory=list)
    temporal_events: list[TemporalEvent] = Field(default_factory=list)

    subject_motions: list[SubjectMotionAnalysis] = Field(default_factory=list)
    camera_motions: list[CameraMotionAnalysis] = Field(default_factory=list)
    post_transforms: list[PostTransformAnalysis] = Field(default_factory=list)
    text_animations: list[TextAnimationAnalysis] = Field(default_factory=list)
    graphic_animations: list[GraphicAnimationAnalysis] = Field(default_factory=list)
    transitions: list[TransitionAnalysis] = Field(default_factory=list)

    question_answers: list[RefinementQuestionAnswer] = Field(default_factory=list)
    second_refinement_requests: list[SecondRefinementRequest] = Field(
        default_factory=list
    )

    initial_state: str = ""
    final_state: str = ""
    continuity_notes: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "FineBatchAnalysis":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if self.fine_batch_id != f"fine_{self.interval_id}":
            # Tolerate but warn via warnings list
            pass
        return self


# ---- Request models ----


class FineContactSheetInput(BaseModel):
    sheet_id: str
    path: Path
    frame_ids: list[str] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    reading_order: Literal["row_major"] = "row_major"
    overlap_with_previous: list[str] = Field(default_factory=list)
    sha256: str = ""

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


class FineDetailFrameInput(BaseModel):
    frame_id: str
    timestamp_seconds: float = Field(ge=0.0)
    path: Path
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    resolution_mode: str = "source_original"
    selection_reason: str = ""
    sha256: str = ""

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


class FineCoarseContext(BaseModel):
    source_batch_ids: list[str] = Field(default_factory=list)
    evidence_frame_ids: list[str] = Field(default_factory=list)
    coarse_summaries: list[str] = Field(default_factory=list)
    coarse_refinement_reasons: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class FineVisionRequest(BaseModel):
    """Unified request sent to a FineVisionProvider for one fine batch."""

    schema_version: str = FINE_SCHEMA_VERSION
    fine_batch_id: str
    video_id: str = "video"
    interval: RefinementSource
    contact_sheets: list[FineContactSheetInput] = Field(default_factory=list)
    detail_frames: list[FineDetailFrameInput] = Field(default_factory=list)
    subtitles: list[SubtitleContext] = Field(default_factory=list)
    subtitle_metadata: SubtitleMetadata = Field(default_factory=SubtitleMetadata)
    user_request: str = ""
    refinement_questions: list[str] = Field(default_factory=list)
    coarse_context: FineCoarseContext = Field(default_factory=FineCoarseContext)
    analysis_dimensions: list[str] = Field(default_factory=list)


# ---- Manifest ----


class FineBatchRunRecord(BaseModel):
    fine_batch_id: str
    interval_id: str
    provider: str
    model: str
    prompt_version: str
    status: Literal["ok", "failed", "cached", "skipped"] = "ok"
    attempts: int = 1
    error: str | None = None
    raw_response_path: str | None = None
    analysis_path: str | None = None
    cache_key: str | None = None
    event_count: int = 0
    question_count: int = 0
    second_refinement_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class FineAnalysisManifest(BaseModel):
    schema_version: str = FINE_SCHEMA_VERSION
    video_id: str = "video"
    provider: str = "mock"
    model: str = ""
    prompt_version: str = "fine-temporal-vision-v1"
    user_request: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    batches: list[FineBatchRunRecord] = Field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0
    cached_count: int = 0
    skipped_count: int = 0
    total_batches: int = 0
    allow_partial: bool = False
    config_fingerprint: str | None = None

    def recompute_counts(self) -> None:
        self.success_count = sum(1 for b in self.batches if b.status == "ok")
        self.cached_count = sum(1 for b in self.batches if b.status == "cached")
        self.failed_count = sum(1 for b in self.batches if b.status == "failed")
        self.skipped_count = sum(1 for b in self.batches if b.status == "skipped")
        self.total_batches = len(self.batches)
        self.updated_at = utc_now()


class FineAnalysisResult(BaseModel):
    schema_version: str = FINE_SCHEMA_VERSION
    status: Literal[
        "completed",
        "completed_no_batches",
        "partial",
        "failed",
    ]
    total_batches: int = 0
    succeeded_batches: int = 0
    failed_batches: int = 0
    cached_batches: int = 0
    total_events: int = 0
    fine_video_analysis_path: Path
    manifest_path: Path
    temporal_events_path: Path
    second_refinement_plan_path: Path | None = None
    failed_batch_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "fine_video_analysis_path",
        "manifest_path",
        "temporal_events_path",
        "second_refinement_plan_path",
        mode="before",
    )
    @classmethod
    def _coerce_paths(cls, value: Any) -> Any:
        if value is None:
            return None
        return Path(value)


# ---- Aggregated full-video analysis ----


class TimeRange(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "TimeRange":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class FineVideoAnalysis(BaseModel):
    schema_version: str = FINE_SCHEMA_VERSION
    video_start: float = 0.0
    video_end: float = 0.0
    analyzed_intervals: list[TimeRange] = Field(default_factory=list)
    analysis_gaps: list[TimeRange] = Field(default_factory=list)

    temporal_events: list[TemporalEvent] = Field(default_factory=list)
    subject_motion_summary: list[str] = Field(default_factory=list)
    camera_motion_summary: list[str] = Field(default_factory=list)
    text_animation_summary: list[str] = Field(default_factory=list)
    graphic_animation_summary: list[str] = Field(default_factory=list)
    transition_summary: list[str] = Field(default_factory=list)

    answered_questions: list[RefinementQuestionAnswer] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    important_events: list[str] = Field(default_factory=list)

    second_refinement_required: bool = False
    second_refinement_plan_path: Path | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("second_refinement_plan_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Any:
        if value is None:
            return None
        return Path(value)


# ---- Second refinement plan ----


class SecondRefinementInterval(BaseModel):
    interval_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    requested_fps: SecondRefinementFps
    priority: PriorityLevel = "medium"
    reasons: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    source_batch_ids: list[str] = Field(default_factory=list)
    evidence_frame_ids: list[str] = Field(default_factory=list)
    exact_detail_timestamps: list[float] = Field(default_factory=list)
    estimated_frame_count: int = 0

    @model_validator(mode="after")
    def _check(self) -> "SecondRefinementInterval":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class SecondRefinementPlan(BaseModel):
    schema_version: str = FINE_SCHEMA_VERSION
    video_duration: float = 0.0
    round1_refinement_duration: float = 0.0
    total_requested_duration: float = 0.0
    budget_duration: float = 0.0
    max_additional_rounds: int = 1
    terminal_after_this_round: bool = True
    intervals: list[SecondRefinementInterval] = Field(default_factory=list)
    deferred_intervals: list[SecondRefinementInterval] = Field(default_factory=list)
    generation_warnings: list[str] = Field(default_factory=list)


__all__ = [
    "BoundaryPrecision",
    "CameraMotionAnalysis",
    "CameraMotionType",
    "EventCategory",
    "FINE_SCHEMA_VERSION",
    "FineAnalysisManifest",
    "FineAnalysisResult",
    "FineBatchAnalysis",
    "FineBatchRunRecord",
    "FineContactSheetInput",
    "FineCoarseContext",
    "FineDetailFrameInput",
    "FineEvidenceReference",
    "FineVideoAnalysis",
    "FineVisionRequest",
    "FrameStateObservation",
    "GraphicAnimationAnalysis",
    "PostTransformAnalysis",
    "PostTransformType",
    "PriorityLevel",
    "RefinementQuestionAnswer",
    "SecondRefinementFps",
    "SecondRefinementInterval",
    "SecondRefinementPlan",
    "SecondRefinementRequest",
    "SecondRefinementRequestType",
    "SubjectMotionAnalysis",
    "TextAnimationAnalysis",
    "TextAnimationType",
    "TemporalBoundary",
    "TemporalEvent",
    "TimeRange",
    "TransitionAnalysis",
    "TransitionType",
]
