"""Data models for the coarse analysis pipeline (Epic 2).

These are the outputs that downstream stages (aggregation, refinement plan)
operate on. They wrap per-batch results from any VisionProvider into a
deterministic, provider-agnostic structure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from video_analyst.models import utc_now
from video_analyst.vision.models import CoarseBatchAnalysis, RefinementRequest

COARSE_SCHEMA_VERSION = "1.0"


# ---- Per-batch result and aggregated run ----


class BatchStatus(str):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"  # reserved for explicit skip (e.g. refresh-batch on missing)


class BatchRunRecord(BaseModel):
    """Persistent record of one batch execution."""

    schema_version: str = COARSE_SCHEMA_VERSION
    batch_id: str
    provider: str
    model: str
    prompt_version: str
    status: Literal["ok", "failed", "skipped"] = "ok"
    attempts: int = 1
    error: str | None = None
    raw_response_path: str | None = None  # relative to output root
    analysis_path: str | None = None  # relative to output root
    selected_frames: int = 0
    omitted_frames: int = 0
    cache_key: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class CoarseAnalysisManifest(BaseModel):
    """Top-level index of all batches in a coarse analysis run."""

    schema_version: str = COARSE_SCHEMA_VERSION
    video_id: str = "video"
    provider: str = "mock"
    model: str = ""
    prompt_version: str = "coarse-vision-v1"
    user_request: str = ""
    analysis_dimensions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    batches: list[BatchRunRecord] = Field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_batches: int = 0
    allow_partial: bool = False
    config_fingerprint: str | None = None

    def recompute_counts(self) -> None:
        self.success_count = sum(1 for b in self.batches if b.status == "ok")
        self.failed_count = sum(1 for b in self.batches if b.status == "failed")
        self.skipped_count = sum(1 for b in self.batches if b.status == "skipped")
        self.total_batches = len(self.batches)
        self.updated_at = utc_now()


class CoarseAnalysisResult(BaseModel):
    schema_version: str = COARSE_SCHEMA_VERSION
    video_id: str
    output_dir: str
    manifest: CoarseAnalysisManifest
    analyses: dict[str, CoarseBatchAnalysis] = Field(default_factory=dict)
    # batch_id -> reason for failure
    failed_batches: dict[str, str] = Field(default_factory=dict)
    structure_path: str | None = None
    refinement_plan_path: str | None = None
    allow_partial: bool = False

    @property
    def total(self) -> int:
        return self.manifest.total_batches

    @property
    def succeeded(self) -> int:
        return self.manifest.success_count

    @property
    def failed(self) -> int:
        return self.manifest.failed_count

    def is_success(self) -> bool:
        if self.allow_partial:
            return self.succeeded > 0
        return self.failed == 0 and self.succeeded > 0


# ---- Full-video aggregation ----


class VideoSection(BaseModel):
    section_id: str
    start: float
    end: float
    title: str
    summary: str
    source_batch_ids: list[str] = Field(default_factory=list)
    important_moments: list["SectionTimeRange"] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class SectionTimeRange(BaseModel):
    start: float
    end: float

    @model_validator(mode="after")
    def _check(self) -> SectionTimeRange:
        if self.end < self.start:
            raise ValueError("end < start")
        return self


class VideoStructure(BaseModel):
    schema_version: str = COARSE_SCHEMA_VERSION
    video_start: float
    video_end: float
    overall_summary: str
    sections: list[VideoSection] = Field(default_factory=list)
    recurring_subjects: list[str] = Field(default_factory=list)
    recurring_visual_patterns: list[str] = Field(default_factory=list)
    subtitle_visual_notes: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    failed_batches: list[str] = Field(default_factory=list)
    gaps: list["VideoGap"] = Field(default_factory=list)


class VideoGap(BaseModel):
    """Marks a contiguous time range with no successful batch coverage."""

    start: float
    end: float
    failed_batch_ids: list[str] = Field(default_factory=list)
    reason: str = "failed_batches"


# ---- Refinement plan ----


class RefinementInterval(BaseModel):
    interval_id: str
    start: float
    end: float
    requested_fps: Literal[5, 10, 15]
    priority: Literal["low", "medium", "high"]
    reasons: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    source_batch_ids: list[str] = Field(default_factory=list)
    evidence_frame_ids: list[str] = Field(default_factory=list)
    estimated_frame_count: int = 0


class RefinementPlan(BaseModel):
    schema_version: str = COARSE_SCHEMA_VERSION
    video_duration: float
    total_requested_duration: float
    budget_duration: float
    intervals: list[RefinementInterval] = Field(default_factory=list)
    deferred_intervals: list[RefinementInterval] = Field(default_factory=list)
    generation_warnings: list[str] = Field(default_factory=list)


# ---- Internal helpers ----


def request_to_interval(
    req: RefinementRequest,
    *,
    interval_id: str,
    source_batch_id: str,
) -> RefinementInterval:
    """Convert a single per-batch refinement request to an internal interval."""
    est = max(0, int(round((req.end - req.start) * req.requested_fps)))
    return RefinementInterval(
        interval_id=interval_id,
        start=req.start,
        end=req.end,
        requested_fps=req.requested_fps,
        priority=req.priority,
        reasons=[req.reason],
        questions=list(req.questions),
        source_batch_ids=[source_batch_id],
        evidence_frame_ids=list(req.evidence_frame_ids),
        estimated_frame_count=est,
    )


__all__ = [
    "BatchRunRecord",
    "CoarseAnalysisManifest",
    "CoarseAnalysisResult",
    "COARSE_SCHEMA_VERSION",
    "RefinementInterval",
    "RefinementPlan",
    "SectionTimeRange",
    "VideoGap",
    "VideoSection",
    "VideoStructure",
    "request_to_interval",
]


# Resolve forward references
VideoSection.model_rebuild()