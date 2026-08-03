"""Data models for Epic 3 — refinement / fine-frame extraction.

All public models are Pydantic v2 with strict validation. Schema versions
are explicit so downstream Epic 4 can refuse to consume unknown contracts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from video_analyst.models import utc_now
from video_analyst.vision.models import SubtitleContext, SubtitleMetadata

EPIC3_SCHEMA_VERSION = "1.0"

AllowedFPS = Literal[5, 10, 15]
PriorityLevel = Literal["low", "medium", "high"]


# ---- Source interval (re-exported plan shape with extra strict fields) ----


class RefinementSource(BaseModel):
    """The subset of a refinement-plan interval that drives fine extraction."""

    interval_id: str
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    requested_fps: AllowedFPS
    priority: PriorityLevel = "medium"
    reasons: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    source_batch_ids: list[str] = Field(default_factory=list)
    evidence_frame_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "RefinementSource":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self


# ---- Frames extracted for a single interval ----


def _hms_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class RefinementFrame(BaseModel):
    """A single fine-grained frame."""

    schema_version: str = EPIC3_SCHEMA_VERSION
    frame_id: str
    interval_id: str
    local_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    timecode: str
    relative_timestamp_seconds: float = Field(ge=0.0)
    requested_fps: AllowedFPS
    path: Path
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=1)
    source_video_sha256: str = Field(min_length=1)
    is_detail: bool = False

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    @classmethod
    def make(
        cls,
        *,
        interval_id: str,
        local_index: int,
        timestamp_seconds: float,
        relative_timestamp_seconds: float,
        requested_fps: AllowedFPS,
        path: Path,
        width: int,
        height: int,
        size_bytes: int,
        sha256: str,
        source_video_sha256: str,
        is_detail: bool = False,
        frame_id: str | None = None,
    ) -> "RefinementFrame":
        if frame_id is None:
            frame_id = (
                f"{interval_id}__f{int(round(timestamp_seconds * 1000)):08d}"
            )
        return cls(
            frame_id=frame_id,
            interval_id=interval_id,
            local_index=local_index,
            timestamp_seconds=round(timestamp_seconds, 6),
            timecode=_hms_ms(timestamp_seconds),
            relative_timestamp_seconds=round(relative_timestamp_seconds, 6),
            requested_fps=requested_fps,
            path=path,
            width=width,
            height=height,
            size_bytes=size_bytes,
            sha256=sha256,
            source_video_sha256=source_video_sha256,
            is_detail=is_detail,
        )


# ---- Contact sheet ----


class TemporalContactSheet(BaseModel):
    """A deterministic 2×N mosaic of objective-labeled frames."""

    schema_version: str = EPIC3_SCHEMA_VERSION
    sheet_id: str
    interval_id: str
    frame_ids: list[str] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    path: Path
    columns: int = Field(ge=1)
    rows: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    overlap_with_previous: list[str] = Field(default_factory=list)
    sha256: str = Field(min_length=1)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    @model_validator(mode="after")
    def _check_consistency(self) -> "TemporalContactSheet":
        if len(self.frame_ids) != len(self.timestamps):
            raise ValueError(
                "frame_ids and timestamps must have the same length"
            )
        if self.rows * self.columns < len(self.frame_ids):
            raise ValueError("rows * columns < frame count")
        return self


# ---- Detail frame (high-resolution reference) ----


class DetailFrameReference(BaseModel):
    frame_id: str
    timestamp_seconds: float
    path: Path
    selection_reason: str
    # Epic 4 遗留修复：detail frame 现在单独从源视频高分辨率抽取。
    source_width: int = Field(default=0, ge=0)
    source_height: int = Field(default=0, ge=0)
    output_width: int = Field(default=0, ge=0)
    output_height: int = Field(default=0, ge=0)
    scaled: bool = False
    resolution_mode: str = "source_original"
    sha256: str = ""

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


# ---- Per-interval artifact (persisted to intervals/interval_XXXX.json) ----


class IntervalArtifact(BaseModel):
    """Full artifact for one refinement interval.

    Persisted to ``analysis/refinement/intervals/<interval_id>.json``.
    """

    schema_version: str = EPIC3_SCHEMA_VERSION
    interval_id: str
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None
    interval: RefinementSource
    frames: list[RefinementFrame] = Field(default_factory=list)
    contact_sheets: list[TemporalContactSheet] = Field(default_factory=list)
    detail_frames: list[DetailFrameReference] = Field(default_factory=list)
    extraction_signature: str
    presentation_signature: str
    extracted_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)


# ---- Fine batch ----


class FineVisualBatch(BaseModel):
    """The Epic 4-facing per-interval fine visual batch.

    Persisted to ``analysis/refinement/visual_inputs/batches/fine_<id>.json``
    and also indexed in ``fine_batches.json``.
    """

    schema_version: str = EPIC3_SCHEMA_VERSION
    fine_batch_id: str
    interval: RefinementSource
    frames: list[RefinementFrame] = Field(default_factory=list)
    contact_sheets: list[TemporalContactSheet] = Field(default_factory=list)
    detail_frames: list[DetailFrameReference] = Field(default_factory=list)
    subtitles: list[SubtitleContext] = Field(default_factory=list)
    subtitle_metadata: SubtitleMetadata = Field(default_factory=SubtitleMetadata)
    user_request: str = ""
    refinement_questions: list[str] = Field(default_factory=list)
    coarse_context: dict[str, Any] = Field(default_factory=dict)


class FineBatchesIndex(BaseModel):
    schema_version: str = EPIC3_SCHEMA_VERSION
    video_id: str = "video"
    total_batches: int = 0
    batches: list[FineVisualBatch] = Field(default_factory=list)


# ---- Epic 3 manifest + result ----


class IntervalManifestEntry(BaseModel):
    interval_id: str
    status: Literal["ok", "failed", "cached", "skipped"] = "ok"
    attempts: int = 1
    frame_count: int = 0
    contact_sheet_count: int = 0
    detail_frame_count: int = 0
    extraction_signature: str | None = None
    presentation_signature: str | None = None
    cache_key: str | None = None
    error: str | None = None
    completed_at: datetime = Field(default_factory=utc_now)


class Epic3Manifest(BaseModel):
    schema_version: str = EPIC3_SCHEMA_VERSION
    video_id: str = "video"
    source_video_sha256: str = ""
    video_duration: float = 0.0
    source_video_path: str = ""
    ffmpeg_version: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    intervals: list[IntervalManifestEntry] = Field(default_factory=list)
    total_intervals: int = 0
    succeeded_intervals: int = 0
    failed_intervals: int = 0
    cached_intervals: int = 0
    skipped_intervals: int = 0
    config_fingerprint: str | None = None
    plan_signature: str | None = None
    # Epic 5：标识这是哪一轮 refinement（round1 / terminal / ...）
    round_name: str = "round1"

    def recompute_counts(self) -> None:
        self.total_intervals = len(self.intervals)
        self.succeeded_intervals = sum(
            1 for e in self.intervals if e.status == "ok"
        )
        self.cached_intervals = sum(
            1 for e in self.intervals if e.status == "cached"
        )
        self.failed_intervals = sum(
            1 for e in self.intervals if e.status == "failed"
        )
        self.skipped_intervals = sum(
            1 for e in self.intervals if e.status == "skipped"
        )
        self.updated_at = utc_now()


class Epic3Result(BaseModel):
    schema_version: str = EPIC3_SCHEMA_VERSION
    status: Literal[
        "completed", "completed_no_intervals", "partial", "failed"
    ]
    total_intervals: int = 0
    succeeded_intervals: int = 0
    failed_intervals: int = 0
    cached_intervals: int = 0
    total_frames: int = 0
    total_contact_sheets: int = 0
    fine_batches_path: Path
    manifest_path: Path
    failed_interval_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("fine_batches_path", "manifest_path", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> Path:
        return Path(value)


__all__ = [
    "AllowedFPS",
    "DetailFrameReference",
    "EPIC3_SCHEMA_VERSION",
    "Epic3Manifest",
    "Epic3Result",
    "FineBatchesIndex",
    "FineVisualBatch",
    "IntervalArtifact",
    "IntervalManifestEntry",
    "PriorityLevel",
    "RefinementFrame",
    "RefinementSource",
    "TemporalContactSheet",
]