"""Shared Pydantic models for Video Analyst (subtitles + phase contracts)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ASREngine(str, Enum):
    GROQ_WHISPER_LARGE_V3_TURBO = "groq-whisper-large-v3-turbo"
    LOCAL_WHISPER_SMALL = "local-whisper-small"


class VideoStreamInfo(BaseModel):
    index: int
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    duration: float | None = None
    bit_rate: int | None = None
    pix_fmt: str | None = None


class AudioStreamInfo(BaseModel):
    index: int
    codec_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    duration: float | None = None
    bit_rate: int | None = None
    language: str | None = None


class MediaInfo(BaseModel):
    path: Path
    format_name: str | None = None
    format_long_name: str | None = None
    duration: float | None = None
    size: int | None = None
    bit_rate: int | None = None
    video_streams: list[VideoStreamInfo] = Field(default_factory=list)
    audio_streams: list[AudioStreamInfo] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


class SubtitleSegment(BaseModel):
    """Unified subtitle entry used across ASR / LLM / active exports."""

    index: int
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str
    language: str | None = None
    is_code_switched: bool = False
    source_segment_ids: list[str] = Field(default_factory=list)
    source: Literal["asr", "llm"] = "asr"
    merge_reason: str | None = None
    calibration_status: str | None = None  # accepted|rejected|needs_review|unchanged

    @field_validator("text", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @model_validator(mode="after")
    def _check_times(self) -> SubtitleSegment:
        if self.start < 0:
            raise ValueError(f"segment {self.index}: start < 0")
        if self.end < self.start:
            raise ValueError(f"segment {self.index}: end < start")
        if not self.text:
            raise ValueError(f"segment {self.index}: empty text")
        return self


# Backward-compatible alias used in specs
SubtitleEntry = SubtitleSegment


class SubtitleDocument(BaseModel):
    version: str = "1.0"
    source_video: str
    audio_path: str | None = None
    language: str | None = None
    engine: ASREngine | str | None = None
    duration: float | None = None
    segments: list[SubtitleSegment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    origin: Literal["asr", "llm", "active"] = "asr"

    def validate_timeline(self, video_duration: float | None = None, tol: float = 0.75) -> None:
        prev_end = -1.0
        for i, seg in enumerate(self.segments, start=1):
            if seg.index != i:
                # allow non-renumbered intermediate docs; only check order of times
                pass
            if seg.start < prev_end - 1e-3:
                raise ValueError(
                    f"non-monotonic timestamps at index={seg.index}: "
                    f"start={seg.start} < prev_end={prev_end}"
                )
            if video_duration is not None and seg.end > video_duration + tol:
                raise ValueError(
                    f"segment {seg.index} ends after video duration "
                    f"({seg.end} > {video_duration}+{tol})"
                )
            prev_end = max(prev_end, seg.end)


class SubtitleVersion(str, Enum):
    ASR = "asr"
    LLM = "llm"


class CalibrationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    UNCHANGED = "unchanged"


class SegmentDiff(BaseModel):
    index: int
    start: float
    end: float
    language: str | None = None
    asr_text: str = ""
    llm_text: str = ""
    active_text: str | None = None
    change_ratio: float = 0.0
    status: CalibrationStatus = CalibrationStatus.UNCHANGED
    risk_reasons: list[str] = Field(default_factory=list)
    note: str | None = None
    changed: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self.active_text is None:
            if self.status in {
                CalibrationStatus.ACCEPTED,
                CalibrationStatus.NEEDS_REVIEW,
            }:
                self.active_text = self.llm_text
            else:
                self.active_text = self.asr_text
        self.changed = self.asr_text.strip() != (self.llm_text or "").strip()


class ASRMeta(BaseModel):
    provider: str = "groq"
    model: str = "whisper-large-v3-turbo"
    multilingual: bool = True
    prompt_version: str = "asr-multilingual-v1"


class LLMMeta(BaseModel):
    enabled: bool = True
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    prompt_version: str = "subtitle-calibration-v1"
    status: str = "completed"  # completed|failed|skipped
    error: str | None = None


class ReviewSummary(BaseModel):
    total_entries: int = 0
    unchanged: int = 0
    accepted: int = 0
    rejected: int = 0
    needs_review: int = 0


class SourceHashes(BaseModel):
    video: str | None = None
    audio: str | None = None
    asr_subtitles: str | None = None
    config: str | None = None


class SubtitleReviewPackage(BaseModel):
    schema_version: str = "1.0"
    version: str = "1.0"  # legacy
    source_video: str
    active: SubtitleVersion = SubtitleVersion.LLM
    reviewed: bool = False
    needs_review: bool = False
    asr: SubtitleDocument
    llm: SubtitleDocument | None = None
    diffs: list[SegmentDiff] = Field(default_factory=list)
    asr_meta: ASRMeta = Field(default_factory=ASRMeta)
    llm_meta: LLMMeta = Field(default_factory=LLMMeta)
    source_hashes: SourceHashes = Field(default_factory=SourceHashes)
    summary: ReviewSummary = Field(default_factory=ReviewSummary)
    # legacy fields
    llm_model: str | None = None
    llm_error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def active_document(self) -> SubtitleDocument:
        if self.active == SubtitleVersion.LLM and self.llm is not None:
            return self.llm
        return self.asr

    def recompute_summary(self) -> None:
        s = ReviewSummary(total_entries=len(self.diffs))
        for d in self.diffs:
            if d.status == CalibrationStatus.ACCEPTED:
                s.accepted += 1
            elif d.status == CalibrationStatus.REJECTED:
                s.rejected += 1
            elif d.status == CalibrationStatus.NEEDS_REVIEW:
                s.needs_review += 1
            else:
                s.unchanged += 1
        self.summary = s
        self.needs_review = s.needs_review > 0 or self.needs_review


class ASRChunkRecord(BaseModel):
    chunk_id: str
    audio_path: str
    absolute_start: float
    absolute_end: float
    overlap_before: float = 0.0
    overlap_after: float = 0.0
    detected_language: str | None = None
    is_code_switched: bool = False
    language_confidence: float | None = None
    provider: str = "groq"
    model: str = "whisper-large-v3-turbo"
    raw_response_path: str | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    error: str | None = None


class ASRManifest(BaseModel):
    schema_version: str = "1.0"
    source_video: str
    audio_path: str
    duration: float | None = None
    multilingual: bool = True
    language_lock: str | None = None
    provider: str = "groq"
    model: str = "whisper-large-v3-turbo"
    prompt_version: str = "asr-multilingual-v1"
    chunks: list[ASRChunkRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class PipelineStage(str, Enum):
    ENV_CHECK = "env_check"
    MEDIA_PROBE = "media_probe"
    AUDIO_EXTRACT = "audio_extract"
    AUDIO_CHUNKING = "audio_chunking"
    ASR_TRANSCRIPTION = "asr_transcription"
    ASR_TIME_OFFSET = "asr_time_offset"
    ASR_OVERLAP_DEDUPE = "asr_overlap_dedupe"
    ASR_LANGUAGE_MERGE = "asr_language_merge"
    SUBTITLE_RENDER_ASR = "subtitle_render_asr"
    LLM_CALIBRATE = "llm_calibrate"
    LLM_VALIDATE = "llm_validate"
    SUBTITLE_ACTIVATE = "subtitle_activate"
    PHASE2_FIXED_FRAMES = "phase2_fixed_frames"
    PHASE2_SCENE_FRAMES = "phase2_scene_frames"
    PHASE2_SUBTITLE_MAPPING = "phase2_subtitle_mapping"
    PHASE2_VISUAL_BATCHES = "phase2_visual_batches"
    # Epic 2 — Vision coarse analysis
    VISION_REQUEST_BUILD = "vision_request_build"
    VISION_BATCH_ANALYSIS = "vision_batch_analysis"
    VISION_BATCH_VALIDATION = "vision_batch_validation"
    COARSE_AGGREGATION = "coarse_aggregation"
    REFINEMENT_PLAN = "refinement_plan"
    # Epic 3 — Refinement / fine-frame extraction
    REFINEMENT_PLAN_LOAD = "refinement_plan_load"
    REFINEMENT_FRAME_EXTRACT = "refinement_frame_extract"
    REFINEMENT_TIMESTAMP_PARSE = "refinement_timestamp_parse"
    REFINEMENT_DEDUPLICATE = "refinement_deduplicate"
    TEMPORAL_CONTACT_SHEET = "temporal_contact_sheet"
    DETAIL_FRAME_SELECT = "detail_frame_select"
    FINE_BATCH_BUILD = "fine_batch_build"
    # Epic 4 — Fine temporal vision analysis
    FINE_REQUEST_BUILD = "fine_request_build"
    FINE_BATCH_ANALYSIS = "fine_batch_analysis"
    FINE_RESPONSE_VALIDATION = "fine_response_validation"
    FINE_EVENT_DEDUPLICATE = "fine_event_deduplicate"
    FINE_VIDEO_AGGREGATION = "fine_video_aggregation"
    SECOND_REFINEMENT_PLAN = "second_refinement_plan"
    # legacy aliases kept for older cache files
    ASR = "asr"
    SUBTITLES = "subtitles"
    DONE = "done"


class PipelineState(BaseModel):
    video_path: str
    output_dir: str
    completed_stages: list[PipelineStage] = Field(default_factory=list)
    media_info_path: str | None = None
    audio_path: str | None = None
    asr_manifest_path: str | None = None
    asr_raw_path: str | None = None
    subtitles_json_path: str | None = None
    subtitles_srt_path: str | None = None
    review_json_path: str | None = None
    active_subtitle_version: SubtitleVersion | None = None
    phase2_batches_path: str | None = None
    coarse_manifest_path: str | None = None
    coarse_structure_path: str | None = None
    refinement_plan_path: str | None = None
    refinement_manifest_path: str | None = None
    fine_batches_path: str | None = None
    fine_analysis_manifest_path: str | None = None
    fine_video_analysis_path: str | None = None
    fine_temporal_events_path: str | None = None
    second_refinement_plan_path: str | None = None
    engine: ASREngine | None = None
    config_fingerprint: str | None = None
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    def mark(self, stage: PipelineStage) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.updated_at = utc_now()
        self.last_error = None

    def has(self, stage: PipelineStage) -> bool:
        return stage in self.completed_stages

    def invalidate_from(self, stages: set[PipelineStage]) -> None:
        self.completed_stages = [s for s in self.completed_stages if s not in stages]
        self.updated_at = utc_now()


# ---- Phase 2 contracts (prep only) ----


class SubtitleMetadata(BaseModel):
    active_source: str = "asr"
    reviewed: bool = False
    needs_review: bool = False
    asr_model: str | None = None
    calibration_model: str | None = None


class VisualBatch(BaseModel):
    batch_id: str
    start: float
    end: float
    frames: list[dict[str, Any]] = Field(default_factory=list)
    subtitles: list[SubtitleSegment] = Field(default_factory=list)
    subtitle_metadata: SubtitleMetadata = Field(default_factory=SubtitleMetadata)
    user_request: str = ""
    analysis_instruction: dict[str, Any] = Field(
        default_factory=lambda: {
            "treat_subtitles_as_supporting_context": True,
            "subtitle_is_not_absolute_ground_truth": True,
        }
    )


class Phase2PrepareResult(BaseModel):
    fixed_frames_dir: str
    scene_frames_dir: str
    fixed_frame_count: int = 0
    scene_frame_count: int = 0
    batches_path: str
    batch_count: int = 0
    subtitle_metadata: SubtitleMetadata = Field(default_factory=SubtitleMetadata)
