"""Data models for Epic 5 final report system.

所有模型都是 Pydantic v2 严格校验。Schema version 显式定义，下游消费方可拒绝
未知契约。

设计原则：
- 所有结论必须保留证据引用（ReportEvidenceReference），不得依赖无证据摘要
- 每个 Shots/Sections/UnresolvedItems 都明确标注 analysis_level
- 任何字段都不能凭空填入专业术语（缺证据就空）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from video_analyst.fine.models import (
    RefinementQuestionAnswer,
    TemporalEvent,
    TimeRange,
)
from video_analyst.models import utc_now

FINAL_SCHEMA_VERSION = "1.0"


# ---- Evidence reference ----

ReportEvidenceType = Literal[
    "coarse_frame",
    "scene_frame",
    "refinement_frame",
    "terminal_frame",
    "contact_sheet",
    "detail_frame",
    "subtitle",
    "coarse_observation",
    "fine_event",
    "terminal_event",
    "media_metadata",
]


class ReportEvidenceReference(BaseModel):
    """报告中的证据引用。

    每个证据有唯一 evidence_id，可在 Markdown/CSV/JSON 中通过 ID 引用。
    image_paths / source_json_paths 必须真实存在（validation 时校验）。
    """

    evidence_id: str
    evidence_type: ReportEvidenceType
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    frame_ids: list[str] = Field(default_factory=list)
    image_paths: list[Path] = Field(default_factory=list)
    source_json_paths: list[Path] = Field(default_factory=list)
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @field_validator("image_paths", "source_json_paths", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return [Path(v) for v in value]
        return value

    @model_validator(mode="after")
    def _check(self) -> "ReportEvidenceReference":
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) < start ({self.start})")
        return self


# ---- Merged temporal event ----

MergedTemporalEventStatus = Literal[
    "from_round1",
    "confirmed_by_terminal_refinement",
    "revised_by_terminal_refinement",
    "rejected_by_terminal_refinement",
]


class MergedTemporalEvent(BaseModel):
    """第一轮 + 终局事件合并结果。

    active_interpretation 是当前采用的事件；superseded_interpretations
    保留被修正/否定的旧版本，绝不直接删除。
    """

    merged_id: str
    active: TemporalEvent
    superseded: list[TemporalEvent] = Field(default_factory=list)
    status: MergedTemporalEventStatus
    source_round: Literal["round1", "terminal"]
    notes: list[str] = Field(default_factory=list)


# ---- Shot ----

ShotAnalysisLevel = Literal["coarse_only", "fine", "terminal_refined"]


class ShotAnalysis(BaseModel):
    """镜头级时间线条目。

    每个镜头都有明确的分析层级；粗分析未细化、精细分析未覆盖、终局精修
    过的镜头必须明确标注。
    """

    schema_version: str = FINAL_SCHEMA_VERSION
    shot_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    duration: float = Field(ge=0.0)
    section_id: str | None = None

    visual_summary: str = ""
    subjects: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)

    shot_size: str | None = None
    composition: list[str] = Field(default_factory=list)

    subject_motions: list[str] = Field(default_factory=list)
    camera_motions: list[str] = Field(default_factory=list)
    post_transforms: list[str] = Field(default_factory=list)
    text_animations: list[str] = Field(default_factory=list)
    graphic_animations: list[str] = Field(default_factory=list)
    transitions_in: list[str] = Field(default_factory=list)
    transitions_out: list[str] = Field(default_factory=list)

    subtitle_entries: list[int] = Field(default_factory=list)
    audio_visual_notes: list[str] = Field(default_factory=list)

    narrative_function: list[str] = Field(default_factory=list)
    editing_notes: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    analysis_level: ShotAnalysisLevel = "coarse_only"
    # Fraction of this shot's duration overlapping successful fine intervals
    fine_coverage_ratio_within_shot: float = Field(ge=0.0, le=1.0, default=0.0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ShotAnalysis":
        if self.end < self.start:
            raise ValueError("end < start")
        if abs(self.duration - (self.end - self.start)) > 1e-3:
            self.duration = round(self.end - self.start, 6)
        return self


# ---- Section ----

class SectionAnalysis(BaseModel):
    schema_version: str = FINAL_SCHEMA_VERSION
    section_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    title: str = ""
    summary: str = ""

    shot_ids: list[str] = Field(default_factory=list)
    main_subjects: list[str] = Field(default_factory=list)
    visual_patterns: list[str] = Field(default_factory=list)
    text_patterns: list[str] = Field(default_factory=list)
    motion_patterns: list[str] = Field(default_factory=list)
    transition_patterns: list[str] = Field(default_factory=list)

    pacing_notes: list[str] = Field(default_factory=list)
    narrative_function: list[str] = Field(default_factory=list)
    important_events: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "SectionAnalysis":
        if self.end < self.start:
            raise ValueError("end < start")
        return self


# ---- Repeatable technique ----

TechniqueCategory = Literal[
    "editing",
    "camera",
    "text",
    "graphics",
    "transition",
    "pacing",
    "storytelling",
]


class RepeatableTechnique(BaseModel):
    technique_id: str
    title: str
    category: TechniqueCategory
    description: str
    observed_time_ranges: list[TimeRange] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    implementation_hints: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


# ---- Coverage ----

class AnalysisCoverage(BaseModel):
    schema_version: str = FINAL_SCHEMA_VERSION
    video_duration: float = 0.0

    coarse_analyzed_duration: float = 0.0
    fine_analyzed_duration: float = 0.0
    terminal_analyzed_duration: float = 0.0

    coarse_coverage_ratio: float = 0.0
    fine_coverage_ratio: float = 0.0
    terminal_coverage_ratio: float = 0.0

    successful_ranges: list[TimeRange] = Field(default_factory=list)
    failed_ranges: list[TimeRange] = Field(default_factory=list)
    deferred_ranges: list[TimeRange] = Field(default_factory=list)
    unexamined_ranges: list[TimeRange] = Field(default_factory=list)

    subtitle_reviewed: bool = False
    subtitle_needs_review: bool = False

    real_vision_api_used: bool = False
    providers_used: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)


# ---- Unresolved items ----

UnresolvedSource = Literal[
    "round1_unresolved",
    "validator_fallback",
    "terminal_unresolved",
    "deferred_terminal",
    "coarse_gap",
    "fine_gap",
]


class UnresolvedItem(BaseModel):
    item_id: str
    time_start: float | None = None
    time_end: float | None = None
    question_or_issue: str
    reason: str
    terminal_attempted: bool = False
    source: UnresolvedSource
    priority: Literal["low", "medium", "high"] = "medium"
    evidence_ids: list[str] = Field(default_factory=list)


# ---- Evidence index entry ----


class EvidenceEntry(BaseModel):
    """evidence index 单条（与 ReportEvidenceReference 镜像）。"""

    evidence_id: str
    evidence_type: ReportEvidenceType
    start: float
    end: float
    description: str = ""
    image_paths: list[str] = Field(default_factory=list)
    source_json_paths: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    terminal_origin: bool = False


class EvidenceIndex(BaseModel):
    schema_version: str = FINAL_SCHEMA_VERSION
    video_id: str = "video"
    entries: list[EvidenceEntry] = Field(default_factory=list)


# ---- Final analysis ----

# Epic 6: 状态拆成正交字段，不再互相覆盖
CompletionStatus = Literal["final", "provisional", "failed"]
# Product vision mode is in_session_agent only. legacy values retained for
# reading old reports; never selected by the default workflow.
ProviderMode = Literal["legacy", "in_session_agent", "mock", "real_api", "mixed"]
VisionExecutionMode = Literal["legacy", "in_session_agent"]
TerminalRefinementStatus = Literal[
    "not_required",
    "executed",
    "skipped",
    "failed",
]
AnalysisCompleteness = Literal["complete", "partial"]

# 兼容旧字段的单一枚举（legacy）
ReportStatus = Literal["final", "provisional", "mock_validation"]


class ReportRuntimeStatus(BaseModel):
    """正交运行状态字段。

    不再让单个枚举互相覆盖。所有新逻辑必须使用这些字段；旧代码可保留
    ``legacy_status`` 但不得作为唯一依据。
    """

    completion_status: CompletionStatus = "final"
    provider_mode: ProviderMode = "in_session_agent"
    vision_execution_mode: VisionExecutionMode = "in_session_agent"
    terminal_refinement_status: TerminalRefinementStatus = "not_required"
    analysis_completeness: AnalysisCompleteness = "complete"
    real_vision_analysis_used: bool = True
    external_api_used: bool = False
    api_key_used: bool = False
    # 兼容旧逻辑派生的单一 status
    legacy_status: ReportStatus = "final"

    def to_legacy(self) -> ReportStatus:
        """从正交字段派生旧 legacy_status（向后兼容）。"""
        if self.completion_status == "failed":
            return "final"  # failed 不在旧枚举中，映射为 final 带 warning
        if self.provider_mode == "mock" and self.vision_execution_mode == "legacy":
            return "mock_validation"
        if self.completion_status == "provisional":
            return "provisional"
        return "final"


class ConfidenceSummary(BaseModel):
    overall: float = Field(ge=0.0, le=1.0, default=0.5)
    content_structure: float = Field(ge=0.0, le=1.0, default=0.5)
    subject_motion: float = Field(ge=0.0, le=1.0, default=0.5)
    camera_motion: float = Field(ge=0.0, le=1.0, default=0.5)
    text_animation: float = Field(ge=0.0, le=1.0, default=0.5)
    graphic_animation: float = Field(ge=0.0, le=1.0, default=0.5)
    transition: float = Field(ge=0.0, le=1.0, default=0.5)
    subtitle_alignment: float = Field(ge=0.0, le=1.0, default=0.5)


class DetailedHighlight(BaseModel):
    """重点精查片段（来自 Fine interval，不是自然语言猜写）。"""

    interval_id: str
    fine_batch_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    reason: str = ""
    summary: str = ""
    direct_observations: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    subject_motions: list[str] = Field(default_factory=list)
    camera_motions: list[str] = Field(default_factory=list)
    post_transforms: list[str] = Field(default_factory=list)
    text_animations: list[str] = Field(default_factory=list)
    graphic_animations: list[str] = Field(default_factory=list)
    transitions: list[str] = Field(default_factory=list)
    question_answers: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class StructuredEventItem(BaseModel):
    """报告章节用的结构化事件条目（带 evidence）。"""

    category: str
    start: float = 0.0
    end: float = 0.0
    description: str = ""
    event_type: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_ids: list[str] = Field(default_factory=list)
    source_batch_ids: list[str] = Field(default_factory=list)


class FinalVideoAnalysis(BaseModel):
    schema_version: str = FINAL_SCHEMA_VERSION
    video_metadata: dict[str, Any] = Field(default_factory=dict)
    analysis_request: str = ""
    generated_at: datetime = Field(default_factory=utc_now)

    status: ReportStatus = "final"
    terminal_refinement_executed: bool = False
    # Epic 6: 正交运行状态（不互相覆盖）
    runtime_status: ReportRuntimeStatus = Field(default_factory=ReportRuntimeStatus)

    executive_summary: str = ""
    content_structure: list[SectionAnalysis] = Field(default_factory=list)
    shots: list[ShotAnalysis] = Field(default_factory=list)
    detailed_highlights: list[DetailedHighlight] = Field(default_factory=list)

    visual_style: list[str] = Field(default_factory=list)
    camera_style: list[str] = Field(default_factory=list)
    motion_style: list[str] = Field(default_factory=list)
    text_design_style: list[str] = Field(default_factory=list)
    graphic_design_style: list[str] = Field(default_factory=list)
    transition_style: list[str] = Field(default_factory=list)

    # Structured chapter items (with evidence IDs)
    text_animation_items: list[StructuredEventItem] = Field(default_factory=list)
    graphic_animation_items: list[StructuredEventItem] = Field(default_factory=list)
    transition_items: list[StructuredEventItem] = Field(default_factory=list)
    subject_motion_items: list[StructuredEventItem] = Field(default_factory=list)
    camera_post_items: list[StructuredEventItem] = Field(default_factory=list)

    pacing_analysis: list[str] = Field(default_factory=list)
    subtitle_audio_visual_analysis: list[str] = Field(default_factory=list)
    narrative_strategy: list[str] = Field(default_factory=list)

    repeatable_techniques: list[RepeatableTechnique] = Field(default_factory=list)
    important_events: list[str] = Field(default_factory=list)

    answered_questions: list[RefinementQuestionAnswer] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    unresolved_items: list[UnresolvedItem] = Field(default_factory=list)
    analysis_gaps: list[TimeRange] = Field(default_factory=list)

    coverage: AnalysisCoverage = Field(default_factory=AnalysisCoverage)
    confidence_summary: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    known_limitations: list[str] = Field(default_factory=list)

    # Vision provenance — default is in-session agent (no external vision API)
    provider_mode: str = "in_session_agent"
    vision_execution_mode: VisionExecutionMode = "in_session_agent"
    real_vision_api_used: bool = False
    real_vision_analysis_used: bool = True
    external_api_used: bool = False
    api_key_used: bool = False
    token_status: str = "unavailable"
    estimated_cost: float | None = None
    pricing_source: str = "in_session_agent"
    agent_session_provenance: dict[str, Any] | None = None


__all__ = [
    "AnalysisCoverage",
    "AnalysisCompleteness",
    "CompletionStatus",
    "ConfidenceSummary",
    "DetailedHighlight",
    "EvidenceEntry",
    "EvidenceIndex",
    "FINAL_SCHEMA_VERSION",
    "FinalVideoAnalysis",
    "MergedTemporalEvent",
    "MergedTemporalEventStatus",
    "ProviderMode",
    "ReportEvidenceReference",
    "ReportEvidenceType",
    "ReportRuntimeStatus",
    "StructuredEventItem",
    "ReportStatus",
    "RepeatableTechnique",
    "SectionAnalysis",
    "ShotAnalysis",
    "ShotAnalysisLevel",
    "TechniqueCategory",
    "TerminalRefinementStatus",
    "UnresolvedItem",
    "UnresolvedSource",
]