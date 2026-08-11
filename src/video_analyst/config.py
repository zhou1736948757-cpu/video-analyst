"""Unified runtime configuration for ASR, calibration, and Phase 2 prep."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class ChunkingConfig:
    target_seconds: float = 25.0
    minimum_seconds: float = 3.0
    maximum_seconds: float = 50.0
    overlap_seconds: float = 0.5
    silence_duration: float = 0.5
    silence_threshold_db: float = -35.0


@dataclass
class MergeConfig:
    auto_merge_gap_seconds: float = 0.8
    conditional_merge_gap_seconds: float = 1.8
    maximum_merged_duration: float = 15.0
    maximum_text_length: int = 160


@dataclass
class ASRConfig:
    provider: str = "auto"  # auto | groq | local
    cloud_model: str = "whisper-large-v3-turbo"
    local_model: str = "small"
    multilingual: bool = True
    language: str | None = None  # None = auto
    prompt_version: str = "asr-multilingual-v1"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)


@dataclass
class CalibrationValidationConfig:
    accept_change_ratio: float = 0.30
    review_change_ratio: float = 0.50
    reject_translation: bool = True
    protect_numbers: bool = True
    protect_named_entities: bool = True
    preserve_fillers: bool = True
    # if fraction of high-risk/rejected entries exceeds this, force active=asr
    max_high_risk_fraction: float = 0.35


@dataclass
class CalibrationConfig:
    enabled: bool = True
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    prompt_version: str = "subtitle-calibration-v1"
    validation: CalibrationValidationConfig = field(
        default_factory=CalibrationValidationConfig
    )


@dataclass
class Phase2Config:
    """Preparation only — no visual model API calls in this phase."""

    enabled: bool = True
    coarse_fps: float = 1.0
    scene_threshold: float = 0.3  # ffmpeg select scene score
    max_scene_frames: int = 200
    batch_seconds: float = 20.0
    # Pixel-only coarse-input reduction.  This is deliberately independent of
    # scene semantics: only the previous kept frame's grayscale thumbnail is
    # compared, while scene and user-pinned frames are protected.
    coarse_dedupe_enabled: bool = True
    coarse_dedupe_threshold: float = 2.0
    coarse_dedupe_algorithm_version: str = "gray-thumbnail-delta-v1"
    coarse_dedupe_thumbnail_size: int = 16


@dataclass
class VisionConfig:
    """Legacy vision-provider config (NOT used by the product path).

    Product vision mode is always ``in_session_agent``: the current
    conversation Agent inspects frames. These fields remain only so older
    experimental modules can import AppConfig without breaking.
    """

    provider: str = "in_session_agent"  # product default; not an HTTP provider
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""  # unused in product path
    base_url_env: str = ""
    model_env: str = ""
    request_timeout: float = 120.0
    max_retries: int = 0
    retry_backoff_base: float = 2.0
    prompt_version: str = "coarse-vision-v1"
    batch_max_frames: int = 20


@dataclass
class CoarseAnalysisConfig:
    """Coarse analysis execution and refinement configuration."""

    allow_partial: bool = False
    refinement_budget_ratio: float = 0.20
    refinement_pad_seconds: float = 0.5
    refinement_max_interval_seconds: float = 15.0
    refinement_max_fps: int = 15
    use_aggregate_model: bool = False  # optional LLM summarization


@dataclass
class FinalAnalysisConfig:
    """Epic 5 final report configuration."""

    schema_version: str = "1.0"
    # evidence image copy mode: copy / symlink / none
    evidence_mode: str = "copy"
    # markdown language: zh / en / ja
    report_language: str = "zh"
    # generate markdown / csv outputs
    generate_markdown: bool = True
    generate_csv: bool = True
    generate_evidence_copy: bool = True
    # confidence aggregation: weight shots by duration
    weight_shots_by_duration: bool = True
    # provisional report allowed when second refinement not executed
    allow_provisional_report: bool = False
    # max number of shots (sane upper bound; 0 = unlimited)
    max_shots: int = 0


@dataclass
class RefinementConfig:
    """Epic 3 refinement configuration (frame extraction + presentation).

    RefinementConfig has two cache signature scopes:
    - Extraction signature: source_video_sha, interval start/end/FPS,
      FFmpeg version, scale (max_frame_long_edge), image format/quality,
      timestamp_parser_version, schema_version
    - Presentation signature: per-interval frame SHA-256 set,
      frames_per_sheet, overlap_frames, sheet columns, label format version,
      subtitle hash, review hash, user_request, refinement_questions,
      schema_version
    """

    # Extraction
    image_format: str = "jpg"  # jpg | png
    jpeg_quality: int = 2  # mapped to FFmpeg -q:v (lower = better)
    max_frame_long_edge: int = 1600  # px
    timestamp_parser_version: str = "showinfo-pts-time-v1"
    ffmpeg_timeout: float = 120.0
    # Deduplication
    dedup_time_threshold_seconds: float = 0.005
    # Detail frames — Epic 4 遗留修复：detail frame 必须是真高分辨率，
    # 单独从源视频抽取，不复用 1600 缩放后的普通 refinement frame。
    # 0 = source-original（不缩放）；>0 = capped_high_resolution 上限。
    detail_max_per_interval: int = 2  # first + last by default
    detail_frame_max_long_edge: int = 4096  # px; 0 = source-original
    detail_frame_jpeg_quality: int = 2
    # Contact sheet
    sheets_enabled: bool = True
    frames_per_sheet: int = 4
    sheet_overlap_frames: int = 1
    sheet_columns: int = 2
    contact_sheet_max_edge: int = 4096
    contact_sheet_label_format_version: str = "objective-v1"
    # Subtitle context for fine batch
    subtitle_context_before: float = 1.0
    subtitle_context_after: float = 1.0
    # Schema version of any Epic 3 artifact (kept here for stable keys)
    schema_version: str = "1.0"
    # Allow partial success
    allow_partial: bool = False
    # Disable sheet generation (still produces fine batches with frames only)
    no_contact_sheets: bool = False


@dataclass
class FineAnalysisConfig:
    """Epic 4 fine temporal vision analysis configuration.

    复用 VisionConfig 的 provider / api_key_env / base_url_env / model_env /
    request_timeout / max_retries / retry_backoff_base。本 dataclass 只描述
    fine 分析特有的配置：prompt 版本、图片发送上限、二轮补帧预算等。
    """

    prompt_version: str = "fine-temporal-vision-v1"
    fine_schema_version: str = "1.0"
    # 图片发送策略：默认全部 contact sheets + 最多 2 张 detail frame
    max_detail_frames_per_request: int = 2
    # 若没有 contact sheets，从精细帧中选代表帧的数量
    representative_frame_count: int = 6
    # 二轮补帧预算：min(video_duration * ratio, round1_total * round1_ratio)
    second_refinement_budget_ratio: float = 0.05
    second_refinement_round1_ratio: float = 0.25
    second_refinement_max_interval_seconds: float = 5.0
    second_refinement_max_fps: int = 15  # 只允许 10 或 15
    # 允许 partial success
    allow_partial: bool = False
    # 事件去重
    event_dedup_time_threshold_seconds: float = 0.5
    # 分析维度（默认全集）
    default_analysis_dimensions: tuple[str, ...] = (
        "subject-motion",
        "camera-motion",
        "post-transform",
        "text-animation",
        "graphic-animation",
        "transition",
    )


@dataclass
class AppConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    subtitle_calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    vision: VisionConfig = field(default_factory=VisionConfig)
    coarse_analysis: CoarseAnalysisConfig = field(default_factory=CoarseAnalysisConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    fine_analysis: FineAnalysisConfig = field(default_factory=FineAnalysisConfig)
    final_analysis: FinalAnalysisConfig = field(default_factory=FinalAnalysisConfig)

    def with_cli_overrides(
        self,
        *,
        language: str | None = None,
        multilingual: bool | None = None,
        llm_calibrate: bool | None = None,
    ) -> AppConfig:
        cfg = replace(
            self,
            asr=replace(self.asr),
            subtitle_calibration=replace(self.subtitle_calibration),
            phase2=replace(self.phase2),
            vision=replace(self.vision),
            coarse_analysis=replace(self.coarse_analysis),
            refinement=replace(self.refinement),
            fine_analysis=replace(self.fine_analysis),
            final_analysis=replace(self.final_analysis),
        )
        if language is not None:
            cfg.asr.language = language
        if multilingual is not None:
            cfg.asr.multilingual = multilingual
        if llm_calibrate is not None:
            cfg.subtitle_calibration.enabled = llm_calibrate
        return cfg

    def fingerprint(self) -> dict[str, Any]:
        """Stable dict for cache invalidation keys (no secrets)."""
        return {
            "asr": {
                "provider": self.asr.provider,
                "cloud_model": self.asr.cloud_model,
                "local_model": self.asr.local_model,
                "multilingual": self.asr.multilingual,
                "language": self.asr.language,
                "prompt_version": self.asr.prompt_version,
                "chunking": self.asr.chunking.__dict__,
                "merge": self.asr.merge.__dict__,
            },
            "subtitle_calibration": {
                "enabled": self.subtitle_calibration.enabled,
                "provider": self.subtitle_calibration.provider,
                "model": self.subtitle_calibration.model,
                "prompt_version": self.subtitle_calibration.prompt_version,
                "validation": self.subtitle_calibration.validation.__dict__,
            },
            "phase2": self.phase2.__dict__,
            "vision": {
                "provider": "in_session_agent",
                "execution_mode": "in_session_agent",
                "prompt_version": self.vision.prompt_version,
                "batch_max_frames": self.vision.batch_max_frames,
            },
            "coarse_analysis": self.coarse_analysis.__dict__,
            "refinement": {
                "image_format": self.refinement.image_format,
                "jpeg_quality": self.refinement.jpeg_quality,
                "max_frame_long_edge": self.refinement.max_frame_long_edge,
                "timestamp_parser_version": self.refinement.timestamp_parser_version,
                "dedup_time_threshold_seconds": self.refinement.dedup_time_threshold_seconds,
                "detail_max_per_interval": self.refinement.detail_max_per_interval,
                "detail_frame_max_long_edge": self.refinement.detail_frame_max_long_edge,
                "detail_frame_jpeg_quality": self.refinement.detail_frame_jpeg_quality,
                "frames_per_sheet": self.refinement.frames_per_sheet,
                "sheet_overlap_frames": self.refinement.sheet_overlap_frames,
                "sheet_columns": self.refinement.sheet_columns,
                "contact_sheet_max_edge": self.refinement.contact_sheet_max_edge,
                "contact_sheet_label_format_version": self.refinement.contact_sheet_label_format_version,
                "subtitle_context_before": self.refinement.subtitle_context_before,
                "subtitle_context_after": self.refinement.subtitle_context_after,
                "schema_version": self.refinement.schema_version,
                "no_contact_sheets": self.refinement.no_contact_sheets,
            },
            "fine_analysis": {
                "prompt_version": self.fine_analysis.prompt_version,
                "fine_schema_version": self.fine_analysis.fine_schema_version,
                "max_detail_frames_per_request": self.fine_analysis.max_detail_frames_per_request,
                "representative_frame_count": self.fine_analysis.representative_frame_count,
                "second_refinement_budget_ratio": self.fine_analysis.second_refinement_budget_ratio,
                "second_refinement_round1_ratio": self.fine_analysis.second_refinement_round1_ratio,
                "second_refinement_max_interval_seconds": self.fine_analysis.second_refinement_max_interval_seconds,
                "second_refinement_max_fps": self.fine_analysis.second_refinement_max_fps,
                "event_dedup_time_threshold_seconds": self.fine_analysis.event_dedup_time_threshold_seconds,
            },
        }


DEFAULT_CONFIG = AppConfig()
