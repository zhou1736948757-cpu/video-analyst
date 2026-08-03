"""Domain-specific exceptions for Video Analyst."""


class VideoAnalystError(Exception):
    """Base error for the package."""


class FFmpegNotFoundError(VideoAnalystError):
    """ffmpeg or ffprobe is missing or not runnable."""


class FFmpegInstallError(VideoAnalystError):
    """FFmpeg installation failed or was declined."""


class EnvironmentBlockedError(VideoAnalystError):
    """Required runtime environment checks failed."""


class MediaProbeError(VideoAnalystError):
    """ffprobe failed to read media metadata."""


class AudioExtractError(VideoAnalystError):
    """FFmpeg audio extraction/compression failed."""


class AudioChunkingError(VideoAnalystError):
    """Audio silence/chunk planning failed."""


class ASRError(VideoAnalystError):
    """Speech recognition failed for all available backends."""


class ASRProviderUnavailableError(ASRError):
    """Requested ASR provider is not available."""


class ASRTranscriptionError(ASRError):
    """ASR transcription of one or more chunks failed."""


class GroqASRError(ASRTranscriptionError):
    """Groq Whisper request failed."""


class LocalWhisperError(ASRTranscriptionError):
    """Local Whisper failed or is unavailable."""


class ASRMergeError(VideoAnalystError):
    """ASR segment merge / time-offset failed."""


class CacheError(VideoAnalystError):
    """Cache read/write failure."""


class LLMCalibrationError(VideoAnalystError):
    """LLM subtitle calibration failed."""


class SubtitleCalibrationError(LLMCalibrationError):
    """Alias for calibration pipeline errors."""


class SubtitleValidationError(VideoAnalystError):
    """Subtitle structure or timestamp validation failed."""


class SubtitleActivationError(VideoAnalystError):
    """Failed to activate ASR/LLM subtitle version."""


class SubtitleReviewError(VideoAnalystError):
    """Review package missing or invalid."""


class Phase2PrepareError(VideoAnalystError):
    """Phase 2 input preparation failed."""


# ---- Epic 2: Vision / Coarse Analysis ----


class VisionProviderUnavailableError(VideoAnalystError):
    """Requested vision provider is not available or misconfigured."""


class VisionRequestError(VideoAnalystError):
    """Vision API request failed (generic)."""


class VisionTimeoutError(VisionRequestError):
    """Vision API request timed out."""


class VisionRateLimitError(VisionRequestError):
    """Vision API returned 429 rate limit."""


class VisionResponseParseError(VideoAnalystError):
    """Vision API response could not be parsed as JSON."""


class VisionResponseValidationError(VideoAnalystError):
    """Vision API response failed schema / semantic validation."""


class VisionBatchAnalysisError(VideoAnalystError):
    """One or more batches failed during coarse analysis."""


class CoarseAggregationError(VideoAnalystError):
    """Full-video coarse structure aggregation failed."""


class RefinementPlanError(VideoAnalystError):
    """Refinement plan generation failed."""


# ---- Epic 3: Refinement / fine-frame extraction ----


class RefinementPlanLoadError(VideoAnalystError):
    """refinement_plan.json missing or unreadable."""


class RefinementPlanValidationError(VideoAnalystError):
    """refinement_plan.json failed contract validation (FPS, ranges, schema)."""


class RefinementFrameExtractionError(VideoAnalystError):
    """FFmpeg failed to extract fine-grained frames for an interval."""


class RefinementTimestampParseError(VideoAnalystError):
    """Could not parse FFmpeg showinfo pts_time for one or more frames."""


class RefinementFrameMismatchError(VideoAnalystError):
    """Extracted frame count does not match the parsed timestamp count."""


class RefinementDeduplicationError(VideoAnalystError):
    """Cross- or intra-interval deduplication failed."""


class TemporalContactSheetError(VideoAnalystError):
    """Failed to assemble a temporal contact sheet (Pillow or layout)."""


class FineBatchBuildError(VideoAnalystError):
    """Failed to assemble FineVisualBatch."""


class Epic3PipelineError(VideoAnalystError):
    """Top-level Epic 3 orchestration failure."""


# ---- Epic 4: Fine temporal vision analysis ----


class SchemaVersionUnsupportedError(VideoAnalystError):
    """A schema with an incompatible major version was encountered.

    Raised when the major version of an artifact (e.g. refinement_plan.json,
    fine_batches.json) differs from the currently supported major version.
    Minor version differences are tolerated with a warning; major version
    differences are a hard stop — we never guess field semantics on an
    unknown major version.
    """


class FineVisionProviderUnavailableError(VideoAnalystError):
    """Requested fine vision provider is not available or misconfigured."""


class FineVisionRequestError(VideoAnalystError):
    """Fine vision API request failed (generic)."""


class FineVisionTimeoutError(FineVisionRequestError):
    """Fine vision API request timed out."""


class FineVisionRateLimitError(FineVisionRequestError):
    """Fine vision API returned 429 rate limit."""


class FineVisionResponseParseError(VideoAnalystError):
    """Fine vision API response could not be parsed as JSON."""


class FineVisionResponseValidationError(VideoAnalystError):
    """Fine vision API response failed schema / semantic validation."""


class FineBatchAnalysisError(VideoAnalystError):
    """One or more fine batches failed during fine analysis."""


class FineEventDeduplicationError(VideoAnalystError):
    """Cross-batch temporal event deduplication failed."""


class FineAggregationError(VideoAnalystError):
    """Full-video fine analysis aggregation failed."""


class SecondRefinementPlanError(VideoAnalystError):
    """Second-round refinement plan generation failed."""


class Epic4PipelineError(VideoAnalystError):
    """Top-level Epic 4 orchestration failure."""


# ---- Epic 5: Terminal refinement + evidence-based report ----


class TerminalRefinementPlanValidationError(VideoAnalystError):
    """second_refinement_plan 不合法（max_additional_rounds/terminal 标志）或不来自本阶段 schema。"""


class TerminalRefinementExecutionError(VideoAnalystError):
    """终局抽帧失败。"""


class TerminalFineAnalysisError(VideoAnalystError):
    """终局精细分析失败。"""


class FinalEventMergeError(VideoAnalystError):
    """第一轮与终局事件合并失败。"""


class ShotTimelineError(VideoAnalystError):
    """镜头级时间线构建失败。"""


class SectionAggregationError(VideoAnalystError):
    """段落级聚合失败。"""


class EvidenceIndexError(VideoAnalystError):
    """报告证据索引构建失败。"""


class ReportBuildError(VideoAnalystError):
    """报告生成失败。"""


class ReportSerializationError(VideoAnalystError):
    """报告序列化失败。"""


class ProvisionalReportNotAllowedError(VideoAnalystError):
    """当 second_refinement_plan 非空但未执行终局时，provisional 报告被禁止。"""


class Epic5PipelineError(VideoAnalystError):
    """Top-level Epic 5 orchestration failure."""
