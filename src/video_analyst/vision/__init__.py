"""Vision subsystem — provider abstraction for coarse visual analysis."""

from video_analyst.vision import response_parser
from video_analyst.vision.anthropic_compatible import AnthropicCompatibleVisionProvider
from video_analyst.vision.base import VisionProvider
from video_analyst.vision.mock import MockVisionProvider
from video_analyst.vision.models import (
    SCHEMA_VERSION,
    CoarseBatchAnalysis,
    EvidenceReference,
    Inference,
    Observation,
    PreviousBatchContext,
    RefinementRequest,
    SubtitleContext,
    SubtitleMetadata,
    TimeRange,
    VisionBatchRequest,
    VisionBatchSelection,
    VisionFrame,
)
from video_analyst.vision.openai_compatible import OpenAICompatibleProvider
from video_analyst.vision.prompts import (
    DEFAULT_ANALYSIS_DIMENSIONS,
    PROMPT_VERSION,
    build_repair_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from video_analyst.vision.registry import (
    create_vision_provider,
    get_provider_class,
    register_provider,
)
from video_analyst.vision.request_builder import (
    build_all_batch_requests,
    build_request_for_batch,
)

__all__ = [
    "SCHEMA_VERSION",
    "PROMPT_VERSION",
    "DEFAULT_ANALYSIS_DIMENSIONS",
    "AnthropicCompatibleVisionProvider",
    "CoarseBatchAnalysis",
    "EvidenceReference",
    "Inference",
    "MockVisionProvider",
    "Observation",
    "OpenAICompatibleProvider",
    "PreviousBatchContext",
    "RefinementRequest",
    "SubtitleContext",
    "SubtitleMetadata",
    "TimeRange",
    "VisionBatchRequest",
    "VisionBatchSelection",
    "VisionFrame",
    "VisionProvider",
    "build_all_batch_requests",
    "build_repair_user_prompt",
    "build_request_for_batch",
    "build_system_prompt",
    "build_user_prompt",
    "create_vision_provider",
    "get_provider_class",
    "register_provider",
    "response_parser",
]