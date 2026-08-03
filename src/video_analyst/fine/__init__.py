"""Fine temporal vision analysis subsystem (Epic 4)."""

from video_analyst.fine.aggregate import (
    aggregate_fine_analysis,
    save_fine_video_analysis,
    save_temporal_events,
)
from video_analyst.fine.anthropic_compatible import AnthropicCompatibleFineProvider
from video_analyst.fine.base import FineVisionProvider
from video_analyst.fine.batch_runner import run_fine_analysis
from video_analyst.fine.deduplicate import (
    DeduplicationReport,
    deduplicate_batch_analyses,
    deduplicate_temporal_events,
)
from video_analyst.fine.manifest import (
    build_fine_cache_key,
    load_fine_batch_analysis,
    load_fine_manifest,
    persist_fine_batch_analysis,
    persist_fine_raw_response,
    save_fine_manifest,
    upsert_fine_batch_record,
)
from video_analyst.fine.mock import MockFineVisionProvider
from video_analyst.fine.models import (
    FINE_SCHEMA_VERSION,
    FineAnalysisManifest,
    FineAnalysisResult,
    FineBatchAnalysis,
    FineBatchRunRecord,
    FineContactSheetInput,
    FineCoarseContext,
    FineDetailFrameInput,
    FineEvidenceReference,
    FineVideoAnalysis,
    FineVisionRequest,
    FrameStateObservation,
    SecondRefinementInterval,
    SecondRefinementPlan,
    SecondRefinementRequest,
    SubjectMotionAnalysis,
    TemporalBoundary,
    TemporalEvent,
    TransitionAnalysis,
)
from video_analyst.fine.openai_compatible import OpenAICompatibleFineProvider
from video_analyst.fine.prompts import (
    PROMPT_VERSION,
    build_fine_repair_user_prompt,
    build_fine_system_prompt,
    build_fine_user_prompt,
)
from video_analyst.fine.registry import (
    create_fine_vision_provider,
    get_fine_provider_class,
    register_fine_provider,
)
from video_analyst.fine.request_builder import (
    RequestBuildReport,
    build_all_fine_requests,
    build_fine_request,
)
from video_analyst.fine.response_parser import (
    parse_and_validate_fine,
    parse_fine_response_payload,
    validate_fine_response,
)
from video_analyst.fine.second_refinement import (
    build_second_refinement_plan,
    load_second_refinement_plan,
    save_second_refinement_plan,
)
from video_analyst.fine.validation import (
    assert_evidence_exists,
    assert_fine_analysis_within_interval,
    assert_question_coverage,
    load_fine_batches_index,
)

__all__ = [
    "FINE_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "AnthropicCompatibleFineProvider",
    "DeduplicationReport",
    "FineAnalysisManifest",
    "FineAnalysisResult",
    "FineBatchAnalysis",
    "FineBatchRunRecord",
    "FineContactSheetInput",
    "FineCoarseContext",
    "FineDetailFrameInput",
    "FineEvidenceReference",
    "FineVideoAnalysis",
    "FineVisionProvider",
    "FineVisionRequest",
    "FrameStateObservation",
    "MockFineVisionProvider",
    "OpenAICompatibleFineProvider",
    "RequestBuildReport",
    "SecondRefinementInterval",
    "SecondRefinementPlan",
    "SecondRefinementRequest",
    "SubjectMotionAnalysis",
    "TemporalBoundary",
    "TemporalEvent",
    "TransitionAnalysis",
    "aggregate_fine_analysis",
    "assert_evidence_exists",
    "assert_fine_analysis_within_interval",
    "assert_question_coverage",
    "build_all_fine_requests",
    "build_fine_cache_key",
    "build_fine_repair_user_prompt",
    "build_fine_request",
    "build_fine_system_prompt",
    "build_fine_user_prompt",
    "build_second_refinement_plan",
    "create_fine_vision_provider",
    "deduplicate_batch_analyses",
    "deduplicate_temporal_events",
    "get_fine_provider_class",
    "load_fine_batch_analysis",
    "load_fine_batches_index",
    "load_fine_manifest",
    "load_second_refinement_plan",
    "parse_and_validate_fine",
    "parse_fine_response_payload",
    "persist_fine_batch_analysis",
    "persist_fine_raw_response",
    "register_fine_provider",
    "run_fine_analysis",
    "save_fine_manifest",
    "save_fine_video_analysis",
    "save_second_refinement_plan",
    "save_temporal_events",
    "upsert_fine_batch_record",
    "validate_fine_response",
]