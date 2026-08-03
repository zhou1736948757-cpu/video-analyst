"""Coarse analysis subsystem — Epic 2 batch runner, aggregation, refinement plan."""

from video_analyst.coarse_analysis.aggregate import aggregate_structure, save_structure
from video_analyst.coarse_analysis.batch_runner import (
    coarse_output_paths,
    ensure_coarse_dirs,
    load_manifest,
    run_coarse_analysis,
    save_manifest,
)
from video_analyst.coarse_analysis.models import (
    BatchRunRecord,
    COARSE_SCHEMA_VERSION,
    CoarseAnalysisManifest,
    CoarseAnalysisResult,
    RefinementInterval,
    RefinementPlan,
    SectionTimeRange,
    VideoGap,
    VideoSection,
    VideoStructure,
    request_to_interval,
)
from video_analyst.coarse_analysis.refinement import (
    build_refinement_plan,
    plan_to_dict,
)
from video_analyst.coarse_analysis.validation import (
    assert_analysis_within_batch,
    assert_refinement_plan_invariants,
    assert_video_structure_invariants,
    load_batch_analysis,
    load_visual_batches,
)

__all__ = [
    "BatchRunRecord",
    "COARSE_SCHEMA_VERSION",
    "CoarseAnalysisManifest",
    "CoarseAnalysisResult",
    "RefinementInterval",
    "RefinementPlan",
    "SectionTimeRange",
    "VideoGap",
    "VideoSection",
    "VideoStructure",
    "aggregate_structure",
    "assert_analysis_within_batch",
    "assert_refinement_plan_invariants",
    "assert_video_structure_invariants",
    "build_refinement_plan",
    "coarse_output_paths",
    "ensure_coarse_dirs",
    "load_batch_analysis",
    "load_manifest",
    "load_visual_batches",
    "plan_to_dict",
    "request_to_interval",
    "run_coarse_analysis",
    "save_manifest",
    "save_structure",
]