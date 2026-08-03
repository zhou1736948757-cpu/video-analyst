"""Refinement subsystem — Epic 3 fine-frame extraction."""

from video_analyst.refinement.batch_builder import (
    build_fine_batch,
    build_fine_batches_index,
    load_fine_batches_index,
    persist_fine_batch,
    persist_fine_batches_index,
)
from video_analyst.refinement.contact_sheet import build_contact_sheets
from video_analyst.refinement.deduplicate import (
    DeduplicatedFrame,
    DeduplicationReport,
    deduplicate_intervals,
)
from video_analyst.refinement.detail_frames import select_detail_frames
from video_analyst.refinement.extractor import (
    ExtractedFrame,
    extract_interval_frames,
    remove_extracted_interval,
)
from video_analyst.refinement.manifest import (
    build_extraction_signature,
    build_presentation_signature,
    load_interval_artifact,
    load_manifest,
    persist_interval_artifact,
    save_manifest,
    upsert_manifest_entry,
)
from video_analyst.refinement.models import (
    DetailFrameReference,
    EPIC3_SCHEMA_VERSION,
    Epic3Manifest,
    Epic3Result,
    FineBatchesIndex,
    FineVisualBatch,
    IntervalArtifact,
    IntervalManifestEntry,
    RefinementFrame,
    RefinementSource,
    TemporalContactSheet,
)
from video_analyst.refinement.plan_loader import (
    PLAN_LOAD_SCHEMA_VERSION,
    load_refinement_plan,
)
from video_analyst.refinement.timestamp_parser import (
    ShowinfoEntry,
    compute_actual_fps,
    expected_interval_seconds,
    pair_with_files,
    parse_showinfo_stderr,
)
from video_analyst.refinement.validation import (
    assert_fine_batches_index_on_disk,
    assert_interval_artifact_on_disk,
    load_fine_batches_index_or_raise,
    load_interval_artifact_or_raise,
    load_manifest_or_raise,
    validate_interval_artifact,
)

__all__ = [
    "DetailFrameReference",
    "DeduplicatedFrame",
    "DeduplicationReport",
    "EPIC3_SCHEMA_VERSION",
    "Epic3Manifest",
    "Epic3Result",
    "ExtractedFrame",
    "FineBatchesIndex",
    "FineVisualBatch",
    "IntervalArtifact",
    "IntervalManifestEntry",
    "PLAN_LOAD_SCHEMA_VERSION",
    "RefinementFrame",
    "RefinementSource",
    "ShowinfoEntry",
    "TemporalContactSheet",
    "assert_fine_batches_index_on_disk",
    "assert_interval_artifact_on_disk",
    "build_contact_sheets",
    "build_extraction_signature",
    "build_fine_batch",
    "build_fine_batches_index",
    "build_presentation_signature",
    "compute_actual_fps",
    "deduplicate_intervals",
    "expected_interval_seconds",
    "extract_interval_frames",
    "load_fine_batches_index",
    "load_fine_batches_index_or_raise",
    "load_interval_artifact",
    "load_interval_artifact_or_raise",
    "load_manifest",
    "load_manifest_or_raise",
    "load_refinement_plan",
    "pair_with_files",
    "parse_showinfo_stderr",
    "persist_fine_batch",
    "persist_fine_batches_index",
    "persist_interval_artifact",
    "remove_extracted_interval",
    "save_manifest",
    "select_detail_frames",
    "upsert_manifest_entry",
    "validate_interval_artifact",
]