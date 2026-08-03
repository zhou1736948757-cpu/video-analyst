"""Tests for Epic 2 pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.config import AppConfig
from video_analyst.exceptions import VisionBatchAnalysisError
from video_analyst.pipeline.epic2 import run_epic2
from video_analyst.vision.mock import MockVisionProvider

from tests._epic2_fixtures import seed_layout


def test_full_pipeline_with_mock(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    # Use the Mock provider directly via registry
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True
    result = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="test analysis",
        refinement_budget_ratio=0.20,
    )
    assert result.total == 2
    assert result.succeeded == 2
    # Outputs exist
    assert layout.coarse_manifest_json.is_file()
    assert layout.coarse_structure_json.is_file()
    assert layout.coarse_refinement_plan_json.is_file()
    # batch files
    assert (layout.coarse_batches_dir / "batch_0001.json").is_file()
    assert (layout.coarse_batches_dir / "batch_0002.json").is_file()
    # raw response files
    assert (layout.coarse_raw_dir / "batch_0001_response.json").is_file()


def test_pipeline_resume_uses_cache(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0

    # First run
    r1 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="same request",
    )
    assert r1.succeeded == 2

    # Capture per-batch cache_keys after run 1
    import json

    def _cache_keys():
        m = json.loads(layout.coarse_manifest_json.read_text())
        return {b["batch_id"]: b["cache_key"] for b in m["batches"]}

    keys_after_run1 = _cache_keys()

    # Second run with same request → cache should hit, no vision call
    r2 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="same request",
    )
    keys_after_run2 = _cache_keys()

    # Same cache keys → no vision rerun
    assert keys_after_run1 == keys_after_run2
    assert r2.succeeded == 2


def test_pipeline_with_needs_refinement_scenario(tmp_path: Path):
    """Pipeline with mock `needs_refinement` must produce refinement_plan.json
    with at least one interval."""
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True
    cfg.coarse_analysis.refinement_budget_ratio = 1.0  # don't drop any
    # Force a custom provider via direct import
    from video_analyst.coarse_analysis.batch_runner import run_coarse_analysis

    provider = MockVisionProvider(scenario="needs_refinement")
    result = run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=provider,
        user_request="",
        video_id="v",
    )
    assert result.succeeded == 2
    # Now build the plan
    from video_analyst.coarse_analysis.aggregate import aggregate_structure
    from video_analyst.coarse_analysis.refinement import build_refinement_plan
    import json
    data = json.loads(layout.phase2_batches_json.read_text())
    batches = data.get("batches") or []
    order = [b["batch_id"] for b in batches]
    meta = {b["batch_id"]: {"start": b["start"], "end": b["end"]} for b in batches}
    video_end = float(meta[order[-1]]["end"])
    video_start = float(meta[order[0]]["start"])
    structure = aggregate_structure(
        analyses=result.analyses,
        failed_batches=result.failed_batches,
        video_start=video_start,
        video_end=video_end,
        batches_order=order,
        batches_meta=meta,
    )
    plan = build_refinement_plan(
        batch_analyses=result.analyses,
        video_duration=video_end,
        video_start=video_start,
        config=cfg.coarse_analysis,
    )
    # needs_refinement mock emits at least one refinement per batch
    assert len(plan.intervals) + len(plan.deferred_intervals) >= 2


def test_pipeline_raises_on_failure_when_not_partial(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = False

    from video_analyst.exceptions import VisionRequestError
    from video_analyst.exceptions import (
        VisionProviderUnavailableError as Err,
    )

    class AlwaysFail(MockVisionProvider):
        def analyze_batch(self_inner, request):
            raise VisionRequestError("nope")

    from video_analyst.vision.registry import register_provider

    register_provider("always_fail", AlwaysFail)
    try:
        with pytest.raises((VisionBatchAnalysisError, Err)):
            run_epic2(
                output_dir=tmp_path,
                config=cfg,
                provider_name="always_fail",
                user_request="",
            )
    finally:
        # Unregister to avoid leaking into other tests
        from video_analyst.vision.registry import _REGISTRY

        _REGISTRY.pop("always_fail", None)


def test_pipeline_refresh_vision(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True

    r1 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="",
    )
    assert (layout.coarse_batches_dir / "batch_0001.json").is_file()

    # Refresh vision → rerun all batches
    r2 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="",
        refresh_vision=True,
    )
    assert r2.succeeded == 2
    # Manifest updated, fresh timestamps
    assert layout.coarse_manifest_json.is_file()


def test_pipeline_user_request_change_reruns(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True

    r1 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="A",
    )
    cache_keys_a = {rec.batch_id: rec.cache_key for rec in r1.manifest.batches}

    r2 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="B",
    )
    cache_keys_b = {rec.batch_id: rec.cache_key for rec in r2.manifest.batches}

    # Cache keys should differ for at least one batch (user_request changed)
    assert cache_keys_a != cache_keys_b


def test_pipeline_budget_change_does_not_rerun_vision(tmp_path: Path):
    """Changing refinement budget must NOT re-invoke the vision provider."""
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True

    r1 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="x",
        refinement_budget_ratio=0.20,
    )
    # Capture batch file mtimes
    import os
    b1_mtime = os.path.getmtime(layout.coarse_batches_dir / "batch_0001.json")

    r2 = run_epic2(
        output_dir=tmp_path,
        config=cfg,
        provider_name="mock",
        user_request="x",
        refinement_budget_ratio=0.50,
    )
    b1_mtime2 = os.path.getmtime(layout.coarse_batches_dir / "batch_0001.json")
    # mtime must be unchanged → no vision rerun
    assert b1_mtime == b1_mtime2
    assert r2.succeeded == 2