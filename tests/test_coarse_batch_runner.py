"""Tests for the coarse analysis batch runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_analyst.config import AppConfig, VisionConfig
from video_analyst.coarse_analysis.batch_runner import (
    coarse_output_paths,
    load_manifest,
    run_coarse_analysis,
    save_manifest,
)
from video_analyst.coarse_analysis.models import CoarseAnalysisManifest
from video_analyst.exceptions import VisionBatchAnalysisError
from video_analyst.vision.mock import MockVisionProvider

from tests._epic2_fixtures import seed_layout


def test_processes_all_batches_in_order(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=60.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    provider = MockVisionProvider(scenario="normal")
    result = run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=provider,
        user_request="x",
        video_id="v",
    )
    assert result.total == 3
    assert result.succeeded == 3
    assert result.failed == 0


def test_second_run_hits_cache_no_reinvocation(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    p1 = MockVisionProvider(scenario="normal")
    r1 = run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p1,
        user_request="same",
        video_id="v",
    )
    assert p1.call_count == 2
    # Second run with same request/user_request → cache hit
    p2 = MockVisionProvider(scenario="normal")
    r2 = run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p2,
        user_request="same",
        video_id="v",
    )
    assert p2.call_count == 0
    assert r2.succeeded == r1.succeeded


def test_changing_user_request_reruns(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    p1 = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p1,
        user_request="A",
        video_id="v",
    )
    p2 = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p2,
        user_request="B",
        video_id="v",
    )
    # Cache key differs because user_request differs
    assert p2.call_count == 2


def test_single_batch_failure_continues_when_partial(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True

    class SelectiveMock(MockVisionProvider):
        def analyze_batch(self_inner, request):
            if request.batch_id == "batch_0001":
                from video_analyst.exceptions import VisionRequestError

                raise VisionRequestError("simulated failure")
            return super().analyze_batch(request)

    p = SelectiveMock(scenario="normal")
    result = run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p,
        user_request="x",
        video_id="v",
    )
    assert result.succeeded == 1
    assert result.failed == 1
    assert "batch_0001" in result.failed_batches


def test_no_partial_raises_on_any_failure(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = False

    class AlwaysFail(MockVisionProvider):
        def analyze_batch(self_inner, request):
            from video_analyst.exceptions import VisionRequestError

            raise VisionRequestError("nope")

    p = AlwaysFail(scenario="normal")
    with pytest.raises(VisionBatchAnalysisError):
        run_coarse_analysis(
            layout=layout,
            config=cfg,
            provider=p,
            user_request="x",
            video_id="v",
        )


def test_refresh_batch_reruns_only_named(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    p1 = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout, config=cfg, provider=p1, user_request="x", video_id="v"
    )
    assert p1.call_count == 2
    p2 = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p2,
        user_request="x",
        video_id="v",
        refresh_batches=["batch_0002"],
    )
    assert p2.call_count == 1


def test_refresh_vision_clears_all(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    p1 = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout, config=cfg, provider=p1, user_request="x", video_id="v"
    )
    p2 = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout,
        config=cfg,
        provider=p2,
        user_request="x",
        video_id="v",
        refresh_vision=True,
    )
    assert p2.call_count == 2


def test_atomic_write_each_batch(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    provider = MockVisionProvider(scenario="normal")
    run_coarse_analysis(
        layout=layout, config=cfg, provider=provider, user_request="x", video_id="v"
    )
    paths = coarse_output_paths(layout)
    assert (paths["batches_dir"] / "batch_0001.json").is_file()
    assert (paths["batches_dir"] / "batch_0002.json").is_file()
    assert (paths["raw_dir"] / "batch_0001_response.json").is_file()
    assert paths["manifest"].is_file()
    m = load_manifest(layout)
    assert m is not None
    assert m.success_count == 2


def test_save_manifest_writes_disk(tmp_path: Path):
    layout = seed_layout(tmp_path)
    paths = coarse_output_paths(layout)
    m = CoarseAnalysisManifest(video_id="v", provider="mock", model="x")
    save_manifest(layout, m)
    assert paths["manifest"].is_file()
    # Round-trip
    loaded = load_manifest(layout)
    assert loaded is not None
    assert loaded.video_id == "v"


def test_retry_then_succeed(tmp_path: Path, monkeypatch):
    """Retry mechanism: first 2 calls per batch fail, third succeeds."""
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 3
    monkeypatch.setattr("time.sleep", lambda s: None)

    from video_analyst.exceptions import VisionRequestError

    class FlakyMock(MockVisionProvider):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._fail_count = 0
            self._success_count = 0
            self._per_batch: dict = {}

        def analyze_batch(self_inner, request):
            bid = request.batch_id
            self_inner._per_batch[bid] = self_inner._per_batch.get(bid, 0) + 1
            calls = self_inner._per_batch[bid]
            if calls <= 2:
                self_inner._fail_count += 1
                raise VisionRequestError("flaky")
            self_inner._success_count += 1
            return super().analyze_batch(request)

    p = FlakyMock(scenario="normal")
    result = run_coarse_analysis(
        layout=layout, config=cfg, provider=p, user_request="x", video_id="v"
    )
    # 2 batches × 2 fails each = 4 fails, 2 successes
    assert p._fail_count == 4
    assert p._success_count == 2
    assert result.succeeded == 2


def test_unrecoverable_parse_error_marks_batch_failed(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 0
    cfg.coarse_analysis.allow_partial = True

    from video_analyst.exceptions import VisionResponseParseError

    class ParseFail(MockVisionProvider):
        def analyze_batch(self_inner, request):
            raise VisionResponseParseError("garbage")

    p = ParseFail(scenario="normal")
    result = run_coarse_analysis(
        layout=layout, config=cfg, provider=p, user_request="x", video_id="v"
    )
    assert result.failed == 2
    assert result.succeeded == 0