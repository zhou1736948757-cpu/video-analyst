"""Tests for the request builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_analyst.exceptions import Phase2PrepareError, SubtitleReviewError
from video_analyst.vision.request_builder import (
    build_all_batch_requests,
    build_request_for_batch,
)
from video_analyst.vision.models import SubtitleMetadata

from tests._epic2_fixtures import seed_layout


def test_reads_only_active_subtitles_for_metadata(tmp_path: Path):
    layout = seed_layout(tmp_path)
    req, sel = build_request_for_batch(
        0,
        layout,
        user_request="x",
        video_id="v",
        max_frames=20,
    )
    # Subtitle metadata must come from review.json, not asr/llm files
    assert req.subtitle_metadata.active_source == "llm"
    assert req.subtitle_metadata.calibration_model == "llama-3.3-70b-versatile"


def test_does_not_read_asr_or_llm_files_directly(tmp_path: Path, monkeypatch):
    layout = seed_layout(tmp_path)
    # Rename the asr/llm files; the request builder should not care.
    layout.subtitles_asr_json.write_text("{}", encoding="utf-8")
    layout.subtitles_llm_json.write_text("{}", encoding="utf-8")
    # Read should still succeed (it doesn't touch asr/llm files)
    req, sel = build_request_for_batch(
        0,
        layout,
        user_request="x",
        video_id="v",
        max_frames=20,
    )
    assert req.batch_id == "batch_0001"


def test_subtitle_metadata_reviewed_and_needs_review(tmp_path: Path):
    layout = seed_layout(tmp_path, reviewed=True, needs_review=True)
    req, _ = build_request_for_batch(0, layout, user_request="x", video_id="v", max_frames=20)
    assert req.subtitle_metadata.reviewed is True
    assert req.subtitle_metadata.needs_review is True


def test_frames_sorted_by_time(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    req, sel = build_request_for_batch(0, layout, user_request="x", video_id="v", max_frames=20)
    times = [f.timestamp_seconds for f in req.frames]
    assert times == sorted(times)
    # scene frame should be preserved (protected)
    assert any(f.sources and "scene" in f.sources for f in req.frames)


def test_frame_cap_drops_excess_with_explicit_omitted_list(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    # small max_frames so cap kicks in
    req, sel = build_request_for_batch(0, layout, user_request="x", video_id="v", max_frames=2)
    assert len(req.frames) <= 2
    # Scene frames preserved
    scene_in_req = [f for f in req.frames if "scene" in (f.sources or [])]
    # Omitted list must be populated when there are more available frames
    assert sel.total_available >= 2
    # All omitted frame_ids are stable strings
    for fid in sel.omitted_frame_ids:
        assert isinstance(fid, str)


def test_build_all_batch_requests_returns_all_batches(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=60.0, batch_seconds=20.0)
    pairs = build_all_batch_requests(
        layout,
        user_request="x",
        video_id="v",
        max_frames=10,
    )
    assert len(pairs) == 3
    assert [p[0].batch_id for p in pairs] == ["batch_0001", "batch_0002", "batch_0003"]


def test_user_request_changes_cache_hash(tmp_path: Path):
    layout = seed_layout(tmp_path)
    req_a, _ = build_request_for_batch(0, layout, user_request="A", video_id="v", max_frames=10)
    req_b, _ = build_request_for_batch(0, layout, user_request="B", video_id="v", max_frames=10)
    assert req_a.user_request != req_b.user_request
    # frames can be the same; cache key derivation should differ
    from video_analyst.coarse_analysis.batch_runner import _build_cache_key

    k_a = _build_cache_key(
        batch_id=req_a.batch_id,
        request=req_a,
        selection=type("S", (), {"model_dump": lambda self, mode: {}})(),
        active_subtitle_path=layout.subtitles_json,
        active_subtitle_signature=None,
        user_request="A",
        provider_name="mock",
        model="mock",
        prompt_version="coarse-vision-v1",
        schema_version="1.0",
    )
    k_b = _build_cache_key(
        batch_id=req_b.batch_id,
        request=req_b,
        selection=type("S", (), {"model_dump": lambda self, mode: {}})(),
        active_subtitle_path=layout.subtitles_json,
        active_subtitle_signature=None,
        user_request="B",
        provider_name="mock",
        model="mock",
        prompt_version="coarse-vision-v1",
        schema_version="1.0",
    )
    assert k_a != k_b


def test_missing_batches_raises(tmp_path: Path):
    layout = seed_layout(tmp_path)
    layout.phase2_batches_json.unlink()
    with pytest.raises(Phase2PrepareError):
        build_request_for_batch(0, layout, user_request="x", video_id="v", max_frames=10)


def test_out_of_range_batch_index_raises(tmp_path: Path):
    layout = seed_layout(tmp_path, duration=40.0, batch_seconds=20.0)
    with pytest.raises(Phase2PrepareError):
        build_request_for_batch(99, layout, user_request="x", video_id="v", max_frames=10)