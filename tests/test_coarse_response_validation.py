"""Tests for vision response validation & repair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_analyst.exceptions import (
    VisionResponseParseError,
    VisionResponseValidationError,
)
from video_analyst.vision.models import VisionBatchRequest, VisionFrame, SubtitleMetadata
from video_analyst.vision.response_parser import (
    parse_and_validate,
    parse_response_payload,
    validate_batch_response,
)


def _request() -> VisionBatchRequest:
    return VisionBatchRequest(
        batch_id="batch_0001",
        video_id="video",
        batch_start=0.0,
        batch_end=10.0,
        frames=[
            VisionFrame(
                frame_id="frame_a",
                timestamp_seconds=1.0,
                timecode="00:00:01.000",
                path=Path("/tmp/a.jpg"),
                sources=["fixed_fps"],
            ),
            VisionFrame(
                frame_id="frame_b",
                timestamp_seconds=5.0,
                timecode="00:00:05.000",
                path=Path("/tmp/b.jpg"),
                sources=["fixed_fps"],
            ),
        ],
        subtitles=[],
        subtitle_metadata=SubtitleMetadata(active_source="llm"),
    )


def _good_payload() -> dict:
    return {
        "schema_version": "1.0",
        "batch_id": "batch_0001",
        "start": 0.0,
        "end": 10.0,
        "summary": "ok",
        "confidence": 0.6,
        "direct_observations": [
            {
                "category": "frame_describe",
                "description": "frame a is dark",
                "time_range": {"start": 0.0, "end": 5.0},
                "evidence": {"frame_ids": ["frame_a"], "timestamps": [1.0]},
                "confidence": 0.7,
            }
        ],
        "refinement_requests": [
            {
                "request_id": "ref_01",
                "start": 4.0,
                "end": 6.0,
                "requested_fps": 10,
                "reason": "fast motion",
                "questions": ["what moves?"],
                "priority": "medium",
                "evidence_frame_ids": ["frame_a"],
                "confidence": 0.6,
            }
        ],
    }


def test_parse_markdown_fenced_json():
    text = "```json\n" + json.dumps(_good_payload()) + "\n```"
    data = parse_response_payload(text)
    assert data["batch_id"] == "batch_0001"


def test_parse_json_with_prose_around():
    text = "Here is the analysis:\n" + json.dumps(_good_payload()) + "\nThanks."
    data = parse_response_payload(text)
    assert data["batch_id"] == "batch_0001"


def test_parse_garbage_raises():
    with pytest.raises(VisionResponseParseError):
        parse_response_payload("this is not json")


def test_parse_empty_raises():
    with pytest.raises(VisionResponseParseError):
        parse_response_payload("")


def test_validate_accepts_good_payload():
    analysis, warnings = parse_and_validate(
        json.dumps(_good_payload()),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    assert analysis.batch_id == "batch_0001"
    assert len(analysis.direct_observations) == 1
    assert analysis.refinement_requests[0].requested_fps == 10


def test_validate_clamps_out_of_range_time_range():
    payload = _good_payload()
    payload["direct_observations"][0]["time_range"] = {"start": -5.0, "end": 999.0}
    payload["direct_observations"][0]["confidence"] = 5.0  # out of range
    analysis, warnings = parse_and_validate(
        json.dumps(payload),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    obs = analysis.direct_observations[0]
    assert obs.time_range.start >= 0.0
    assert obs.time_range.end <= 10.0
    assert 0.0 <= obs.confidence <= 1.0
    assert any("clamp" in w or "confidence" in w.lower() for w in warnings)


def test_validate_drops_unknown_frame_ids():
    payload = _good_payload()
    payload["direct_observations"][0]["evidence"]["frame_ids"] = [
        "frame_a",
        "GHOST_FRAME",
    ]
    analysis, warnings = parse_and_validate(
        json.dumps(payload),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    obs = analysis.direct_observations[0]
    assert obs.evidence.frame_ids == ["frame_a"]
    assert any("GHOST_FRAME" in w or "unknown" in w for w in warnings)


def test_validate_snap_invalid_fps():
    payload = _good_payload()
    payload["refinement_requests"][0]["requested_fps"] = 7  # not 5/10/15
    analysis, warnings = parse_and_validate(
        json.dumps(payload),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    assert analysis.refinement_requests[0].requested_fps in {5, 10, 15}


def test_validate_rejects_empty_reason_refinement():
    payload = _good_payload()
    payload["refinement_requests"][0]["reason"] = "  "
    with pytest.raises(VisionResponseValidationError):
        validate_batch_response(
            payload, request=_request(), video_start=0.0, video_end=10.0
        )


def test_validate_keeps_observations_and_inferences_separate():
    payload = _good_payload()
    payload["inferred_interpretations"] = [
        {
            "description": "likely a hand-drawn animation",
            "supporting_observation_ids": [],
            "confidence": 0.5,
            "alternatives": [],
        }
    ]
    analysis, _ = parse_and_validate(
        json.dumps(payload),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    # Inferences are in their own field, NOT mixed into direct_observations
    assert len(analysis.direct_observations) == 1
    assert len(analysis.inferred_interpretations) == 1
    assert all(
        not hasattr(o, "supporting_observation_ids")
        for o in analysis.direct_observations
    )


def test_validate_fixes_missing_items_with_warning():
    payload = {"schema_version": "1.0"}
    analysis, warnings = parse_and_validate(
        json.dumps(payload),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    assert analysis.batch_id == "batch_0001"
    assert analysis.direct_observations == []
    assert analysis.refinement_requests == []


def test_validate_batch_id_mismatch_clamped_with_warning():
    payload = _good_payload()
    payload["batch_id"] = "wrong_batch"
    analysis, warnings = parse_and_validate(
        json.dumps(payload),
        request=_request(),
        video_start=0.0,
        video_end=10.0,
    )
    assert analysis.batch_id == "batch_0001"
    assert any("batch_id" in w for w in warnings)