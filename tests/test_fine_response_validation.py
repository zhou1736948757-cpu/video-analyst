"""Tests for fine response validation (Epic 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.exceptions import (
    FineVisionResponseParseError,
    FineVisionResponseValidationError,
)
from video_analyst.fine.models import FineVisionRequest
from video_analyst.fine.response_parser import (
    parse_and_validate_fine,
    parse_fine_response_payload,
    validate_fine_response,
)

from tests._epic4_fixtures import has_pillow, make_fine_visual_batch, seed_fine_batches

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _request(tmp_path: Path) -> FineVisionRequest:
    from video_analyst.config import FineAnalysisConfig
    from video_analyst.fine.request_builder import build_fine_request

    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10)
    req, _ = build_fine_request(batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x")
    return req


def _valid_payload(request: FineVisionRequest) -> dict:
    sheet = request.contact_sheets[0]
    first_frame = sheet.frame_ids[0]
    last_frame = sheet.frame_ids[-1]
    first_ts = sheet.timestamps[0]
    last_ts = sheet.timestamps[-1]
    return {
        "schema_version": "1.0",
        "fine_batch_id": request.fine_batch_id,
        "interval_id": request.interval.interval_id,
        "start": request.interval.start,
        "end": request.interval.end,
        "summary": "valid",
        "frame_states": [
            {
                "observation_id": f"obs_{first_frame}",
                "timestamp_seconds": first_ts,
                "frame_id": first_frame,
                "description": "first frame state",
                "visible_subjects": ["a"],
                "visible_text": [],
                "composition_notes": [],
                "confidence": 0.7,
            }
        ],
        "temporal_events": [
            {
                "event_id": "ev_001",
                "category": "subject_motion",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "description": "subject moved",
                "direct_observations": [f"obs_{first_frame}"],
                "inference": "subject moved between samples",
                "evidence": {
                    "contact_sheet_ids": [sheet.sheet_id],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["camera moved"],
                "confidence": 0.6,
            }
        ],
        "subject_motions": [],
        "camera_motions": [],
        "post_transforms": [],
        "text_animations": [],
        "graphic_animations": [],
        "transitions": [],
        "question_answers": [
            {
                "question": q,
                "answer": f"answer to {q}",
                "status": "answered",
                "evidence": {
                    "contact_sheet_ids": [sheet.sheet_id],
                    "frame_ids": [first_frame],
                    "timestamps": [first_ts],
                },
                "confidence": 0.6,
            }
            for q in request.refinement_questions
        ],
        "second_refinement_requests": [],
        "initial_state": "initial",
        "final_state": "final",
        "continuity_notes": [],
        "unresolved_items": [],
        "confidence": 0.6,
        "warnings": [],
    }


def test_valid_response(tmp_path: Path):
    req = _request(tmp_path)
    analysis, warnings = validate_fine_response(_valid_payload(req), request=req)
    assert analysis.fine_batch_id == req.fine_batch_id
    assert len(analysis.temporal_events) == 1
    assert len(analysis.question_answers) == len(req.refinement_questions)


def test_code_fence_stripped(tmp_path: Path):
    req = _request(tmp_path)
    import json

    raw = "```json\n" + json.dumps(_valid_payload(req)) + "\n```"
    analysis, _ = parse_and_validate_fine(raw, request=req)
    assert analysis.fine_batch_id == req.fine_batch_id


def test_extra_text_around_json(tmp_path: Path):
    req = _request(tmp_path)
    import json

    raw = "Here is the analysis:\n" + json.dumps(_valid_payload(req)) + "\nDone."
    analysis, _ = parse_and_validate_fine(raw, request=req)
    assert analysis.fine_batch_id == req.fine_batch_id


def test_unknown_frame_id_dropped(tmp_path: Path):
    req = _request(tmp_path)
    payload = _valid_payload(req)
    payload["temporal_events"][0]["evidence"]["frame_ids"].append("NOT_A_REAL_FRAME")
    analysis, warnings = validate_fine_response(payload, request=req)
    assert "NOT_A_REAL_FRAME" not in analysis.temporal_events[0].evidence.frame_ids
    assert any("unknown frame_ids" in w for w in warnings)


def test_unknown_sheet_id_dropped(tmp_path: Path):
    req = _request(tmp_path)
    payload = _valid_payload(req)
    payload["temporal_events"][0]["evidence"]["contact_sheet_ids"].append("NOT_A_REAL_SHEET")
    analysis, warnings = validate_fine_response(payload, request=req)
    assert "NOT_A_REAL_SHEET" not in analysis.temporal_events[0].evidence.contact_sheet_ids


def test_time_out_of_interval_clamped(tmp_path: Path):
    req = _request(tmp_path)
    payload = _valid_payload(req)
    payload["temporal_events"][0]["boundary"]["earliest"] = req.interval.start - 10
    payload["temporal_events"][0]["boundary"]["latest"] = req.interval.end + 10
    analysis, warnings = validate_fine_response(payload, request=req)
    ev = analysis.temporal_events[0]
    assert ev.boundary.earliest >= req.interval.start
    assert ev.boundary.latest <= req.interval.end


def test_unknown_observation_id_dropped(tmp_path: Path):
    req = _request(tmp_path)
    payload = _valid_payload(req)
    payload["temporal_events"][0]["direct_observations"].append("NOT_A_REAL_OBS")
    analysis, warnings = validate_fine_response(payload, request=req)
    assert "NOT_A_REAL_OBS" not in analysis.temporal_events[0].direct_observations


def test_unanswered_question_filled_unresolved(tmp_path: Path):
    req = _request(tmp_path)
    payload = _valid_payload(req)
    # 删除第一个问题的回答
    payload["question_answers"] = payload["question_answers"][1:]
    analysis, warnings = validate_fine_response(payload, request=req)
    # 仍覆盖所有问题，缺的标 unresolved
    assert len(analysis.question_answers) == len(req.refinement_questions)
    statuses = {qa.status for qa in analysis.question_answers}
    assert "unresolved" in statuses


def test_confidence_out_of_range_clamped(tmp_path: Path):
    req = _request(tmp_path)
    payload = _valid_payload(req)
    payload["confidence"] = 1.5
    payload["temporal_events"][0]["confidence"] = 2.0
    analysis, warnings = validate_fine_response(payload, request=req)
    assert analysis.confidence == 1.0
    assert analysis.temporal_events[0].confidence == 1.0


def test_second_refinement_invalid_fps_rejected(tmp_path: Path):
    """二轮补帧 FPS 只能是 10 或 15。"""
    req = _request(tmp_path)
    payload = _valid_payload(req)
    sheet = req.contact_sheets[0]
    payload["second_refinement_requests"] = [
        {
            "request_id": "2nd_001",
            "request_type": "higher_temporal_density",
            "start": req.interval.start,
            "end": req.interval.end,
            "requested_fps": 20,  # 非法
            "reason": "need more",
            "questions": [],
            "priority": "high",
            "evidence": {
                "contact_sheet_ids": [sheet.sheet_id],
                "frame_ids": [sheet.frame_ids[0]],
                "timestamps": [sheet.timestamps[0]],
            },
            "confidence": 0.6,
        }
    ]
    analysis, warnings = validate_fine_response(payload, request=req)
    # fps=20 被规范化
    assert analysis.second_refinement_requests[0].requested_fps in (10, 15)


def test_second_refinement_15fps_cannot_upgrade(tmp_path: Path):
    """当前 15 FPS 时不能再升级 FPS。"""
    from video_analyst.config import FineAnalysisConfig
    from video_analyst.fine.request_builder import build_fine_request

    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=15)
    req, _ = build_fine_request(batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x")
    payload = _valid_payload(req)
    sheet = req.contact_sheets[0]
    payload["second_refinement_requests"] = [
        {
            "request_id": "2nd_001",
            "request_type": "higher_temporal_density",
            "start": req.interval.start,
            "end": req.interval.end,
            "requested_fps": 15,  # 当前已 15，不能再 higher_temporal_density
            "reason": "need more",
            "questions": [],
            "priority": "high",
            "evidence": {
                "contact_sheet_ids": [sheet.sheet_id],
                "frame_ids": [sheet.frame_ids[0]],
                "timestamps": [sheet.timestamps[0]],
            },
            "confidence": 0.6,
        }
    ]
    analysis, warnings = validate_fine_response(payload, request=req)
    # 请求被丢弃
    assert len(analysis.second_refinement_requests) == 0
    assert any("cannot upgrade" in w for w in warnings)


def test_inference_and_observation_separate(tmp_path: Path):
    """推断和观察在不同字段。"""
    req = _request(tmp_path)
    payload = _valid_payload(req)
    analysis, _ = validate_fine_response(payload, request=req)
    ev = analysis.temporal_events[0]
    # inference 是顶层字段，不是 direct_observations
    assert ev.inference is not None
    assert ev.inference not in ev.direct_observations
    # frame_states 只有 description（观察），不含 inference
    for fs in analysis.frame_states:
        assert not hasattr(fs, "inference")


def test_empty_response_returns_minimal(tmp_path: Path):
    """空响应也能通过（模型允许列表为空）。"""
    req = _request(tmp_path)
    payload = {
        "schema_version": "1.0",
        "fine_batch_id": req.fine_batch_id,
        "interval_id": req.interval.interval_id,
        "start": req.interval.start,
        "end": req.interval.end,
        "summary": "empty",
        "frame_states": [],
        "temporal_events": [],
        "subject_motions": [],
        "camera_motions": [],
        "post_transforms": [],
        "text_animations": [],
        "graphic_animations": [],
        "transitions": [],
        "question_answers": [],
        "second_refinement_requests": [],
        "confidence": 0.3,
    }
    analysis, warnings = validate_fine_response(payload, request=req)
    # 问题被补 unresolved
    assert len(analysis.question_answers) == len(req.refinement_questions)
    assert all(qa.status == "unresolved" for qa in analysis.question_answers)


def test_parse_empty_raises():
    with pytest.raises(FineVisionResponseParseError):
        parse_fine_response_payload("")


def test_parse_no_json_raises():
    with pytest.raises(FineVisionResponseParseError):
        parse_fine_response_payload("just plain text no json")


def test_parse_non_object_raises_in_validate(tmp_path: Path):
    """parse 解析成功（合法 JSON 数组），但 validate 阶段拒绝非对象。"""
    req = _request(tmp_path)
    data = parse_fine_response_payload("[1, 2, 3]")
    assert data == [1, 2, 3]  # parse 只做 JSON 提取，不强制对象
    with pytest.raises(FineVisionResponseParseError):
        validate_fine_response(data, request=req)


def test_invalid_category_coerced(tmp_path: Path):
    """非法 category 被规范化为 uncertain_change。"""
    req = _request(tmp_path)
    payload = _valid_payload(req)
    payload["temporal_events"][0]["category"] = "bogus_category"
    analysis, warnings = validate_fine_response(payload, request=req)
    assert analysis.temporal_events[0].category == "uncertain_change"


def test_exact_boundary_collapses_range(tmp_path: Path):
    """precision=exact 时，latest 被折叠到 earliest。"""
    req = _request(tmp_path)
    payload = _valid_payload(req)
    sheet = req.contact_sheets[0]
    payload["temporal_events"][0]["boundary"] = {
        "earliest": sheet.timestamps[0],
        "latest": sheet.timestamps[-1],
        "precision": "exact",
    }
    analysis, warnings = validate_fine_response(payload, request=req)
    ev = analysis.temporal_events[0]
    assert ev.boundary.precision == "exact"
    assert ev.boundary.earliest == ev.boundary.latest
