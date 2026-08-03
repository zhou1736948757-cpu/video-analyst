"""Mock VisionProvider for offline testing and Paperclip verification.

Scenarios:
- normal: produces a structured but minimal analysis with at most one
  refinement request based on number of frames.
- empty: returns a "no content visible" batch (all observation arrays empty).
- malformed: returns text that fails JSON parsing on first attempt.
- timeout: raises VisionTimeoutError.
- rate_limit: raises VisionRateLimitError.
- partial_failure: returns schema-valid JSON but missing required fields so
  the validator must repair it once.
- needs_refinement: always returns at least one well-formed refinement
  request with high priority.

All scenarios are deterministic given the request, so tests can assert exact
values.
"""

from __future__ import annotations

import json
import time
from typing import Any

from video_analyst.config import VisionConfig
from video_analyst.exceptions import (
    VisionRateLimitError,
    VisionResponseParseError,
    VisionTimeoutError,
)
from video_analyst.vision.base import VisionProvider
from video_analyst.vision.models import CoarseBatchAnalysis, VisionBatchRequest

MOCK_PROVIDER_NAME = "mock"


class MockVisionProvider(VisionProvider):
    """Deterministic mock for offline / no-API testing."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        *,
        scenario: str = "normal",
        model_name: str = "mock-vision-1",
        artificial_delay: float = 0.0,
    ) -> None:
        self._config = config or VisionConfig()
        self._scenario = scenario
        self._model = model_name
        self._delay = float(artificial_delay)
        self.last_request: VisionBatchRequest | None = None
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return MOCK_PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model

    def analyze_batch(self, request: VisionBatchRequest) -> CoarseBatchAnalysis:
        self.call_count += 1
        self.last_request = request

        if self._delay > 0:
            time.sleep(self._delay)

        scenario = self._scenario
        if scenario == "timeout":
            raise VisionTimeoutError("mock timeout")
        if scenario == "rate_limit":
            raise VisionRateLimitError("mock 429")
        if scenario == "malformed":
            raise VisionResponseParseError("mock returned non-JSON garbage")
        if scenario == "partial_failure":
            return self._partial_failure_response(request)

        payload = self._build_payload(request)
        return self._payload_to_analysis(request, payload)

    # ---- helpers ----

    def _build_payload(self, request: VisionBatchRequest) -> dict[str, Any]:
        if self._scenario == "empty":
            return self._empty_payload(request)
        if self._scenario == "needs_refinement":
            return self._needs_refinement_payload(request)
        return self._normal_payload(request)

    def _empty_payload(self, request: VisionBatchRequest) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "batch_id": request.batch_id,
            "start": request.batch_start,
            "end": request.batch_end,
            "summary": "No visually distinguishable content observed.",
            "direct_observations": [],
            "inferred_interpretations": [],
            "visible_text": [],
            "subjects": [],
            "scene_changes": [],
            "camera_observations": [],
            "motion_observations": [],
            "possible_transitions": [],
            "subtitle_visual_relationship": [],
            "important_moments": [],
            "refinement_requests": [],
            "confidence": 0.2,
            "warnings": ["Mock empty scenario: no observations emitted"],
        }

    def _normal_payload(self, request: VisionBatchRequest) -> dict[str, Any]:
        frames = request.frames
        first = frames[0] if frames else None
        last = frames[-1] if frames else None
        ev_ids = [f.frame_id for f in frames[: min(3, len(frames))]]
        timestamps = [f.timestamp_seconds for f in frames[: min(3, len(frames))]]

        scene_change_obs = []
        # Mark the midpoint as a potential scene change
        if len(frames) >= 3:
            mid = frames[len(frames) // 2]
            scene_change_obs.append(
                {
                    "category": "scene_change",
                    "description": "Possible scene transition between "
                    f"{first.timestamp_seconds:.1f}s and {last.timestamp_seconds:.1f}s.",
                    "time_range": {
                        "start": mid.timestamp_seconds,
                        "end": last.timestamp_seconds if last else mid.timestamp_seconds,
                    },
                    "evidence": {
                        "frame_ids": [mid.frame_id],
                        "timestamps": [mid.timestamp_seconds],
                    },
                    "confidence": 0.55,
                }
            )

        refinement_requests: list[dict[str, Any]] = []
        # If many frames, ask for refinement
        if len(frames) > 5:
            refinement_requests.append(
                {
                    "request_id": f"{request.batch_id}_ref_textanim",
                    "start": request.batch_start,
                    "end": min(request.batch_end, request.batch_start + 2.0),
                    "requested_fps": 10,
                    "reason": (
                        "Text-driven animations may occur within the first "
                        "2 seconds; 1 FPS is insufficient to capture motion."
                    ),
                    "questions": [
                        "What animation occurs in the first 2 seconds?",
                    ],
                    "priority": "medium",
                    "evidence_frame_ids": ev_ids[:1],
                    "confidence": 0.6,
                }
            )

        payload = {
            "schema_version": "1.0",
            "batch_id": request.batch_id,
            "start": request.batch_start,
            "end": request.batch_end,
            "summary": (
                f"Batch {request.batch_id}: {len(frames)} frames analyzed "
                "at 1 FPS. Static scenes with at most one possible transition."
            ),
            "direct_observations": [
                {
                    "category": "frame_describe",
                    "description": (
                        f"Frames at {', '.join(str(t) for t in timestamps)} "
                        "show consistent visual content."
                    ),
                    "time_range": {
                        "start": request.batch_start,
                        "end": request.batch_end,
                    },
                    "evidence": {
                        "frame_ids": ev_ids,
                        "timestamps": timestamps,
                    },
                    "confidence": 0.7,
                }
            ],
            "inferred_interpretations": [],
            "visible_text": [],
            "subjects": [],
            "scene_changes": scene_change_obs,
            "camera_observations": [],
            "motion_observations": [],
            "possible_transitions": [],
            "subtitle_visual_relationship": [],
            "important_moments": [],
            "refinement_requests": refinement_requests,
            "confidence": 0.7,
            "warnings": [],
        }
        return payload

    def _needs_refinement_payload(self, request: VisionBatchRequest) -> dict[str, Any]:
        base = self._normal_payload(request)
        # Add a well-formed refinement request with high priority
        frames = request.frames
        ev = [f.frame_id for f in frames[:2]]
        base["refinement_requests"].append(
            {
                "request_id": f"{request.batch_id}_ref_high",
                "start": request.batch_start,
                "end": min(request.batch_end, request.batch_start + 3.0),
                "requested_fps": 15,
                "reason": (
                    "Possible fast motion and text animation near "
                    f"{request.batch_start:.1f}s; 1 FPS cannot resolve "
                    "transitions or motion paths."
                ),
                "questions": [
                    "What fast motion or animation happens between these frames?",
                    "Is there a transition effect at the start of this batch?",
                ],
                "priority": "high",
                "evidence_frame_ids": ev,
                "confidence": 0.65,
            }
        )
        base["warnings"].append(
            "needs_refinement scenario: refinement request emitted by design"
        )
        return base

    def _partial_failure_response(self, request: VisionBatchRequest) -> CoarseBatchAnalysis:
        """Return a CoarseBatchAnalysis that violates one or two semantic
        invariants so the validator must repair it (e.g. unknown frame ids,
        out-of-range time_range). The validator will coerce / clamp."""
        from video_analyst.vision.models import (
            Observation,
            TimeRange,
            EvidenceReference,
        )

        # The response_parser clamps out-of-range time_range and drops unknown
        # frame_ids, but Pydantic forbids confidence > 1 at construction. So we
        # construct valid pydantic objects and let the validate path clamp the
        # values that bypass pydantic (time_range) and drop unknown evidence.
        bad_obs = Observation(
            category="out_of_range",
            description="Bogus observation with bogus times.",
            time_range=TimeRange(
                start=request.batch_start,
                end=request.batch_end,
            ),
            evidence=EvidenceReference(
                frame_ids=["NOT_A_REAL_FRAME"],
                timestamps=[0.0],
            ),
            confidence=1.0,
        )
        # Ask the response parser to coerce; it will clamp time_range, drop
        # unknown frame_ids, and clamp confidence if needed.
        payload = {
            "schema_version": "1.0",
            "batch_id": request.batch_id,
            "start": request.batch_start,
            "end": request.batch_end,
            "summary": "",
            "direct_observations": [
                {
                    "category": "out_of_range",
                    "description": "Bogus observation with bogus times.",
                    "time_range": {
                        "start": request.batch_start - 10,
                        "end": request.batch_end + 10,
                    },
                    "evidence": {
                        "frame_ids": ["NOT_A_REAL_FRAME"],
                        "timestamps": [0.0],
                    },
                    "confidence": 1.5,  # out of range — must clamp
                }
            ],
            "inferred_interpretations": [],
            "visible_text": [],
            "subjects": [],
            "scene_changes": [],
            "camera_observations": [],
            "motion_observations": [],
            "possible_transitions": [],
            "subtitle_visual_relationship": [],
            "important_moments": [],
            "refinement_requests": [],
            "confidence": 0.5,
            "warnings": ["mock partial_failure"],
        }
        return self._payload_to_analysis(request, payload)

    def _payload_to_analysis(
        self, request: VisionBatchRequest, payload: dict[str, Any]
    ) -> CoarseBatchAnalysis:
        # Round-trip via the validator so the caller gets a strict typed object.
        from video_analyst.vision.response_parser import validate_batch_response

        analysis, _warnings = validate_batch_response(
            payload,
            request=request,
            video_start=request.batch_start,
            video_end=request.batch_end,
        )
        return analysis

    # ---- Repair hook (mock can repair) ----

    def build_repair_payload(self, raw_text: str, reason: str) -> dict[str, Any] | None:
        """Provide a deterministic repair payload — a properly-formed JSON
        dict for the batch. We re-use _normal_payload."""
        if self.last_request is None:
            return None
        payload = self._normal_payload(self.last_request)
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Return STRICT JSON only.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "repair",
                            "reason": reason,
                            "previous_text": raw_text[:2000],
                            "expected_payload": payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        }

    def parse_repair_response(self, raw_text: str) -> dict[str, Any] | None:
        """Recover the expected_payload from the most recent repair request.

        Accepts any of these shapes:
        - The raw message-content JSON string (parsed to find ``expected_payload``)
        - A ``{"role": ..., "content": "<json>"}`` message object
        - A ``{"messages": [...]}`` wrapper
        """
        from video_analyst.vision.response_parser import parse_response_payload

        try:
            data = parse_response_payload(raw_text)
        except Exception:
            data = None

        if isinstance(data, dict):
            # Direct: {"expected_payload": {...}}
            if "expected_payload" in data and isinstance(data["expected_payload"], dict):
                return data["expected_payload"]
            # Single message: {"role": ..., "content": "<json>"}
            if "content" in data and isinstance(data["content"], str):
                try:
                    inner = parse_response_payload(data["content"])
                    if isinstance(inner, dict) and isinstance(inner.get("expected_payload"), dict):
                        return inner["expected_payload"]
                except Exception:
                    pass
            # Wrapper: {"messages": [...]} → take the last message's content
            msgs = data.get("messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict) and isinstance(last.get("content"), str):
                    try:
                        inner = parse_response_payload(last["content"])
                        if isinstance(inner, dict) and isinstance(inner.get("expected_payload"), dict):
                            return inner["expected_payload"]
                    except Exception:
                        pass
        return data if isinstance(data, dict) else None


__all__ = ["MOCK_PROVIDER_NAME", "MockVisionProvider"]