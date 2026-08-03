"""Mock FineVisionProvider for offline testing and end-to-end verification.

Scenarios:
- normal: 基础合法分析
- subject_motion: 主体动作
- camera_motion: 摄像机运动
- text_animation: 文字动画
- graphic_animation: 图形动效
- transition: 转场
- ambiguous: 不确定场景（输出 uncertain + alternatives）
- needs_more_frames: 返回合法的二次补帧请求
- malformed: 返回非法 JSON（测试响应修复）
- timeout: 抛 FineVisionTimeoutError
- rate_limit: 抛 FineVisionRateLimitError
- partial_failure: 返回带非法字段的结果（测试验证器修复一次）

所有场景确定性，不调用外部 API。
"""

from __future__ import annotations

import json
from typing import Any

from video_analyst.config import VisionConfig
from video_analyst.exceptions import (
    FineVisionRateLimitError,
    FineVisionResponseParseError,
    FineVisionTimeoutError,
)
from video_analyst.fine.base import FineVisionProvider
from video_analyst.fine.models import FineBatchAnalysis, FineVisionRequest

MOCK_FINE_PROVIDER_NAME = "mock"


class MockFineVisionProvider(FineVisionProvider):
    """Deterministic mock for fine temporal vision offline testing."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        *,
        scenario: str = "normal",
        model_name: str = "mock-fine-vision-1",
        artificial_delay: float = 0.0,
    ) -> None:
        self._config = config or VisionConfig()
        self._scenario = scenario
        self._model = model_name
        self._delay = float(artificial_delay)
        self.last_request: FineVisionRequest | None = None
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return MOCK_FINE_PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model

    def analyze_fine_batch(self, request: FineVisionRequest) -> FineBatchAnalysis:
        self.call_count += 1
        self.last_request = request

        if self._delay > 0:
            import time

            time.sleep(self._delay)

        scenario = self._scenario
        if scenario == "timeout":
            raise FineVisionTimeoutError("mock fine timeout")
        if scenario == "rate_limit":
            raise FineVisionRateLimitError("mock fine 429")
        if scenario == "malformed":
            raise FineVisionResponseParseError("mock fine returned non-JSON garbage")
        if scenario == "partial_failure":
            return self._partial_failure_response(request)

        payload = self._build_payload(request)
        return self._payload_to_analysis(request, payload)

    # ---- helpers ----

    def _build_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        if self._scenario == "subject_motion":
            return self._subject_motion_payload(request)
        if self._scenario == "camera_motion":
            return self._camera_motion_payload(request)
        if self._scenario == "text_animation":
            return self._text_animation_payload(request)
        if self._scenario == "graphic_animation":
            return self._graphic_animation_payload(request)
        if self._scenario == "transition":
            return self._transition_payload(request)
        if self._scenario == "ambiguous":
            return self._ambiguous_payload(request)
        if self._scenario == "needs_more_frames":
            return self._needs_more_frames_payload(request)
        return self._normal_payload(request)

    def _base_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        """基础合法 payload，各场景在其上扩展。"""
        sheets = request.contact_sheets
        frame_ids: list[str] = []
        sheet_ids: list[str] = []
        timestamps: list[float] = []
        for s in sheets:
            sheet_ids.append(s.sheet_id)
            frame_ids.extend(s.frame_ids)
            timestamps.extend(s.timestamps)
        # 若没有 contact sheet，用 detail frame
        if not frame_ids:
            for df in request.detail_frames:
                frame_ids.append(df.frame_id)
                timestamps.append(df.timestamp_seconds)

        first_ts = timestamps[0] if timestamps else request.interval.start
        last_ts = timestamps[-1] if timestamps else request.interval.end
        first_frame = frame_ids[0] if frame_ids else ""
        last_frame = frame_ids[-1] if frame_ids else ""

        # frame_states：每张帧一个直接观察
        frame_states = []
        for fid, ts in zip(frame_ids, timestamps):
            frame_states.append(
                {
                    "observation_id": f"obs_{fid}",
                    "timestamp_seconds": float(ts),
                    "frame_id": fid,
                    "description": f"Frame at {ts:.3f}s shows static composition.",
                    "visible_subjects": ["subject_a"],
                    "visible_text": [],
                    "composition_notes": ["centered"],
                    "confidence": 0.7,
                }
            )

        return {
            "schema_version": "1.0",
            "fine_batch_id": request.fine_batch_id,
            "interval_id": request.interval.interval_id,
            "start": request.interval.start,
            "end": request.interval.end,
            "summary": f"Fine analysis of {request.fine_batch_id}.",
            "frame_states": frame_states,
            "temporal_events": [],
            "subject_motions": [],
            "camera_motions": [],
            "post_transforms": [],
            "text_animations": [],
            "graphic_animations": [],
            "transitions": [],
            "question_answers": [],
            "second_refinement_requests": [],
            "initial_state": f"Initial state at {first_ts:.3f}s.",
            "final_state": f"Final state at {last_ts:.3f}s.",
            "continuity_notes": [],
            "unresolved_items": [],
            "confidence": 0.6,
            "warnings": [],
            # 内部用的证据引用素材
            "_sheet_ids": sheet_ids,
            "_frame_ids": frame_ids,
            "_timestamps": timestamps,
            "_first_ts": first_ts,
            "_last_ts": last_ts,
            "_first_frame": first_frame,
            "_last_frame": last_frame,
        }

    def _answer_questions(self, request: FineVisionRequest, status: str = "answered") -> list[dict[str, Any]]:
        """逐条回答 refinement_questions。"""
        out = []
        for q in request.refinement_questions:
            out.append(
                {
                    "question": q,
                    "answer": f"Mock answer for: {q}",
                    "status": status,
                    "evidence": {
                        "contact_sheet_ids": [s.sheet_id for s in request.contact_sheets[:1]],
                        "frame_ids": [f for f in (request.contact_sheets[0].frame_ids if request.contact_sheets else [])][:1],
                        "timestamps": [request.interval.start],
                    },
                    "confidence": 0.6,
                }
            )
        return out

    def _normal_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        payload = self._base_payload(request)
        payload["question_answers"] = self._answer_questions(request)
        payload["summary"] = (
            f"Fine batch {request.fine_batch_id}: {len(payload['_frame_ids'])} frames "
            "analyzed. No significant temporal changes detected at this sampling rate."
        )
        return payload

    def _subject_motion_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        payload = self._base_payload(request)
        first_ts = payload["_first_ts"]
        last_ts = payload["_last_ts"]
        first_frame = payload["_first_frame"]
        last_frame = payload["_last_frame"]
        sheet_ids = payload["_sheet_ids"]
        frame_ids = payload["_frame_ids"]
        payload["subject_motions"] = [
            {
                "event_id": f"{request.fine_batch_id}_sm_001",
                "subject": "person_a",
                "action": "head_turn",
                "direction": "right",
                "start_state": f"Facing forward at {first_ts:.3f}s.",
                "end_state": f"Facing right at {last_ts:.3f}s.",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["subject may have turned between samples"],
                "confidence": 0.6,
            }
        ]
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_001",
                "category": "subject_motion",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "description": "Subject head turned from forward to right.",
                "direct_observations": [f"obs_{first_frame}", f"obs_{last_frame}"],
                "inference": "Subject likely turned head to the right between samples.",
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["turn happened between samples"],
                "confidence": 0.6,
                "source_batch_ids": list(request.coarse_context.source_batch_ids),
            }
        ]
        payload["question_answers"] = self._answer_questions(request)
        payload["summary"] = "Subject head turn detected between first and last frame."
        return payload

    def _camera_motion_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        payload = self._base_payload(request)
        first_ts = payload["_first_ts"]
        last_ts = payload["_last_ts"]
        first_frame = payload["_first_frame"]
        last_frame = payload["_last_frame"]
        sheet_ids = payload["_sheet_ids"]
        payload["camera_motions"] = [
            {
                "event_id": f"{request.fine_batch_id}_cm_001",
                "motion_type": "pan_right",
                "description": "Camera pans right across the interval.",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["subject moved left instead (relative motion ambiguous)"],
                "confidence": 0.55,
            }
        ]
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_001",
                "category": "camera_motion",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "description": "Apparent rightward motion of scene content.",
                "direct_observations": [f"obs_{first_frame}", f"obs_{last_frame}"],
                "inference": "Camera likely panned right (could be subject moving left).",
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["subject moved left", "post position transform"],
                "confidence": 0.5,
                "source_batch_ids": list(request.coarse_context.source_batch_ids),
            }
        ]
        payload["question_answers"] = self._answer_questions(request)
        payload["summary"] = "Camera pan right inferred from scene shift."
        return payload

    def _text_animation_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        payload = self._base_payload(request)
        first_ts = payload["_first_ts"]
        last_ts = payload["_last_ts"]
        first_frame = payload["_first_frame"]
        last_frame = payload["_last_frame"]
        sheet_ids = payload["_sheet_ids"]
        # 修改 frame_states：第一帧无文字，末帧有文字
        for fs in payload["frame_states"]:
            if fs["frame_id"] == first_frame:
                fs["visible_text"] = []
                fs["description"] = f"Frame at {first_ts:.3f}s: no visible text."
            if fs["frame_id"] == last_frame:
                fs["visible_text"] = ["HELLO"]
                fs["description"] = f"Frame at {last_ts:.3f}s: text 'HELLO' visible."
        payload["text_animations"] = [
            {
                "event_id": f"{request.fine_batch_id}_ta_001",
                "text_content": "HELLO",
                "animation_types": ["appear"],
                "start_state": f"No text at {first_ts:.3f}s.",
                "end_state": f"Text 'HELLO' visible at {last_ts:.3f}s.",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": [
                    "text may have faded in between samples",
                    "text may have flown in from off-screen",
                ],
                "confidence": 0.6,
            }
        ]
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_001",
                "category": "text_animation",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "description": "Text 'HELLO' appeared between samples.",
                "direct_observations": [f"obs_{first_frame}", f"obs_{last_frame}"],
                "inference": "Text appeared; entry path cannot be determined at this FPS.",
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["fade", "fly-in", "type-on"],
                "confidence": 0.55,
                "source_batch_ids": list(request.coarse_context.source_batch_ids),
            }
        ]
        payload["question_answers"] = self._answer_questions(request)
        payload["summary"] = "Text 'HELLO' appeared between first and last frame."
        return payload

    def _graphic_animation_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        payload = self._base_payload(request)
        first_ts = payload["_first_ts"]
        last_ts = payload["_last_ts"]
        first_frame = payload["_first_frame"]
        last_frame = payload["_last_frame"]
        sheet_ids = payload["_sheet_ids"]
        payload["graphic_animations"] = [
            {
                "event_id": f"{request.fine_batch_id}_ga_001",
                "graphic_subject": "progress_bar",
                "animation_types": ["position", "scale"],
                "start_state": f"Progress bar at 20% at {first_ts:.3f}s.",
                "end_state": f"Progress bar at 80% at {last_ts:.3f}s.",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["bar may have jumped discretely"],
                "confidence": 0.6,
            }
        ]
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_001",
                "category": "graphic_animation",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "description": "Progress bar advanced from 20% to 80%.",
                "direct_observations": [f"obs_{first_frame}", f"obs_{last_frame}"],
                "inference": "Progress bar filled between samples.",
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": ["discrete jump"],
                "confidence": 0.6,
                "source_batch_ids": list(request.coarse_context.source_batch_ids),
            }
        ]
        payload["question_answers"] = self._answer_questions(request)
        payload["summary"] = "Progress bar graphic advanced between samples."
        return payload

    def _transition_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        payload = self._base_payload(request)
        frame_ids = payload["_frame_ids"]
        timestamps = payload["_timestamps"]
        sheet_ids = payload["_sheet_ids"]
        # 用中间帧作为转场点
        if len(frame_ids) >= 2:
            mid_idx = len(frame_ids) // 2
            mid_frame = frame_ids[mid_idx]
            prev_frame = frame_ids[mid_idx - 1]
            mid_ts = timestamps[mid_idx]
            prev_ts = timestamps[mid_idx - 1]
        else:
            mid_frame = frame_ids[-1] if frame_ids else ""
            prev_frame = frame_ids[0] if frame_ids else ""
            mid_ts = timestamps[-1] if timestamps else request.interval.end
            prev_ts = timestamps[0] if timestamps else request.interval.start
        payload["transitions"] = [
            {
                "event_id": f"{request.fine_batch_id}_tr_001",
                "transition_type": "cut",
                "outgoing_state": f"Scene A before {mid_ts:.3f}s.",
                "incoming_state": f"Scene B at {mid_ts:.3f}s.",
                "boundary": {
                    "earliest": prev_ts,
                    "latest": mid_ts,
                    "precision": "bounded",
                    "before_frame_id": prev_frame,
                    "after_frame_id": mid_frame,
                },
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [prev_frame, mid_frame],
                    "timestamps": [prev_ts, mid_ts],
                },
                "alternatives": ["very fast dissolve"],
                "confidence": 0.7,
            }
        ]
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_001",
                "category": "transition",
                "boundary": {
                    "earliest": prev_ts,
                    "latest": mid_ts,
                    "precision": "bounded",
                    "before_frame_id": prev_frame,
                    "after_frame_id": mid_frame,
                },
                "description": "Hard cut between scene A and scene B.",
                "direct_observations": [f"obs_{prev_frame}", f"obs_{mid_frame}"],
                "inference": "Likely a cut (no intermediate blend state observed).",
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [prev_frame, mid_frame],
                    "timestamps": [prev_ts, mid_ts],
                },
                "alternatives": ["very fast dissolve"],
                "confidence": 0.7,
                "source_batch_ids": list(request.coarse_context.source_batch_ids),
            }
        ]
        payload["question_answers"] = self._answer_questions(request)
        payload["summary"] = "Hard cut detected at midpoint."
        return payload

    def _ambiguous_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        """不确定场景：输出 uncertain + 多 alternatives。"""
        payload = self._base_payload(request)
        first_ts = payload["_first_ts"]
        last_ts = payload["_last_ts"]
        first_frame = payload["_first_frame"]
        last_frame = payload["_last_frame"]
        sheet_ids = payload["_sheet_ids"]
        payload["camera_motions"] = [
            {
                "event_id": f"{request.fine_batch_id}_cm_001",
                "motion_type": "uncertain",
                "description": "Frame content scaled up between samples; cannot distinguish zoom/dolly/post-scale.",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": [
                    "optical zoom_in",
                    "dolly_in",
                    "post_transform scale",
                ],
                "confidence": 0.3,
            }
        ]
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_001",
                "category": "uncertain_change",
                "boundary": {
                    "earliest": first_ts,
                    "latest": last_ts,
                    "precision": "bounded",
                    "before_frame_id": first_frame,
                    "after_frame_id": last_frame,
                },
                "description": "Content appears larger at end; cause undetermined.",
                "direct_observations": [f"obs_{first_frame}", f"obs_{last_frame}"],
                "inference": None,
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "alternatives": [
                    "optical zoom_in",
                    "dolly_in",
                    "post_transform scale",
                ],
                "confidence": 0.3,
                "source_batch_ids": list(request.coarse_context.source_batch_ids),
            }
        ]
        payload["question_answers"] = self._answer_questions(request, status="partially_answered")
        payload["summary"] = "Ambiguous scale change; cause undetermined."
        payload["confidence"] = 0.3
        return payload

    def _needs_more_frames_payload(self, request: FineVisionRequest) -> dict[str, Any]:
        """返回结构合法的二次补帧请求。"""
        payload = self._base_payload(request)
        first_ts = payload["_first_ts"]
        last_ts = payload["_last_ts"]
        first_frame = payload["_first_frame"]
        last_frame = payload["_last_frame"]
        sheet_ids = payload["_sheet_ids"]
        current_fps = int(request.interval.requested_fps)
        # 升级规则：5→10, 10→15, 15→缩窄区间
        if current_fps < 10:
            req_fps = 10
            req_type = "higher_temporal_density"
            req_start = request.interval.start
            req_end = min(request.interval.end, request.interval.start + 2.0)
        elif current_fps < 15:
            req_fps = 15
            req_type = "higher_temporal_density"
            req_start = request.interval.start
            req_end = min(request.interval.end, request.interval.start + 2.0)
        else:
            # 15 FPS：缩窄区间
            req_fps = 15
            req_type = "narrower_interval"
            mid = (request.interval.start + request.interval.end) / 2.0
            req_start = max(request.interval.start, mid - 0.5)
            req_end = min(request.interval.end, mid + 0.5)
        payload["second_refinement_requests"] = [
            {
                "request_id": f"{request.fine_batch_id}_2nd_001",
                "request_type": req_type,
                "start": req_start,
                "end": req_end,
                "requested_fps": req_fps,
                "exact_detail_timestamps": [],
                "reason": (
                    f"Current {current_fps} FPS cannot resolve the fast change "
                    f"between {first_ts:.3f}s and {last_ts:.3f}s."
                ),
                "questions": [
                    "What exactly happens between these frames at higher FPS?",
                ],
                "priority": "high",
                "evidence": {
                    "contact_sheet_ids": sheet_ids[:1],
                    "frame_ids": [first_frame, last_frame],
                    "timestamps": [first_ts, last_ts],
                },
                "confidence": 0.6,
            }
        ]
        payload["question_answers"] = self._answer_questions(request, status="partially_answered")
        payload["summary"] = "Insufficient temporal density; second refinement requested."
        payload["unresolved_items"] = ["fast change path undetermined at current FPS"]
        return payload

    def _partial_failure_response(self, request: FineVisionRequest) -> FineBatchAnalysis:
        """返回带非法字段的结果，测试验证器修复一次。

        验证器会：裁剪时间到 interval、移除未知 frame_id、裁剪 confidence。
        """
        payload = self._base_payload(request)
        # 注入非法字段
        payload["temporal_events"] = [
            {
                "event_id": f"{request.fine_batch_id}_ev_bad",
                "category": "subject_motion",
                "boundary": {
                    "earliest": request.interval.start - 10,  # 越界
                    "latest": request.interval.end + 10,  # 越界
                    "precision": "exact",
                },
                "description": "Bogus event with out-of-range boundary.",
                "direct_observations": ["NOT_A_REAL_OBS"],  # 未知
                "inference": "bogus",
                "evidence": {
                    "contact_sheet_ids": ["NOT_A_REAL_SHEET"],  # 未知
                    "frame_ids": ["NOT_A_REAL_FRAME"],  # 未知
                    "timestamps": [0.0],
                },
                "alternatives": [],
                "confidence": 1.5,  # 越界
            }
        ]
        payload["confidence"] = 1.5  # 越界
        return self._payload_to_analysis(request, payload)

    def _payload_to_analysis(
        self, request: FineVisionRequest, payload: dict[str, Any]
    ) -> FineBatchAnalysis:
        """通过验证器把 payload 转成 FineBatchAnalysis。"""
        from video_analyst.fine.response_parser import validate_fine_response

        # 移除内部辅助字段
        clean = {k: v for k, v in payload.items() if not k.startswith("_")}
        analysis, _warnings = validate_fine_response(clean, request=request)
        return analysis

    # ---- Repair hook ----

    def build_repair_payload(self, raw_text: str, reason: str) -> dict[str, Any] | None:
        """提供确定性 repair payload。"""
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
                            "expected_payload": {k: v for k, v in payload.items() if not k.startswith("_")},
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        }

    def parse_repair_response(self, raw_text: str) -> dict[str, Any] | None:
        """从 repair 响应中恢复 expected_payload。"""
        from video_analyst.fine.response_parser import parse_fine_response_payload

        try:
            data = parse_fine_response_payload(raw_text)
        except Exception:
            data = None

        if isinstance(data, dict):
            if "expected_payload" in data and isinstance(data["expected_payload"], dict):
                return data["expected_payload"]
            if "content" in data and isinstance(data["content"], str):
                try:
                    inner = parse_fine_response_payload(data["content"])
                    if isinstance(inner, dict) and isinstance(inner.get("expected_payload"), dict):
                        return inner["expected_payload"]
                except Exception:
                    pass
            msgs = data.get("messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict) and isinstance(last.get("content"), str):
                    try:
                        inner = parse_fine_response_payload(last["content"])
                        if isinstance(inner, dict) and isinstance(inner.get("expected_payload"), dict):
                            return inner["expected_payload"]
                    except Exception:
                        pass
        return data if isinstance(data, dict) else None


__all__ = ["MOCK_FINE_PROVIDER_NAME", "MockFineVisionProvider"]
