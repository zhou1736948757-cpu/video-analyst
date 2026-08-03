"""OpenAI-compatible fine temporal vision provider (Epic 4).

复用 Epic 2 的 ``OpenAICompatibleProvider`` HTTP 传输层（重试 / 退避 / 超时 /
API Key 读取 / 脱敏），不复制新的网络客户端。本类只覆盖：
- 消息构建（fine system/user prompt）
- 响应解析（fine response_parser）
- 一次格式修复请求
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from video_analyst.config import VisionConfig
from video_analyst.exceptions import (
    FineVisionProviderUnavailableError,
    FineVisionResponseParseError,
    FineVisionResponseValidationError,
)
from video_analyst.fine.base import FineVisionProvider
from video_analyst.fine.models import FineBatchAnalysis, FineVisionRequest
from video_analyst.fine.prompts import (
    PROMPT_VERSION,
    build_fine_repair_user_prompt,
    build_fine_system_prompt,
    build_fine_user_prompt,
)
from video_analyst.logging.setup import get_logger
from video_analyst.vision.openai_compatible import OpenAICompatibleProvider

log = get_logger("fine.openai_compatible")

# 单张图片 base64 预算上限（与 coarse 一致，避免巨型图片拖垮请求）
_MAX_IMAGE_BYTES = 1_500_000


class OpenAICompatibleFineProvider(OpenAICompatibleProvider, FineVisionProvider):
    """Fine temporal vision provider for any OpenAI-compatible endpoint.

    继承 ``OpenAICompatibleProvider`` 的全部 HTTP 传输能力（``__init__`` /
    ``_send_with_retry`` / ``_extract_message_text`` / ``_redact_bearer``），
    只覆盖消息构建与响应解析以适配 fine 分析契约。
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__(config)

    # --- identity ---

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    # --- main API ---

    def analyze_fine_batch(self, request: FineVisionRequest) -> FineBatchAnalysis:
        if not self._api_key:
            raise FineVisionProviderUnavailableError(
                f"Missing API key for fine vision provider (env: {self._config.api_key_env})"
            )
        images = self._read_images(request)
        messages = self._build_messages(request, images)
        raw = self._chat(messages)
        try:
            payload = self._extract_json_payload(raw)
        except FineVisionResponseParseError:
            # 一次确定性修复
            if self.supports_repair_request():
                try:
                    repaired_text = self._chat_repair(raw, "previous response was not JSON")
                    payload = self._extract_json_payload(repaired_text)
                except (FineVisionResponseParseError, FineVisionResponseValidationError) as exc:
                    raise FineVisionResponseValidationError(
                        f"repair attempt did not yield valid JSON: {exc}"
                    ) from exc
            else:
                raise
        from video_analyst.fine.response_parser import validate_fine_response

        analysis, _warnings = validate_fine_response(payload, request=request)
        return analysis

    # --- payload assembly ---

    def _read_images(self, request: FineVisionRequest) -> list[dict[str, Any]]:
        """读取 contact sheets + detail frames 为 base64。

        顺序：先全部 contact sheets，再 detail frames（与 prompt 描述顺序一致）。
        每张图片附带 frame_id / timestamp / role，供日志与调试。
        """
        out: list[dict[str, Any]] = []
        for sheet in request.contact_sheets:
            path = Path(sheet.path)
            if not path.is_file():
                raise FineVisionProviderUnavailableError(
                    f"contact sheet file missing: {path}"
                )
            data = path.read_bytes()
            if len(data) > _MAX_IMAGE_BYTES:
                raise FineVisionProviderUnavailableError(
                    f"contact sheet {sheet.sheet_id} too large: {len(data)} bytes"
                )
            b64 = base64.b64encode(data).decode("ascii")
            out.append(
                {
                    "role": "contact_sheet",
                    "sheet_id": sheet.sheet_id,
                    "frame_ids": list(sheet.frame_ids),
                    "timestamps": list(sheet.timestamps),
                    "b64": b64,
                }
            )
        for df in request.detail_frames:
            path = Path(df.path)
            if not path.is_file():
                raise FineVisionProviderUnavailableError(
                    f"detail frame file missing: {path}"
                )
            data = path.read_bytes()
            if len(data) > _MAX_IMAGE_BYTES:
                raise FineVisionProviderUnavailableError(
                    f"detail frame {df.frame_id} too large: {len(data)} bytes"
                )
            b64 = base64.b64encode(data).decode("ascii")
            out.append(
                {
                    "role": "detail_frame",
                    "frame_id": df.frame_id,
                    "timestamp_seconds": df.timestamp_seconds,
                    "b64": b64,
                }
            )
        return out

    def _build_messages(
        self, request: FineVisionRequest, images: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        # 图片按 contact sheets → detail frames 顺序发送
        for img in images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img['b64']}",
                        "detail": "high",
                    },
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": build_fine_user_prompt(request),
            }
        )
        return [
            {"role": "system", "content": build_fine_system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # --- HTTP（复用父类 _send_with_retry / _extract_message_text） ---

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        # 复用父类传输层（含 429/5xx 重试、超时、脱敏日志）
        return self._send_with_retry(url, body)

    def _chat_repair(self, previous_text: str, reason: str) -> str:
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": build_fine_system_prompt()},
                {
                    "role": "user",
                    "content": build_fine_repair_user_prompt(previous_text, reason),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        return self._send_with_retry(url, body)

    def _extract_json_payload(self, raw_text: str) -> dict[str, Any]:
        """复用 coarse 的 parse_response_payload 做 JSON 提取。

        coarse 的 ``parse_response_payload`` 只做 JSON 解析（去 fence / 去前后文本），
        与 fine 无关，可直接复用。
        """
        from video_analyst.vision.response_parser import parse_response_payload
        from video_analyst.exceptions import VisionResponseParseError

        try:
            return parse_response_payload(raw_text)
        except VisionResponseParseError as exc:
            raise FineVisionResponseParseError(str(exc)) from exc

    # --- repair hooks ---

    def supports_repair_request(self) -> bool:
        return True


__all__ = ["OpenAICompatibleFineProvider"]
