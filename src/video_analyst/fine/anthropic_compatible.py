"""Anthropic-compatible fine temporal vision provider (Epic 4).

复用 AnthropicCompatibleVisionProvider 的 /v1/messages + x-api-key 传输层
（重试/退避/超时/脱敏/Telemetry），不复制第二套网络客户端。

只覆盖：
- 消息构建（fine system/user prompt）
- 响应解析（fine response_parser）
- 一次格式修复请求（text-only）
"""

from __future__ import annotations

import base64
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
    build_fine_repair_user_prompt,
    build_fine_system_prompt,
    build_fine_user_prompt,
)
from video_analyst.logging.setup import get_logger
from video_analyst.vision.anthropic_compatible import (
    AnthropicCompatibleVisionProvider,
    _media_type_for,
)

log = get_logger("fine.anthropic_compatible")

# 与 coarse provider 一致的单图 base64 预算上限
_MAX_IMAGE_BYTES = 1_500_000


class AnthropicCompatibleFineProvider(AnthropicCompatibleVisionProvider, FineVisionProvider):
    """Fine temporal vision provider for any Anthropic Messages API endpoint.

    继承 ``AnthropicCompatibleVisionProvider`` 的全部 /v1/messages 传输能力
    （headers、retry、backoff、timeout、x-api-key 脱敏），只覆盖消息构建与
    响应解析以适配 fine 分析契约。
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__(config)

    # --- identity ---

    @property
    def provider_name(self) -> str:
        return "anthropic-compatible"

    # --- main API ---

    def analyze_fine_batch(self, request: FineVisionRequest) -> FineBatchAnalysis:
        if not self._api_key:
            raise FineVisionProviderUnavailableError(
                "Missing API key for Anthropic-compatible fine vision provider. "
                "Set VIDEO_ANALYST_ANTHROPIC_API_KEY (preferred) or "
                f"{self._config.api_key_env}."
            )
        images = self._read_images(request)
        messages = self._build_messages(request, images)
        raw = self._messages(messages)
        try:
            payload = self._extract_json_payload(raw)
        except FineVisionResponseParseError:
            if self.supports_repair_request():
                try:
                    repaired = self._messages_repair(raw, "previous response was not JSON")
                    payload = self._extract_json_payload(repaired)
                except (
                    FineVisionResponseParseError,
                    FineVisionResponseValidationError,
                ) as exc:
                    raise FineVisionResponseValidationError(
                        f"repair attempt did not yield valid JSON: {exc}"
                    ) from exc
            else:
                raise
        from video_analyst.fine.response_parser import validate_fine_response

        analysis, _warnings = validate_fine_response(payload, request=request)
        return analysis

    # --- payload assembly (fine-specific) ---

    def _read_images(self, request: FineVisionRequest) -> list[dict[str, Any]]:
        """读取 contact sheets + detail frames 为 base64。

        顺序：先 contact sheets，再 detail frames（与 prompt 描述顺序一致）。
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
                    f"contact sheet {sheet.sheet_id} too large: "
                    f"{len(data)} bytes > {_MAX_IMAGE_BYTES}"
                )
            b64 = base64.b64encode(data).decode("ascii")
            out.append(
                {
                    "role": "contact_sheet",
                    "sheet_id": sheet.sheet_id,
                    "frame_ids": list(sheet.frame_ids),
                    "timestamps": list(sheet.timestamps),
                    "b64": b64,
                    "media_type": _media_type_for(path),
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
                    f"detail frame {df.frame_id} too large: "
                    f"{len(data)} bytes > {_MAX_IMAGE_BYTES}"
                )
            b64 = base64.b64encode(data).decode("ascii")
            out.append(
                {
                    "role": "detail_frame",
                    "frame_id": df.frame_id,
                    "timestamp_seconds": df.timestamp_seconds,
                    "b64": b64,
                    "media_type": _media_type_for(path),
                }
            )
        return out

    def _build_messages(
        self, request: FineVisionRequest, images: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """按 Anthropic Messages API 构造 content blocks。"""
        content: list[dict[str, Any]] = []
        for img in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["b64"],
                    },
                }
            )
        content.append({"type": "text", "text": build_fine_user_prompt(request)})
        return [{"role": "user", "content": content}]

    # --- HTTP（复用父类 _send_with_retry / _extract_message_text） ---

    def _messages(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/messages"
        body = {
            "model": self._model,
            "max_tokens": 4096,
            "system": build_fine_system_prompt(),
            "messages": messages,
        }
        return self._send_with_retry(url, body)

    def _messages_repair(self, previous_text: str, reason: str) -> str:
        url = f"{self._base_url}/v1/messages"
        body = {
            "model": self._model,
            "max_tokens": 4096,
            "system": build_fine_system_prompt(),
            "messages": [
                {"role": "user", "content": build_fine_repair_user_prompt(previous_text, reason)},
            ],
        }
        return self._send_with_retry(url, body)

    def _extract_json_payload(self, raw_text: str) -> dict[str, Any]:
        """复用 coarse 的 parse_response_payload 做 JSON 提取。

        coarse 的 ``parse_response_payload`` 只做 JSON 解析（去 fence / 去前后
        文本），与 fine 无关，可直接复用。
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


# Re-import helper to avoid leaking from module top-level; this keeps the import
# in the function so the file works even if a sibling helper is missing.

__all__ = ["AnthropicCompatibleFineProvider"]