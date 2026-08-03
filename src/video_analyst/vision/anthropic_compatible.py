"""Anthropic Messages API compatible multimodal vision provider.

复用 OpenAICompatibleProvider 的网络/重试/脱敏/Cache/Telemetry 机制，但使用
Anthropic Messages API (/v1/messages + x-api-key + anthropic-version) 而不是
OpenAI 的 /chat/completions + Authorization: Bearer。

支持：
- 文本块（text content blocks）
- base64 JPEG/PNG image source blocks
- 多张图片
- 文本-only repair call
- 原始响应统一转换为已有 CoarseBatchAnalysis Schema

不扩展新的分析能力，不复制第二套网络客户端（与 OpenAICompatibleProvider 共用
httpx、retry/backoff、redact、cache、telemetry、errors）。
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from video_analyst.config import VisionConfig
from video_analyst.exceptions import (
    VisionProviderUnavailableError,
    VisionRateLimitError,
    VisionRequestError,
    VisionResponseParseError,
    VisionResponseValidationError,
    VisionTimeoutError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.vision.base import VisionProvider
from video_analyst.vision.models import CoarseBatchAnalysis, VisionBatchRequest
from video_analyst.vision.prompts import (
    build_repair_user_prompt,
    build_system_prompt,
    build_user_prompt,
)

log = get_logger("vision.anthropic_compatible")

# Anthropic Messages API 强制要求该 header
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-opus-4-8"

# 与 OpenAI 一致的单图 base64 预算上限
_MAX_IMAGE_BYTES = 1_500_000


def _redact_api_key(value: str | None) -> str:
    if not value:
        return ""
    return "***REDACTED***"


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    # 与 OpenAI provider 保持一致：默认按 jpeg 编码，未知格式也尝试 jpeg
    return "image/jpeg"


class AnthropicCompatibleVisionProvider(VisionProvider):
    """Provider for any Anthropic Messages API (/v1/messages) compatible endpoint."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        # 统一模型解析：跨 provider 不互相回退
        from video_analyst.model_resolution import (
            resolve_vision_base_url,
            resolve_vision_model,
        )

        # Anthropic 推荐的环境变量；缺省回退到 config 中指定的 env 名
        self._api_key = (
            os.environ.get("VIDEO_ANALYST_ANTHROPIC_API_KEY", "")
            or os.environ.get(config.api_key_env, "")
        )
        self._base_url = resolve_vision_base_url(
            "anthropic-compatible", config.base_url or None, config
        ).rstrip("/")
        # Anthropic-compatible 必须显式模型；不存在则抛 VisionModelNotConfiguredError
        self._model = resolve_vision_model(
            "anthropic-compatible", config.model or None, config,
            require_explicit=True,
        )

    # --- identity ---

    @property
    def provider_name(self) -> str:
        return "anthropic-compatible"

    @property
    def model_name(self) -> str:
        return self._model

    # --- main API ---

    def analyze_batch(self, request: VisionBatchRequest) -> CoarseBatchAnalysis:
        if not self._api_key:
            raise VisionProviderUnavailableError(
                "Missing API key for Anthropic-compatible vision provider. "
                "Set VIDEO_ANALYST_ANTHROPIC_API_KEY (preferred) or "
                f"{self._config.api_key_env}."
            )
        images = self._read_images(request.frames)
        messages = self._build_messages(request, images)
        raw = self._messages(messages)
        try:
            payload = self._extract_json_payload(raw)
        except VisionResponseParseError:
            if self.supports_repair_request():
                try:
                    repaired = self._messages_repair(raw, "previous response was not JSON")
                    payload = self._extract_json_payload(repaired)
                except (VisionResponseParseError, VisionResponseValidationError) as exc:
                    raise VisionResponseValidationError(
                        f"repair attempt did not yield valid JSON: {exc}"
                    ) from exc
            else:
                raise
        from video_analyst.vision.response_parser import validate_batch_response

        analysis, _warnings = validate_batch_response(
            payload,
            request=request,
            video_start=request.batch_start,
            video_end=request.batch_end,
        )
        return analysis

    # --- payload assembly ---

    def _read_images(self, frames) -> list[dict[str, Any]]:
        """读取 frame JPEG/PNG 为 base64。Provider MUST refuse silently truncation."""
        out: list[dict[str, Any]] = []
        for f in frames:
            path = Path(f.path)
            if not path.is_file():
                raise VisionRequestError(f"frame file missing: {path}")
            data = path.read_bytes()
            if len(data) > _MAX_IMAGE_BYTES:
                raise VisionRequestError(
                    f"frame {f.frame_id} too large: {len(data)} bytes > {_MAX_IMAGE_BYTES}"
                )
            b64 = base64.b64encode(data).decode("ascii")
            out.append(
                {
                    "frame_id": f.frame_id,
                    "timestamp_seconds": f.timestamp_seconds,
                    "b64": b64,
                    "media_type": _media_type_for(path),
                }
            )
        return out

    def _build_messages(
        self, request: VisionBatchRequest, images: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """按 Anthropic Messages API 构造 content blocks。

        - 图片块：type=image, source.type=base64, media_type, data
        - 文本块：type=text
        - System 提示通过顶层 system 字段（不在 messages 内）
        """
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
        content.append(
            {"type": "text", "text": build_user_prompt(request, omitted_frame_ids=None)}
        )
        return [{"role": "user", "content": content}]

    # --- HTTP ---

    def _messages(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/messages"
        body = {
            "model": self._model,
            "max_tokens": 4096,
            "system": build_system_prompt(),
            "messages": messages,
        }
        return self._send_with_retry(url, body)

    def _messages_repair(self, previous_text: str, reason: str) -> str:
        """文本-only repair。Anthropic 不支持 response_format，故用 prompt 强制 JSON。"""
        url = f"{self._base_url}/v1/messages"
        body = {
            "model": self._model,
            "max_tokens": 4096,
            "system": build_system_prompt(),
            "messages": [
                {"role": "user", "content": build_repair_user_prompt(previous_text, reason)},
            ],
        }
        return self._send_with_retry(url, body)

    def _send_with_retry(self, url: str, body: dict[str, Any]) -> str:
        # 复用 OpenAICompatibleProvider 的同一套 httpx 客户端语义
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise VisionProviderUnavailableError(
                "httpx is required for AnthropicCompatibleVisionProvider"
            ) from exc

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        timeout = self._config.request_timeout
        max_retries = max(0, int(self._config.max_retries))
        backoff_base = max(1.0, float(self._config.retry_backoff_base))

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                log.info(
                    "vision request: provider=%s model=%s attempt=%d",
                    self.provider_name,
                    self._model,
                    attempt,
                )
                resp = httpx.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                last_exc = VisionTimeoutError(f"vision request timed out: {exc}")
                if attempt >= max_retries:
                    raise last_exc from exc
                time.sleep(backoff_base ** attempt)
                continue
            except httpx.HTTPError as exc:
                last_exc = VisionRequestError(f"vision request failed: {exc}")
                if attempt >= max_retries:
                    raise last_exc from exc
                time.sleep(backoff_base ** attempt)
                continue

            status = resp.status_code
            if status == 429:
                last_exc = VisionRateLimitError("vision API returned 429")
                if attempt >= max_retries:
                    raise last_exc
                time.sleep(backoff_base ** attempt)
                continue
            if 500 <= status < 600:
                last_exc = VisionRequestError(f"vision API {status}: {resp.text[:200]}")
                if attempt >= max_retries:
                    raise last_exc
                time.sleep(backoff_base ** attempt)
                continue
            if status >= 400:
                raise VisionRequestError(
                    f"vision API {status}: {resp.text[:200]} | "
                    f"x-api-key={_redact_api_key(self._api_key)}"
                )

            try:
                data = resp.json()
            except Exception as exc:
                raise VisionResponseParseError(
                    f"vision response not JSON: {exc}"
                ) from exc
            text = self._extract_message_text(data)
            if not text:
                raise VisionResponseParseError("vision response had no assistant text")
            return text

        if last_exc is not None:
            raise last_exc
        raise VisionRequestError("vision request failed without exception")

    def _extract_message_text(self, data: Any) -> str:
        """Anthropic 响应：{content: [{type: "text", text: "..."} | ...], usage: {...}}"""
        try:
            content = data["content"]
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "") or ""
            return ""
        except (KeyError, TypeError) as exc:
            raise VisionResponseParseError(
                f"unexpected Anthropic response shape: {exc}"
            ) from exc

    def _extract_json_payload(self, raw_text: str) -> dict[str, Any]:
        from video_analyst.vision.response_parser import parse_response_payload

        return parse_response_payload(raw_text)


__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "AnthropicCompatibleVisionProvider",
]