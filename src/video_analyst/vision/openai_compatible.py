"""Generic OpenAI-compatible multimodal vision provider.

Reads credentials and base_url from environment or config. NO hard-coded
API keys, base URLs, or model names.

Sends a strict JSON schema response_format when supported. The model is asked
to return a single JSON object matching the CoarseBatchAnalysis schema.

Retries 429 and 5xx with exponential backoff. 4xx (other than 429) is
treated as non-retryable. Network timeouts are retried.
"""

from __future__ import annotations

import base64
import json
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
from video_analyst.vision.models import CoarseBatchAnalysis, SCHEMA_VERSION, VisionBatchRequest
from video_analyst.vision.prompts import build_system_prompt, build_user_prompt

log = get_logger("vision.openai_compatible")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Rough per-image base64 budget (≈ 1 MB of JPEG) — enough for ~30 frames.
_MAX_IMAGE_BYTES = 1_500_000


def _redact_bearer(value: str | None) -> str:
    if not value:
        return ""
    return "***REDACTED***"


class OpenAICompatibleProvider(VisionProvider):
    """Provider for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        # 统一模型解析：跨 provider 不互相回退
        from video_analyst.model_resolution import (
            resolve_vision_base_url,
            resolve_vision_model,
        )

        self._api_key = os.environ.get(config.api_key_env, "")
        self._base_url = resolve_vision_base_url(
            "openai-compatible", config.base_url or None, config
        ).rstrip("/")
        self._model = resolve_vision_model(
            "openai-compatible", config.model or None, config
        )

    # --- identity ---

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    @property
    def model_name(self) -> str:
        return self._model

    # --- main API ---

    def analyze_batch(self, request: VisionBatchRequest) -> CoarseBatchAnalysis:
        if not self._api_key:
            raise VisionProviderUnavailableError(
                f"Missing API key for vision provider (env: {self._config.api_key_env})"
            )
        # Read images once; if any file is missing, raise immediately so the
        # batch is marked failed instead of partial.
        images = self._read_images(request.frames)
        messages = self._build_messages(request, images)
        raw = self._chat(messages)
        try:
            payload = self._extract_json_payload(raw)
        except VisionResponseParseError:
            # One deterministic repair attempt
            if self.supports_repair_request():
                try:
                    repaired_text = self._chat_repair(raw, "previous response was not JSON")
                    payload = self._extract_json_payload(repaired_text)
                except (VisionResponseParseError, VisionResponseValidationError) as exc:
                    raise VisionResponseValidationError(
                        f"repair attempt did not yield valid JSON: {exc}"
                    ) from exc
            else:
                raise
        from video_analyst.vision.response_parser import validate_batch_response

        # We don't know the global video_start/video_end here; assume batch bounds.
        # The pipeline aggregate stage will reconcile / clip further if needed.
        analysis, _warnings = validate_batch_response(
            payload,
            request=request,
            video_start=request.batch_start,
            video_end=request.batch_end,
        )
        return analysis

    # --- payload assembly ---

    def _read_images(self, frames) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for f in frames:
            path = Path(f.path)
            if not path.is_file():
                raise VisionRequestError(f"frame file missing: {path}")
            data = path.read_bytes()
            if len(data) > _MAX_IMAGE_BYTES:
                # The provider MUST refuse to silently truncate.
                raise VisionRequestError(
                    f"frame {f.frame_id} too large: {len(data)} bytes > "
                    f"{_MAX_IMAGE_BYTES}"
                )
            b64 = base64.b64encode(data).decode("ascii")
            out.append(
                {
                    "frame_id": f.frame_id,
                    "timestamp_seconds": f.timestamp_seconds,
                    "b64": b64,
                }
            )
        return out

    def _build_messages(
        self, request: VisionBatchRequest, images: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        for img in images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img['b64']}",
                        "detail": "low",
                    },
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": build_user_prompt(
                    request,
                    omitted_frame_ids=None,
                ),
            }
        )
        return [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # --- HTTP ---

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        return self._send_with_retry(url, body)

    def _chat_repair(self, previous_text: str, reason: str) -> str:
        from video_analyst.vision.prompts import build_repair_user_prompt

        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {
                    "role": "user",
                    "content": build_repair_user_prompt(previous_text, reason),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        return self._send_with_retry(url, body)

    def _send_with_retry(self, url: str, body: dict[str, Any]) -> str:
        # Lazy import: tests that only construct the provider shouldn't require httpx.
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise VisionProviderUnavailableError(
                "httpx is required for OpenAICompatibleProvider"
            ) from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
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
                # 4xx (except 429) — do NOT retry
                raise VisionRequestError(
                    f"vision API {status}: {resp.text[:200]} | auth={_redact_bearer(self._api_key)}"
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

        # Should not reach here, but raise defensively.
        if last_exc is not None:
            raise last_exc
        raise VisionRequestError("vision request failed without exception")

    def _extract_message_text(self, data: Any) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionResponseParseError(
                f"unexpected response shape: {exc}"
            ) from exc

    def _extract_json_payload(self, raw_text: str) -> dict[str, Any]:
        from video_analyst.vision.response_parser import parse_response_payload

        return parse_response_payload(raw_text)


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "OpenAICompatibleProvider"]