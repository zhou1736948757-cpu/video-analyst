"""Provider Capability Probe (Epic 6).

发送一个最小测试请求探测 Provider 能力，不上传用户视频。
检查：
- API Key 是否配置
- Endpoint 是否可访问
- Model 是否存在
- Vision 是否支持
- JSON mode 是否支持
- Usage 是否可读取

结果缓存但允许刷新。
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_analyst.config import VisionConfig
from video_analyst.exceptions import (
    FineVisionProviderUnavailableError,
    VisionProviderUnavailableError,
)
from video_analyst.logging.setup import get_logger

log = get_logger("vision.provider_check")


@dataclass
class ProviderCapabilities:
    """Provider 能力模型。"""

    supports_images: bool = False
    supports_multiple_images: bool = False
    supports_json_mode: bool = False
    supports_json_schema: bool = False
    returns_usage: bool = False
    max_images_known: int | None = None
    max_request_bytes_known: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ProviderCheckResult:
    """provider-check 结果。"""

    provider: str
    model: str
    api_key_configured: bool = False
    endpoint_reachable: bool = False
    authentication_accepted: bool = False
    model_available: bool = False
    text_request_supported: bool = False
    image_input_supported: bool = False
    multiple_images_supported: bool = False
    json_output_supported: str | None = None  # "native" | "prompt_forced_json" | None
    usage_returned: bool = False
    # legacy aliases
    model_exists: bool = False
    vision_supported: bool = False
    json_mode_supported: bool = False
    usage_available: bool = False
    # network evidence
    network_request_sent: bool = False
    network_scope: str | None = None  # "none" | "local_mock" | "external_provider"
    response_received: bool = False
    latency_ms: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    # local artifact path for external evidence
    artifact_path: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.api_key_configured
            and self.endpoint_reachable
            and self.model_available
            and self.image_input_supported
        )


def _make_test_image_bytes() -> bytes:
    """生成一张无敏感信息测试图片（PNG）。"""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (64, 64), (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.rectangle([16, 16, 48, 48], fill=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # 无 Pillow 时返回最小 PNG（1x1 灰）
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63f8cf00000003000100184e6e440000000049454e44ae426082"
        )


def _append_json_mode_issue(result: ProviderCheckResult, text: str) -> None:
    """把 JSON mode 不支持的提示加入 notes。"""
    lowered = text.lower()
    if "json" in lowered or "response_format" in lowered:
        result.capabilities.notes.append(
            "JSON mode may not be supported; will retry without response_format"
        )


def check_openai_compatible_provider(
    config: VisionConfig,
    *,
    timeout: float = 30.0,
) -> ProviderCheckResult:
    """探测 OpenAI-compatible Provider 能力。

    只发送一个最小测试请求（一张小测试图片 + 极简 prompt），不上传用户视频。
    """
    import os
    import time

    import httpx

    from video_analyst.model_resolution import resolve_vision_model

    # 统一解析：跨 provider 不互相回退
    model = resolve_vision_model("openai-compatible", config=config)
    base_url = (
        config.base_url
        or os.environ.get(config.base_url_env, "")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    api_key = os.environ.get(config.api_key_env, "")

    result = ProviderCheckResult(provider="openai-compatible", model=model)

    if not api_key:
        result.api_key_configured = False
        result.error_type = "missing_api_key"
        result.error_message = f"API key not set in env {config.api_key_env}"
        return result
    result.api_key_configured = True

    test_img = _make_test_image_bytes()
    b64 = base64.b64encode(test_img).decode("ascii")

    # 尝试带 JSON mode 的请求
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one word. Return JSON: {\"word\": \"...\"}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Classify network scope
    from video_analyst.telemetry.network_scope import classify_network_scope
    net_scope, ep_host = classify_network_scope(url)
    result.network_scope = net_scope

    start = time.monotonic()
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        result.network_request_sent = True
    except httpx.TimeoutException:
        result.network_request_sent = True
        result.error_type = "timeout"
        result.error_message = f"request timed out after {timeout}s"
        return result
    except httpx.HTTPError as exc:
        result.network_request_sent = True
        result.error_type = "network_error"
        result.error_message = str(exc)[:200]
        return result
    finally:
        result.latency_ms = (time.monotonic() - start) * 1000.0

    result.endpoint_reachable = resp.status_code < 500
    result.response_received = True

    if resp.status_code == 404:
        result.error_type = "model_not_found"
        result.error_message = f"model {model} not found (404)"
        return result
    if resp.status_code == 401 or resp.status_code == 403:
        result.error_type = "authentication"
        result.error_message = f"auth failed ({resp.status_code})"
        return result
    if resp.status_code == 400:
        # 可能是 JSON mode 不支持
        text = resp.text[:500]
        result.error_type = "bad_request"
        result.error_message = text
        _append_json_mode_issue(result, text)
        # 降级重试（无 JSON mode）
        return _retry_without_json_mode(base_url, api_key, model, b64, result, timeout)
    if resp.status_code >= 400:
        result.error_type = f"http_{resp.status_code}"
        result.error_message = resp.text[:200]
        return result

    # 成功
    result.model_exists = True
    result.model_available = True
    result.authentication_accepted = True
    result.vision_supported = True
    result.image_input_supported = True
    result.multiple_images_supported = True
    result.json_mode_supported = True
    result.json_output_supported = "native"
    result.text_request_supported = True
    result.network_request_sent = True
    result.response_received = True
    try:
        data = resp.json()
        usage = data.get("usage")
        if usage and (usage.get("prompt_tokens") or usage.get("total_tokens")):
            result.returns_usage = True
            result.usage_available = True
            result.usage_returned = True
    except Exception:
        result.returns_usage = False

    result.capabilities = ProviderCapabilities(
        supports_images=True,
        supports_multiple_images=True,  # 假设支持（多数 OpenAI-compatible 支持）
        supports_json_mode=True,
        supports_json_schema=False,
        returns_usage=result.returns_usage,
        notes=[],
    )
    return result


def _retry_without_json_mode(
    base_url: str,
    api_key: str,
    model: str,
    b64: str,
    result: ProviderCheckResult,
    timeout: float,
) -> ProviderCheckResult:
    """降级重试：移除 response_format。"""
    import httpx

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one word."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    except Exception as exc:
        result.error_type = "network_error"
        result.error_message = str(exc)[:200]
        return result
    if resp.status_code < 400:
        result.model_exists = True
        result.model_available = True
        result.authentication_accepted = True
        result.vision_supported = True
        result.image_input_supported = True
        result.json_mode_supported = False
        result.json_output_supported = "prompt_forced_json"
        result.text_request_supported = True
        result.network_request_sent = True
        result.response_received = True
        result.capabilities = ProviderCapabilities(
            supports_images=True,
            supports_multiple_images=True,
            supports_json_mode=False,
            returns_usage=bool(resp.json().get("usage")),
            notes=["JSON mode not supported; using plain text + parser fallback"],
        )
        result.error_type = None
        result.error_message = None
    return result


def check_mock_provider(config: VisionConfig) -> ProviderCheckResult:
    """Mock provider 始终 ready（不需要 API Key）。"""
    result = ProviderCheckResult(provider="mock", model="mock-vision-1")
    result.api_key_configured = True
    result.endpoint_reachable = True
    result.authentication_accepted = True
    result.model_available = True
    result.model_exists = True
    result.text_request_supported = True
    result.image_input_supported = True
    result.multiple_images_supported = True
    result.json_output_supported = "native"
    result.vision_supported = True
    result.json_mode_supported = True
    result.usage_available = False
    result.usage_returned = False
    result.network_request_sent = False
    result.network_scope = "none"
    result.response_received = False
    result.capabilities = ProviderCapabilities(
        supports_images=True,
        supports_multiple_images=True,
        supports_json_mode=True,
        returns_usage=False,
        notes=["Mock provider; no network calls"],
    )
    return result


def check_anthropic_compatible_provider(
    config: VisionConfig,
    *,
    timeout: float = 30.0,
) -> ProviderCheckResult:
    """探测 Anthropic Messages API Provider 能力。

    探测步骤：
    1. Text-only 请求 → 验证 endpoint / auth / model / text support
    2. Image+text 请求 → 验证 image_input_supported
    3. Multiple images 请求 → 验证 multiple_images_supported
    4. 记录 network_scope / endpoint_host / response_received

    只发送程序生成的 64×64 无敏感测试图片，不上传用户视频。
    Anthropic 不支持 ``response_format`` JSON mode；
    ``json_output_supported`` 标记为 ``prompt_forced_json``。
    """
    import os
    import time

    import httpx

    from video_analyst.model_resolution import (
        ANTHROPIC_API_KEY_ENV,
        ANTHROPIC_BASE_URL_ENV,
        resolve_vision_base_url,
        resolve_vision_model,
    )
    from video_analyst.telemetry.network_scope import classify_network_scope
    from video_analyst.vision.anthropic_compatible import ANTHROPIC_VERSION

    # 统一解析（不再硬编码 gpt-4o-mini 默认）
    model = resolve_vision_model("anthropic-compatible", config=config)
    base_url = resolve_vision_base_url("anthropic-compatible", config=config)
    api_key = (
        os.environ.get(ANTHROPIC_API_KEY_ENV, "")
        or os.environ.get(config.api_key_env, "")
    )

    result = ProviderCheckResult(provider="anthropic-compatible", model=model)

    if not api_key:
        result.api_key_configured = False
        result.error_type = "missing_api_key"
        result.error_message = (
            f"API key not set in env {ANTHROPIC_API_KEY_ENV} or {config.api_key_env}"
        )
        return result
    result.api_key_configured = True

    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }

    # Classify network scope from URL
    net_scope, ep_host = classify_network_scope(url)
    result.network_scope = net_scope
    result.endpoint_host = ep_host  # type: ignore[attr-defined]

    # ---- Step 1: Text-only probe ----
    text_body: dict[str, Any] = {
        "model": model,
        "max_tokens": 32,
        "system": "You are a capability probe. Respond in plain text only.",
        "messages": [
            {"role": "user", "content": "Reply with the word OK."}
        ],
    }

    start = time.monotonic()
    try:
        resp = httpx.post(url, headers=headers, json=text_body, timeout=timeout)
        result.network_request_sent = True
        result.response_received = True
    except httpx.TimeoutException:
        result.network_request_sent = True
        result.error_type = "timeout"
        result.error_message = f"request timed out after {timeout}s"
        result.latency_ms = (time.monotonic() - start) * 1000.0
        return result
    except httpx.HTTPError as exc:
        result.network_request_sent = True
        result.error_type = "network_error"
        result.error_message = str(exc)[:200]
        result.latency_ms = (time.monotonic() - start) * 1000.0
        return result
    finally:
        if result.latency_ms is None:
            result.latency_ms = (time.monotonic() - start) * 1000.0

    result.endpoint_reachable = resp.status_code < 500

    if resp.status_code == 401 or resp.status_code == 403:
        result.error_type = "authentication"
        result.error_message = f"auth failed ({resp.status_code})"
        return result
    if resp.status_code == 404:
        result.error_type = "model"
        result.error_message = f"model {model} not found (404)"
        return result
    if resp.status_code == 429:
        result.error_type = "rate_limit"
        result.error_message = "rate limited (429)"
        return result
    if 500 <= resp.status_code < 600:
        result.error_type = "server_error"
        result.error_message = f"server error ({resp.status_code})"
        return result
    if resp.status_code >= 400:
        result.error_type = "invalid_response"
        result.error_message = resp.text[:200]
        return result

    # Text request succeeded
    result.authentication_accepted = True
    result.model_available = True
    result.model_exists = True
    result.text_request_supported = True
    try:
        data = resp.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
            result.returns_usage = True
            result.usage_available = True
            result.usage_returned = True
    except Exception:
        pass

    # ---- Step 2: Image probe ----
    test_img = _make_test_image_bytes()
    b64 = base64.b64encode(test_img).decode("ascii")

    img_body: dict[str, Any] = {
        "model": model,
        "max_tokens": 64,
        "system": "You are a capability probe. Respond in plain text only.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one word.",
                    },
                ],
            }
        ],
    }

    img_start = time.monotonic()
    try:
        img_resp = httpx.post(url, headers=headers, json=img_body, timeout=timeout)
    except httpx.TimeoutException:
        result.error_type = "timeout"
        result.error_message = "image probe timed out"
        result.latency_ms = (time.monotonic() - img_start) * 1000.0
        result.capabilities.notes.append(
            "endpoint is reachable but not suitable for Video Analyst vision workloads"
        )
        return result
    except httpx.HTTPError as exc:
        result.error_type = "image_support"
        result.error_message = f"image probe failed: {str(exc)[:200]}"
        result.capabilities.notes.append(
            "endpoint is reachable but not suitable for Video Analyst vision workloads"
        )
        return result

    if img_resp.status_code == 401 or img_resp.status_code == 403:
        result.error_type = "authentication"
        result.error_message = "image probe auth failed"
        return result
    if img_resp.status_code == 400:
        result.error_type = "image_support"
        result.error_message = img_resp.text[:300]
        result.capabilities.notes.append(
            "endpoint is reachable but not suitable for Video Analyst vision workloads"
        )
        return result
    if img_resp.status_code >= 400:
        result.error_type = f"http_{img_resp.status_code}"
        result.error_message = img_resp.text[:200]
        return result

    # Image probe succeeded
    result.image_input_supported = True
    result.vision_supported = True
    result.latency_ms = (time.monotonic() - img_start) * 1000.0

    # Check usage from image response too
    try:
        img_data = img_resp.json()
        img_usage = img_data.get("usage") if isinstance(img_data, dict) else None
        if img_usage and (img_usage.get("input_tokens") or img_usage.get("output_tokens")):
            result.returns_usage = True
            result.usage_available = True
            result.usage_returned = True
    except Exception:
        pass

    # ---- Step 3: Multiple images probe ----
    multi_body: dict[str, Any] = {
        "model": model,
        "max_tokens": 64,
        "system": "You are a capability probe. Respond in plain text only.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe these images in one word each.",
                    },
                ],
            }
        ],
    }

    try:
        multi_resp = httpx.post(url, headers=headers, json=multi_body, timeout=timeout)
        if multi_resp.status_code < 400:
            result.multiple_images_supported = True
        else:
            result.multiple_images_supported = False
            result.capabilities.notes.append(
                "multiple images not supported; single-image batches only"
            )
    except Exception:
        result.multiple_images_supported = False

    # Anthropic has no native JSON response_format; coarse provider uses prompt-forced JSON
    result.json_output_supported = "prompt_forced_json"
    result.json_mode_supported = False

    result.capabilities = ProviderCapabilities(
        supports_images=result.image_input_supported,
        supports_multiple_images=result.multiple_images_supported,
        supports_json_mode=False,
        supports_json_schema=False,
        returns_usage=result.returns_usage,
        notes=[
            "Anthropic Messages API: no JSON response_format; coarse provider "
            "uses prompt-forced JSON + parse_response_payload.",
            "x-api-key + anthropic-version headers (not Authorization: Bearer).",
        ],
    )

    # Save artifact for external evidence
    _save_provider_check_artifact(result, config)

    return result


def _save_provider_check_artifact(
    result: ProviderCheckResult, config: VisionConfig
) -> None:
    """保存 provider check 结果到本地 artifact 文件。"""
    try:
        from pathlib import Path

        artifact_dir = Path("validation/anthropic")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "provider_check_real.json"

        payload = {
            "provider": result.provider,
            "model": result.model,
            "api_key_configured": result.api_key_configured,
            "endpoint_reachable": result.endpoint_reachable,
            "authentication_accepted": result.authentication_accepted,
            "model_available": result.model_available,
            "text_request_supported": result.text_request_supported,
            "image_input_supported": result.image_input_supported,
            "multiple_images_supported": result.multiple_images_supported,
            "json_output_supported": result.json_output_supported,
            "usage_returned": result.usage_returned,
            "network_request_sent": result.network_request_sent,
            "network_scope": result.network_scope,
            "response_received": result.response_received,
            "latency_ms": result.latency_ms,
            "ready": result.ready,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
        artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        result.artifact_path = str(artifact_path)
    except Exception:
        # Artifact saving must not break provider check
        pass


def run_provider_check(
    provider: str,
    config: VisionConfig,
) -> ProviderCheckResult:
    """统一入口：根据 provider 类型执行探测。"""
    if provider == "mock":
        return check_mock_provider(config)
    if provider == "openai-compatible":
        return check_openai_compatible_provider(config)
    if provider == "anthropic-compatible":
        return check_anthropic_compatible_provider(config)
    if provider == "gemini":
        result = ProviderCheckResult(provider="gemini", model=config.model or "")
        result.error_type = "not_implemented"
        result.error_message = "Gemini provider is a fail-fast stub"
        return result
    result = ProviderCheckResult(provider=provider, model=config.model or "")
    result.error_type = "unknown_provider"
    result.error_message = f"unknown provider: {provider}"
    return result


__all__ = [
    "ProviderCapabilities",
    "ProviderCheckResult",
    "check_anthropic_compatible_provider",
    "check_mock_provider",
    "check_openai_compatible_provider",
    "run_provider_check",
]