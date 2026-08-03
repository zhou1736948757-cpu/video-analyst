"""Unified vision-model resolver (Epic 1-6 compatible).

为不同的 vision provider 提供统一的"模型名解析"函数，避免出现：

* Anthropic-compatible provider 错误回退到 OpenAI 默认 ``gpt-4o-mini``
* Dry Run 看到的模型与实际请求体的模型不一致
* Provider Check、Dry Run、Telemetry、缓存键使用不同模型名

**优先级（Anthropic-compatible）**::

    CLI --vision-model
    > VIDEO_ANALYST_ANTHROPIC_MODEL
    > config.vision.model（若已经是 Anthropic provider 的设置）
    > Anthropic Provider 默认模型

**优先级（OpenAI-compatible）**::

    CLI --vision-model
    > VIDEO_ANALYST_VISION_MODEL
    > config.vision.model
    > OpenAI Provider 默认模型

不在解析路径上跨 provider 互相回退：Anthropic 解析只看 Anthropic 自己的源，
OpenAI 解析只看 OpenAI 自己的源。
"""

from __future__ import annotations

import os
from typing import Mapping

from video_analyst.config import VisionConfig

# Per-provider env var overrides (different from the generic config env names)
ANTHROPIC_MODEL_ENV = "VIDEO_ANALYST_ANTHROPIC_MODEL"
ANTHROPIC_BASE_URL_ENV = "VIDEO_ANALYST_ANTHROPIC_BASE_URL"
ANTHROPIC_API_KEY_ENV = "VIDEO_ANALYST_ANTHROPIC_API_KEY"


# 已知 provider 的默认模型（与各 Provider 内部 DEFAULT_MODEL 一致）
_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "mock": "mock-vision-1",
    "openai-compatible": "gpt-4o-mini",
    "anthropic-compatible": "claude-opus-4-8",
    "gemini": "gemini-1.5-flash",
}

# 已知 provider 的默认 base URL（与各 Provider 内部 DEFAULT_BASE_URL 一致）
_PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "mock": "",
    "openai-compatible": "https://api.openai.com/v1",
    "anthropic-compatible": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}


def provider_default_model(provider_name: str) -> str:
    """返回指定 provider 的默认模型名。未知 provider 走通用兜底。"""
    return _PROVIDER_DEFAULT_MODELS.get(
        provider_name, _PROVIDER_DEFAULT_MODELS["openai-compatible"]
    )


def provider_default_base_url(provider_name: str) -> str:
    """返回指定 provider 的默认 base URL。"""
    return _PROVIDER_DEFAULT_BASE_URLS.get(provider_name, "")


def resolve_vision_model(
    provider_name: str,
    cli_model: str | None = None,
    config: VisionConfig | None = None,
    environ: Mapping[str, str] | None = None,
    require_explicit: bool = False,
) -> str:
    """统一解析 vision 模型名。

    严格不跨 provider 回退：Anthropic 不会回退到 OpenAI 默认，OpenAI 不会回退
    到 Anthropic 默认。

    优先级：

    1. ``cli_model``（CLI ``--vision-model``）
    2. provider-specific env（Anthropic 用 ``VIDEO_ANALYST_ANTHROPIC_MODEL``，
       其他用 ``VIDEO_ANALYST_VISION_MODEL``）
    3. ``config.vision.model``（若与 provider 不一致也接受，但优先用 env）
    4. provider 默认模型（OpenAI / mock 保留；Anthropic 三源缺失时显式失败）

    ``require_explicit=True`` 时：Anthropic-compatible 三源均缺失抛
    ``VisionModelNotConfiguredError``，不允许默认回退。
    """
    from video_analyst.budget_guard import VisionModelNotConfiguredError

    env = environ if environ is not None else os.environ
    # 1. CLI 参数最高优先级
    if cli_model:
        return cli_model
    # 2. provider-specific env
    if provider_name == "anthropic-compatible":
        env_value = env.get(ANTHROPIC_MODEL_ENV, "")
        if env_value:
            return env_value
    else:
        env_value = env.get(config.model_env if config else "VIDEO_ANALYST_VISION_MODEL", "")
        if env_value:
            return env_value
    # 3. config.vision.model
    if config is not None and config.model:
        return config.model
    # 4. provider 默认
    if provider_name == "anthropic-compatible" and require_explicit:
        raise VisionModelNotConfiguredError(provider_name)
    return provider_default_model(provider_name)


def resolve_vision_base_url(
    provider_name: str,
    cli_base_url: str | None = None,
    config: VisionConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """统一解析 vision base URL。同模型解析：跨 provider 不互相回退。"""
    env = environ if environ is not None else os.environ
    if cli_base_url:
        return cli_base_url
    if provider_name == "anthropic-compatible":
        env_value = env.get(ANTHROPIC_BASE_URL_ENV, "")
        if env_value:
            return env_value
    else:
        env_value = env.get(
            config.base_url_env if config else "VIDEO_ANALYST_VISION_BASE_URL", ""
        )
        if env_value:
            return env_value
    if config is not None and config.base_url:
        return config.base_url
    return provider_default_base_url(provider_name)


__all__ = [
    "ANTHROPIC_API_KEY_ENV",
    "ANTHROPIC_BASE_URL_ENV",
    "ANTHROPIC_MODEL_ENV",
    "provider_default_base_url",
    "provider_default_model",
    "resolve_vision_base_url",
    "resolve_vision_model",
]