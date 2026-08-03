"""Provider registry and factory for fine temporal vision providers (Epic 4).

复用 Epic 2 的 VisionConfig（provider / api_key_env / base_url_env / model_env /
request_timeout / max_retries / retry_backoff_base）。Epic 4 的 FineAnalysisConfig
只补充 fine 特有的 prompt 版本与图片发送上限。
"""

from __future__ import annotations

from video_analyst.config import VisionConfig
from video_analyst.exceptions import FineVisionProviderUnavailableError
from video_analyst.fine.base import FineVisionProvider

_REGISTRY: dict[str, type[FineVisionProvider]] = {}


def register_fine_provider(name: str, cls: type[FineVisionProvider]) -> None:
    """Register a fine provider class under a given name."""
    _REGISTRY[name] = cls


def get_fine_provider_class(name: str) -> type[FineVisionProvider]:
    """Look up a registered fine provider class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise FineVisionProviderUnavailableError(
            f"Unknown fine vision provider '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def create_fine_vision_provider(
    provider: str,
    config: VisionConfig,
) -> FineVisionProvider:
    """Factory: instantiate a fine vision provider by name with config."""
    cls = get_fine_provider_class(provider)
    return cls(config)  # type: ignore[call-arg]


def _register_builtins() -> None:
    """Register all built-in fine providers."""
    from video_analyst.fine.mock import MockFineVisionProvider
    from video_analyst.fine.openai_compatible import OpenAICompatibleFineProvider
    from video_analyst.fine.anthropic_compatible import AnthropicCompatibleFineProvider

    register_fine_provider("mock", MockFineVisionProvider)
    register_fine_provider("openai-compatible", OpenAICompatibleFineProvider)
    register_fine_provider("anthropic-compatible", AnthropicCompatibleFineProvider)

    try:
        from video_analyst.fine.gemini import GeminiFineVisionProvider

        register_fine_provider("gemini", GeminiFineVisionProvider)
    except ImportError:
        pass  # gemini dependencies not installed


_register_builtins()


__all__ = [
    "create_fine_vision_provider",
    "get_fine_provider_class",
    "register_fine_provider",
]
