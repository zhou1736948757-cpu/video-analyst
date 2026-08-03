"""Provider registry and factory for vision providers."""

from __future__ import annotations

from video_analyst.config import VisionConfig
from video_analyst.exceptions import VisionProviderUnavailableError
from video_analyst.vision.base import VisionProvider

_REGISTRY: dict[str, type[VisionProvider]] = {}


def register_provider(name: str, cls: type[VisionProvider]) -> None:
    """Register a provider class under a given name."""
    _REGISTRY[name] = cls


def get_provider_class(name: str) -> type[VisionProvider]:
    """Look up a registered provider class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise VisionProviderUnavailableError(
            f"Unknown vision provider '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def create_vision_provider(
    provider: str,
    config: VisionConfig,
) -> VisionProvider:
    """Factory: instantiate a vision provider by name with config."""
    cls = get_provider_class(provider)
    return cls(config)  # type: ignore[call-arg]


def _register_builtins() -> None:
    """Register all built-in providers."""
    from video_analyst.vision.mock import MockVisionProvider
    from video_analyst.vision.openai_compatible import OpenAICompatibleProvider
    from video_analyst.vision.anthropic_compatible import AnthropicCompatibleVisionProvider

    register_provider("mock", MockVisionProvider)
    register_provider("openai-compatible", OpenAICompatibleProvider)
    register_provider("anthropic-compatible", AnthropicCompatibleVisionProvider)

    try:
        from video_analyst.vision.gemini import GeminiVisionProvider

        register_provider("gemini", GeminiVisionProvider)
    except ImportError:
        pass  # gemini dependencies not installed


_register_builtins()
