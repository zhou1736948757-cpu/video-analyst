"""Gemini provider stub.

The Gemini multimodal API has a different shape (parts, file references,
inline_data) and depends on an optional `google-generativeai` client.

Epic 2 deliberately does NOT require Gemini support. We expose the class so
the registry stays stable, but ``analyze_batch`` raises
``VisionProviderUnavailableError`` until a verified client is added.

When a future Epic wants real Gemini support, only this file changes.
"""

from __future__ import annotations

from video_analyst.config import VisionConfig
from video_analyst.exceptions import VisionProviderUnavailableError
from video_analyst.vision.base import VisionProvider
from video_analyst.vision.models import CoarseBatchAnalysis, VisionBatchRequest


class GeminiVisionProvider(VisionProvider):
    """Placeholder provider. Will fail-fast with a clear message."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._config.model or "gemini-1.5-flash"

    def analyze_batch(self, request: VisionBatchRequest) -> CoarseBatchAnalysis:
        raise VisionProviderUnavailableError(
            "Gemini provider is not implemented in Epic 2. "
            "Use --vision-provider mock or openai-compatible."
        )


__all__ = ["GeminiVisionProvider"]