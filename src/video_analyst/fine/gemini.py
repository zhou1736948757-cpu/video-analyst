"""Gemini fine vision provider — fail-fast stub (Epic 4).

与 Epic 2 的 gemini stub 一致：保留接口与配置，不伪装完成。若 Google
GenAI SDK 不可用或未配置，构造时即明确失败。
"""

from __future__ import annotations

from video_analyst.config import VisionConfig
from video_analyst.exceptions import FineVisionProviderUnavailableError
from video_analyst.fine.base import FineVisionProvider
from video_analyst.fine.models import FineBatchAnalysis, FineVisionRequest


class GeminiFineVisionProvider(FineVisionProvider):
    """Fail-fast stub for Google Gemini fine temporal vision.

    Not implemented. Constructing this provider raises
    ``FineVisionProviderUnavailableError`` so callers fail loudly instead of
    silently falling back. Remove this stub only when a real implementation
    exists.
    """

    def __init__(self, config: VisionConfig) -> None:
        raise FineVisionProviderUnavailableError(
            "GeminiFineVisionProvider is not implemented. "
            "Use --vision-provider mock or openai-compatible."
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return ""

    def analyze_fine_batch(self, request: FineVisionRequest) -> FineBatchAnalysis:  # pragma: no cover
        raise FineVisionProviderUnavailableError(
            "GeminiFineVisionProvider.analyze_fine_batch not implemented"
        )


__all__ = ["GeminiFineVisionProvider"]
