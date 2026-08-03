"""Abstract base class for vision providers.

Providers are responsible ONLY for:
- Constructing and sending model requests
- Receiving raw responses
- Converting to unified CoarseBatchAnalysis structure
- Raising clear Provider exceptions

Providers are NOT responsible for:
- Deciding which batches to analyze
- Deciding refinement budgets
- Aggregating full-video structure
- Managing pipeline cache
- Generating final reports
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from video_analyst.vision.models import CoarseBatchAnalysis, VisionBatchRequest


class VisionProvider(ABC):
    """Unified interface for all vision model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier (e.g. 'mock', 'openai-compatible')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name being used. May be empty for providers that allow many."""
        ...

    @abstractmethod
    def analyze_batch(self, request: VisionBatchRequest) -> CoarseBatchAnalysis:
        """Analyze a single batch and return structured results.

        The provider is responsible for converting its raw response into the
        unified CoarseBatchAnalysis shape. It MUST NOT return raw JSON or
        provider-specific structures.

        Raises:
            VisionTimeoutError: Request timed out.
            VisionRateLimitError: Rate limited (429).
            VisionRequestError: Other API errors.
            VisionResponseParseError: Response not valid JSON.
            VisionResponseValidationError: Response failed validation.
        """
        ...

    # --- Optional hooks used by the batch runner. Default impls are sensible. ---

    def supports_repair_request(self) -> bool:
        """Return True if the provider can service a 'fix the JSON format'
        follow-up request. The Mock provider can; many real providers can too."""
        return True

    def build_repair_payload(self, raw_text: str, reason: str) -> dict[str, Any] | None:
        """Build a payload (text-only) for the repair call. Returning None means
        the batch runner should skip the repair attempt and mark the batch as
        failed without retrying."""
        return None

    def parse_repair_response(self, raw_text: str) -> dict[str, Any] | None:
        """Parse the provider response for a repair call back into JSON dict."""
        from video_analyst.vision.response_parser import parse_response_payload

        return parse_response_payload(raw_text)


__all__ = ["VisionProvider"]