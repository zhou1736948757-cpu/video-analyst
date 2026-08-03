"""Abstract base class for fine temporal vision providers (Epic 4).

Provider 只负责：
- 构造或发送模型请求
- 接收原始响应
- 转换为统一 FineBatchAnalysis 结构
- 抛出明确的 Provider 异常

Provider 不负责：
- 决定哪些 fine batch 需要分析
- 决定二轮补帧预算
- 汇总全片
- 管理 Pipeline 缓存
- 生成最终报告
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from video_analyst.fine.models import FineBatchAnalysis, FineVisionRequest


class FineVisionProvider(ABC):
    """Unified interface for all fine temporal vision model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier (e.g. 'mock', 'openai-compatible')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name being used."""
        ...

    @abstractmethod
    def analyze_fine_batch(self, request: FineVisionRequest) -> FineBatchAnalysis:
        """Analyze one fine batch and return structured results.

        The provider converts its raw response into the unified
        FineBatchAnalysis shape via the fine response_parser. It MUST NOT
        return raw JSON or provider-specific structures.

        Raises:
            FineVisionTimeoutError: Request timed out.
            FineVisionRateLimitError: Rate limited (429).
            FineVisionRequestError: Other API errors.
            FineVisionResponseParseError: Response not valid JSON.
            FineVisionResponseValidationError: Response failed validation.
        """
        ...

    # --- Optional hooks used by the batch runner ---

    def supports_repair_request(self) -> bool:
        """Return True if the provider can service a 'fix the JSON format'
        follow-up request."""
        return True

    def build_repair_payload(self, raw_text: str, reason: str) -> dict[str, Any] | None:
        """Build a payload for the repair call. Returning None means skip
        the repair attempt and mark the batch as failed."""
        return None

    def parse_repair_response(self, raw_text: str) -> dict[str, Any] | None:
        """Parse the provider response for a repair call back into JSON dict."""
        from video_analyst.fine.response_parser import parse_fine_response_payload

        try:
            return parse_fine_response_payload(raw_text)
        except Exception:
            return None


__all__ = ["FineVisionProvider"]
