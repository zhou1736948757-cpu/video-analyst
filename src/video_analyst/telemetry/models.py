"""Telemetry data models (Epic 6).

所有字段严格不包含密钥。

``real_vision_api_used`` 的判定（严格）：

* ``network_request_sent = true``
* ``network_scope = external_provider``（local mock / mock transport / cache hit
  / dry run 都不算）
* ``response_received = true``
* ``success = true``
* ``stage ∈ {coarse_vision, fine_vision, terminal_vision}``

``network_scope`` 的取值：

* ``none``：未发送任何 HTTP（Mock provider / cache hit / Dry Run）
* ``local_mock``：本地 mock（127.0.0.1 / localhost / ::1 / MockTransport /
  monkeypatched transport）
* ``external_provider``：实际访问了非本地 hostname

``endpoint_host`` 只保存 hostname，不保存路径 / query / key。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from video_analyst.models import utc_now

CallStage = Literal[
    "asr",
    "subtitle_calibration",
    "coarse_vision",
    "fine_vision",
    "terminal_vision",
]

TokenStatus = Literal["available", "estimated", "unavailable"]

NetworkScope = Literal["none", "local_mock", "external_provider"]

ProviderMode = Literal["mock", "real_api", "mixed"]


class ProviderCallRecord(BaseModel):
    """单次 Provider 调用的结构化记录。

    只有 ``network_request_sent=true`` **且** ``network_scope=external_provider``
    **且** ``response_received=true`` **且** ``success=true`` 的视觉调用
    才能让 ``real_vision_api_used=true``。Mock / local mock / cache hit /
    Dry Run 永远不算真实调用。
    """

    call_id: str
    provider: str
    model: str
    endpoint_host: str | None = None

    stage: CallStage
    batch_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    request_attempts: int = 1
    retry_count: int = 0

    input_image_count: int = 0
    input_audio_seconds: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    token_status: TokenStatus = "unavailable"

    estimated_cost: float | None = None
    currency: str | None = None
    pricing_source: str | None = None
    pricing_snapshot_date: str | None = None

    cache_hit: bool = False
    network_request_sent: bool = False
    response_received: bool = False
    success: bool = False
    http_status_code: int | None = None
    error_type: str | None = None

    network_scope: NetworkScope = "none"
    transport_metadata: dict | None = None


class TelemetrySummary(BaseModel):
    """所有调用记录的汇总。"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    cache_hits: int = 0
    network_calls_sent: int = 0
    responses_received: int = 0

    real_vision_api_used: bool = False
    provider_mode: ProviderMode = "mock"
    external_network_calls: int = 0
    local_mock_calls: int = 0
    providers_used: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    stages_used: list[CallStage] = Field(default_factory=list)

    total_retries: int = 0
    total_input_images: int = 0
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_estimated_cost: float | None = None
    currency: str | None = None

    calls_by_stage: dict[str, int] = Field(default_factory=dict)
    calls_by_provider: dict[str, int] = Field(default_factory=dict)
    errors_by_type: dict[str, int] = Field(default_factory=dict)

    token_status_summary: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "CallStage",
    "NetworkScope",
    "ProviderCallRecord",
    "ProviderMode",
    "TelemetrySummary",
    "TokenStatus",
]