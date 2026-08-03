"""Telemetry recorder (Epic 6).

统一记录所有 Provider 调用。支持上下文管理器模式记录单次调用的开始/结束/耗时。

设计：
- 全局 recorder（单例），供 coarse/fine/terminal Provider 复用
- 线程不安全（单线程 pipeline 使用）
- 记录只写本地 output/telemetry/，不发送任何网络
- API Key、Authorization、base64 图片绝不进入记录
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from video_analyst.logging.setup import get_logger
from video_analyst.models import utc_now
from video_analyst.telemetry.models import (
    CallStage,
    NetworkScope,
    ProviderCallRecord,
    TelemetrySummary,
)
from video_analyst.telemetry.network_scope import classify_network_scope
from video_analyst.telemetry.pricing import (
    PricingSnapshot,
    estimate_cost,
    load_pricing_snapshot,
)
from video_analyst.telemetry.redaction import redact_secrets

log = get_logger("telemetry.recorder")

_global_recorder: "TelemetryRecorder | None" = None
_global_lock = threading.Lock()


class TelemetryRecorder:
    """记录所有 Provider 调用。

    使用：
        recorder = get_global_recorder(output_dir)
        with recorder.record_call(provider="mock", model="m", stage="coarse_vision") as rec:
            rec.input_image_count = 5
            # ... do work ...
            rec.success = True
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        pricing_snapshot: PricingSnapshot | None = None,
    ) -> None:
        self._records: list[ProviderCallRecord] = []
        self._output_dir = output_dir
        self._pricing = pricing_snapshot or load_pricing_snapshot()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """清空内存记录（不影响已写盘的文件）。"""
        with self._lock:
            self._records = []

    @property
    def records(self) -> list[ProviderCallRecord]:
        with self._lock:
            return list(self._records)

    @property
    def pricing(self) -> PricingSnapshot:
        return self._pricing

    @contextmanager
    def record_call(
        self,
        *,
        provider: str,
        model: str,
        stage: CallStage,
        batch_id: str | None = None,
        endpoint_host: str | None = None,
        endpoint_url: str | None = None,
        transport_metadata: dict | None = None,
        cache_hit: bool = False,
    ) -> Iterator[ProviderCallRecord]:
        """上下文管理器：自动记录开始/结束/耗时。

        调用方在 with 块内设置 input_image_count / token / success 等字段。
        ``endpoint_url`` 用于 network_scope 分类；只取 hostname，不存路径。
        ``transport_metadata`` 传入 ``{"transport": "mock"}`` 等用于标识 mock
        transport。
        ``cache_hit=True`` 时直接标记 network_scope="none"。
        """
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        started = utc_now()
        if cache_hit:
            network_scope: NetworkScope = "none"
            resolved_host = None
        else:
            network_scope, resolved_host = classify_network_scope(
                endpoint_url, transport_metadata
            )
        rec = ProviderCallRecord(
            call_id=call_id,
            provider=redact_secrets(provider),
            model=redact_secrets(model),
            stage=stage,
            batch_id=batch_id,
            endpoint_host=redact_secrets(resolved_host) if resolved_host else (
                redact_secrets(endpoint_host) if endpoint_host else None
            ),
            cache_hit=cache_hit,
            network_scope=network_scope,
            transport_metadata=transport_metadata,
            started_at=started,
        )
        try:
            yield rec
        finally:
            rec.completed_at = utc_now()
            if rec.completed_at is not None:
                rec.duration_seconds = (
                    rec.completed_at - rec.started_at
                ).total_seconds()
            # 费用估算
            if (
                rec.input_tokens is not None
                and rec.output_tokens is not None
                and rec.estimated_cost is None
            ):
                cost, currency, src = estimate_cost(
                    provider=provider,
                    model=model,
                    input_tokens=rec.input_tokens,
                    output_tokens=rec.output_tokens,
                    snapshot=self._pricing,
                )
                if cost is not None:
                    rec.estimated_cost = cost
                    rec.currency = currency
                    rec.pricing_source = src
                    rec.pricing_snapshot_date = self._pricing.snapshot_date
            with self._lock:
                self._records.append(rec)

    def add_record(self, record: ProviderCallRecord) -> None:
        """直接添加一条已完成记录。"""
        with self._lock:
            self._records.append(record)

    def real_vision_api_used(self) -> bool:
        """判断是否使用了真实视觉 API。

        严格判定：network_request_sent=true **且** network_scope=external_provider
        **且** response_received=true **且** success=true **且** stage 是视觉阶段。
        本地 Mock HTTP、MockTransport、cache hit、Dry Run 都不算。
        """
        vision_stages = {"coarse_vision", "fine_vision", "terminal_vision"}
        with self._lock:
            for rec in self._records:
                if (
                    rec.stage in vision_stages
                    and rec.network_request_sent
                    and rec.response_received
                    and rec.success
                    and rec.network_scope == "external_provider"
                ):
                    return True
        return False

    def provider_mode(self) -> str:
        """返回 mock / real_api / mixed。"""
        from video_analyst.telemetry.models import ProviderMode

        has_external = False
        has_other = False
        with self._lock:
            for rec in self._records:
                if rec.network_scope == "external_provider" and rec.network_request_sent:
                    has_external = True
                elif rec.network_scope == "local_mock" and rec.network_request_sent:
                    has_other = True
        if has_external and has_other:
            return "mixed"
        if has_external:
            return "real_api"
        return "mock"

    def providers_used(self) -> list[str]:
        with self._lock:
            seen: list[str] = []
            for rec in self._records:
                if rec.provider not in seen:
                    seen.append(rec.provider)
            return seen

    def models_used(self) -> list[str]:
        with self._lock:
            seen: list[str] = []
            for rec in self._records:
                if rec.model and rec.model not in seen:
                    seen.append(rec.model)
            return seen


def get_global_recorder(output_dir: Path | None = None) -> TelemetryRecorder:
    """获取或创建全局 recorder。

    首次调用时传入 output_dir；后续调用可省略，返回已存在的实例。
    """
    global _global_recorder
    with _global_lock:
        if _global_recorder is None:
            _global_recorder = TelemetryRecorder(output_dir=output_dir)
        return _global_recorder


def set_global_recorder(recorder: TelemetryRecorder) -> None:
    """替换全局 recorder（测试用）。"""
    global _global_recorder
    with _global_lock:
        _global_recorder = recorder


def reset_global_recorder() -> None:
    """清空全局 recorder（测试用）。"""
    global _global_recorder
    with _global_lock:
        if _global_recorder is not None:
            _global_recorder.reset()


__all__ = [
    "TelemetryRecorder",
    "get_global_recorder",
    "reset_global_recorder",
    "set_global_recorder",
]