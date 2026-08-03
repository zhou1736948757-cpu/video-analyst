"""Telemetry aggregation (Epic 6).

从 ProviderCallRecord 列表生成 TelemetrySummary。

严格判定 real_vision_api_used：

* network_request_sent=true
* network_scope=external_provider
* response_received=true
* success=true
* stage ∈ {coarse_vision, fine_vision, terminal_vision}

本地 Mock / MockTransport / cache hit / Dry Run 一律不算真实 API。
"""

from __future__ import annotations

from video_analyst.telemetry.models import (
    ProviderCallRecord,
    ProviderMode,
    TelemetrySummary,
)


def _resolve_provider_mode(records: list[ProviderCallRecord]) -> ProviderMode:
    has_external = any(
        r.network_request_sent and r.network_scope == "external_provider"
        for r in records
    )
    has_local_mock = any(
        r.network_request_sent and r.network_scope == "local_mock"
        for r in records
    )
    if has_external and has_local_mock:
        return "mixed"
    if has_external:
        return "real_api"
    return "mock"


def aggregate_summary(records: list[ProviderCallRecord]) -> TelemetrySummary:
    """汇总所有调用记录。"""
    if not records:
        return TelemetrySummary()

    total = len(records)
    successful = sum(1 for r in records if r.success)
    failed = total - successful
    cache_hits = sum(1 for r in records if r.cache_hit)
    network_sent = sum(1 for r in records if r.network_request_sent)
    responses = sum(1 for r in records if r.response_received)
    external_calls = sum(
        1 for r in records
        if r.network_request_sent and r.network_scope == "external_provider"
    )
    local_mock_calls = sum(
        1 for r in records
        if r.network_request_sent and r.network_scope == "local_mock"
    )

    providers: list[str] = []
    models: list[str] = []
    stages: list[str] = []
    for r in records:
        if r.provider and r.provider not in providers:
            providers.append(r.provider)
        if r.model and r.model not in models:
            models.append(r.model)
        if r.stage and r.stage not in stages:
            stages.append(r.stage)

    total_retries = sum(r.retry_count for r in records)
    total_images = sum(r.input_image_count for r in records)

    # Token 汇总（只有 available / estimated 的计入）
    token_records = [
        r for r in records if r.token_status in ("available", "estimated")
    ]
    total_input = sum(r.input_tokens or 0 for r in token_records) if token_records else None
    total_output = sum(r.output_tokens or 0 for r in token_records) if token_records else None
    total_tokens = sum(r.total_tokens or 0 for r in token_records) if token_records else None

    # 费用汇总
    cost_records = [r for r in records if r.estimated_cost is not None]
    total_cost = sum(r.estimated_cost or 0.0 for r in cost_records) if cost_records else None
    currency = None
    if cost_records:
        currency = cost_records[0].currency

    calls_by_stage: dict[str, int] = {}
    calls_by_provider: dict[str, int] = {}
    errors_by_type: dict[str, int] = {}
    token_status_summary: dict[str, int] = {}
    for r in records:
        calls_by_stage[r.stage] = calls_by_stage.get(r.stage, 0) + 1
        calls_by_provider[r.provider] = calls_by_provider.get(r.provider, 0) + 1
        if r.error_type:
            errors_by_type[r.error_type] = errors_by_type.get(r.error_type, 0) + 1
        token_status_summary[r.token_status] = token_status_summary.get(r.token_status, 0) + 1

    # real_vision_api_used: 严格判定
    vision_stages = {"coarse_vision", "fine_vision", "terminal_vision"}
    real_api = any(
        r.stage in vision_stages
        and r.network_request_sent
        and r.response_received
        and r.success
        and r.network_scope == "external_provider"
        for r in records
    )

    return TelemetrySummary(
        total_calls=total,
        successful_calls=successful,
        failed_calls=failed,
        cache_hits=cache_hits,
        network_calls_sent=network_sent,
        responses_received=responses,
        real_vision_api_used=real_api,
        provider_mode=_resolve_provider_mode(records),
        external_network_calls=external_calls,
        local_mock_calls=local_mock_calls,
        providers_used=providers,
        models_used=models,
        stages_used=stages,  # type: ignore[arg-type]
        total_retries=total_retries,
        total_input_images=total_images,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        total_estimated_cost=total_cost,
        currency=currency,
        calls_by_stage=calls_by_stage,
        calls_by_provider=calls_by_provider,
        errors_by_type=errors_by_type,
        token_status_summary=token_status_summary,
    )


__all__ = ["aggregate_summary"]