"""Telemetry subsystem (Epic 6).

统一记录所有 Provider 调用：次数 / 图片数 / Token / 费用 / 运行时间 / 重试 /
缓存命中 / 错误分类。绝不记录 API Key、Authorization Header、完整 base64 图片。

输出：
- output/telemetry/calls.jsonl   每次调用一条 JSON
- output/telemetry/summary.json  汇总
- output/telemetry/errors.jsonl  错误调用单独文件
"""

from video_analyst.telemetry.models import (
    CallStage,
    ProviderCallRecord,
    TelemetrySummary,
    TokenStatus,
)
from video_analyst.telemetry.recorder import (
    TelemetryRecorder,
    get_global_recorder,
    reset_global_recorder,
    set_global_recorder,
)
from video_analyst.telemetry.pricing import (
    PricingSnapshot,
    estimate_cost,
    load_pricing_snapshot,
)
from video_analyst.telemetry.aggregation import aggregate_summary
from video_analyst.telemetry.redaction import redact_secrets
from video_analyst.telemetry.report import save_telemetry_outputs

__all__ = [
    "CallStage",
    "PricingSnapshot",
    "ProviderCallRecord",
    "TelemetryRecorder",
    "TelemetrySummary",
    "TokenStatus",
    "aggregate_summary",
    "estimate_cost",
    "get_global_recorder",
    "load_pricing_snapshot",
    "redact_secrets",
    "reset_global_recorder",
    "save_telemetry_outputs",
    "set_global_recorder",
]