"""Telemetry output writers (Epic 6).

写入：
- output/telemetry/calls.jsonl   每次调用一条 JSON
- output/telemetry/summary.json  汇总
- output/telemetry/errors.jsonl  错误调用单独文件
"""

from __future__ import annotations

import json
from pathlib import Path

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.telemetry.aggregation import aggregate_summary
from video_analyst.telemetry.models import ProviderCallRecord, TelemetrySummary
from video_analyst.telemetry.recorder import TelemetryRecorder


def _telemetry_dir(layout: OutputLayout) -> Path:
    d = layout.root / "telemetry"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_telemetry_outputs(
    layout: OutputLayout,
    recorder: TelemetryRecorder,
) -> dict[str, Path]:
    """把 recorder 的所有记录写入磁盘。

    Returns: {calls_jsonl, summary_json, errors_jsonl} 路径。
    """
    tdir = _telemetry_dir(layout)
    calls_path = tdir / "calls.jsonl"
    errors_path = tdir / "errors.jsonl"
    summary_path = tdir / "summary.json"

    records = recorder.records
    cache = CacheStore(layout)

    # calls.jsonl
    lines = []
    error_lines = []
    for rec in records:
        line = json.dumps(rec.model_dump(mode="json"), ensure_ascii=False)
        lines.append(line)
        if not rec.success or rec.error_type:
            error_lines.append(line)
    calls_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    errors_path.write_text(
        "\n".join(error_lines) + ("\n" if error_lines else ""), encoding="utf-8"
    )

    # summary.json
    summary = aggregate_summary(records)
    cache.write_json(summary_path, summary.model_dump(mode="json"))

    return {
        "calls_jsonl": calls_path,
        "summary_json": summary_path,
        "errors_jsonl": errors_path,
    }


__all__ = ["save_telemetry_outputs"]