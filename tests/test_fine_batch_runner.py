"""Tests for fine batch runner (Epic 4).

覆盖：多 batch 顺序、缓存命中、refresh 单 batch、refresh 全部、
中断恢复、单 batch 重试、partial success、原子写入。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from video_analyst.config import AppConfig
from video_analyst.fine.batch_runner import run_fine_analysis
from video_analyst.fine.mock import MockFineVisionProvider
from video_analyst.pipeline.epic4 import run_epic4

from tests._epic4_fixtures import has_pillow, seed_fine_batches

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _cfg(allow_partial: bool = True) -> AppConfig:
    cfg = AppConfig()
    cfg.vision.provider = "mock"
    cfg.vision.max_retries = 1
    cfg.fine_analysis.allow_partial = allow_partial
    return cfg


def test_multiple_batches_sequential(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p = MockFineVisionProvider(scenario="normal")
    result = run_fine_analysis(
        layout=layout, config=cfg, provider=p, user_request="x"
    )
    assert result.total_batches == 2
    assert result.succeeded_batches == 2
    assert result.failed_batches == 0
    # 每个 batch 输出存在
    for fb_id in ["fine_interval_0001", "fine_interval_0002"]:
        assert (layout.fine_batches_out_dir / f"{fb_id}.json").is_file()
        assert (layout.fine_raw_dir / f"{fb_id}_response.json").is_file()
    assert layout.fine_manifest_json.is_file()
    assert layout.fine_video_analysis_json.is_file()
    assert layout.fine_temporal_events_json.is_file()
    assert layout.fine_second_refinement_plan_json.is_file()


def test_second_run_hits_cache(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p1 = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p1, user_request="x")
    assert p1.call_count == 2
    # 第二次运行 → 缓存命中
    p2 = MockFineVisionProvider(scenario="normal")
    r2 = run_fine_analysis(layout=layout, config=cfg, provider=p2, user_request="x")
    assert p2.call_count == 0
    assert r2.cached_batches == 2


def test_refresh_single_batch(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p1 = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p1, user_request="x")
    # refresh 单 batch
    p2 = MockFineVisionProvider(scenario="normal")
    r2 = run_fine_analysis(
        layout=layout, config=cfg, provider=p2, user_request="x",
        refresh_batches=["fine_interval_0001"],
    )
    # 只重跑 1 个
    assert p2.call_count == 1
    assert r2.succeeded_batches == 1
    assert r2.cached_batches == 1


def test_refresh_all_batches(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p1 = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p1, user_request="x")
    p2 = MockFineVisionProvider(scenario="normal")
    r2 = run_fine_analysis(
        layout=layout, config=cfg, provider=p2, user_request="x", refresh=True
    )
    assert p2.call_count == 2
    assert r2.succeeded_batches == 2


def test_resume_after_partial_failure(tmp_path: Path, monkeypatch):
    """第一个 batch 失败，第二个成功；第二次运行恢复第一个。"""
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg(allow_partial=True)

    import video_analyst.fine.batch_runner as br
    real = br._analyze_with_provider
    state = {"calls": 0}

    def selective(*args, **kwargs):
        req = args[1]
        if req.fine_batch_id == "fine_interval_0001":
            state["calls"] += 1
            if state["calls"] == 1:
                from video_analyst.exceptions import FineVisionResponseParseError
                raise FineVisionResponseParseError("simulated first-attempt failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(br, "_analyze_with_provider", selective)
    p1 = MockFineVisionProvider(scenario="normal")
    r1 = run_fine_analysis(layout=layout, config=cfg, provider=p1, user_request="x")
    # interval_0001 failed, interval_0002 ok
    assert r1.failed_batches == 1
    assert "fine_interval_0001" in r1.failed_batch_ids
    assert (layout.fine_batches_out_dir / "fine_interval_0002.json").is_file()

    # 第二次运行（无失败注入）→ interval_0001 恢复
    monkeypatch.undo()
    p2 = MockFineVisionProvider(scenario="normal")
    r2 = run_fine_analysis(layout=layout, config=cfg, provider=p2, user_request="x")
    assert r2.failed_batches == 0
    assert (layout.fine_batches_out_dir / "fine_interval_0001.json").is_file()


def test_no_partial_raises_on_failure(tmp_path: Path, monkeypatch):
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg(allow_partial=False)

    import video_analyst.fine.batch_runner as br
    from video_analyst.exceptions import FineVisionResponseParseError

    def always_fail(*args, **kwargs):
        raise FineVisionResponseParseError("always fails")

    monkeypatch.setattr(br, "_analyze_with_provider", always_fail)
    p = MockFineVisionProvider(scenario="normal")
    from video_analyst.exceptions import FineBatchAnalysisError
    with pytest.raises(FineBatchAnalysisError):
        run_fine_analysis(layout=layout, config=cfg, provider=p, user_request="x")


def test_atomic_writes(tmp_path: Path):
    """每个 batch 完成后立即原子写入 manifest。"""
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p, user_request="x")
    manifest = json.loads(layout.fine_manifest_json.read_text(encoding="utf-8"))
    assert manifest["total_batches"] == 2
    assert manifest["success_count"] == 2
    # manifest 中 batch 按 fine_batch_id 排序
    ids = [b["fine_batch_id"] for b in manifest["batches"]]
    assert ids == sorted(ids)


def test_empty_plan_completes_no_batches(tmp_path: Path):
    """fine_batches.json 为空时正常完成。"""
    layout = seed_fine_batches(tmp_path, intervals=[])
    # 写空 fine_batches.json
    from video_analyst.cache.store import CacheStore
    CacheStore(layout).write_json(
        layout.refinement_fine_batches_json,
        {"schema_version": "1.0", "video_id": "v", "total_batches": 0, "batches": []},
    )
    cfg = _cfg()
    p = MockFineVisionProvider(scenario="normal")
    result = run_fine_analysis(layout=layout, config=cfg, provider=p, user_request="x")
    assert result.status == "completed_no_batches"
    assert result.total_batches == 0


def test_epic4_pipeline_end_to_end(tmp_path: Path):
    """通过 run_epic4 端到端验证。"""
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    result = run_epic4(
        video_path=Path("/tmp/v.mp4"),
        output_dir=tmp_path,
        config=cfg,
        user_request="analyze motion",
    )
    assert result.status == "completed"
    assert result.total_batches == 2


def test_failed_batch_does_not_delete_siblings(tmp_path: Path, monkeypatch):
    """一个 batch 失败不删除已成功的其他 batch。"""
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg(allow_partial=True)

    import video_analyst.fine.batch_runner as br
    from video_analyst.exceptions import FineVisionResponseParseError

    real = br._analyze_with_provider

    def fail_first(*args, **kwargs):
        req = args[1]
        if req.fine_batch_id == "fine_interval_0001":
            raise FineVisionResponseParseError("always fails for 0001")
        return real(*args, **kwargs)

    monkeypatch.setattr(br, "_analyze_with_provider", fail_first)
    p = MockFineVisionProvider(scenario="normal")
    r = run_fine_analysis(layout=layout, config=cfg, provider=p, user_request="x")
    assert r.failed_batches == 1
    assert r.succeeded_batches == 1
    # interval_0002 结果保留
    assert (layout.fine_batches_out_dir / "fine_interval_0002.json").is_file()
    # interval_0001 无结果文件
    assert not (layout.fine_batches_out_dir / "fine_interval_0001.json").is_file()


def test_user_request_change_invalidates_cache(tmp_path: Path):
    """改 user_request → 缓存失效，重调 provider。"""
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p1 = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p1, user_request="A")
    p2 = MockFineVisionProvider(scenario="normal")
    r2 = run_fine_analysis(layout=layout, config=cfg, provider=p2, user_request="B")
    # user_request 变化 → 缓存失效，重调
    assert p2.call_count == 2
    assert r2.succeeded_batches == 2


def test_budget_change_does_not_reinvoke_provider(tmp_path: Path):
    """只改二轮补帧预算 → 不重调 provider，只重建 plan。"""
    layout = seed_fine_batches(tmp_path)
    cfg = _cfg()
    p1 = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p1, user_request="x")
    # 改预算
    cfg.fine_analysis.second_refinement_budget_ratio = 0.02
    p2 = MockFineVisionProvider(scenario="normal")
    run_fine_analysis(layout=layout, config=cfg, provider=p2, user_request="x")
    # 缓存命中，不重调
    assert p2.call_count == 0
