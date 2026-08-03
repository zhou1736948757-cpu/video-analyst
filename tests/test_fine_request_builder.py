"""Tests for fine request builder (Epic 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyst.config import FineAnalysisConfig
from video_analyst.fine.request_builder import (
    build_all_fine_requests,
    build_fine_request,
)

from tests._epic4_fixtures import has_pillow, make_fine_visual_batch, seed_fine_batches

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def test_contact_sheets_in_time_order(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    # 读回 fine_batches.json 的第一个 batch
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10, sheet_count=2)
    req, report = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    # contact sheet 按时间顺序
    assert len(req.contact_sheets) == 2
    assert req.contact_sheets[0].sheet_id < req.contact_sheets[1].sheet_id
    # 每张 sheet 的 frame_ids 非空
    for cs in req.contact_sheets:
        assert cs.frame_ids
        assert cs.timestamps
        assert len(cs.frame_ids) == len(cs.timestamps)
        assert cs.reading_order == "row_major"


def test_overlap_frame_metadata_preserved(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10, sheet_count=3)
    req, _ = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    # 第二张及之后 sheet 有 overlap_with_previous
    assert req.contact_sheets[0].overlap_with_previous == []
    assert len(req.contact_sheets[1].overlap_with_previous) >= 1


def test_detail_frame_paths_correct(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10)
    req, report = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    assert len(req.detail_frames) == 2
    for df in req.detail_frames:
        assert df.path.is_file()
        assert df.resolution_mode == "source_original"
        assert df.sha256
    # detail frame 与普通 contact sheet 路径不同
    cs_paths = {str(cs.path) for cs in req.contact_sheets}
    df_paths = {str(df.path) for df in req.detail_frames}
    assert cs_paths.isdisjoint(df_paths)


def test_only_active_subtitles_used(tmp_path: Path):
    """字幕只读 active subtitles.json + review.json。"""
    layout = seed_fine_batches(tmp_path, active_source="llm")
    # 写一个 subtitles_asr.json 干扰项
    layout.subtitles_asr_json.write_text(
        '{"version":"1.0","source_video":"/tmp/v.mp4","segments":[{"index":99,"start":0.0,"end":10.0,"text":"asr-only","language":"Japanese","is_code_switched":false,"source_segment_ids":[],"source":"asr","merge_reason":null,"calibration_status":null}],"metadata":{},"origin":"asr"}',
        encoding="utf-8",
    )
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10)
    req, _ = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    # 不能出现 "asr-only"
    for s in req.subtitles:
        assert "asr-only" not in s.text
    assert req.subtitle_metadata.active_source == "llm"


def test_refinement_questions_complete(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    questions = ["q1", "q2", "q3"]
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10, questions=questions)
    req, _ = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    assert req.refinement_questions == questions


def test_coarse_context_complete(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10)
    req, _ = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    cc = req.coarse_context
    assert cc.source_batch_ids == ["batch_0001"]
    # 从 coarse structure 读取 summary
    assert any("coarse" in s for s in cc.coarse_summaries)
    # 从 refinement plan 读取 reasons
    assert cc.coarse_refinement_reasons
    # 从 coarse structure 读取 unresolved
    assert cc.unresolved_questions


def test_no_contact_sheets_uses_representative_frames(tmp_path: Path):
    """没有 contact sheets 时，从精细帧选代表帧（记录省略）。"""
    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10, sheet_count=0, detail_count=0)
    # 手动注入精细帧
    from video_analyst.refinement.models import RefinementFrame

    batch.frames = [
        RefinementFrame.make(
            interval_id="interval_0001",
            local_index=i,
            timestamp_seconds=5.0 + i * 0.1,
            relative_timestamp_seconds=i * 0.1,
            requested_fps=10,  # type: ignore[arg-type]
            path=layout.refinement_frames_dir / "interval_0001" / f"f{i}.jpg",
            width=320,
            height=240,
            size_bytes=1024,
            sha256=f"sha{i}",
            source_video_sha256="vsha",
        )
        for i in range(10)
    ]
    # 为这些 frame 创建真实文件
    from tests._epic4_fixtures import make_contact_sheet_jpeg

    for i, fr in enumerate(batch.frames):
        make_contact_sheet_jpeg(fr.path, color=(i * 20, 100, 100))
        fr.sha256 = __import__("video_analyst.util.hashing", fromlist=["sha256_file"]).sha256_file(fr.path)

    req, report = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    assert report.used_fallback_representatives is True
    assert len(report.representative_frame_ids) > 0
    assert len(report.omitted_representative_frame_ids) > 0


def test_image_hash_enters_cache_key(tmp_path: Path):
    """图片签名（contact sheet sha + detail frame sha）进入缓存键。"""
    layout = seed_fine_batches(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10)
    req, report = build_fine_request(
        batch, layout=layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    assert report.image_signature
    assert len(report.image_signature) == 64

    from video_analyst.fine.manifest import build_fine_cache_key
    from video_analyst.config import VisionConfig

    key1 = build_fine_cache_key(
        request=req,
        report=report,
        layout=layout,
        provider_name="mock",
        model="mock-fine-vision-1",
        cfg=FineAnalysisConfig(),
        vision_cfg=VisionConfig(),
    )
    # 改 user_request → 缓存键变化
    req2 = req.model_copy(update={"user_request": "different"})
    key2 = build_fine_cache_key(
        request=req2,
        report=report,
        layout=layout,
        provider_name="mock",
        model="mock-fine-vision-1",
        cfg=FineAnalysisConfig(),
        vision_cfg=VisionConfig(),
    )
    assert key1 != key2


def test_omitted_detail_frames_recorded(tmp_path: Path):
    """超过上限的 detail frame 被记录到 omitted。"""
    layout = seed_fine_batches(tmp_path)
    from video_analyst.refinement.models import DetailFrameReference

    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=5.0, end=8.0, fps=10, detail_count=0)
    # 注入 5 个 detail frame，但配置上限 2
    for i in range(5):
        batch.detail_frames.append(
            DetailFrameReference(
                frame_id=f"interval_0001__detail_{i:02d}",
                timestamp_seconds=5.0 + i,
                path=layout.refinement_detail_frames_dir / "interval_0001" / f"d{i}.jpg",
                selection_reason="interval_start",
            )
        )
    cfg = FineAnalysisConfig(max_detail_frames_per_request=2)
    req, report = build_fine_request(
        batch, layout=layout, cfg=cfg, user_request="x"
    )
    assert len(report.detail_frame_ids) == 2
    assert len(report.omitted_detail_frame_ids) == 3


def test_build_all_fine_requests_orders_by_time(tmp_path: Path):
    layout = seed_fine_batches(tmp_path)
    results = build_all_fine_requests(
        layout, cfg=FineAnalysisConfig(), user_request="x"
    )
    assert len(results) == 2
    # 按 interval.start 排序
    starts = [req.interval.start for req, _ in results]
    assert starts == sorted(starts)
