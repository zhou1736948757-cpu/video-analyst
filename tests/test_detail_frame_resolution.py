"""Tests that detail frame resolution > refinement frame resolution (Epic 4 legacy fix 2.1).

Creates a small 4K-source video and runs Epic 3 + Epic 4 extraction, then
asserts that the detail frame's output dimensions are strictly larger than
the refinement frame's dimensions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig
from video_analyst.pipeline.epic3 import run_epic3

from tests._epic3_fixtures import has_ffmpeg, has_pillow, make_test_video, make_refinement_plan_json

pytestmark = pytest.mark.skipif(
    not (has_ffmpeg() and has_pillow()),
    reason="ffmpeg + Pillow required for detail-frame resolution test",
)


def _make_4k_video(out_path: Path, *, duration: float = 4.0) -> Path:
    """生成 4K (3840x2160) 测试视频。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        out_path.unlink()
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=size=3840x2160:rate=30:duration={duration}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, shell=False, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"4K test video failed: {r.stderr[-300:]}")
    return out_path


def test_detail_frame_exceeds_refinement_frame_resolution(tmp_path: Path):
    """detail frame 实际尺寸大于普通 refinement frame。"""
    # 4K 源视频
    video = tmp_path / "4k.mp4"
    _make_4k_video(video, duration=4.0)

    # Seed Epic 3 输入 + refinement plan
    layout = OutputLayout.create(tmp_path)
    layout.media_info_json.write_text(
        json.dumps(
            {
                "format_name": "mp4", "duration": 4.0, "size": 1, "bit_rate": 1,
                "video_streams": [{"index": 0, "codec_name": "h264", "width": 3840, "height": 2160, "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "duration": 4.0}],
                "audio_streams": [], "raw": {},
            }
        ),
        encoding="utf-8",
    )
    layout.subtitles_json.write_text(
        json.dumps({"version": "1.0", "source_video": str(video), "duration": 4.0, "segments": [], "metadata": {}, "origin": "asr"}),
        encoding="utf-8",
    )
    layout.review_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0", "version": "1.0", "source_video": str(video),
                "active": "asr", "reviewed": False, "needs_review": False,
                "asr": {"version": "1.0", "source_video": str(video), "duration": 4.0, "segments": [], "metadata": {}, "origin": "asr"},
                "llm": None, "diffs": [],
                "asr_meta": {"provider": "groq", "model": "whisper-large-v3-turbo", "multilingual": True, "prompt_version": "asr-multilingual-v1"},
                "llm_meta": {"enabled": False, "provider": "groq", "model": "", "prompt_version": "", "status": "skipped", "error": None},
                "source_hashes": {}, "summary": {"total_entries": 0, "unchanged": 0, "accepted": 0, "rejected": 0, "needs_review": 0},
                "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    layout.phase2_batches_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "batches": [
                    {
                        "batch_id": "batch_0001", "start": 0.5, "end": 3.0,
                        "frames": [{"path": "phase2/frames_1fps/frame_00000500.jpg", "time": 0.5, "kind": "fixed_fps", "fps": 1.0}],
                        "subtitles": [],
                        "subtitle_metadata": {"active_source": "asr", "reviewed": False, "needs_review": False, "asr_model": "whisper", "calibration_model": None},
                        "user_request": "",
                        "analysis_instruction": {"treat_subtitles_as_supporting_context": True, "subtitle_is_not_absolute_ground_truth": True},
                    }
                ],
                "subtitle_metadata": {"active_source": "asr", "reviewed": False, "needs_review": False, "asr_model": "whisper", "calibration_model": None},
                "note": "Epic4 detail-frame fixture",
            }
        ),
        encoding="utf-8",
    )
    make_refinement_plan_json(
        layout.coarse_refinement_plan_json,
        duration=4.0,
        intervals=[
            {"interval_id": "interval_0001", "start": 0.5, "end": 3.0, "requested_fps": 10, "priority": "high",
             "reasons": ["need hi-res detail"], "questions": ["what happens here?"]}
        ],
    )

    cfg = AppConfig()
    cfg.refinement.max_frame_long_edge = 1600  # 普通 frame 上限
    cfg.refinement.detail_frame_max_long_edge = 4096  # detail frame 上限（> 1600）
    cfg.refinement.sheets_enabled = False  # 简化测试，不生成拼图

    result = run_epic3(video_path=video, output_dir=tmp_path, config=cfg, user_request="x")
    assert result.succeeded_intervals == 1

    # 读取 fine batch 找 detail frame
    fb_path = layout.refinement_batches_dir / "fine_interval_0001.json"
    assert fb_path.is_file()
    fb = json.loads(fb_path.read_text(encoding="utf-8"))

    detail_frames = fb.get("detail_frames", [])
    assert len(detail_frames) >= 1, "should have at least 1 detail frame"

    # 读取普通 refinement frame（来自 fine batch.frames）
    refinement_frames = fb.get("frames", [])
    assert len(refinement_frames) >= 1, "should have at least 1 refinement frame"

    # 验证分辨率：detail frame 实际尺寸 > 普通 frame
    from PIL import Image
    for df in detail_frames:
        df_path = Path(df["path"])
        assert df_path.is_file()
        with Image.open(df_path) as img:
            df_w, df_h = img.size
        # metadata 一致性
        assert df["output_width"] == df_w
        assert df["output_height"] == df_h
        # 4K 源视频 detail frame 应远大于普通 1600 上限
        assert df_w >= 3200 or df_h >= 1800, (
            f"detail frame should be > 1600 long edge, got {df_w}x{df_h}"
        )

    # 普通 frame 长边 ≤ 1600
    for rf in refinement_frames:
        rf_path = Path(rf["path"])
        if not rf_path.is_file():
            continue
        with Image.open(rf_path) as img:
            rf_w, rf_h = img.size
        assert max(rf_w, rf_h) <= 1600, (
            f"refinement frame should be ≤ 1600, got {rf_w}x{rf_h}"
        )

    # 断言 detail frame 至少一边大于普通 frame
    detail_max_w = 0
    detail_max_h = 0
    for df in detail_frames:
        with Image.open(df["path"]) as img:
            w, h = img.size
            detail_max_w = max(detail_max_w, w)
            detail_max_h = max(detail_max_h, h)
    assert detail_max_w > 1600 or detail_max_h > 1600


def test_detail_frame_resolution_mode_source_original(tmp_path: Path):
    """源视频长边不超过 4096 时，resolution_mode=source_original，scaled=false。"""
    video = tmp_path / "hd.mp4"
    _make_4k_video(video, duration=3.0)
    layout = OutputLayout.create(tmp_path)
    layout.media_info_json.write_text(
        json.dumps(
            {
                "format_name": "mp4", "duration": 3.0, "size": 1, "bit_rate": 1,
                "video_streams": [{"index": 0, "codec_name": "h264", "width": 3840, "height": 2160, "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "duration": 3.0}],
                "audio_streams": [], "raw": {},
            }
        ),
        encoding="utf-8",
    )
    layout.subtitles_json.write_text(
        json.dumps({"version": "1.0", "source_video": str(video), "duration": 3.0, "segments": [], "metadata": {}, "origin": "asr"}),
        encoding="utf-8",
    )
    layout.review_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0", "version": "1.0", "source_video": str(video),
                "active": "asr", "reviewed": False, "needs_review": False,
                "asr": {"version": "1.0", "source_video": str(video), "duration": 3.0, "segments": [], "metadata": {}, "origin": "asr"},
                "llm": None, "diffs": [],
                "asr_meta": {"provider": "groq", "model": "whisper-large-v3-turbo", "multilingual": True, "prompt_version": "asr-multilingual-v1"},
                "llm_meta": {"enabled": False, "provider": "groq", "model": "", "prompt_version": "", "status": "skipped", "error": None},
                "source_hashes": {}, "summary": {"total_entries": 0, "unchanged": 0, "accepted": 0, "rejected": 0, "needs_review": 0},
                "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    layout.phase2_batches_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "batches": [
                    {
                        "batch_id": "batch_0001", "start": 0.5, "end": 2.5,
                        "frames": [{"path": "phase2/frames_1fps/frame_00000500.jpg", "time": 0.5, "kind": "fixed_fps", "fps": 1.0}],
                        "subtitles": [],
                        "subtitle_metadata": {"active_source": "asr", "reviewed": False, "needs_review": False, "asr_model": "whisper", "calibration_model": None},
                        "user_request": "",
                        "analysis_instruction": {"treat_subtitles_as_supporting_context": True, "subtitle_is_not_absolute_ground_truth": True},
                    }
                ],
                "subtitle_metadata": {"active_source": "asr", "reviewed": False, "needs_review": False, "asr_model": "whisper", "calibration_model": None},
                "note": "Epic4 detail-mode fixture",
            }
        ),
        encoding="utf-8",
    )
    make_refinement_plan_json(
        layout.coarse_refinement_plan_json,
        duration=3.0,
        intervals=[{"interval_id": "interval_0001", "start": 0.5, "end": 2.5, "requested_fps": 10, "priority": "high"}],
    )

    cfg = AppConfig()
    cfg.refinement.max_frame_long_edge = 1600
    cfg.refinement.detail_frame_max_long_edge = 4096
    cfg.refinement.sheets_enabled = False

    run_epic3(video_path=video, output_dir=tmp_path, config=cfg, user_request="x")

    fb = json.loads((layout.refinement_batches_dir / "fine_interval_0001.json").read_text(encoding="utf-8"))
    for df in fb["detail_frames"]:
        # 3840x2160 source < 4096 上限 → 不缩放
        assert df["resolution_mode"] == "source_original"
        assert df["scaled"] is False
        assert df["source_width"] == 3840
        assert df["source_height"] == 2160


def test_detail_frame_resolution_mode_capped(tmp_path: Path):
    """源视频超过 detail 上限时，resolution_mode=capped_high_resolution，scaled=true。"""
    # 生成 8K（7680x4320）视频
    video = tmp_path / "8k.mp4"
    out_path = video
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=size=7680x4320:rate=10:duration=2.0",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, shell=False, check=False)
    if r.returncode != 0:
        pytest.skip(f"8K video generation failed: {r.stderr[-200:]}")

    layout = OutputLayout.create(tmp_path)
    layout.media_info_json.write_text(
        json.dumps(
            {
                "format_name": "mp4", "duration": 2.0, "size": 1, "bit_rate": 1,
                "video_streams": [{"index": 0, "codec_name": "h264", "width": 7680, "height": 4320, "avg_frame_rate": "10/1", "r_frame_rate": "10/1", "duration": 2.0}],
                "audio_streams": [], "raw": {},
            }
        ),
        encoding="utf-8",
    )
    layout.subtitles_json.write_text(
        json.dumps({"version": "1.0", "source_video": str(video), "duration": 2.0, "segments": [], "metadata": {}, "origin": "asr"}),
        encoding="utf-8",
    )
    layout.review_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0", "version": "1.0", "source_video": str(video),
                "active": "asr", "reviewed": False, "needs_review": False,
                "asr": {"version": "1.0", "source_video": str(video), "duration": 2.0, "segments": [], "metadata": {}, "origin": "asr"},
                "llm": None, "diffs": [],
                "asr_meta": {"provider": "groq", "model": "whisper-large-v3-turbo", "multilingual": True, "prompt_version": "asr-multilingual-v1"},
                "llm_meta": {"enabled": False, "provider": "groq", "model": "", "prompt_version": "", "status": "skipped", "error": None},
                "source_hashes": {}, "summary": {"total_entries": 0, "unchanged": 0, "accepted": 0, "rejected": 0, "needs_review": 0},
                "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    layout.phase2_batches_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "batches": [
                    {
                        "batch_id": "batch_0001", "start": 0.5, "end": 1.5,
                        "frames": [{"path": "phase2/frames_1fps/frame_00000500.jpg", "time": 0.5, "kind": "fixed_fps", "fps": 1.0}],
                        "subtitles": [],
                        "subtitle_metadata": {"active_source": "asr", "reviewed": False, "needs_review": False, "asr_model": "whisper", "calibration_model": None},
                        "user_request": "",
                        "analysis_instruction": {"treat_subtitles_as_supporting_context": True, "subtitle_is_not_absolute_ground_truth": True},
                    }
                ],
                "subtitle_metadata": {"active_source": "asr", "reviewed": False, "needs_review": False, "asr_model": "whisper", "calibration_model": None},
                "note": "Epic4 8K fixture",
            }
        ),
        encoding="utf-8",
    )
    make_refinement_plan_json(
        layout.coarse_refinement_plan_json,
        duration=2.0,
        intervals=[{"interval_id": "interval_0001", "start": 0.5, "end": 1.5, "requested_fps": 10, "priority": "high"}],
    )

    cfg = AppConfig()
    cfg.refinement.max_frame_long_edge = 1600
    cfg.refinement.detail_frame_max_long_edge = 4096  # 源视频 7680 > 4096 → 触发 capped
    cfg.refinement.sheets_enabled = False

    run_epic3(video_path=video, output_dir=tmp_path, config=cfg, user_request="x")

    fb = json.loads((layout.refinement_batches_dir / "fine_interval_0001.json").read_text(encoding="utf-8"))
    for df in fb["detail_frames"]:
        assert df["resolution_mode"] == "capped_high_resolution"
        assert df["scaled"] is True
        assert df["source_width"] == 7680
        assert df["source_height"] == 4320
        # 输出尺寸 ≤ 4096
        assert df["output_width"] <= 4096
        assert df["output_height"] <= 4096
        # 但仍 > 1600（普通 frame 上限）
        assert max(df["output_width"], df["output_height"]) > 1600