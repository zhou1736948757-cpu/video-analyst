"""Tests for the FFmpeg fine-frame extractor.

These tests generate a tiny MP4 using lavfi (no internet), then verify:
- 5/10/15 FPS extraction produces reasonable frame counts
- timestamps are monotonically increasing
- timestamps stay within interval bounds
- frame count matches showinfo record count (no filename-based inference)
- CFR and VFR sources both work
- landscape and portrait sources both work
- filenames with spaces / non-ASCII work
- precise interval at start and end of video work
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from video_analyst.config import RefinementConfig
from video_analyst.env.detect import detect_ffmpeg_environment
from video_analyst.exceptions import RefinementFrameExtractionError
from video_analyst.refinement.extractor import (
    ExtractedFrame,
    extract_interval_frames,
)
from video_analyst.refinement.models import RefinementSource
from video_analyst.refinement.timestamp_parser import compute_actual_fps

from tests._epic3_fixtures import (
    ffprobe_duration,
    has_ffmpeg,
    has_pillow,
    make_test_video,
)


pytestmark = pytest.mark.skipif(
    not (has_ffmpeg() and has_pillow()),
    reason="ffmpeg + Pillow required for extractor tests",
)


@pytest.fixture(scope="module")
def ffmpeg_env():
    return detect_ffmpeg_environment()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    out = tmp_path / "v.mp4"
    make_test_video(out, duration=6.0, fps=30, width=320, height=240)
    return out


def _src(start: float, end: float, fps: int, *, interval_id: str = "interval_0001") -> RefinementSource:
    return RefinementSource(
        interval_id=interval_id,
        start=start,
        end=end,
        requested_fps=fps,  # type: ignore[arg-type]
        priority="high",
    )


def _cfg(**overrides) -> RefinementConfig:
    cfg = RefinementConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _assert_monotonic_increasing(frames: list[ExtractedFrame]):
    last = -1.0
    for f in frames:
        assert f.absolute_timestamp >= last - 1e-6, (
            f"timestamp {f.absolute_timestamp} decreased after {last}"
        )
        last = f.absolute_timestamp


def _assert_within_interval(frames: list[ExtractedFrame], start: float, end: float, tol: float = 0.2):
    for f in frames:
        # allow a small tolerance for seek boundary
        assert f.absolute_timestamp >= start - tol, (
            f"frame ts {f.absolute_timestamp} < start {start} - tol"
        )
        assert f.absolute_timestamp <= end + tol, (
            f"frame ts {f.absolute_timestamp} > end {end} + tol"
        )


def test_extract_5fps(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 3.0, 5)
    out_dir = tmp_path / "frames"
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=out_dir,
        cfg=_cfg(),
    )
    # 2 seconds @ 5 FPS → ~10-11 frames
    assert 8 <= len(frames) <= 12
    _assert_monotonic_increasing(frames)
    _assert_within_interval(frames, 1.0, 3.0)
    # Inter-frame interval ~0.2s
    if len(frames) >= 2:
        gap = frames[1].absolute_timestamp - frames[0].absolute_timestamp
        assert 0.1 <= gap <= 0.4, f"unexpected gap {gap}"


def test_extract_10fps(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 3.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    assert 18 <= len(frames) <= 22
    _assert_monotonic_increasing(frames)
    _assert_within_interval(frames, 1.0, 3.0)


def test_extract_15fps(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 3.0, 15)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    assert 28 <= len(frames) <= 32
    _assert_monotonic_increasing(frames)
    _assert_within_interval(frames, 1.0, 3.0)


def test_extract_at_video_start(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(0.0, 1.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    assert len(frames) >= 1
    _assert_monotonic_increasing(frames)
    _assert_within_interval(frames, 0.0, 1.0, tol=0.05)


def test_extract_at_video_end(tmp_path: Path, video: Path, ffmpeg_env):
    duration = ffprobe_duration(video)
    interval = _src(max(0.0, duration - 1.0), duration, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    assert len(frames) >= 1
    _assert_monotonic_increasing(frames)


def test_extract_vfr_source(tmp_path: Path, ffmpeg_env):
    video = tmp_path / "vfr.mp4"
    make_test_video(video, duration=6.0, fps=30, vfr=True)
    interval = _src(1.0, 3.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    # VFR may produce fewer frames; that's acceptable
    assert len(frames) >= 1
    _assert_monotonic_increasing(frames)


def test_extract_portrait(tmp_path: Path, ffmpeg_env):
    video = tmp_path / "portrait.mp4"
    make_test_video(video, duration=4.0, fps=30, width=240, height=320)
    interval = _src(0.5, 2.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    assert len(frames) >= 1
    for f in frames:
        assert f.width > 0 and f.height > 0


def test_extract_with_spaces_in_path(tmp_path: Path, ffmpeg_env):
    video = tmp_path / "my video file.mp4"
    make_test_video(video, duration=4.0, fps=30)
    interval = _src(1.0, 2.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames with spaces",
        cfg=_cfg(),
    )
    assert len(frames) >= 1


def test_extract_with_non_ascii_path(tmp_path: Path, ffmpeg_env):
    video = tmp_path / "视频 测试.mp4"
    make_test_video(video, duration=4.0, fps=30)
    interval = _src(1.0, 2.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "帧目录",
        cfg=_cfg(),
    )
    assert len(frames) >= 1


def test_extract_frame_count_matches_files(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 3.0, 10)
    out_dir = tmp_path / "frames"
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=out_dir,
        cfg=_cfg(),
    )
    files = sorted(out_dir.glob("frame*.jpg"))
    assert len(files) == len(frames), (
        f"file count {len(files)} != frame record count {len(frames)}"
    )


def test_extract_produces_sha256(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 2.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    for f in frames:
        assert f.sha256 and len(f.sha256) == 64
        assert f.size_bytes > 0


def test_extract_failure_on_missing_video(tmp_path: Path, ffmpeg_env):
    interval = _src(1.0, 2.0, 10)
    with pytest.raises(RefinementFrameExtractionError):
        extract_interval_frames(
            ffmpeg_env=ffmpeg_env,
            source_video=tmp_path / "does_not_exist.mp4",
            interval=interval,
            out_dir=tmp_path / "frames",
            cfg=_cfg(),
        )


def test_extract_jpeg_quality_config(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 2.0, 10)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=out_a,
        cfg=_cfg(jpeg_quality=2),
    )
    extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=out_b,
        cfg=_cfg(jpeg_quality=15),
    )
    fa = sorted(out_a.glob("frame*.jpg"))
    fb = sorted(out_b.glob("frame*.jpg"))
    if fa and fb:
        size_a = fa[0].stat().st_size
        size_b = fb[0].stat().st_size
        # Higher quality (lower q:v) → larger file (most of the time)
        # Allow equality for tiny synthetic sources.
        assert size_a >= size_b


def test_extract_long_edge_respected(tmp_path: Path, ffmpeg_env):
    video = tmp_path / "big.mp4"
    make_test_video(video, duration=4.0, fps=30, width=640, height=480)
    interval = _src(1.0, 2.0, 10)
    out_dir = tmp_path / "frames"
    extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=out_dir,
        cfg=_cfg(max_frame_long_edge=320),
    )
    files = sorted(out_dir.glob("frame*.jpg"))
    assert files
    from PIL import Image

    with Image.open(files[0]) as im:
        assert max(im.width, im.height) <= 320 + 2  # allow rounding


def test_extract_actual_fps_close_to_requested(tmp_path: Path, video: Path, ffmpeg_env):
    interval = _src(1.0, 3.0, 10)
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=tmp_path / "frames",
        cfg=_cfg(),
    )
    if len(frames) >= 2:
        from video_analyst.refinement.extractor import ExtractedFrame

        # Reconstruct ShowinfoEntry list for compute_actual_fps
        from video_analyst.refinement.timestamp_parser import ShowinfoEntry

        entries = [
            ShowinfoEntry(pts=None, pts_time=f.relative_timestamp)
            for f in frames
        ]
        actual = compute_actual_fps(entries)
        assert actual is not None
        # Should be within 15% of 10 FPS
        assert abs(actual - 10.0) <= 2.0, f"actual fps {actual} too far from 10"


def _make_gop_video(out_path: Path, *, duration: float = 8.0, gop: int = 60) -> Path:
    """生成 GOP 明显大于 1 秒的测试视频（30fps + gop=60 → 2s GOP）。

    关键帧落在 0/2/4/6 秒，因此 2.37s 是非关键帧。用于验证 input-side
    `-ss` seek + showinfo pts_time 不会错误落到前一个关键帧。
    """
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        out_path.unlink()
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=size=320x240:rate=30:duration={duration}",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=False, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"gop video generation failed: {r.stderr[-300:]}")
    return out_path


def test_extract_from_non_keyframe_start(tmp_path: Path, ffmpeg_env):
    """非关键帧起点抽帧准确性验收。

    视频关键帧在 0/2/4/6 秒，interval 从 2.37s（非关键帧）开始，10 FPS 抽帧。
    验证：
    1. 第一帧绝对时间接近 2.37s（不落到 2.0s 关键帧）
    2. interval.start + pts_time 与实际时间轴一致
    3. 时间戳单调递增
    """
    video = tmp_path / "gop.mp4"
    _make_gop_video(video, duration=8.0, gop=60)

    # 确认关键帧位置（0/2/4/6）
    import subprocess

    r = subprocess.run(
        ["ffprobe", "-loglevel", "error", "-select_streams", "v:0",
         "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, timeout=30, shell=False, check=False,
    )
    keyframe_times = []
    for line in r.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        # 行格式形如 "2.000000," 取首个逗号前的数字
        s = s.split(",")[0].strip()
        try:
            keyframe_times.append(float(s))
        except ValueError:
            continue
    assert 0.0 in keyframe_times
    assert 2.0 in keyframe_times
    assert 4.0 in keyframe_times
    assert 2.37 not in keyframe_times, "2.37s should NOT be a keyframe"

    interval_start = 2.37
    interval_end = 3.57  # 1.2s 区间
    interval = _src(interval_start, interval_end, 10)
    out_dir = tmp_path / "frames"
    frames = extract_interval_frames(
        ffmpeg_env=ffmpeg_env,
        source_video=video,
        interval=interval,
        out_dir=out_dir,
        cfg=_cfg(),
    )
    assert len(frames) >= 1, "should produce at least one frame"

    # 第一帧绝对时间必须接近 2.37s，不能落到 2.0s 关键帧
    first_ts = frames[0].absolute_timestamp
    # 容许 0.15s 编码误差（一个 10fps 帧间隔 = 0.1s）
    tolerance = 0.15
    assert abs(first_ts - interval_start) <= tolerance, (
        f"first frame ts {first_ts} too far from interval start {interval_start} "
        f"(|Δ|={abs(first_ts - interval_start):.4f} > {tolerance}); "
        f"may have snapped to previous keyframe at 2.0s"
    )
    # 明确不等于 2.0（前一个关键帧）
    assert first_ts > 2.2, (
        f"first frame ts {first_ts} snapped to previous keyframe at 2.0s"
    )

    # 时间戳单调递增
    _assert_monotonic_increasing(frames)

    # interval.start + pts_time == absolute_timestamp（时间戳换算一致性）
    for f in frames:
        recomputed = interval_start + f.relative_timestamp
        assert abs(recomputed - f.absolute_timestamp) < 1e-6, (
            f"interval.start({interval_start}) + pts_time({f.relative_timestamp}) "
            f"= {recomputed} != absolute_timestamp({f.absolute_timestamp})"
        )

    # 相邻间隔接近 0.1s（10 FPS）
    if len(frames) >= 2:
        gaps = [
            frames[i + 1].absolute_timestamp - frames[i].absolute_timestamp
            for i in range(len(frames) - 1)
        ]
        avg_gap = sum(gaps) / len(gaps)
        assert 0.08 <= avg_gap <= 0.12, f"avg gap {avg_gap} not close to 0.1s"

