"""Focused regression tests for the local read-video capability fusion.

All tests are deterministic and mock media/subprocess boundaries; no network,
cookie store, external vision API, or real-video extraction is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from video_analyst.config import AppConfig
from video_analyst.inspection.prepare import _validate_scope, parse_timestamp, prepare_inspection
from video_analyst.phase2.deduplicate import deduplicate_frames
from video_analyst.source.importer import build_ytdlp_command, import_url


def _image(path: Path, value: int) -> Path:
    Image.new("RGB", (32, 24), (value, value, value)).save(path)
    return path


def test_coarse_dedupe_keeps_first_last_scene_and_pinned(tmp_path: Path):
    same = _image(tmp_path / "same.jpg", 20)
    candidates = [
        {"path": str(same), "time": 0.0, "kind": "fixed_fps"},
        {"path": str(same), "time": 1.0, "kind": "fixed_fps"},
        {"path": str(same), "time": 2.0, "kind": "scene"},
        {"path": str(same), "time": 3.0, "kind": "pinned", "pinned": True},
        {"path": str(same), "time": 4.0, "kind": "fixed_fps"},
    ]

    result = deduplicate_frames(candidates, threshold=2.0, pinned_timestamps=[3.0])

    assert result.candidate_count == 5
    assert result.kept_count == 4
    assert result.dropped_count == 1
    assert [frame["time"] for frame in result.kept] == [0.0, 2.0, 3.0, 4.0]
    assert result.dropped[0]["kept_candidate_index"] == 0


def test_coarse_dedupe_threshold_and_disabled_mode(tmp_path: Path):
    dark = _image(tmp_path / "dark.jpg", 10)
    near = _image(tmp_path / "near.jpg", 11)
    candidates = [
        {"path": str(dark), "time": 0.0},
        {"path": str(near), "time": 1.0},
        {"path": str(dark), "time": 2.0},
    ]

    assert deduplicate_frames(candidates, threshold=0.5).kept_count == 3
    assert deduplicate_frames(candidates, threshold=2.0).kept_count == 2
    assert deduplicate_frames(candidates, threshold=0.0, enabled=False).kept_count == 3


def test_phase2_dedupe_config_changes_cache_fingerprint():
    first = AppConfig().fingerprint()
    changed = AppConfig()
    changed.phase2.coarse_dedupe_threshold = 9.0
    changed.phase2.coarse_dedupe_enabled = False
    assert first["phase2"] != changed.fingerprint()["phase2"]


def test_timestamp_parsing_and_strict_bounds():
    assert parse_timestamp("12.5") == pytest.approx(12.5)
    assert parse_timestamp("01:02.5") == pytest.approx(62.5)
    assert parse_timestamp("01:02:03") == pytest.approx(3723.0)
    with pytest.raises(ValueError):
        parse_timestamp("1:60")
    with pytest.raises(ValueError):
        parse_timestamp("-1")
    with pytest.raises(Exception):
        _validate_scope(duration=10.0, start="2", end="8", at=["1"])


def test_inspection_sidecar_does_not_touch_existing_state(monkeypatch, tmp_path: Path):
    output = tmp_path / "output"
    state_path = output / "cache" / "workflow_state.json"
    state_path.parent.mkdir(parents=True)
    original_state = '{"status":"waiting_for_coarse_agent_analysis"}\n'
    state_path.write_text(original_state, encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"local-test-video")
    frame = _image(tmp_path / "frame.jpg", 30)

    monkeypatch.setattr("video_analyst.inspection.prepare.validate_video_input", lambda *_a, **_k: video)
    monkeypatch.setattr(
        "video_analyst.inspection.prepare.detect_ffmpeg_environment",
        lambda: SimpleNamespace(ffmpeg_path=Path("ffmpeg"), ffprobe_path=Path("ffprobe")),
    )
    monkeypatch.setattr(
        "video_analyst.inspection.prepare.probe_media",
        lambda *_a, **_k: SimpleNamespace(duration=5.0),
    )

    def fake_regular(*, out_dir: Path, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "candidate_000000.jpg"
        frame.replace(target)
        return [{"candidate_index": 0, "path": str(target), "time": 1.0, "kind": "inspection_fps", "requested_fps": 5}]

    monkeypatch.setattr("video_analyst.inspection.prepare._extract_interval_candidates", fake_regular)
    monkeypatch.setattr(
        "video_analyst.inspection.prepare._extract_pinned_candidates",
        lambda **_kwargs: [],
    )

    result = prepare_inspection(video, output, start="0", end="2", fps=5)

    assert result.manifest_path.is_file()
    assert state_path.read_text(encoding="utf-8") == original_state
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["frames"][0]["timestamp"] == 1.0
    assert manifest["instruction"]["does_not_change_coarse_or_fine_state"] is True


def test_bilibili_command_and_language_selection_without_cookies(tmp_path: Path):
    command = build_ytdlp_command(
        "https://www.bilibili.com/video/BV1public",
        tmp_path / "video.%(ext)s",
        subtitle_language="zh-Hant",
    )
    assert "--no-playlist" in command
    assert "--cookies" not in command
    assert "--cookies-from-browser" not in command
    assert "zh-Hant" in command[command.index("--sub-langs") + 1]


def test_import_url_uses_mocked_ytdlp_and_writes_provenance(tmp_path: Path):
    def fake_runner(command, **kwargs):
        assert "--no-playlist" in command
        assert "--cookies" not in command
        assert kwargs["shell"] is False
        source_dir = Path(kwargs["cwd"])
        video = source_dir / "video.mp4"
        video.write_bytes(b"mock-video")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "extractor_key": "BiliBili",
                    "title": "public",
                    "duration": 12.0,
                    "_filename": str(video),
                }
            ),
            stderr="",
        )

    result = import_url(
        "https://www.bilibili.com/video/BV1public",
        tmp_path,
        subtitle_language="zh",
        runner=fake_runner,
    )
    assert result.video_path.name == "video.mp4"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["original_url"].startswith("https://www.bilibili.com")
    assert manifest["extractor"] == "BiliBili"
    assert manifest["subtitle_provenance"]["requested_language"] == "zh"
    assert manifest["cookies_used"] is False
