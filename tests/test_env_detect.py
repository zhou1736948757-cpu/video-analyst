from pathlib import Path
from unittest.mock import patch

import pytest

from video_analyst.env.detect import (
    detect_ffmpeg_environment,
    detect_python_ok,
    get_os_family,
)
from video_analyst.exceptions import FFmpegNotFoundError
from video_analyst.process import CommandResult


def test_python_ok():
    ok, version = detect_python_ok()
    assert ok is True
    assert version.count(".") >= 2


def test_os_family():
    family = get_os_family()
    assert family in {"macos", "windows", "linux"} or isinstance(family, str)


def test_detect_ffmpeg_missing():
    with (
        patch("video_analyst.env.detect._which_command", return_value=None),
    ):
        with pytest.raises(FFmpegNotFoundError):
            detect_ffmpeg_environment()


def test_detect_ffmpeg_ok(tmp_path: Path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("")
    ffprobe.write_text("")

    def fake_which(name: str) -> Path | None:
        return ffmpeg if name == "ffmpeg" else ffprobe

    def fake_run(args, **kwargs):
        name = Path(args[0]).name
        return CommandResult(
            args=[str(a) for a in args],
            returncode=0,
            stdout=f"{name} version 6.0 test\n",
            stderr="",
        )

    with (
        patch("video_analyst.env.detect._which_command", side_effect=fake_which),
        patch("video_analyst.env.detect.run_command", side_effect=fake_run),
    ):
        env = detect_ffmpeg_environment()
        assert env.ffmpeg_path == ffmpeg
        assert env.ffprobe_path == ffprobe
        assert "ffmpeg version" in env.ffmpeg_version
