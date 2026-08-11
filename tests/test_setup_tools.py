"""Deterministic tests for shared FFmpeg/yt-dlp setup behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from video_analyst.cli import app
from video_analyst.env.setup_ytdlp import YTDLP_RELEASE_URL, install_ytdlp
from video_analyst.source.importer import import_url

runner = CliRunner()


def test_install_ytdlp_uses_injected_downloader_and_shared_tools_dir(
    monkeypatch, tmp_path: Path
):
    tools_root = tmp_path / "tools"
    monkeypatch.setenv("VIDEO_ANALYST_TOOLS_DIR", str(tools_root))
    monkeypatch.setattr("video_analyst.env.detect.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "video_analyst.env.detect._candidate_bin_dirs",
        lambda: [tools_root / "bin"],
    )
    calls: list[str] = []

    def fake_download(url: str, destination: Path) -> None:
        calls.append(url)
        destination.write_text(
            "#!/bin/sh\n" + ("# test payload\n" * 120) + "printf '2026.07.04\\n'\n",
            encoding="utf-8",
        )

    result = install_ytdlp(confirm=False, downloader=fake_download)

    assert calls == [YTDLP_RELEASE_URL]
    assert result.path == tools_root / "bin" / "yt-dlp"
    assert result.version == "2026.07.04"
    assert result.path.stat().st_mode & 0o111


def test_setup_can_skip_both_downloads(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr("video_analyst.cli.ensure_tools_directory", lambda: tmp_path)
    monkeypatch.setattr(
        "video_analyst.cli.install_ffmpeg",
        lambda **_kwargs: calls.append("ffmpeg"),
    )
    monkeypatch.setattr(
        "video_analyst.cli.install_ytdlp",
        lambda **_kwargs: calls.append("yt-dlp"),
    )
    monkeypatch.setattr("video_analyst.cli._print_doctor", lambda: 0)

    result = runner.invoke(app, ["setup", "--no-ffmpeg", "--no-yt-dlp"])

    assert result.exit_code == 0
    assert calls == []
    assert "setup skipped" in result.output


def test_setup_defaults_to_both_tools(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr("video_analyst.cli.ensure_tools_directory", lambda: tmp_path)
    monkeypatch.setattr(
        "video_analyst.cli.install_ffmpeg",
        lambda **_kwargs: calls.append("ffmpeg")
        or SimpleNamespace(ffmpeg_path=tmp_path / "ffmpeg", ffprobe_path=tmp_path / "ffprobe"),
    )
    monkeypatch.setattr(
        "video_analyst.cli.install_ytdlp",
        lambda **_kwargs: calls.append("yt-dlp")
        or SimpleNamespace(path=tmp_path / "yt-dlp"),
    )
    monkeypatch.setattr("video_analyst.cli._print_doctor", lambda: 0)

    result = runner.invoke(app, ["setup", "--yes"])

    assert result.exit_code == 0
    assert calls == ["ffmpeg", "yt-dlp"]


def test_doctor_json_exposes_optional_url_import(monkeypatch):
    monkeypatch.setattr(
        "video_analyst.cli.collect_environment_report",
        lambda: SimpleNamespace(
            python_ok=True,
            python_version="3.13.0",
            ffmpeg=None,
            yt_dlp=None,
            groq_api_key_configured=False,
            local_whisper_installed=False,
            os_family="macos",
            ready=False,
            url_import_ready=False,
        ),
    )

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 1
    assert '"yt_dlp": false' in result.output
    assert '"url_import_ready": false' in result.output


def test_import_url_resolves_shared_ytdlp_for_real_runner(monkeypatch, tmp_path: Path):
    executable = tmp_path / "shared" / "yt-dlp"
    seen: list[str] = []
    monkeypatch.setattr(
        "video_analyst.source.importer.detect_ytdlp_environment",
        lambda: SimpleNamespace(path=executable),
    )

    def fake_runner(command, **kwargs):
        seen.append(command[0])
        source_dir = Path(kwargs["cwd"])
        video = source_dir / "video.mp4"
        video.write_bytes(b"mock-video")
        return SimpleNamespace(
            returncode=0,
            stdout='{"extractor_key":"BiliBili","title":"public"}',
            stderr="",
        )

    # Patch subprocess.run rather than passing runner so production resolution
    # selects the shared executable before invocation.
    monkeypatch.setattr("video_analyst.source.importer.subprocess.run", fake_runner)
    result = import_url("https://www.bilibili.com/video/BV1public", tmp_path, runner=None)
    assert result.video_path.is_file()
    assert seen == [str(executable)]
