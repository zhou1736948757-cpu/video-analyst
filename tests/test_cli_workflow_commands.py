"""CLI smoke tests for the in-session agent workflow commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from video_analyst.cli import app

runner = CliRunner()


def test_doctor_no_vision_key_required():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code in (0, 1)  # 1 if ffmpeg missing
    assert "vision_api_key_required" in result.output
    assert "false" in result.output.lower() or '"vision_api_key_required": false' in result.output
    assert "in_session_agent" in result.output


def test_status_on_empty_dir(tmp_path: Path):
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 0
    assert "not_started" in result.output
    assert "in_session_agent" in result.output


def test_validate_coarse_missing_prepare(tmp_path: Path):
    result = runner.invoke(app, ["validate-coarse", str(tmp_path)])
    assert result.exit_code == 1
    assert "FAILED" in result.output or "failed" in result.output.lower() or "No coarse" in result.output


def test_run_missing_video_exits_nonzero(tmp_path: Path):
    missing = tmp_path / "nope.mp4"
    result = runner.invoke(app, ["run", str(missing), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0


def test_provider_modes_removed():
    for cmd in ("provider-check", "coarse-analyze", "fine-analyze"):
        result = runner.invoke(app, [cmd])
        assert result.exit_code == 2
        assert "removed" in result.output.lower()
