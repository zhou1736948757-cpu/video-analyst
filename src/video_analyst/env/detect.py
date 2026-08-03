"""Runtime environment detection (Python, FFmpeg, ASR backends)."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from video_analyst.exceptions import FFmpegNotFoundError
from video_analyst.process import run_command


def tools_root() -> Path:
    """Shared tools directory usable by all agents on this machine."""
    override = os.environ.get("VIDEO_ANALYST_TOOLS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".video-analyst" / "tools").resolve()


def tools_bin_dir() -> Path:
    return tools_root() / "bin"


def get_os_family() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system


def detect_python_ok(min_major: int = 3, min_minor: int = 11) -> tuple[bool, str]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = (sys.version_info.major, sys.version_info.minor) >= (min_major, min_minor)
    return ok, version


@dataclass(frozen=True)
class FFmpegEnvironment:
    ffmpeg_path: Path
    ffprobe_path: Path
    ffmpeg_version: str
    ffprobe_version: str


@dataclass(frozen=True)
class EnvironmentReport:
    os_family: str
    python_ok: bool
    python_version: str
    ffmpeg: FFmpegEnvironment | None
    ffmpeg_error: str | None
    groq_api_key_configured: bool
    local_whisper_installed: bool
    tools_bin: Path

    @property
    def ready(self) -> bool:
        return self.python_ok and self.ffmpeg is not None


def _candidate_bin_dirs() -> list[Path]:
    dirs: list[Path] = [tools_bin_dir()]
    # macOS Homebrew locations
    dirs.extend(
        [
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
        ]
    )
    # Common Windows install locations (checked only if present)
    if get_os_family() == "windows":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        dirs.extend(
            [
                Path(program_files) / "ffmpeg" / "bin",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
            ]
        )
    return dirs


def _which_command(name: str) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found).resolve()

    for directory in _candidate_bin_dirs():
        candidate = directory / name
        if get_os_family() == "windows":
            for suffix in ("", ".exe"):
                win_candidate = directory / f"{name}{suffix}"
                if win_candidate.is_file():
                    return win_candidate.resolve()
        elif candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _version_first_line(binary: Path) -> str:
    result = run_command([binary, "-version"], timeout=30)
    if not result.ok:
        raise FFmpegNotFoundError(
            f"{binary.name} exited with code {result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:500]}"
        )
    line = (result.stdout or result.stderr).splitlines()
    if not line:
        raise FFmpegNotFoundError(f"{binary.name} produced empty version output")
    return line[0].strip()


def detect_ffmpeg_environment() -> FFmpegEnvironment:
    """
    Locate and verify ffmpeg + ffprobe.

    Uses shutil.which, then known Homebrew / tools paths.
    Does not modify the system environment.
    """
    ffmpeg = _which_command("ffmpeg")
    ffprobe = _which_command("ffprobe")

    missing: list[str] = []
    if ffmpeg is None:
        missing.append("ffmpeg")
    if ffprobe is None:
        missing.append("ffprobe")
    if missing:
        raise FFmpegNotFoundError(
            "Missing required tools: "
            + ", ".join(missing)
            + ". Run `video-analyst setup` or install FFmpeg via your OS package manager."
        )

    assert ffmpeg is not None and ffprobe is not None
    ffmpeg_version = _version_first_line(ffmpeg)
    ffprobe_version = _version_first_line(ffprobe)

    return FFmpegEnvironment(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
    )


def is_local_whisper_installed() -> bool:
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def is_groq_api_key_configured() -> bool:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return bool(key)


def collect_environment_report() -> EnvironmentReport:
    python_ok, python_version = detect_python_ok()
    ffmpeg: FFmpegEnvironment | None = None
    ffmpeg_error: str | None = None
    try:
        ffmpeg = detect_ffmpeg_environment()
    except FFmpegNotFoundError as exc:
        ffmpeg_error = str(exc)

    return EnvironmentReport(
        os_family=get_os_family(),
        python_ok=python_ok,
        python_version=python_version,
        ffmpeg=ffmpeg,
        ffmpeg_error=ffmpeg_error,
        groq_api_key_configured=is_groq_api_key_configured(),
        local_whisper_installed=is_local_whisper_installed(),
        tools_bin=tools_bin_dir(),
    )


def ensure_ready_for_media(
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> FFmpegEnvironment:
    """Raise EnvironmentBlockedError-style messages via VideoAnalystError subclasses."""
    from video_analyst.exceptions import EnvironmentBlockedError

    python_ok, python_version = detect_python_ok()
    if not python_ok:
        raise EnvironmentBlockedError(
            f"Python 3.11+ required, found {python_version}"
        )

    env = detect_ffmpeg_environment()

    if input_path is not None:
        path = input_path.expanduser().resolve()
        if not path.is_file():
            raise EnvironmentBlockedError(f"Input file not readable: {path}")
        if not os.access(path, os.R_OK):
            raise EnvironmentBlockedError(f"Input file not readable: {path}")

    if output_dir is not None:
        out = output_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        if not os.access(out, os.W_OK):
            raise EnvironmentBlockedError(f"Output directory not writable: {out}")

    return env
