"""Cross-platform FFmpeg installation helpers (confirmation required)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from video_analyst.env.detect import (
    FFmpegEnvironment,
    detect_ffmpeg_environment,
    get_os_family,
    tools_bin_dir,
    tools_root,
)
from video_analyst.exceptions import FFmpegInstallError, FFmpegNotFoundError
from video_analyst.logging.setup import get_logger
from video_analyst.process import run_command

log = get_logger("env.setup")

HOMEBREW_INSTALL_CMD = (
    '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
)


@dataclass(frozen=True)
class InstallPlan:
    os_family: str
    commands: list[list[str]]
    notes: list[str]
    requires_admin: bool = False


def _command_available(name: str) -> bool:
    return shutil.which(name) is not None


def _detect_linux_distro() -> str:
    os_release = Path("/etc/os-release")
    if not os_release.is_file():
        return "unknown"
    data: dict[str, str] = {}
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            data[key] = value.strip().strip('"')
    return (data.get("ID") or data.get("ID_LIKE") or "unknown").lower()


def recommended_install_commands() -> InstallPlan:
    """Return OS-specific install commands without executing them."""
    os_family = get_os_family()
    tools = tools_bin_dir()

    if os_family == "macos":
        if _command_available("brew"):
            return InstallPlan(
                os_family=os_family,
                commands=[["brew", "install", "ffmpeg"]],
                notes=[
                    "Uses Homebrew (trusted package manager).",
                    f"Optional tools directory for agents: {tools}",
                ],
                requires_admin=False,
            )
        return InstallPlan(
            os_family=os_family,
            commands=[],
            notes=[
                "Homebrew is not installed. Install Homebrew first (do not silent-install unknown software):",
                HOMEBREW_INSTALL_CMD,
                "Then run: brew install ffmpeg",
                f"Tools directory (created for agents): {tools}",
            ],
            requires_admin=False,
        )

    if os_family == "windows":
        if _command_available("winget"):
            return InstallPlan(
                os_family=os_family,
                commands=[
                    [
                        "winget",
                        "install",
                        "--id=Gyan.FFmpeg",
                        "-e",
                        "--accept-source-agreements",
                        "--accept-package-agreements",
                    ]
                ],
                notes=[
                    "Uses winget package Gyan.FFmpeg.",
                    "Re-open the terminal after install, then re-run video-analyst doctor.",
                ],
                requires_admin=True,
            )
        if _command_available("choco"):
            return InstallPlan(
                os_family=os_family,
                commands=[["choco", "install", "ffmpeg", "-y"]],
                notes=[
                    "Uses existing Chocolatey installation.",
                    "Chocolatey will not be auto-installed.",
                ],
                requires_admin=True,
            )
        return InstallPlan(
            os_family=os_family,
            commands=[],
            notes=[
                "No winget/choco available. Manual install required:",
                "1. Download a trusted FFmpeg Windows build from the official FFmpeg site or a well-known maintainer (e.g. gyan.dev / BtbN GitHub releases).",
                "2. Extract to a fixed directory (example: %USERPROFILE%\\.video-analyst\\tools).",
                "3. Place ffmpeg.exe and ffprobe.exe in the bin folder of that directory.",
                f"4. Either add that bin folder to PATH (only if you allow it) or set VIDEO_ANALYST_TOOLS_DIR to the tools root.",
                "5. Re-open the terminal and run: video-analyst doctor",
                "Do not download FFmpeg from random blogs or unverified mirrors.",
            ],
            requires_admin=False,
        )

    if os_family == "linux":
        distro = _detect_linux_distro()
        if distro in {"ubuntu", "debian"} or "debian" in distro or "ubuntu" in distro:
            return InstallPlan(
                os_family=os_family,
                commands=[
                    ["sudo", "apt", "update"],
                    ["sudo", "apt", "install", "-y", "ffmpeg"],
                ],
                notes=["Uses apt (Ubuntu/Debian). Requires sudo."],
                requires_admin=True,
            )
        if distro in {"fedora"} or "fedora" in distro:
            return InstallPlan(
                os_family=os_family,
                commands=[["sudo", "dnf", "install", "-y", "ffmpeg"]],
                notes=[
                    "Uses dnf (Fedora).",
                    "If the default repos do not provide FFmpeg, do not add unknown third-party repos automatically.",
                ],
                requires_admin=True,
            )
        if distro in {"arch", "manjaro"} or "arch" in distro:
            return InstallPlan(
                os_family=os_family,
                commands=[["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"]],
                notes=["Uses pacman (Arch)."],
                requires_admin=True,
            )
        return InstallPlan(
            os_family=os_family,
            commands=[],
            notes=[
                f"Unsupported or unknown Linux distro ({distro}).",
                "Install the ffmpeg package with your distro package manager so that both ffmpeg and ffprobe are available.",
            ],
            requires_admin=False,
        )

    return InstallPlan(
        os_family=os_family,
        commands=[],
        notes=[f"Unsupported OS family: {os_family}. Install FFmpeg manually."],
        requires_admin=False,
    )


def ensure_tools_directory() -> Path:
    """Create the shared tools directory used by agents."""
    root = tools_root()
    bin_dir = tools_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Video Analyst shared tools directory.\n"
            "Place portable ffmpeg/ffprobe/yt-dlp binaries in bin/ if not using a system package manager.\n"
            "`video-analyst setup` installs missing FFmpeg and yt-dlp by default.\n"
            "Set VIDEO_ANALYST_TOOLS_DIR to override this location.\n",
            encoding="utf-8",
        )
    return bin_dir


def _sanitize_error(text: str) -> str:
    """Keep install errors informative but strip obvious secrets."""
    cleaned = text.replace("\r", "")
    # Truncate huge dumps
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000] + "\n...[truncated]..."
    return cleaned


def install_ffmpeg(*, confirm: bool = True, assume_yes: bool = False) -> FFmpegEnvironment:
    """
    Install FFmpeg if missing.

    Does not reinstall when already available.
    Does not run privileged commands without confirmation unless assume_yes=True.
    """
    ensure_tools_directory()

    try:
        env = detect_ffmpeg_environment()
        log.info("FFmpeg already available: %s", env.ffmpeg_path)
        return env
    except FFmpegNotFoundError:
        pass

    plan = recommended_install_commands()
    if not plan.commands:
        notes = "\n".join(plan.notes)
        raise FFmpegInstallError(
            "Cannot auto-install FFmpeg on this system.\n" + notes
        )

    print("FFmpeg / ffprobe not found.")
    print(f"OS: {plan.os_family}")
    print("Planned install command(s):")
    for cmd in plan.commands:
        print("  $ " + " ".join(cmd))
    for note in plan.notes:
        print(f"  note: {note}")
    if plan.requires_admin:
        print("  warning: these commands may require administrator privileges.")

    if confirm and not assume_yes:
        answer = input("Proceed with installation? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise FFmpegInstallError("Installation cancelled by user.")

    for cmd in plan.commands:
        log.info("Running install command: %s", " ".join(cmd))
        result = run_command(cmd, timeout=600)
        if not result.ok:
            detail = _sanitize_error(result.stderr or result.stdout or "")
            raise FFmpegInstallError(
                f"Install command failed ({result.returncode}): {' '.join(cmd)}\n{detail}"
            )

    # Re-verify — never pretend success
    try:
        env = detect_ffmpeg_environment()
    except FFmpegNotFoundError as exc:
        raise FFmpegInstallError(
            "Installation finished but ffmpeg/ffprobe still not detectable. "
            "Re-open the terminal (PATH refresh) and run video-analyst doctor.\n"
            f"Detail: {exc}"
        ) from exc

    log.info("FFmpeg installed and verified: %s / %s", env.ffmpeg_path, env.ffprobe_path)
    return env
