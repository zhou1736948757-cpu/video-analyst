"""Install the official yt-dlp executable into the shared tools directory."""

from __future__ import annotations

import os
import stat
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from video_analyst.env.detect import (
    YtDlpEnvironment,
    detect_ytdlp_environment,
    get_os_family,
)
from video_analyst.env.setup_ffmpeg import ensure_tools_directory
from video_analyst.exceptions import YtDlpInstallError, YtDlpNotFoundError
from video_analyst.logging.setup import get_logger

log = get_logger("env.setup_ytdlp")

YTDLP_RELEASE_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"
YTDLP_WINDOWS_RELEASE_URL = (
    "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
)
DownloadFile = Callable[[str, Path], None]


def _release_url() -> str:
    if get_os_family() == "windows":
        return YTDLP_WINDOWS_RELEASE_URL
    return YTDLP_RELEASE_URL


def _download_official(url: str, destination: Path) -> None:
    """Download one official release asset without shell or cookie state."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "video-analyst-setup"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if len(payload) < 1024:
        raise YtDlpInstallError("Downloaded yt-dlp asset is unexpectedly small.")
    destination.write_bytes(payload)


def _make_executable(path: Path) -> None:
    if get_os_family() == "windows":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_ytdlp(
    *,
    confirm: bool = True,
    assume_yes: bool = False,
    downloader: DownloadFile | None = None,
) -> YtDlpEnvironment:
    """Install yt-dlp if missing, then verify its version.

    The package installation itself stays side-effect free.  This explicit
    setup action downloads the official release asset into the shared tools
    directory, and never reads browser sessions or cookie files.
    """

    tools_bin = ensure_tools_directory()
    target = tools_bin / ("yt-dlp.exe" if get_os_family() == "windows" else "yt-dlp")

    try:
        env = detect_ytdlp_environment()
        log.info("yt-dlp already available: %s", env.path)
        return env
    except YtDlpNotFoundError:
        pass

    url = _release_url()
    print("yt-dlp not found.")
    print(f"Planned download: {url}")
    print(f"Destination: {target}")
    print("No browser cookies or cookie files will be read.")
    if confirm and not assume_yes:
        answer = input("Proceed with yt-dlp download? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise YtDlpInstallError("yt-dlp installation cancelled by user.")

    downloader = downloader or _download_official
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=tools_bin,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        downloader(url, temporary)
        if not temporary.is_file() or temporary.stat().st_size < 1024:
            raise YtDlpInstallError("Downloaded yt-dlp asset is empty or incomplete.")
        _make_executable(temporary)
        temporary.replace(target)
    except YtDlpInstallError:
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise YtDlpInstallError(f"yt-dlp download failed: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    try:
        env = detect_ytdlp_environment()
    except YtDlpNotFoundError as exc:
        raise YtDlpInstallError(
            "yt-dlp download finished but the executable could not be verified. "
            "Re-open the terminal or run `video-analyst doctor`."
        ) from exc
    log.info("yt-dlp installed and verified: %s", env.path)
    return env


__all__ = [
    "YTDLP_RELEASE_URL",
    "YTDLP_WINDOWS_RELEASE_URL",
    "install_ytdlp",
]
