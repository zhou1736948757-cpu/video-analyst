"""Explicit yt-dlp import for public video URLs.

The core Video Analyst pipeline remains local-file-only: callers run this
command first, then pass the resulting local path to ``prepare``.  This module
never discovers or uses browser sessions/cookies implicitly, and the optional
cookie file is never written to the source manifest.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from video_analyst.env.detect import detect_ytdlp_environment
from video_analyst.exceptions import SourceImportError, YtDlpNotFoundError

SUPPORTED_SUBTITLE_LANGUAGES = ("zh-Hans", "zh-Hant", "zh", "ja", "en", "all")
_SUBTITLE_SELECTORS = {
    # Keep the selection explicit and data-driven.  In particular, do not
    # hardcode an English-only selector for all sources.
    "zh-Hans": "zh-Hans,zh-CN,zh-SG,zh",
    "zh-Hant": "zh-Hant,zh-TW,zh-HK,zh-MO,zh",
    "zh": "zh.*,cmn",
    "ja": "ja.*,jpn",
    "en": "en.*,en-US,en-GB",
    "all": "all",
}
_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv")


@dataclass
class SourceImportResult:
    status: str
    output_dir: Path
    video_path: Path
    manifest_path: Path
    source_manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_dir": str(self.output_dir),
            "video_path": str(self.video_path),
            "manifest_path": str(self.manifest_path),
            "source_manifest": self.source_manifest,
        }


def is_public_url(value: str) -> bool:
    """Return true for an HTTP(S) URL; authentication is never inferred."""

    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def subtitle_selector(language: str) -> str:
    """Map a supported subtitle language choice to a yt-dlp selector."""

    normalized = str(language or "all").strip()
    if normalized not in _SUBTITLE_SELECTORS:
        choices = ", ".join(SUPPORTED_SUBTITLE_LANGUAGES)
        raise ValueError(f"unsupported subtitle language {language!r}; choose one of {choices}")
    return _SUBTITLE_SELECTORS[normalized]


def build_ytdlp_command(
    url: str,
    output_template: Path,
    *,
    subtitle_language: str = "all",
    no_playlist: bool = True,
    playlist: bool = False,
    part: int | None = None,
    cookies: Path | None = None,
    yt_dlp_bin: str = "yt-dlp",
) -> list[str]:
    """Build a side-effect-free yt-dlp command for unit testing.

    ``--no-playlist`` is the default.  A caller must opt into ``playlist=True``
    to download a playlist, and ``part`` maps to yt-dlp's explicit
    ``--playlist-items`` selection.  Cookie files are accepted only when
    explicitly supplied by the caller.
    """

    if not is_public_url(url):
        raise ValueError("source must be a public http(s) URL")
    if playlist and no_playlist:
        # The public API's defaults are intentionally unambiguous.  Callers
        # that opt into a playlist can omit no_playlist or pass False.
        no_playlist = False
    if part is not None:
        try:
            part_int = int(part)
        except (TypeError, ValueError) as exc:
            raise ValueError("part must be a positive integer") from exc
        if part_int <= 0:
            raise ValueError("part must be a positive integer")
    else:
        part_int = None
    if cookies is not None:
        cookie_path = Path(cookies).expanduser().resolve()
        if not cookie_path.is_file():
            raise ValueError(f"explicit cookie file not found: {cookie_path}")
    else:
        cookie_path = None

    cmd = [
        str(yt_dlp_bin),
        "--no-warnings",
        "--print-json",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        subtitle_selector(subtitle_language),
        "--sub-format",
        "vtt",
        "--convert-subs",
        "vtt",
        "--merge-output-format",
        "mp4",
    ]
    if no_playlist:
        cmd.append("--no-playlist")
    elif playlist:
        cmd.append("--yes-playlist")
    if part_int is not None:
        cmd.extend(["--playlist-items", str(part_int)])
    if cookie_path is not None:
        cmd.extend(["--cookies", str(cookie_path)])
    cmd.extend(["-o", str(output_template), "--", url])
    return cmd


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _read_info_files(out_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.info.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _candidate_video_paths(out_dir: Path, info: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for key in ("_filename", "filepath", "filename"):
        value = info.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
                candidates.append(path.resolve())
    for item in info.get("requested_downloads") or []:
        if not isinstance(item, dict):
            continue
        for key in ("filepath", "filename", "_filename"):
            value = item.get(key)
            if value:
                path = Path(str(value)).expanduser()
                if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
                    candidates.append(path.resolve())
    for ext in _VIDEO_EXTENSIONS:
        candidates.extend(sorted(out_dir.glob(f"video*{ext}")))
    # Deterministic de-duplication while preserving yt-dlp metadata order.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _duration(info: dict[str, Any]) -> float | None:
    try:
        value = float(info.get("duration"))
    except (TypeError, ValueError):
        return None
    return round(value, 6) if math.isfinite(value) and value >= 0 else None


def import_url(
    url: str,
    output_dir: Path,
    *,
    subtitle_language: str = "all",
    playlist: bool = False,
    no_playlist: bool = True,
    part: int | None = None,
    cookies: Path | None = None,
    yt_dlp_bin: str | None = None,
    timeout: float = 1800.0,
    runner: Callable[..., Any] | None = None,
) -> SourceImportResult:
    """Download one public URL (ordinary Bilibili URLs included).

    ``runner`` is injectable so unit tests can mock yt-dlp without touching the
    network.  It must have the same call shape as ``subprocess.run``.
    """

    if not is_public_url(url):
        raise SourceImportError("source must be a public http(s) URL")
    try:
        selector = subtitle_selector(subtitle_language)
    except ValueError as exc:
        raise SourceImportError(str(exc)) from exc
    if yt_dlp_bin is None:
        # Tests inject a runner and do not need a local executable. Real calls
        # resolve the shared tools directory so PATH customization is not
        # required for every Agent process.
        if runner is None:
            try:
                yt_dlp_bin = str(detect_ytdlp_environment().path)
            except YtDlpNotFoundError as exc:
                raise SourceImportError(str(exc)) from exc
        else:
            yt_dlp_bin = "yt-dlp"
    try:
        command = build_ytdlp_command(
            url,
            Path(output_dir).expanduser().resolve() / "source" / "video.%(ext)s",
            subtitle_language=subtitle_language,
            no_playlist=no_playlist,
            playlist=playlist,
            part=part,
            cookies=cookies,
            yt_dlp_bin=yt_dlp_bin,
        )
    except ValueError as exc:
        raise SourceImportError(str(exc)) from exc

    root = Path(output_dir).expanduser().resolve()
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    invoke = runner or subprocess.run
    try:
        completed = invoke(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
            cwd=str(source_dir),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceImportError(f"yt-dlp could not be started: {exc}") from exc

    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    metadata_records = [*_json_lines(stdout), *_read_info_files(source_dir)]
    info = metadata_records[-1] if metadata_records else {}
    videos = _candidate_video_paths(source_dir, info)
    if not videos:
        detail = stderr.strip()[-500:] or "no local video was produced"
        raise SourceImportError(
            f"yt-dlp did not produce a local video (exit {getattr(completed, 'returncode', '?')}): {detail}"
        )
    if int(getattr(completed, "returncode", 1) or 0) != 0:
        # Subtitle endpoints may fail while a usable video is present.  Keep
        # the import successful, but preserve the warning in the manifest.
        warning = stderr.strip()[-500:]
    else:
        warning = None

    subtitle_paths = sorted(source_dir.glob("video*.vtt"))
    extractor = (
        info.get("extractor_key")
        or info.get("extractor")
        or info.get("ie_key")
        or "unknown"
    )
    selected_part: dict[str, Any] = {
        "strategy": "playlist" if playlist else "single",
        "playlist_index": info.get("playlist_index"),
        "part": int(part) if part is not None else None,
    }
    manifest = {
        "schema_version": "1.0",
        "original_url": url,
        "extractor": str(extractor),
        "title": info.get("title"),
        "duration": _duration(info),
        "selected_part": selected_part,
        "video_path": str(videos[0]),
        "video_paths": [str(path) for path in videos],
        "subtitle_provenance": {
            "requested_language": subtitle_language,
            "selector": selector,
            "files": [str(path) for path in subtitle_paths],
            "source": "yt-dlp_native_or_automatic_captions",
        },
        "playlist_strategy": selected_part["strategy"],
        "cookies_used": cookies is not None,
        "cookies_source": "explicit_file" if cookies is not None else None,
        "warning": warning,
        "note": (
            "Explicit public URL import only. Pass the resulting local video "
            "path to video-analyst prepare; the core pipeline is local-file-only."
        ),
    }
    manifest_path = source_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    # A root-level alias makes the source manifest discoverable without knowing
    # the sidecar directory, while both files contain the same provenance.
    root_manifest_path = root / "source_manifest.json"
    _write_json(root_manifest_path, manifest)
    return SourceImportResult(
        status="source_imported",
        output_dir=source_dir,
        video_path=videos[0],
        manifest_path=manifest_path,
        source_manifest=manifest,
    )


download_source = import_url


__all__ = [
    "SUPPORTED_SUBTITLE_LANGUAGES",
    "SourceImportResult",
    "build_ytdlp_command",
    "download_source",
    "import_url",
    "is_public_url",
    "subtitle_selector",
]
