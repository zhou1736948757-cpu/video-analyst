#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video-analyst common helpers — stdlib only, zero third-party deps.

PIL is never imported here; near-duplicate digests and tiles are produced by
ffmpeg itself (scale + format=gray rawvideo, tile filter). Nothing in this
module ever prints or persists an API key value.
"""

import json
import os
import re
import shutil
import subprocess
import sys

PROJECT = "video-analyst"
PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")

CONFIG_DIR = os.path.expanduser("~/.config/video-analyst")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

GROQ_KEY_GUIDANCE = (
    "需要 GROQ_API_KEY（Groq Whisper 云端 ASR）\n"
    "申请地址：https://console.groq.com/keys\n"
    "把 key 发给当前 Agent，Agent 会帮你写入 ~/.config/video-analyst/config.json"
)
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

# --------------------------------------------------------------------------- io


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def dump_json(path, data, indent=2):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")


def relpath_from(base, target):
    """Deterministic relative path (used in generated manifests, not absolute env leaks)."""
    try:
        return os.path.relpath(target, base)
    except ValueError:
        return target


def iso_now():
    import datetime

    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_num(x, digits=3):
    """Deterministic float formatting for reports."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return str(x)


# ------------------------------------------------------------------ subprocess


def run(argv, timeout=180, check=True, binary=False, **kw):
    """Run a command. Returns CompletedProcess. check=False -> failed probe returns rc!=0.

    binary=True captures stdout/stderr as bytes (e.g. ffmpeg rawvideo digests).
    """
    cp = subprocess.run(
        [str(a) for a in argv], capture_output=True, timeout=timeout, **kw
    )
    if not binary:
        cp.stdout = cp.stdout.decode("utf-8", errors="replace")
        cp.stderr = cp.stderr.decode("utf-8", errors="replace")
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"command failed ({cp.returncode}): {' '.join(str(a) for a in argv)}\n"
            f"stdout: {(cp.stdout or b'')[:2000] if isinstance(cp.stdout, bytes) else cp.stdout[:2000]}\n"
            f"stderr: {(cp.stderr or b'')[:2000] if isinstance(cp.stderr, bytes) else cp.stderr[:2000]}"
        )
    return cp


def which(names):
    """First executable found among names -> (name, path) or (None, None)."""
    if isinstance(names, str):
        names = [names]
    for n in names:
        p = shutil.which(n)
        if p:
            return n, p
    return None, None


def tool_version(name):
    try:
        cp = run([name, "-version"], timeout=30, check=False)
        first = (cp.stdout or cp.stderr).splitlines()[0] if (cp.stdout or cp.stderr) else ""
        return first.strip()
    except Exception:
        return "unavailable"


def ffprobe_json(path):
    cp = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    data = json.loads(cp.stdout)
    return data


def parse_fraction(s):
    try:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


# --------------------------------------------------------------- time parsing


_TIME_SEC = re.compile(r"^(\d+(?:\.\d+)?)$")
_TIME_MMSS = re.compile(r"^(\d+):([0-5]\d(?:\.\d+)?)$")
_TIME_HMS = re.compile(r"^(\d+):([0-5]\d):([0-5]\d(?:\.\d+)?)$")


def parse_time(s):
    """Parse absolute seconds, MM:SS or HH:MM:SS (HH/MM parts may exceed 59)."""
    s = s.strip()
    m = _TIME_SEC.match(s)
    if m:
        return float(m.group(1))
    m = _TIME_MMSS.match(s)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    m = _TIME_HMS.match(s)
    if m:
        return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    raise ValueError(f"invalid time format: {s!r} (use seconds, MM:SS or HH:MM:SS)")


def format_hms(t):
    """Deterministic HH:MM:SS.mmm for report rows."""
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ------------------------------------------------------------------ provenance


def provenance(asr_engine=None, extra=None):
    """Hard rule 13 provenance block. Never contains API key values."""
    p = {
        "vision_execution_mode": "in_session_agent",
        "external_api_used": False,
        "api_key_used": False,
        "token_status": "unavailable",
        "estimated_cost": None,
        "asr_engine": asr_engine,
    }
    if extra:
        p.update(extra)
    return p


def deps_summary():
    """Environment/tooling facts for provenance. No keys, no secrets."""
    ffmpeg_v = tool_version("ffmpeg") or "unavailable"
    ffprobe_v = tool_version("ffprobe") or "unavailable"
    yt = which("yt-dlp")
    pil = "not_used"
    try:
        import PIL  # noqa: F401  (optional enhancement only)

        pil = "available"
    except Exception:
        pil = "not_installed"
    return {
        "python": sys.version.split()[0],
        "ffmpeg": ffmpeg_v,
        "ffprobe": ffprobe_v,
        "yt_dlp": yt[1] if yt[0] else "missing",
        "pil": pil,
    }