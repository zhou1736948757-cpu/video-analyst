#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""doctor — environment self-check.

Covers ffmpeg / ffprobe / yt-dlp / PIL(optional) / local-whisper ASR probe /
Groq key presence / config writability. Groq key is reported as present|MISSING
ONLY — the value is never printed. When MISSING, the fixed onboarding guidance
(SKILL.md「密钥与外部服务」section, identical wording) is printed.
"""

import os
import shutil
import sys

from . import config as config_mod
from .asr import LOCAL_WHISPER_BINARIES, probe_asr
from .common import GROQ_KEY_GUIDANCE, CONFIG_PATH, which


def _check_exec(names, label):
    n, p = which(names)
    if n:
        return "OK", f"{label}: {p}"
    return "MISSING", f"{label}: none of {names} on PATH"


def collect():
    ffmpeg, ffmpeg_p = _check_exec("ffmpeg", "ffmpeg")
    ffprobe, ffprobe_p = _check_exec("ffprobe", "ffprobe")
    ytdlp, ytdlp_p = _check_exec("yt-dlp", "yt-dlp")
    whisper_name, whisper_path = which(LOCAL_WHISPER_BINARIES)
    pil = "available" if _pil_imports() else "MISSING"
    asr_probe = probe_asr()
    key_status = "present" if config_mod.key_present() else "MISSING"
    cfg_writable = config_mod.config_writable()
    cfg_perms = config_mod.config_permission_ok()
    return {
        "python": f"{sys.version.split()[0]} ({sys.executable})",
        "ffmpeg": ffmpeg, "ffmpeg_path": ffmpeg_p,
        "ffprobe": ffprobe, "ffprobe_path": ffprobe_p,
        "yt_dlp": ytdlp, "yt_dlp_path": ytdlp_p,
        "PIL": pil,
        "local_whisper": whisper_name or "MISSING",
        "local_whisper_path": whisper_path or "",
        "groq_api_key": key_status,
        "groq_guidance": GROQ_KEY_GUIDANCE if key_status == "MISSING" else None,
        "config_path": CONFIG_PATH,
        "config_writable": "yes" if cfg_writable else "NO",
        "config_permission_600": "yes" if cfg_perms else "NO (see notes)",
        "asr": asr_probe["mode"],
        "asr_details": asr_probe["details"],
        "route_notes": _route_notes(asr_probe),
    }


def _pil_imports():
    try:
        import PIL  # noqa: F401

        return True
    except Exception:
        return False


def _route_notes(probe):
    notes = []
    if probe["mode"] == "unavailable":
        notes.append("ASR unavailable: no local whisper and no GROQ_API_KEY — "
                     "prepare() will record asr=unavailable and skip transcription.")
    elif probe["engine"] == "local_whisper":
        notes.append("ASR will use local whisper.")
    else:
        notes.append("ASR will use Groq Whisper (cloud). Audio will be uploaded only on demand "
                     "during prepare().")
    return notes


def render_doctor(d):
    lines = ["=== video-analyst doctor ==="]
    lines.append(f"python          : OK ({d['python']})")
    lines.append(f"ffmpeg          : {d['ffmpeg']} {'' if d['ffmpeg'] == 'MISSING' else d['ffmpeg_path']}")
    lines.append(f"ffprobe         : {d['ffprobe']} {'' if d['ffprobe'] == 'MISSING' else d['ffprobe_path']}")
    lines.append(f"yt-dlp          : {d['yt_dlp']} {'' if d['yt_dlp'] == 'MISSING' else d['yt_dlp_path']}")
    lines.append(f"PIL (optional)  : {d['PIL']} (optional enhancement; ffmpeg fallback active)")
    lines.append(f"local whisper   : {d['local_whisper']} "
                 f"{d['local_whisper_path']}")
    lines.append(f"groq_api_key    : {d['groq_api_key']}")
    if d["groq_guidance"]:
        lines.append("")
        lines.append(d["groq_guidance"])
        lines.append("")
    lines.append(f"config path     : {d['config_path']} "
                 f"(writable={d['config_writable']}, perm600={d['config_permission_600']})")
    lines.append(f"asr             : {d['asr']}")
    for note in d["route_notes"]:
        lines.append(f"note            : {note}")
    return "\n".join(lines) + "\n"