#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR probe chain (SKILL.md ASR section + VAF-002 key protocol).

Probe chain:
  1. local whisper  -> `whisper` / `whisper.cpp` / `faster-whisper` executable on PATH
  2. Groq Whisper   -> GROQ_API_KEY present (env first, then ~/.config/video-analyst/config.json);
                       stdlib urllib multipart POST to api.groq.com, model=whisper-large-v3
  3. none           -> asr=unavailable; prepare skips ASR and records it honestly.

Offline discipline (task VAF-002): this module NEVER calls any cloud endpoint when no
key exists. The Groq branch is fully implemented and compilable, but is only reachable
when a key is present. The key value is never logged, printed, or persisted.
"""

import os
import uuid

from . import config as config_mod
from .common import GROQ_KEY_GUIDANCE, ensure_dir, iso_now, run, which

LOCAL_WHISPER_BINARIES = ("whisper", "whisper.cpp", "whisper-cpp", "faster-whisper")


def detect_local_whisper():
    """Return (name, path) of first local whisper executable, else (None, None)."""
    for name in LOCAL_WHISPER_BINARIES:
        found, path = which(name)
        if found:
            return name, path
    return None, None


def probe_asr():
    """Deterministic ASR capability probe. Never transmits audio without a key.

    Returns dict:
      {"engine": "local_whisper"|"groq_whisper"|null,
       "mode": "available"|"unavailable",
       "details": {...}, "guidance": str|None}
    """
    details = {}
    name, path = detect_local_whisper()
    if name:
        details["local_whisper"] = f"{name} ({path})"
        return {
            "engine": "local_whisper",
            "mode": "available",
            "details": details,
            "guidance": None,
        }
    details["local_whisper"] = "missing"
    if config_mod.key_present():
        details["groq"] = "key=present"
        return {
            "engine": "groq_whisper",
            "mode": "available",
            "details": details,
            "guidance": None,
        }
    details["groq"] = "key=MISSING"
    return {
        "engine": None,
        "mode": "unavailable",
        "details": details,
        "guidance": GROQ_KEY_GUIDANCE,
    }


# ------------------------------------------------------- local whisper runner


def run_local_whisper(audio_wav, subtitle_dir, engine_name):
    """whisper CLI with default multilingual ASR (no forced --language)."""
    ensure_dir(subtitle_dir)
    doc_id = uuid.uuid4().hex[:12]
    prefix = os.path.join(subtitle_dir, doc_id)
    cmd = [
        engine_name,
        audio_wav,
        "--model", "base",
        "--output_format", "json,srt",
        "--output_dir", subtitle_dir,
        "--output_name", doc_id,
    ]
    cp = run(cmd, timeout=1800)
    if cp.returncode != 0:
        raise RuntimeError(f"local whisper failed: {cp.stderr[-800:]}")
    json_path = prefix + ".json"
    srt_path = prefix + ".srt"
    json_url = json_path if os.path.exists(json_path) else None
    srt = srt_path if os.path.exists(srt_path) else None
    text, segments = extract_segments(json_url)
    return {"id": doc_id, "engine": "local_whisper", "json": json_url, "srt": srt,
            "text": text, "segments": segments}


# ------------------------------------------------------------ Groq Whisper


def _multipart_body(audio_path, model, language):
    """Build multipart/form-data body (urllib only). Returns (body, boundary)."""
    boundary = "----videoanalyst" + uuid.uuid4().hex
    CRLF = b"\r\n"
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()
    fname = os.path.basename(audio_path)
    body = bytearray()
    # file part
    body += b"--" + boundary.encode() + CRLF
    body += (f'Content-Disposition: form-data; name="file"; filename="{fname}"'
             f"\"{CRLF}Content-Type: application/octet-stream{CRLF}{CRLF}").encode()
    body += audio_bytes + CRLF
    # model part
    body += b"--" + boundary.encode() + CRLF
    body += (f'Content-Disposition: form-data; name="model"{CRLF}{CRLF}'
             f"{model}{CRLF}").encode()
    if language:
        body += b"--" + boundary.encode() + CRLF
        body += (f'Content-Disposition: form-data; name="language"{CRLF}{CRLF}'
                 f"{language}{CRLF}").encode()
    body += b"--" + boundary.encode() + b"--" + CRLF
    return bytes(body), boundary


def groq_transcribe(audio_path, api_key, model="whisper-large-v3", language=None):
    """Real Groq Whisper transcription (only called when a key is present).

    Returns dict with segments; writes .json + .srt under subtitles/<uuid>.
    The api_key is passed to urllib headers only; never stored or logged.
    """
    import json
    import urllib.request

    from .common import GROQ_ENDPOINT, ensure_dir

    subtitle_dir = os.path.join(os.path.dirname(audio_path), "..", "subtitles")
    ensure_dir(subtitle_dir)
    doc_id = uuid.uuid4().hex[:12]

    body, boundary = _multipart_body(audio_path, model, language)
    req = urllib.request.Request(
        GROQ_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "video-analyst/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload.get("text", "")
    segs = payload.get("segments") or []
    segs_norm = [
        {
            "id": s.get("id"),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
        }
        for s in segs
    ]
    json_path = os.path.join(subtitle_dir, f"{doc_id}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"id": doc_id, "engine": "groq_whisper", "text": text,
                   "segments": segs_norm}, fh, ensure_ascii=False, indent=2)
    srt_path = os.path.join(subtitle_dir, f"{doc_id}.srt")
    write_srt(segs_norm, srt_path)
    return {"id": doc_id, "engine": "groq_whisper", "json": json_path, "srt": srt_path,
            "text": text, "segments": segs_norm}


def write_srt(segments, srt_path):
    srt_lines = []
    for i, s in enumerate(segments, start=1):
        srt_lines.append(str(i))
        srt_lines.append(_fmt_srt_time(s["start"]) + " --> " + _fmt_srt_time(s["end"]))
        srt_lines.append(s.get("text", ""))
        srt_lines.append("")
    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(srt_lines))


def _fmt_srt_time(t):
    ms = int(round((t - int(t)) * 1000))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_segments(json_path):
    """Load segments from a whisper/groq json doc. Returns (text, segments)."""
    from .common import load_json

    if not json_path or not os.path.exists(json_path):
        return None, []
    data = load_json(json_path, default={}) or {}
    return data.get("text") or None, data.get("segments") or []


# ------------------------------------------------------- active subtitle doc


def write_active_subtitles(subtitle_dir, documents, asr_info, note=None):
    """Downstream contract (hard rule 4): finalize reads ONLY subtitles/subtitles.json."""
    ensure_dir(subtitle_dir)
    active = None
    if documents:
        active = documents[0]["id"]
    doc = {
        "active": active,
        "documents": documents,
        "asr": asr_info,
        "note": note or "Downstream reads only this file + review.json.",
        "updated_at": iso_now(),
    }
    path = os.path.join(subtitle_dir, "subtitles.json")
    from .common import dump_json

    dump_json(path, doc)
    return path


def run_asr_pipeline(video_path, output_dir):
    """Prepare-stage ASR entry. Returns (asr_result, subtitle_dir).

    asr_result: {"mode", "engine", "probe", "reason", "guidance"}
    Never uploads audio when no usable engine/key is present.
    """
    subtitle_dir = os.path.join(output_dir, "subtitles")
    probe = probe_asr()
    if probe["mode"] == "unavailable":
        return (
            {
                "mode": "unavailable",
                "engine": None,
                "probe": probe["details"],
                "reason": "no local whisper executable and no GROQ_API_KEY",
                "guidance": probe["guidance"],
            },
            subtitle_dir,
        )
    # audio extract once: 16k mono wav into a temp path
    audio_dir = os.path.join(output_dir, ".audio_work")
    ensure_dir(audio_dir)
    wav = os.path.join(audio_dir, "audio_16k_mono.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-vn", "-ac", "1",
         "-ar", "16000", wav])
    docs = []
    if probe["engine"] == "local_whisper":
        res = run_local_whisper(wav, subtitle_dir, probe["details"]["local_whisper"].split(" ")[0])
        docs.append(res)
    else:
        api_key = config_mod.read_key()
        res = groq_transcribe(wav, api_key)
        docs.append(res)
    active_path = write_active_subtitles(subtitle_dir, docs, probe["details"])
    return (
        {
            "mode": "available",
            "engine": probe["engine"],
            "probe": probe["details"],
            "active_doc": os.path.basename(active_path),
            "reason": None,
            "guidance": None,
        },
        subtitle_dir,
    )