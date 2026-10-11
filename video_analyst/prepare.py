#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 prepare — precheck, ASR probe, coarse inputs generation.

  1. ffprobe precheck (video stream required; audio optional -> ASR gate)
  2. ASR probe chain (local whisper -> Groq Whisper -> unavailable)
  3. 1 FPS coarse frames + scene-change frames (ffmpeg select/scene)
  4. near-duplicate removal via ffmpeg scale=160:90 gray rawvideo pixel compare
     (first/last/scene retained; PIL optional enhancement never required)
  5. contact sheets per batch via ffmpeg tile
  6. writes analysis/coarse/inputs/{manifest.json,batches,frames,contact_sheets}
     + subtitles/subtitles.json (downstream contract)
"""

import glob as _glob
import os
import shutil
import uuid

from . import asr as asr_mod
from . import state as state_mod
from .common import (deps_summary, dump_json, ensure_dir, ffprobe_json, fmt_num,
                     iso_now, parse_fraction, provenance, relpath_from)

BATCH_SIZE = 12          # frames per coarse batch
COARSE_WIDTH = 960       # coarse frame target width
DIGEST_W, DIGEST_H = 160, 90
DUP_THRESHOLD = 1.5      # mean abs gray difference below this -> near duplicate
SCENE_THRESHOLD = 0.3    # ffmpeg select scene score threshold


def _png_vf(width):
    return f"scale='min({width},iw)':-2"


def extract_frame(video, ts, out_png, width=COARSE_WIDTH):
    ensure_dir(os.path.dirname(out_png))
    from .common import run

    run(["ffmpeg", "-y", "-v", "error", "-ss", fmt_num(ts, 3), "-i", video,
         "-frames:v", "1", "-vf", _png_vf(width), "-compression_level", "6", out_png])


def gray_digest(video, ts):
    """Raw 160x90 gray bytes for a frame (ffmpeg only, no PIL)."""
    from .common import run

    cp = run(
        ["ffmpeg", "-v", "error", "-ss", fmt_num(ts, 3), "-i", video, "-frames:v", "1",
         "-vf", f"scale={DIGEST_W}:{DIGEST_H},format=gray", "-f", "rawvideo", "-"],
        check=False, binary=True,
    )
    if cp.returncode != 0 or len(cp.stdout) != DIGEST_W * DIGEST_H:
        return None
    return cp.stdout


def mean_abs_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    total = 0
    for i in range(0, n):
        total += abs(a[i] - b[i])
    return total / n


def detect_scene_changes(video, threshold=SCENE_THRESHOLD):
    """ffmpeg select=gt(scene,thr),metadata=print -> list of scene timestamps."""
    from .common import run

    log = os.path.join(os.path.dirname(video) or ".", f".scene_{uuid.uuid4().hex[:8]}.log")
    try:
        run(["ffmpeg", "-v", "error", "-i", video,
             "-vf", f"select='gt(scene,{threshold})',metadata=print:file={log}",
             "-an", "-f", "null", "-"], check=False)
        ts = []
        if os.path.exists(log):
            for line in open(log, encoding="utf-8", errors="replace"):
                if "pts_time" in line:
                    try:
                        ts.append(float(line.split("=", 1)[1].strip()))
                    except ValueError:
                        pass
        return sorted(set(round(t, 3) for t in ts))
    finally:
        if os.path.exists(log):
            os.remove(log)


def parse_probe(path):
    """ffprobe summary -> dict with duration/streams/codec etc."""
    data = ffprobe_json(path)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if vstream is None:
        raise RuntimeError(f"{path}: no video stream (ffprobe) — cannot analyze")
    dur = float(fmt.get("duration") or 0)
    try:
        fps = parse_fraction(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/1")
    except Exception:
        fps = 0.0
    return {
        "duration_s": round(dur, 3),
        "size_bytes": int(fmt.get("size") or 0),
        "format_long": fmt.get("format_long_name") or "",
        "video": {
            "codec": vstream.get("codec_name"),
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "pix_fmt": vstream.get("pix_fmt"),
            "fps": round(fps, 3) if fps else None,
        },
        "audio": {
            "streams": sum(1 for s in streams if s.get("codec_type") == "audio"),
            "codec": (astream or {}).get("codec_name"),
        },
    }


def build_contact_sheet(frames_dir, first_seq, count, out_png, cols=4):
    """Tile `count` consecutive coarse-%06d.png frames from first_seq into one sheet."""
    from .common import run

    if count <= 0:
        return
    rows = (count + cols - 1) // cols
    pattern = os.path.join(frames_dir, "coarse-%06d.png")
    run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-start_number", str(first_seq),
         "-i", pattern, "-frames:v", str(count),
         "-vf", f"scale=320:-2,tile={cols}x{rows}:padding=4:margin=4",
         "-compression_level", "6", out_png]
    )


def group_batches(frames, batch_size=BATCH_SIZE):
    batches = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        batch_id = f"coarse-batch-{i // batch_size + 1:02d}"
        batches.append({
            "batch_id": batch_id,
            "frame_ids": [f["frame_id"] for f in chunk],
            "start": min(f["timestamp"] for f in chunk),
            "end": max(f["timestamp"] for f in chunk),
            "contact_sheet": None,
        })
    return batches


# --------------------------------------------------------------- prepare core


def run_prepare(video_path, output_dir):
    """Full Stage 1. Returns dict summary (also persisted in manifest). Does NOT do vision."""
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise RuntimeError(f"{video_path}: file not found")
    ensure_dir(output_dir)
    st = state_mod.transition(output_dir, "prepare_start", source=video_path,
                              note=f"prepare {video_path}")
    try:
        return _prepare_inner(video_path, output_dir)
    except Exception as exc:
        state_mod._fail(output_dir, state_mod.load(output_dir), "prepare_fail",
                        "prepare pipeline failed", exc)
        raise


def _prepare_inner(video_path, output_dir):
    from . import state as state_mod

    try:
        probe = parse_probe(video_path)
    except Exception as exc:
        raise RuntimeError(f"precheck failed: {exc}")

    # ---- ASR (probe chain; offline->unavailable, never fake)
    asr_result, subtitle_dir = asr_mod.run_asr_pipeline(video_path, output_dir)

    # ---- coarse frame candidates: 1FPS + scene changes
    tmp = os.path.join(output_dir, ".frames_tmp")
    ensure_dir(tmp)
    frames_dir = os.path.join(output_dir, "analysis/coarse/inputs/frames")
    ensure_dir(frames_dir)

    candidates = []  # (ts, type, tmpfile)
    duration = probe["duration_s"]
    n_regular = max(1, int(duration) + (1 if duration > int(duration) else 0))
    for i in range(n_regular):
        ts = float(i)
        if ts >= duration - 1e-9 and i > 0:
            break
        tmp_png = os.path.join(tmp, f"r-{i + 1:04d}.png")
        extract_frame(video_path, min(ts, max(0.0, duration - 0.001)), tmp_png)
        candidates.append([round(ts, 3), "regular", tmp_png])

    for sts in detect_scene_changes(video_path):
        near = [c for c in candidates if abs(c[0] - sts) < 0.5]
        if near:
            near[0][1] = "scene_change"
        else:
            tmp_png = os.path.join(tmp, f"s-{len(candidates):04d}.png")
            extract_frame(video_path, min(sts, max(0.0, duration - 0.001)), tmp_png)
            candidates.append([round(sts, 3), "scene_change", tmp_png])

    candidates.sort(key=lambda c: c[0])
    if len(candidates) > 1:
        candidates[0][1] = "first"
        candidates[-1][1] = "last"

    # ---- near-duplicate removal (keep first/last/scene_change/pinned)
    digests = {}
    kept = []
    removed = []
    prev_digest = None
    for idx, (ts, ftype, tfile) in enumerate(candidates):
        is_last = idx == len(candidates) - 1
        protected = ftype in ("first", "last", "scene_change", "pinned")
        digest = gray_digest(video_path, ts)
        digests[ts] = digest
        dup_of = None
        if prev_digest is not None and digest is not None and prev_digest is not None:
            diff = mean_abs_diff(prev_digest, digest)
            if diff < DUP_THRESHOLD:
                dup_of = prev_ts
        if dup_of is not None and not protected and not is_last:
            removed.append({"ts": ts, "type": ftype, "dup_of_ts": dup_of,
                            "reason": "near_duplicate"})
            continue
        prev_digest = digest
        prev_ts = ts
        kept.append([ts, ftype, tfile])

    # ---- renumber kept -> coarse-%06d.png
    frames = []
    for seq, (ts, ftype, tfile) in enumerate(kept, start=1):
        fname = f"coarse-{seq:06d}.png"
        shutil.move(tfile, os.path.join(frames_dir, fname))
        frames.append({
            "frame_id": f"coarse-{seq:06d}",
            "path": relpath_from(output_dir, os.path.join(frames_dir, fname)),
            "timestamp": ts,
            "type": ftype,
            "batch_id": None,
            "pinned": ftype in ("pinned",),
        })

    # ---- batches + contact sheets
    batch_defs = group_batches(frames)
    sheets_dir = os.path.join(output_dir, "analysis/coarse/inputs/contact_sheets")
    ensure_dir(sheets_dir)
    contact_sheets = []
    for b in batch_defs:
        idxs = [f["frame_id"] for f in frames if f["frame_id"] in b["frame_ids"]]
        first_seq = min(int(fid.split("-")[1]) for fid in idxs)
        count = len(idxs)
        sid = f"coarse-sheet-{b['batch_id'].split('-')[-1]}"
        sheet_rel = f"analysis/coarse/inputs/contact_sheets/{sid}.png"
        build_contact_sheet(frames_dir, first_seq, count, os.path.join(output_dir, sheet_rel))
        b["contact_sheet"] = sheet_rel
        contact_sheets.append({"sheet_id": sid, "path": sheet_rel,
                               "frame_ids": idxs, "batch_id": b["batch_id"]})
        for f in frames:
            if f["frame_id"] in idxs:
                f["batch_id"] = b["batch_id"]

    # ---- batch definition files (agent writes its own analysis elsewhere)
    batches_dir = os.path.join(output_dir, "analysis/coarse/inputs/batches")
    ensure_dir(batches_dir)
    for b in batch_defs:
        payload = {
            "batch_id": b["batch_id"],
            "frames": b["frame_ids"],
            "contact_sheet": b["contact_sheet"],
            "start": b["start"],
            "end": b["end"],
            "instructions": (
                "Open and inspect every frame + the contact sheet. Write your analysis to "
                "analysis/coarse/batches/<batch_id>.json per CoarseBatchAnalysis schema. "
                "Evidence must cite only frame_ids / image paths that exist."
            ),
        }
        dump_json(os.path.join(batches_dir, f"{b['batch_id']}.json"), payload)

    # ---- subtitles contract
    sub_payload = {
        "active": None,
        "documents": [],
        "asr": asr_result,
        "note": "Downstream reads only this file + review.json.",
    }
    dump_json(os.path.join(subtitle_dir, "subtitles.json"), sub_payload)

    # ---- manifest
    near_dup = {
        "method": "ffmpeg scale=160:90 format=gray rawvideo pixel mean-abs-diff",
        "threshold": DUP_THRESHOLD,
        "removed": removed,
        "kept_first_last_scene": [{"ts": c[0], "type": c[1]} for c in kept
                                  if c[1] in ("first", "last", "scene_change")],
    }
    manifest = {
        "format": "video-analyst/coarse-inputs-manifest/1",
        "source": {"path": video_path, **probe},
        "preparation": {
            "frame_rate": 1,
            "scene_threshold": SCENE_THRESHOLD,
            "frame_sampler": "ffmpeg -ss (input-seek) -frames:v 1",
            "near_duplicate": near_dup,
            "probe_timestamp": iso_now(),
        },
        "coarse": {
            "frames": frames,
            "batches": batch_defs,
            "contact_sheets": contact_sheets,
        },
        "asr": asr_result,
        "deps": deps_summary(),
        "provenance": provenance(
            asr_engine=asr_result.get("engine"),
            extra={
                "asr_mode": asr_result.get("mode"),
                "asr_reason": asr_result.get("reason"),
                "generation_batch_ids": ["coarse-1fps", "scene-change",
                                         "contact-sheets", "subtitles"],
            },
        ),
        "generated": {
            "batches": [b["batch_id"] for b in batch_defs],
            "frames": [f["frame_id"] for f in frames],
            "contact_sheets": [c["sheet_id"] for c in contact_sheets],
            "subtitles": "subtitles/subtitles.json",
        },
    }
    dump_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), manifest)

    # cleanup
    shutil.rmtree(tmp, ignore_errors=True)

    state_mod.transition(output_dir, "prepare_done", note="coarse inputs written")
    _remove_stale_analysis(output_dir)
    return {
        "state": state_mod.current(output_dir),
        "duration_s": probe["duration_s"],
        "frames": len(frames),
        "scene_frames": sum(1 for f in frames if f["type"] == "scene_change"),
        "batches": len(batch_defs),
        "contact_sheets": len(contact_sheets),
        "near_duplicates_removed": len(removed),
        "asr_mode": asr_result.get("mode"),
        "asr_engine": asr_result.get("engine"),
        "asr_guidance": asr_result.get("guidance"),
    }


def _remove_stale_analysis(output_dir):
    """After regeneration, drop stage outputs that no longer correspond to fresh inputs."""
    import shutil as _sh

    for rel in ("analysis/coarse/batches", "analysis/coarse/video_structure.json",
                "analysis/coarse/refinement_plan.json", "analysis/fine", "analysis/terminal_fine",
                "analysis/inspection", "analysis/refinement"):
        p = os.path.join(output_dir, rel)
        if os.path.isdir(p):
            _sh.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)