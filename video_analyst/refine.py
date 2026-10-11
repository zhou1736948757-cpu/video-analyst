#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 refine-prepare — adaptive 5/10/15 FPS detail extraction per RefinementPlan.

Reads analysis/coarse/refinement_plan.json, validates interval fps against the
allowed 5|10|15 buckets (hard rule 10), extracts detail frames + fine contact
sheets under analysis/refinement/visual_inputs/, and writes fine_batches.json.
"""

import os

from . import state as state_mod
from .state import RejectionError
from .common import (dump_json, ensure_dir, fmt_num, load_json, relpath_from, run)

DETAIL_WIDTH = 1280
DETAIL_DIR = "analysis/refinement/visual_inputs/detail_frames"
SHEET_DIR = "analysis/refinement/visual_inputs/contact_sheets"
ALLOWED_FPS = (5, 10, 15)


def extract_interval_frames(video, start, end, fps, out_png_dir, prefix, rel_base=None):
    """Extract frames t = start + k/fps (k=0..n) while t <= end. Returns [{frame_id,timestamp,path}]."""
    ensure_dir(out_png_dir)
    if rel_base is None:
        rel_base = os.path.dirname(out_png_dir)
    n = max(0, int(round((end - start) * fps)))
    rows = []
    k = 0
    while True:
        t = start + k / fps
        if t > end + 1e-9 or t < 0:
            break
        fname = f"{prefix}-{k + 1:04d}.png"
        out = os.path.join(out_png_dir, fname)
        run(["ffmpeg", "-y", "-v", "error", "-ss", fmt_num(t, 3), "-i", video,
             "-frames:v", "1", "-vf", f"scale='min({DETAIL_WIDTH},iw)':-2",
             "-compression_level", "6", out])
        rows.append({"frame_id": f"{prefix}-{k + 1:04d}",
                     "timestamp": round(t, 3),
                     "abs_path": out,
                     "path": relpath_from(rel_base, out)})
        k += 1
        if k > 100000:
            break
    return rows


def build_fine_sheet(interval_id, rows, out_sheet):
    """Tile detail rows (sequential names) into one contact sheet."""
    if not rows:
        return None
    d = os.path.dirname(rows[0]["abs_path"])
    pattern = os.path.join(d, f"{interval_id}-%04d.png")
    cols = 4
    cnt = len(rows)
    g = (cnt + cols - 1) // cols
    run(["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-start_number", "1",
         "-i", pattern, "-frames:v", str(cnt),
         "-vf", f"scale=320:-2,tile={cols}x{g}:padding=4:margin=4",
         "-compression_level", "6", out_sheet])
    return out_sheet


def _manifest_frames(output_dir):
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    return man.get("coarse", {}).get("frames", []), man.get("coarse", {}).get("batches", [])


def validate_refinement_plan(output_dir, plan):
    """Semantic checks: fps bucket, interval bounds, evidence existence (hard rules 9/10)."""
    errs = []
    duration = _media_duration(output_dir)
    frame_ids = {f["frame_id"] for f in _manifest_frames(output_dir)[0]}
    batch_ids = {b["batch_id"] for b in _manifest_frames(output_dir)[1]}
    for iv in plan.get("intervals", []):
        if iv.get("fps", iv.get("requested_fps")) not in ALLOWED_FPS:
            errs.append(f"interval {iv.get('interval_id')}: fps must be 5|10|15")
        if iv.get("start", 0) < 0 or iv.get("end", 0) < iv.get("start", 0):
            errs.append(f"interval {iv.get('interval_id')}: invalid start/end")
        if duration and iv.get("end", 0) > duration + 1e-6:
            errs.append(f"interval {iv.get('interval_id')}: end {iv.get('end')} > duration {duration}")
        missing = [e for e in iv.get("evidence_frame_ids", []) if e not in frame_ids]
        if missing:
            errs.append(f"interval {iv.get('interval_id')}: evidence_frame_ids not in manifest: {missing}")
        missing_b = [b for b in iv.get("source_batch_ids", []) if b not in batch_ids]
        if missing_b:
            errs.append(f"interval {iv.get('interval_id')}: source_batch_ids not in manifest: {missing_b}")
    return errs


def _media_duration(output_dir):
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    src = man.get("source", {})
    return src.get("duration_s")


def run_refine(video_path, output_dir):
    """Stage 3. Requires coarse_analysis_complete. Returns summary with interval counts."""
    try:
        return _refine_inner(video_path, output_dir)
    except RejectionError:
        raise
    except Exception as exc:
        state_mod._fail(output_dir, state_mod.load(output_dir), "refine_fail",
                        "refine-prepare failed", exc)
        raise


def _refine_inner(video_path, output_dir):
    load_json(state_mod.state_path(output_dir)) or state_mod.ensure_state(output_dir)
    plan_path = os.path.join(output_dir, "analysis/coarse/refinement_plan.json")
    plan = load_json(plan_path)
    if plan is None:
        raise RejectionError("refinement_plan.json missing — agent must write coarse analysis first")
    sem_errs = validate_refinement_plan(output_dir, plan)
    if sem_errs:
        raise RejectionError("refinement plan invalid:\n  - " + "\n  - ".join(sem_errs))

    state_mod.transition(output_dir, "refine_start", note="reading refinement plan")
    video_path = os.path.abspath(video_path)
    detail_dir = os.path.join(output_dir, DETAIL_DIR)
    sheet_dir = os.path.join(output_dir, SHEET_DIR)
    ensure_dir(detail_dir)
    ensure_dir(sheet_dir)

    out_intervals = []
    for idx, iv in enumerate(plan.get("intervals", []), start=1):
        interval_id = iv.get("interval_id") or f"interval-{idx}"
        fps = iv.get("fps", iv.get("requested_fps"))
        rows = extract_interval_frames(video_path, iv["start"], iv["end"], fps,
                                       os.path.join(detail_dir, interval_id), interval_id,
                                       rel_base=output_dir)
        sheet_rel = os.path.join(SHEET_DIR, f"{interval_id}-sheet.jpg")
        sheet_path = os.path.join(output_dir, sheet_rel)
        build_fine_sheet(interval_id, rows, sheet_path)
        out_intervals.append({
            "interval_id": interval_id,
            "batch_id": f"fine-batch-{idx:02d}",
            "start": iv["start"],
            "end": iv["end"],
            "fps": fps,
            "priority": iv.get("priority", "medium"),
            "reason": iv.get("reason", iv.get("reasons", "")),
            "source_batch_ids": iv.get("source_batch_ids", []),
            "evidence_frame_ids": iv.get("evidence_frame_ids", []),
            "refinement_questions": iv.get("refinement_questions", iv.get("questions", [])),
            "detail_frames": rows,
            "contact_sheets": [sheet_rel],
        })

    fine_batches = {
        "format": "video-analyst/fine-batches/1",
        "plan_id": plan.get("plan_id"),
        "probe_timestamp": _manifest_probe(output_dir),
        "intervals": out_intervals,
    }
    dump_json(os.path.join(output_dir, "analysis/refinement/visual_inputs/fine_batches.json"),
              fine_batches)
    state_mod.transition(output_dir, "refine_done",
                         note=f"detail frames written ({sum(len(i['detail_frames']) for i in out_intervals)})")
    return {
        "state": state_mod.current(output_dir),
        "intervals": len(out_intervals),
        "detail_frames": sum(len(i["detail_frames"]) for i in out_intervals),
        "contact_sheets": len(out_intervals),
    }


def _manifest_probe(output_dir):
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    return man.get("preparation", {}).get("probe_timestamp")