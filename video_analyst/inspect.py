#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect-prepare — deterministic absolute-time inspection inputs (SKILL.md).

video-analyst inspect-prepare VIDEO -o OUTPUT --at T [--at T ...]
video-analyst inspect-prepare VIDEO -o OUTPUT --start S --end E --fps 5|10|15

* --at / --start / --end accept absolute seconds, MM:SS or HH:MM:SS.
* --at frames are pinned: never removed by near-duplicate filtering.
* Artifacts live under analysis/inspection/; state is NOT changed; finalize does
  NOT consume them.
"""

import os

from .common import dump_json, ensure_dir, fmt_num, parse_time, relpath_from, run
from .prepare import extract_frame
from .refine import ALLOWED_FPS


def _resolve_selection(at_times, start, end, fps):
    """-> (list of timestamps, meta)"""
    ts = []
    for t in at_times:
        ts.append((parse_time(t), "pinned"))
    if start is not None and end is not None:
        s, e = parse_time(start), parse_time(end)
        if e < s:
            raise RuntimeError("--end must be >= --start")
        if fps not in ALLOWED_FPS:
            raise RuntimeError(f"--fps must be one of {ALLOWED_FPS}")
        n = int(round((e - s) * fps))
        for k in range(n + 1):
            ts.append((s + k / fps, "sequence"))
    elif start is not None or end is not None:
        raise RuntimeError("--start and --end must be given together")
    return ts


def run_inspect(video_path, output_dir, at_times=None, start=None, end=None, fps=5):
    at_times = at_times or []
    ts_list = _resolve_selection(at_times, start, end, fps)
    if not ts_list:
        raise RuntimeError("no inspection timestamps: use --at T ... or --start S --end E --fps F")
    ts_list.sort(key=lambda x: x[0])
    inspect_dir = os.path.join(output_dir, "analysis/inspection")
    frames_dir = os.path.join(inspect_dir, "frames")
    ensure_dir(frames_dir)
    rows = []
    for i, (t, kind) in enumerate(ts_list, start=1):
        fname = f"inspect-{i:04d}.png"
        out = os.path.join(frames_dir, fname)
        extract_frame(os.path.abspath(video_path), t, out, width=1280)
        rows.append({
            "frame_id": f"inspect-{i:04d}",
            "path": relpath_from(output_dir, out),
            "timestamp": round(t, 3),
            "kind": kind,
            "pinned": kind == "pinned",
        })
    manifest = {
        "format": "video-analyst/inspection-manifest/1",
        "frames": rows,
        "note": "Inspection inputs do not change workflow state and are not consumed by finalize.",
    }
    dump_json(os.path.join(inspect_dir, "inspection_manifest.json"), manifest)
    return {"state": _current_or_none(output_dir), "frames": rows,
            "paths": sorted(r["path"] for r in rows)}


def _current_or_none(output_dir):
    from . import state as state_mod

    return state_mod.current(output_dir)