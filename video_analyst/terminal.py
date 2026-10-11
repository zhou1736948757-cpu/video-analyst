#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""terminal-prepare — optional terminal refinement, at most once (hard rule 11).

Reads analysis/fine/second_refinement_plan.json (terminal_refinement_status must be
"required"). A second terminal-prepare call is rejected: third round forbidden.
Terminal inputs land under analysis/terminal_fine/.
"""

import os

from . import state as state_mod
from .state import RejectionError
from .common import ensure_dir, load_json, dump_json
from .refine import ALLOWED_FPS, build_fine_sheet, extract_interval_frames

TERMINAL_MSG = "terminal refinement already performed once — third round forbidden (hard rule 11): at most one terminal preparation."


def run_terminal(video_path, output_dir):
    try:
        return _terminal_inner(video_path, output_dir)
    except RejectionError:
        raise
    except Exception as exc:
        state_mod._fail(output_dir, state_mod.load(output_dir), "terminal_fail",
                        "terminal-prepare failed", exc)
        raise


def _terminal_inner(video_path, output_dir):
    st = load_json(state_mod.state_path(output_dir)) or state_mod.ensure_state(output_dir)
    count = int(st.get("terminal_refinement_count", 0) or 0)
    if count >= 1:
        raise RejectionError(TERMINAL_MSG)
    sec_path = os.path.join(output_dir, "analysis/fine/second_refinement_plan.json")
    sec = load_json(sec_path)
    if sec is None or sec.get("terminal_refinement_status") != "required":
        raise RejectionError(
            "terminal refinement not required: analysis/fine/second_refinement_plan.json "
            "missing or terminal_refinement_status != 'required'"
        )
    state_mod.transition(output_dir, "terminal_start", note="terminal refinement start")
    video_path = os.path.abspath(video_path)
    detail_dir = os.path.join(output_dir, "analysis/terminal_fine/detail_frames")
    sheet_dir = os.path.join(output_dir, "analysis/terminal_fine/contact_sheets")
    ensure_dir(detail_dir)
    ensure_dir(sheet_dir)

    intervals = sec.get("intervals") or []
    summary = []
    for idx, iv in enumerate(intervals, start=1):
        interval_id = iv.get("interval_id") or f"terminal-{idx}"
        fps = iv.get("fps", iv.get("requested_fps")) or 10
        if fps not in ALLOWED_FPS:
            raise RuntimeError(f"terminal interval {interval_id}: fps must be 5|10|15")
        rows = extract_interval_frames(video_path, iv["start"], iv["end"], fps,
                                       os.path.join(detail_dir, interval_id), interval_id,
                                       rel_base=output_dir)
        sheet_rel = os.path.join("analysis/terminal_fine/contact_sheets",
                                 f"{interval_id}-sheet.jpg")
        build_fine_sheet(interval_id, rows, os.path.join(output_dir, sheet_rel))
        summary.append({"interval_id": interval_id, "batch_id": f"terminal-batch-{idx:02d}",
                        "fps": fps,
                        "detail_frames": [{"frame_id": r["frame_id"], "timestamp": r["timestamp"],
                                           "path": r["path"]} for r in rows],
                        "contact_sheet": sheet_rel})

    dump_json(os.path.join(output_dir, "analysis/terminal_fine/terminal_inputs.json"),
              {"format": "video-analyst/terminal-inputs/1", "intervals": summary})
    st2 = state_mod.load(output_dir)
    st2["terminal_refinement_count"] = 1
    state_mod.save(output_dir, st2)
    return {
        "state": state_mod.current(output_dir),
        "terminal_refinement_count": 1,
        "intervals": len(summary),
        "detail_frames": sum(len(s["detail_frames"]) for s in summary),
    }