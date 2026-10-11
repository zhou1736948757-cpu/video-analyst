#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""status OUTPUT — read state.json and report the 12-state position + progress."""

import os

from . import state as state_mod
from .common import load_json

PROGRESS_KEYS = [
    ("analysis/coarse/inputs/manifest.json", "coarse inputs"),
    ("analysis/coarse/batches", "coarse agent analysis"),
    ("analysis/coarse/refinement_plan.json", "refinement plan"),
    ("analysis/refinement/visual_inputs/fine_batches.json", "refinement visuals"),
    ("analysis/fine/batches", "fine agent analysis"),
    ("analysis/fine/second_refinement_plan.json", "second refinement plan"),
    ("analysis/terminal_fine", "terminal inputs"),
    ("analysis/final/final_report.json", "final report"),
]


def build_status(output_dir):
    st = load_json(state_mod.state_path(output_dir)) or {"state": "not_started"}
    progress = []
    for rel, label in PROGRESS_KEYS:
        p = os.path.join(output_dir, rel)
        if os.path.isdir(p):
            items = [f for f in os.listdir(p) if not f.startswith(".")]
            progress.append({"label": label, "status": "present", "detail": f"{len(items)} item(s)"})
        elif os.path.isfile(p):
            progress.append({"label": label, "status": "present", "detail": "file"})
        else:
            progress.append({"label": label, "status": "absent", "detail": ""})
    return {
        "output": output_dir,
        "state": st.get("state", "not_started"),
        "terminal_refinement_count": st.get("terminal_refinement_count", 0),
        "source": st.get("source"),
        "history_count": len(st.get("history", [])),
        "progress": progress,
    }


def render_status(s):
    lines = ["=== video-analyst status ===",
             f"OUTPUT : {s['output']}",
             f"state  : {s['state']}",
             f"terminal_refinement_count : {s['terminal_refinement_count']}",
             f"source  : {s.get('source') or '(none)'}",
             "progress:"]
    for pr in s["progress"]:
        lines.append(f"  {pr['status']:<7} {pr['label']} {pr['detail']}")
    return "\n".join(lines) + "\n"