#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clean OUTPUT --all-generated — remove generated artifacts.

Deletes everything the CLI generated (analysis/, subtitles/, .audio_work,
.frames_tmp), PRESERVING state.json and the user's source/import inputs.
"""

import os
import shutil

REMOVABLE = ["analysis", "subtitles", ".audio_work", ".frames_tmp"]


def run_clean(output_dir):
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        raise RuntimeError(f"{output_dir}: not a directory")
    removed = []
    for rel in REMOVABLE:
        p = os.path.join(output_dir, rel)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(rel + "/")
        elif os.path.isfile(p):
            os.remove(p)
            removed.append(rel)
    kept = [f for f in sorted(os.listdir(output_dir))
            if not f.startswith(".") and f not in ("state.json",)]
    return {
        "removed": removed,
        "kept": kept,
        "state_preserved": os.path.isfile(os.path.join(output_dir, "state.json")),
        "note": "state.json and user inputs preserved; run prepare again to regenerate inputs",
    }