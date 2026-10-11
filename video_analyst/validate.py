#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate-coarse / validate-fine / validate-terminal.

Strict schema validation (self-written stdlib validator, video_analyst/schemas/*.json)
PLUS evidence-truth checks (hard rules 8/9/10/12): schema fields are validated and
evidence must reference real, existing frames/batches/image paths — never schema
empty-value filler. A passing validate advances the state machine.
"""

import os

from . import state as state_mod
from .common import load_json
from .schemas import load_schema, validate as _validate

STAGES = {
    "coarse": {
        "dirs": ["analysis/coarse/batches", "analysis/coarse"],
        "files": {
            "analysis/coarse/video_structure.json": "video-structure.schema.json",
            "analysis/coarse/refinement_plan.json": "refinement-plan.schema.json",
        },
        "batch_pattern": "analysis/coarse/batches/*.json",
        "batch_schema": "coarse-batch-analysis.schema.json",
    },
    "fine": {
        "dirs": ["analysis/fine/batches", "analysis/fine"],
        "files": {
            "analysis/fine/temporal_events.json": "temporal-events.schema.json",
            "analysis/fine/fine_video_analysis.json": "fine-video-analysis.schema.json",
            "analysis/fine/second_refinement_plan.json": "refinement-plan.schema.json",
        },
        "batch_pattern": "analysis/fine/batches/*.json",
        "batch_schema": "fine-batch-analysis.schema.json",
    },
    "terminal": {
        "dirs": ["analysis/terminal_fine"],
        "files": {},
        "batch_pattern": "analysis/terminal_fine/*-*.json",
        "batch_schema": "fine-batch-analysis.schema.json",
        "exclude": {"analysis/terminal_fine/terminal_inputs.json"},
    },
}


def _glob_match(output_dir, pattern):
    import glob

    return sorted(glob.glob(os.path.join(output_dir, pattern)))


def _expect_non_empty(output_dir, rel):
    p = os.path.join(output_dir, rel)
    if not os.path.isfile(p):
        return f"{rel}: file missing"
    if os.path.getsize(p) == 0:
        return f"{rel}: empty file"
    return None


def _schema_errors_for(path, filename):
    data = load_json(path)
    if data is None:
        return [f"{os.path.relpath(path)}: not readable JSON"]
    schema = load_schema(filename)
    if not isinstance(data, dict):
        return [f"{os.path.relpath(path)}: expected JSON object"]
    errs = _validate(data, schema, schema)
    return [f"{os.path.relpath(path)}: {e}" for e in errs]


def _manifest_frames(output_dir):
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    return man.get("coarse", {}).get("frames", [])


def _known_fine_ids(output_dir):
    fb = load_json(os.path.join(output_dir, "analysis/refinement/visual_inputs/fine_batches.json"),
                   default={})
    ids = set()
    for iv in fb.get("intervals", []):
        for f in iv.get("detail_frames", []):
            ids.add(f["frame_id"])
        if iv.get("batch_id"):
            ids.add(iv["batch_id"])
    ti = load_json(os.path.join(output_dir, "analysis/terminal_fine/terminal_inputs.json"),
                   default={}) or {}
    for iv in ti.get("intervals", []):
        for f in iv.get("detail_frames", []):
            ids.add(f["frame_id"])
        if iv.get("batch_id"):
            ids.add(iv["batch_id"])
    return ids


def _evidence_checks(output_dir, evidence, kind, batch_id=None):
    """Hard rule 9: evidence must point at real things."""
    errs = []
    if not isinstance(evidence, dict):
        return ["evidence: must be an object"]
    frame_ids = {f["frame_id"] for f in _manifest_frames(output_dir)}
    known_fine = _known_fine_ids(output_dir)
    for fid in evidence.get("frame_ids") or []:
        if fid not in frame_ids and fid not in known_fine:
            errs.append(f"evidence.frame_ids {fid!r}: no such frame in visual inputs")
    for p in evidence.get("image_paths") or []:
        full = os.path.join(output_dir, p)
        if not os.path.isfile(full):
            errs.append(f"evidence.image_paths {p!r}: file does not exist")
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    sheets = {c["sheet_id"] for c in man.get("coarse", {}).get("contact_sheets", [])}
    for sid in evidence.get("contact_sheet_ids") or []:
        if sid not in sheets:
            errs.append(f"evidence.contact_sheet_ids {sid!r}: no such contact sheet")
    return errs


def _batch_id_known(output_dir, batch_id):
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    known = {b["batch_id"] for b in man.get("coarse", {}).get("batches", [])}
    known |= _known_fine_ids(output_dir)
    return batch_id in known


def validate_stage(output_dir, stage):
    spec = STAGES[stage]
    errs = []
    # every expected non-empty file exists
    for rel in spec["dirs"]:
        d = os.path.join(output_dir, rel)
        if not os.path.isdir(d):
            errs.append(f"{rel}: directory missing")
    # fixed files
    for rel, schema_name in spec["files"].items():
        e = _expect_non_empty(output_dir, rel)
        if e:
            errs.append(e)
        elif os.path.isfile(os.path.join(output_dir, rel)):
            errs += _schema_errors_for(os.path.join(output_dir, rel), schema_name)
    # batch files
    if not _glob_match(output_dir, spec["batch_pattern"]):
        errs.append(f"{spec['batch_pattern']}: no batch files found for {stage} stage")
    for path in _glob_match(output_dir, spec["batch_pattern"]):
        base = os.path.basename(path)
        if spec.get("exclude") and path in {os.path.join(output_dir, e) for e in spec["exclude"]}:
            continue
        errs += _schema_errors_for(path, spec["batch_schema"])
        data = load_json(path) or {}
        bid = data.get("batch_id")
        if bid and not _batch_id_known(output_dir, bid):
            errs.append(f"{os.path.relpath(path)}: batch_id {bid!r} not in visual inputs")
        if isinstance(data.get("evidence"), dict):
            errs += _evidence_checks(output_dir, data["evidence"], stage, bid)
    # semantic: shot segments bounded by media duration
    man = load_json(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json"), default={})
    dur = (man.get("source") or {}).get("duration_s")
    if stage == "fine" and dur:
        fva = load_json(os.path.join(output_dir, "analysis/fine/fine_video_analysis.json"), default={})
        for s in (fva or {}).get("shots") or []:
            if s.get("start", 0) > dur + 1e-6 or s.get("end", 0) > dur + 1e-6:
                errs.append(f"fine_video_analysis.shots {s.get('shot_id')}: exceeds duration {dur}")
    return errs


def run_validate(output_dir, stage):
    errs = validate_stage(output_dir, stage)
    if errs:
        return {"ok": False, "stage": stage, "errors": errs}
    # advance state machine deterministically
    if stage == "coarse":
        state_mod.transition(output_dir, "coarse_validated", note="coarse validation PASS")
    elif stage == "fine":
        state_mod.transition(output_dir, "fine_validated", note="fine validation PASS")
        sec = load_json(os.path.join(output_dir, "analysis/fine/second_refinement_plan.json"),
                        default={}) or {}
        if sec.get("terminal_refinement_status") == "not_required":
            state_mod.transition(output_dir, "terminal_ready", note="terminal not required")
    elif stage == "terminal":
        state_mod.transition(output_dir, "terminal_validated", note="terminal validation PASS")
        state_mod.transition(output_dir, "terminal_ready", note="terminal complete")
    return {"ok": True, "stage": stage, "errors": [],
            "state": state_mod.current(output_dir)}