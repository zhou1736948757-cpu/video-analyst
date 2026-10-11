#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5 finalize — deterministic-only aggregation (no re-vision).

Aggregates coarse + binary fine + active subtitles into final_report.md/json/csv.
Determinism guarantee: pure function of existing files under OUTPUT — no wall
clock, no pid, no random; two runs on identical OUTPUT produce identical md5.
"""

import csv
import os

from . import state as state_mod
from .common import (dump_json, fmt_num, format_hms, load_json, provenance,
                     relpath_from)


def _read_opt(rel):
    if os.path.isfile(rel):
        return load_json(rel)
    return None


def _sorted_files(dirpath, pattern=""):
    if not os.path.isdir(dirpath):
        return []
    return sorted(p for p in os.listdir(dirpath) if p.endswith(pattern))


def collect_outputs(output_dir):
    """Read every consumable artifact in deterministic order."""
    coarse_b = [
        {"batch_file": f, "data": load_json(os.path.join(output_dir, "analysis/coarse/batches", f))}
        for f in _sorted_files(os.path.join(output_dir, "analysis/coarse/batches"), ".json")
    ]
    fine_b = [
        {"batch_file": f, "data": load_json(os.path.join(output_dir, "analysis/fine/batches", f))}
        for f in _sorted_files(os.path.join(output_dir, "analysis/fine/batches"), ".json")
    ]
    terminal_b = [
        {"batch_file": f, "data": load_json(os.path.join(output_dir, "analysis/terminal_fine", f))}
        for f in _sorted_files(os.path.join(output_dir, "analysis/terminal_fine"), ".json")
        if f != "terminal_inputs.json"
    ]
    return {
        "manifest": _read_opt(os.path.join(output_dir, "analysis/coarse/inputs/manifest.json")),
        "coarse_batches": coarse_b,
        "video_structure": _read_opt(os.path.join(output_dir, "analysis/coarse/video_structure.json")),
        "refinement_plan": _read_opt(os.path.join(output_dir, "analysis/coarse/refinement_plan.json")),
        "fine_batches": fine_b,
        "temporal_events": _read_opt(os.path.join(output_dir, "analysis/fine/temporal_events.json")),
        "fine_video_analysis": _read_opt(os.path.join(output_dir, "analysis/fine/fine_video_analysis.json")),
        "second_refinement_plan": _read_opt(os.path.join(output_dir,
                                                         "analysis/fine/second_refinement_plan.json")),
        "terminal_inputs": _read_opt(os.path.join(output_dir, "analysis/terminal_fine/terminal_inputs.json")),
        "terminal_batches": terminal_b,
        "subtitles": _read_opt(os.path.join(output_dir, "subtitles/subtitles.json")),
    }


def _report_id(manifest):
    """Deterministic id from source basename + duration (no time, no pid).

    确定性约束：manifest 中任何被 finalize 透传的字段必须是 pure 派生值；
    禁止新增 wall-clock 字段进入 manifest 且被 finalize 读取（会破坏两次运行
    md5 一致）。probe_timestamp 等 wall-clock 字段只可写入 preparations 而
    finalize 不得读取（rv-VAF-002 FR-003 固化）。
    """
    if not manifest:
        return "video-analyst-report"
    src = manifest.get("source", {})
    base = os.path.basename(src.get("path") or "video")
    return f"{base}-{src.get('duration_s', 0)}"


def _shot_rows(fine, coarse_struct, events):
    """Deterministic flat rows for CSV: shots then events sorted by (start, key)."""
    rows = []
    shots = ((fine or {}).get("shots") or []) if fine and isinstance(fine.get("shots"), list) else []
    for s in sorted(shots, key=lambda x: (x.get("start", 0), x.get("shot_id", ""))):
        rows.append({
            "kind": "shot",
            "id": s.get("shot_id", ""),
            "start": fmt_num(s.get("start", " "), 3),
            "end": fmt_num(s.get("end", " "), 3),
            "description": s.get("description", ""),
            "source": "fine_video_analysis",
        })
    events_list = ((events or {}).get("events") or []) if isinstance(events, dict) else []
    for ev in sorted(events_list, key=lambda x: (x.get("timestamp", 0), x.get("event_id", ""))):
        rows.append({
            "kind": ev.get("kind", "event"),
            "id": ev.get("event_id", ""),
            "start": fmt_num(ev.get("timestamp", " "), 3),
            "end": "",
            "description": ev.get("description", ""),
            "source": "temporal_events",
        })
    if coarse_struct and isinstance(coarse_struct.get("segments"), list):
        for seg in sorted(coarse_struct["segments"],
                          key=lambda x: (x.get("start", 0), x.get("segment_id", ""))):
            rows.append({
                "kind": "segment",
                "id": seg.get("segment_id", ""),
                "start": fmt_num(seg.get("start", " "), 3),
                "end": fmt_num(seg.get("end", " "), 3),
                "description": seg.get("description", ""),
                "source": "video_structure",
            })
    return rows


def render_md(report_id, data, rows, provenance_block):
    lines = []
    lines.append(f"# Video Analysis Report — {report_id}")
    lines.append("")
    lines.append("Generated deterministically from existing analysis inputs (finalize does no re-vision).")
    lines.append("")
    lines.append("## Source")
    src = (data["manifest"] or {}).get("source", {})
    if src:
        lines.append(f"- path: {src.get('path')}")
        lines.append(f"- duration: {src.get('duration_s')} s")
        vid = src.get("video", {})
        lines.append(f"- video: {vid.get('width')}x{vid.get('height')} "
                     f"{vid.get('codec')} {vid.get('fps')} fps")
        lines.append(f"- audio streams: {src.get('audio', {}).get('streams')}")
    lines.append("")
    lines.append("## Coarse analysis (batches)")
    for cb in data["coarse_batches"]:
        d = cb["data"] or {}
        lines.append(f"- **{cb['batch_file']}** ({d.get('batch_id', '?')}): "
                     f"{d.get('summary', '(missing)')}")
    lines.append("")
    lines.append("## Video structure")
    vs = data["video_structure"]
    if vs:
        lines.append(f"- structure_type: {vs.get('structure_type')}")
        for seg in vs.get("segments", []):
            lines.append(f"- {seg.get('segment_id')} {format_hms(seg.get('start'))} → "
                         f"{format_hms(seg.get('end'))}: {seg.get('description')}")
    lines.append("")
    lines.append("## Refinement plan")
    rp = data["refinement_plan"]
    if rp:
        lines.append(f"- plan: {rp.get('plan_id')} / terminal="
                     f"{rp.get('terminal_refinement_status')}")
        for iv in rp.get("intervals", []):
            lines.append(f"- {iv.get('interval_id')}: {format_hms(iv.get('start'))} → "
                         f"{format_hms(iv.get('end'))} @ {iv.get('fps', iv.get('requested_fps'))} fps "
                         f"[{iv.get('priority')}]")
    lines.append("")
    lines.append("## Fine analysis (batches)")
    for fb in data["fine_batches"]:
        d = fb["data"] or {}
        lines.append(f"- **{fb['batch_file']}** ({d.get('batch_id', '?')}): "
                     f"{d.get('summary', '(missing)')}")
    lines.append("")
    lines.append("## Fine video analysis")
    fva = data["fine_video_analysis"]
    if fva:
        lines.append(f"- summary: {fva.get('summary')}")
        for s in fva.get("shots", []):
            lines.append(f"- shot {s.get('shot_id')} {format_hms(s.get('start'))} → "
                         f"{format_hms(s.get('end'))}: {s.get('description')}")
    lines.append("")
    lines.append("## Temporal events")
    te = data["temporal_events"]
    if te:
        for e in te.get("events", []):
            lines.append(f"- {e.get('event_id')} @ {format_hms(e.get('timestamp'))} "
                         f"[{e.get('kind')}]: {e.get('description')}")
    lines.append("")
    lines.append("## Subtitles")
    subs = data["subtitles"]
    if subs:
        lines.append(f"- active: {subs.get('active') or '(none)'}")
        lines.append(f"- asr: {subs.get('asr', {}).get('mode', 'unavailable')}")
        for doc in subs.get("documents", []):
            lines.append(f"- doc {doc.get('id')} ({doc.get('engine')})")
    lines.append("")
    lines.append("## Provenance")
    for k, v in provenance_block.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def run_finalize(output_dir):
    """Finalize: aggregate + write final_report.json/md/csv. Returns outputs dict."""
    output_dir = os.path.abspath(output_dir)
    st = load_json(state_mod.state_path(output_dir)) or {}
    cur = st.get("state")
    if cur not in ("ready_to_finalize", "terminal_analysis_complete", "fine_analysis_complete",
                   "completed"):
        raise RuntimeError(
            f"finalize requires ready_to_finalize (now: {cur}) — run validate-fine/validate-terminal "
            "or check the workflow state"
        )
    data = collect_outputs(output_dir)
    manifest = data["manifest"] or {}
    report_id = _report_id(manifest)
    rows = _shot_rows(data["fine_video_analysis"], data["video_structure"],
                      data["temporal_events"])
    src = manifest.get("source") or {}
    prov = provenance(
        asr_engine=((data["subtitles"] or {}).get("asr") or {}).get("engine"),
        extra={
            "source": src.get("path"),
            "duration_s": src.get("duration_s"),
            "input_manifest": os.path.join("analysis/coarse/inputs/manifest.json"),
            "deps": manifest.get("deps"),
        },
    )
    final = {
        "format": "video-analyst/final-report/1",
        "report_id": report_id,
        "source": src,
        "coarse": {
            "manifest": manifest.get("coarse"),
            "batches": [b["data"] for b in data["coarse_batches"] if b["data"]],
            "video_structure": data["video_structure"],
            "refinement_plan": data["refinement_plan"],
        },
        "fine": {
            "batches": [b["data"] for b in data["fine_batches"] if b["data"]],
            "temporal_events": data["temporal_events"],
            "fine_video_analysis": data["fine_video_analysis"],
            "second_refinement_plan": data["second_refinement_plan"],
        },
        "terminal": {
            "inputs": data["terminal_inputs"],
            "batches": [b["data"] for b in data["terminal_batches"] if b["data"]],
        },
        "subtitles": {
            "active": (data["subtitles"] or {}).get("active"),
            "asr": ((data["subtitles"] or {}).get("asr") or {}),
            "documents": ((data["subtitles"] or {}).get("documents") or []),
        },
        "rows": rows,
        "provenance": prov,
    }

    out_dir = os.path.join(output_dir, "analysis/final")
    from .common import ensure_dir

    ensure_dir(out_dir)
    json_path = os.path.join(out_dir, "final_report.json")
    md_path = os.path.join(out_dir, "final_report.md")
    csv_path = os.path.join(out_dir, "final_report.csv")
    dump_json(json_path, final)
    md_rel = relpath_from(output_dir, md_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_md(report_id, data, rows, prov))
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "id", "start", "end", "description", "source"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    state_mod.transition(output_dir, "finalize_done", note=f"final reports written to {out_dir}")
    return {
        "state": state_mod.current(output_dir),
        "report_id": report_id,
        "json": json_path,
        "md": md_path,
        "csv": csv_path,
        "rows": len(rows),
    }