"""Report consistency validation (Production Readiness Gate).

复用 FinalVideoAnalysis 模型，不创建第二套 Schema。

用法：python scripts/validate_report_consistency.py OUTPUT_DIR
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(output_dir: Path) -> tuple[bool, list[str], list[str]]:
    """返回 (passed, errors, warnings)。"""
    errors: list[str] = []
    warnings: list[str] = []

    report_dir = output_dir / "report"
    report_json = report_dir / "report.json"
    report_md = report_dir / "report.md"
    evidence_index = report_dir / "evidence" / "index.json"
    fine_events = output_dir / "analysis" / "fine" / "temporal_events.json"
    fine_batches_dir = output_dir / "analysis" / "fine" / "batches"

    if not report_json.is_file():
        errors.append("report.json missing — cannot validate")
        return False, errors, warnings

    try:
        report = _load_json(report_json)
    except Exception as exc:
        errors.append(f"report.json not valid JSON: {exc}")
        return False, errors, warnings

    video_duration = float(
        (report.get("video_metadata") or {}).get("duration")
        or (report.get("coverage") or {}).get("video_duration")
        or 0.0
    )

    # --- Fine event counts from source ---
    n_temporal = 0
    n_text = 0
    n_graphic = 0
    n_transition = 0
    if fine_events.is_file():
        te = _load_json(fine_events)
        if isinstance(te, list):
            events = te
        else:
            events = te.get("events") or te.get("temporal_events") or []
        n_temporal = len(events)
        for e in events:
            cat = e.get("category")
            if cat == "text_animation":
                n_text += 1
            elif cat == "graphic_animation":
                n_graphic += 1
            elif cat == "transition":
                n_transition += 1
    if fine_batches_dir.is_dir() and n_temporal == 0:
        for p in fine_batches_dir.glob("*.json"):
            try:
                d = _load_json(p)
            except Exception:
                continue
            n_text += len(d.get("text_animations") or [])
            n_graphic += len(d.get("graphic_animations") or [])
            n_transition += len(d.get("transitions") or [])
            n_temporal += len(d.get("temporal_events") or [])

    # Prefer report structured counts when present
    rep_text = report.get("text_animation_items") or []
    rep_graphic = report.get("graphic_animation_items") or []
    rep_tr = report.get("transition_items") or []
    rep_highlights = report.get("detailed_highlights") or []
    important = report.get("important_events") or []
    shots = report.get("shots") or []
    techniques = report.get("repeatable_techniques") or []
    coverage = report.get("coverage") or {}
    fine_ratio = float(coverage.get("fine_coverage_ratio") or 0.0)

    # 1. temporal_events > 0 → detailed analysis not all empty
    source_event_count = n_temporal or len(important)
    if source_event_count > 0:
        if not rep_highlights and not any(
            (s.get("text_animations") or s.get("graphic_animations") or s.get("transitions_in") or s.get("transitions_out"))
            for s in shots
        ):
            errors.append(
                "Fine temporal events exist but detailed analysis sections are empty "
                "(no detailed_highlights and no shot-level motion/text/graphic)"
            )

    # 2. text animation events → text chapter non-empty
    if n_text > 0 or rep_text:
        has_text = bool(rep_text) or any(s.get("text_animations") for s in shots)
        if not has_text:
            errors.append("text_animation events exist but text animation chapter is empty")

    # 3. transition events → transition chapter non-empty
    if n_transition > 0 or rep_tr:
        has_tr = bool(rep_tr) or any(
            (s.get("transitions_in") or s.get("transitions_out")) for s in shots
        )
        if not has_tr:
            errors.append("transition events exist but transition chapter is empty")

    # graphic check
    if n_graphic > 0 or rep_graphic:
        has_g = bool(rep_graphic) or any(s.get("graphic_animations") for s in shots)
        if not has_g:
            errors.append("graphic_animation events exist but graphic chapter is empty")

    # 4. fine coverage > 0 → not all shots coarse_only
    if fine_ratio > 0 and shots:
        levels = {s.get("analysis_level") for s in shots}
        if levels == {"coarse_only"}:
            errors.append(
                "fine_coverage_ratio > 0 but all shots are coarse_only"
            )

    # 5. shot timeline covers full video
    if shots and video_duration > 0:
        first = float(shots[0].get("start") or 0)
        last = float(shots[-1].get("end") or 0)
        if abs(first - 0.0) > 0.05:
            errors.append(f"shots[0].start={first} != 0")
        if abs(last - video_duration) > 0.05:
            errors.append(
                f"shots[-1].end={last} != video_duration={video_duration}"
            )
        # continuity
        for i in range(1, len(shots)):
            prev_end = float(shots[i - 1].get("end") or 0)
            cur_start = float(shots[i].get("start") or 0)
            if abs(prev_end - cur_start) > 0.05:
                errors.append(
                    f"shot gap/overlap between {shots[i-1].get('shot_id')} "
                    f"end={prev_end} and {shots[i].get('shot_id')} start={cur_start}"
                )
        total = sum(float(s.get("duration") or 0) for s in shots)
        if abs(total - video_duration) > 0.1:
            errors.append(
                f"sum(shot.duration)={total} != video_duration={video_duration}"
            )

    # 6. Terminal text consistency
    runtime = report.get("runtime_status") or {}
    terminal_status = runtime.get("terminal_refinement_status") or (
        "executed" if report.get("terminal_refinement_executed") else "not_required"
    )
    if report_md.is_file():
        md = report_md.read_text(encoding="utf-8")
        if terminal_status == "not_required" and "终局精修" in md and "round1 + 终局" in md:
            errors.append(
                "terminal not_required but markdown claims round1 + 终局精修"
            )
        if "来自 round1 + 终局精修" in md and terminal_status != "executed":
            errors.append(
                "markdown claims 终局精修 but terminal was not executed"
            )
        # 9. Chinese report should not contain English empty placeholders
        lang = "zh"
        english_empties = [
            "_No detailed analytical highlights._",
            "_No subject motion recorded._",
            "_No camera / post-transform events recorded._",
            "_No text / graphic animation recorded._",
            "_No transition events recorded._",
            "_Pacing analysis not generated._",
            "_Subtitle/A/V analysis not generated._",
            "_No repeatable techniques extracted._",
            "_No sections identified._",
            "_No shots built._",
        ]
        for phrase in english_empties:
            if phrase in md:
                errors.append(f"Chinese report contains English empty placeholder: {phrase}")

        # 7/8. EVID references
        # Match real IDs like EVID-EV-e1 / EVID-SHOT-shot_0000; ignore doc placeholders
        evid_in_md = set(
            re.findall(r"EVID-(?:EV|CS|DF|SHOT)-[A-Za-z0-9_\-]+", md)
        )
        evid_index_ids: set[str] = set()
        if evidence_index.is_file():
            try:
                idx = _load_json(evidence_index)
                for e in idx.get("entries") or []:
                    if e.get("evidence_id"):
                        evid_index_ids.add(e["evidence_id"])
            except Exception as exc:
                errors.append(f"evidence index invalid: {exc}")
        if source_event_count > 0 and not evid_in_md and not any(
            s.get("evidence_ids") for s in shots
        ):
            errors.append("important fine events exist but no EVID references found")
        for eid in evid_in_md:
            if evid_index_ids and eid not in evid_index_ids:
                errors.append(f"markdown references missing evidence id {eid}")

        # structured events should have evidence
        for field in (
            "text_animation_items",
            "graphic_animation_items",
            "transition_items",
            "detailed_highlights",
        ):
            for item in report.get(field) or []:
                eids = item.get("evidence_ids") or []
                if not eids:
                    errors.append(f"{field} item missing evidence_ids: {item.get('description') or item.get('summary') or item.get('interval_id')}")

    # 10. repeatable techniques consistent with available fine events
    if (n_text or n_graphic or n_transition or rep_text or rep_graphic or rep_tr) and not techniques:
        errors.append(
            "fine text/graphic/transition events exist but repeatable_techniques is empty"
        )

    # Provider / API flags for in_session_agent
    if report.get("external_api_used") is True:
        errors.append("external_api_used must be false for in_session_agent product path")
    if report.get("api_key_used") is True:
        errors.append("api_key_used must be false for in_session_agent product path")

    # evidence image paths if declared
    if evidence_index.is_file():
        try:
            ev_idx = _load_json(evidence_index)
        except Exception as exc:
            errors.append(f"evidence index not valid JSON: {exc}")
            ev_idx = {"entries": []}
        for entry in ev_idx.get("entries") or []:
            eid = entry.get("evidence_id")
            eend = float(entry.get("end") or 0.0)
            if video_duration > 0 and eend > video_duration + 1.0:
                errors.append(
                    f"evidence {eid} end={eend} exceeds video duration={video_duration}"
                )
            for ip in entry.get("image_paths") or []:
                ip_path = Path(ip)
                if not ip_path.is_absolute():
                    ip_path = output_dir / ip_path
                if ip and not ip_path.exists():
                    warnings.append(f"evidence {eid} image path missing: {ip}")

    for shot in shots:
        send = float(shot.get("end") or 0.0)
        if video_duration > 0 and send > video_duration + 1.0:
            errors.append(
                f"shot {shot.get('shot_id')} end={send} exceeds duration={video_duration}"
            )

    # subtitles not reviewed warning preferred
    if coverage and not coverage.get("subtitle_reviewed"):
        if report_md.is_file() and "未人工审核" not in report_md.read_text(encoding="utf-8"):
            warnings.append("subtitles not reviewed but no explicit warning in report")

    return len(errors) == 0, errors, warnings


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python scripts/validate_report_consistency.py OUTPUT_DIR")
        return 2
    out = Path(argv[0]).expanduser().resolve()
    print(f"Output: {out}")
    ok, errors, warnings = validate(out)
    if ok:
        print("Consistency check: PASSED")
    else:
        print("Consistency check: FAILED")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
