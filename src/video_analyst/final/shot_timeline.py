"""Shot-level timeline construction (Epic 5).

镜头边界来源优先级：
1. 明确转场事件（transition temporal events）
2. Coarse section boundaries
3. 视频开始 / 结束

约束：
- 时间线必须连续覆盖 [0, video_duration]
- sum(shot.duration) == video_duration（浮点容差）
- 与 fine interval 重叠 → analysis_level=fine（部分覆盖也算 fine）
- 与 terminal 成功区间重叠 → terminal_refined
- 结构化 Fine 字段映射进 ShotAnalysis
"""

from __future__ import annotations

from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.final.models import (
    MergedTemporalEvent,
    ReportEvidenceReference,
    ShotAnalysis,
    ShotAnalysisLevel,
)
from video_analyst.fine.models import FineBatchAnalysis

_MIN_SHOT_SECONDS = 0.05
_BOUNDARY_MERGE_THRESHOLD = 0.5
_TIME_TOL = 1e-3


def _collect_candidate_boundaries(layout: OutputLayout) -> list[tuple[float, str, float]]:
    """收集候选边界 (timestamp, source, confidence)。"""
    candidates: list[tuple[float, str, float]] = []
    cache = CacheStore(layout)

    if layout.coarse_structure_json.is_file():
        try:
            data = cache.read_json(layout.coarse_structure_json)
            for sc in data.get("sections") or []:
                try:
                    candidates.append((float(sc["start"]), "coarse_section_start", 0.5))
                    candidates.append((float(sc["end"]), "coarse_section_end", 0.5))
                except Exception:
                    pass
            for gap in data.get("gaps") or []:
                try:
                    candidates.append((float(gap["start"]), "coarse_gap_start", 0.8))
                    candidates.append((float(gap["end"]), "coarse_gap_end", 0.8))
                except Exception:
                    pass
        except Exception:
            pass
    return candidates


def _dedupe_boundaries(
    candidates: list[tuple[float, str, float]],
    *,
    threshold: float = _BOUNDARY_MERGE_THRESHOLD,
) -> list[tuple[float, list[str], float]]:
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda x: x[0])
    merged: list[tuple[float, list[str], float]] = []
    cur_ts, cur_sources, cur_conf = sorted_c[0][0], [sorted_c[0][1]], sorted_c[0][2]
    for ts, src, conf in sorted_c[1:]:
        if ts - cur_ts <= threshold:
            if src not in cur_sources:
                cur_sources.append(src)
            cur_conf = max(cur_conf, conf)
        else:
            merged.append((cur_ts, cur_sources, cur_conf))
            cur_ts, cur_sources, cur_conf = ts, [src], conf
    merged.append((cur_ts, cur_sources, cur_conf))
    return merged


def _clamp_bounds(bounds: list[float], video_duration: float) -> list[float]:
    """Force continuous coverage [0, video_duration] with no exterior points."""
    if video_duration <= 0:
        return [0.0]
    cleaned: list[float] = []
    for b in bounds:
        if b < -_TIME_TOL:
            continue
        if b > video_duration + _TIME_TOL:
            continue
        cleaned.append(min(max(0.0, b), video_duration))
    cleaned = sorted(set(round(x, 6) for x in cleaned))
    if not cleaned or cleaned[0] > _TIME_TOL:
        cleaned = [0.0] + cleaned
    else:
        cleaned[0] = 0.0
    if cleaned[-1] < video_duration - _TIME_TOL:
        cleaned.append(round(video_duration, 6))
    else:
        cleaned[-1] = round(video_duration, 6)
    # remove near-duplicates after clamp
    out: list[float] = [cleaned[0]]
    for b in cleaned[1:]:
        if b - out[-1] > _TIME_TOL:
            out.append(b)
        else:
            out[-1] = b
    if out[-1] < video_duration - _TIME_TOL:
        out.append(round(video_duration, 6))
    else:
        out[-1] = round(video_duration, 6)
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_shot_timeline(
    *,
    layout: OutputLayout,
    video_duration: float,
    fine_analyses: dict[str, FineBatchAnalysis],
    merged_events: list[MergedTemporalEvent],
    terminal_ranges: list[tuple[float, float]] | None = None,
    evidence_index_helpers: dict[str, ReportEvidenceReference] | None = None,
) -> list[ShotAnalysis]:
    """构建连续覆盖 [0, video_duration] 的镜头时间线。"""
    evidence_index_helpers = evidence_index_helpers or {}
    terminal_ranges = terminal_ranges or []

    transition_times: list[tuple[float, str, float]] = []
    for m in merged_events:
        if m.active.category == "transition":
            t = m.active.boundary.earliest
            transition_times.append((t, "transition_event", m.active.confidence))

    other_boundaries = _collect_candidate_boundaries(layout) if layout is not None else []
    # Interior section ends that equal video end should not create a phantom
    # boundary past duration; clamp later.
    all_candidates = transition_times + other_boundaries
    deduped = _dedupe_boundaries(all_candidates)

    raw_bounds = [0.0] + [ts for ts, _, _ in deduped] + [video_duration]
    bounds = _clamp_bounds(raw_bounds, video_duration)

    shots: list[ShotAnalysis] = []
    for i in range(len(bounds) - 1):
        s = bounds[i]
        e = bounds[i + 1]
        if e - s < _MIN_SHOT_SECONDS:
            # Merge ultra-short gap into previous shot when possible
            if shots:
                prev = shots[-1]
                shots[-1] = prev.model_copy(
                    update={
                        "end": round(e, 6),
                        "duration": round(e - prev.start, 6),
                    }
                )
            continue
        shot = _build_one_shot(
            layout=layout,
            shot_id=f"shot_{i:04d}",
            start=s,
            end=e,
            fine_analyses=fine_analyses,
            merged_events=merged_events,
            terminal_ranges=terminal_ranges,
        )
        shots.append(shot)

    if not shots and video_duration > 0:
        shots.append(
            _build_one_shot(
                layout=layout,
                shot_id="shot_0000",
                start=0.0,
                end=video_duration,
                fine_analyses=fine_analyses,
                merged_events=merged_events,
                terminal_ranges=terminal_ranges,
            )
        )

    # Final continuity repair: force last.end == video_duration
    if shots and video_duration > 0:
        last = shots[-1]
        if abs(last.end - video_duration) > _TIME_TOL:
            shots[-1] = last.model_copy(
                update={
                    "end": round(video_duration, 6),
                    "duration": round(video_duration - last.start, 6),
                }
            )
        if abs(shots[0].start) > _TIME_TOL:
            first = shots[0]
            shots[0] = first.model_copy(
                update={
                    "start": 0.0,
                    "duration": round(first.end - 0.0, 6),
                }
            )
        # Fix residual gaps/overlaps by snapping adjacent ends
        for i in range(1, len(shots)):
            prev, cur = shots[i - 1], shots[i]
            if abs(prev.end - cur.start) > _TIME_TOL:
                shots[i - 1] = prev.model_copy(
                    update={
                        "end": round(cur.start, 6),
                        "duration": round(cur.start - prev.start, 6),
                    }
                )

    return shots


def _build_one_shot(
    *,
    layout: OutputLayout,
    shot_id: str,
    start: float,
    end: float,
    fine_analyses: dict[str, FineBatchAnalysis],
    merged_events: list[MergedTemporalEvent],
    terminal_ranges: list[tuple[float, float]],
) -> ShotAnalysis:
    warnings: list[str] = []
    duration = end - start

    overlapping_batches: list[FineBatchAnalysis] = []
    fine_overlap = 0.0
    for a in fine_analyses.values():
        ov = _overlap(start, end, a.start, a.end)
        if ov > _TIME_TOL:
            overlapping_batches.append(a)
            fine_overlap += ov

    fine_coverage_ratio = min(1.0, fine_overlap / duration) if duration > 0 else 0.0

    terminal_overlap = 0.0
    for t0, t1 in terminal_ranges:
        terminal_overlap += _overlap(start, end, t0, t1)

    # Events overlapping shot
    shot_events: list[MergedTemporalEvent] = []
    for ev in merged_events:
        if ev.active.boundary.earliest < end - _TIME_TOL and ev.active.boundary.latest > start + _TIME_TOL:
            shot_events.append(ev)

    has_terminal_event = any(ev.source_round == "terminal" for ev in shot_events)

    analysis_level: ShotAnalysisLevel = "coarse_only"
    if terminal_overlap > _TIME_TOL or has_terminal_event:
        analysis_level = "terminal_refined"
    elif fine_overlap > _TIME_TOL:
        analysis_level = "fine"
        if fine_coverage_ratio < 0.999:
            warnings.append(
                f"镜头仅部分被 Fine 覆盖 ({fine_coverage_ratio:.1%})"
            )

    evidence_ids: list[str] = []
    for ev in shot_events:
        raw_id = ev.active.event_id
        eid = f"EVID-EV-{raw_id}"
        for candidate in (eid, raw_id):
            if candidate not in evidence_ids:
                evidence_ids.append(candidate)
        for fid in ev.active.evidence.frame_ids:
            if fid not in evidence_ids:
                evidence_ids.append(fid)

    subjects: list[str] = []
    visible_text: list[str] = []
    subject_motions: list[str] = []
    camera_motions: list[str] = []
    post_transforms: list[str] = []
    text_animations: list[str] = []
    graphic_animations: list[str] = []
    transitions_in: list[str] = []
    transitions_out: list[str] = []

    # Prefer structured FineBatchAnalysis fields
    for a in overlapping_batches:
        for fs in a.frame_states:
            for s in fs.visible_subjects:
                if s and s not in subjects:
                    subjects.append(s)
            for t in fs.visible_text:
                if t and t not in visible_text:
                    visible_text.append(t)
        for sm in a.subject_motions:
            desc = getattr(sm, "description", None) or str(sm)
            if hasattr(sm, "motion_type"):
                desc = f"{sm.motion_type}: {getattr(sm, 'description', '')}".strip(": ")
            if desc and desc not in subject_motions:
                subject_motions.append(desc)
        for cm in a.camera_motions:
            desc = getattr(cm, "description", None) or str(cm)
            if hasattr(cm, "motion_type"):
                desc = f"{cm.motion_type}: {getattr(cm, 'description', '')}".strip(": ")
            if desc and desc not in camera_motions:
                camera_motions.append(desc)
        for pt in a.post_transforms:
            desc = getattr(pt, "description", None) or str(pt)
            if hasattr(pt, "transform_type"):
                desc = f"{pt.transform_type}: {getattr(pt, 'description', '')}".strip(": ")
            if desc and desc not in post_transforms:
                post_transforms.append(desc)
        for ta in a.text_animations:
            parts = []
            if getattr(ta, "text_content", None):
                parts.append(str(ta.text_content))
            if getattr(ta, "animation_types", None):
                parts.append("/".join(str(x) for x in ta.animation_types))
            if getattr(ta, "start_state", None) or getattr(ta, "end_state", None):
                parts.append(f"{ta.start_state}→{ta.end_state}")
            desc = " | ".join(p for p in parts if p) or "text animation"
            if desc not in text_animations:
                text_animations.append(desc)
            eid = f"EVID-EV-{ta.event_id}"
            if eid not in evidence_ids:
                evidence_ids.append(eid)
        for ga in a.graphic_animations:
            parts = [getattr(ga, "graphic_subject", "") or "graphic"]
            if getattr(ga, "animation_types", None):
                parts.append("/".join(str(x) for x in ga.animation_types))
            if getattr(ga, "start_state", None) or getattr(ga, "end_state", None):
                parts.append(f"{ga.start_state}→{ga.end_state}")
            desc = " | ".join(p for p in parts if p)
            if desc not in graphic_animations:
                graphic_animations.append(desc)
            eid = f"EVID-EV-{ga.event_id}"
            if eid not in evidence_ids:
                evidence_ids.append(eid)
        for tr in a.transitions:
            desc = f"{tr.transition_type}: {tr.outgoing_state} → {tr.incoming_state}".strip()
            t = tr.boundary.earliest
            if abs(t - start) <= 0.55:
                if desc not in transitions_in:
                    transitions_in.append(desc)
            if abs(t - end) <= 0.55:
                if desc not in transitions_out:
                    transitions_out.append(desc)
            # also keep mid-shot transitions as out notes
            if abs(t - start) > 0.55 and abs(t - end) > 0.55:
                if desc not in transitions_out:
                    transitions_out.append(desc)
            eid = f"EVID-EV-{tr.event_id}"
            if eid not in evidence_ids:
                evidence_ids.append(eid)

    # Supplement from temporal events when structured lists empty
    for ev in shot_events:
        cat = ev.active.category
        desc = ev.active.description
        conf = ev.active.confidence
        if cat == "subject_motion" and desc not in subject_motions:
            subject_motions.append(desc)
        elif cat == "camera_motion" and desc not in camera_motions:
            camera_motions.append(desc)
        elif cat == "post_transform" and desc not in post_transforms:
            post_transforms.append(desc)
        elif cat == "text_animation" and desc not in text_animations:
            text_animations.append(desc)
        elif cat == "graphic_animation" and desc not in graphic_animations:
            graphic_animations.append(desc)
        elif cat == "transition":
            t = ev.active.boundary.earliest
            if abs(t - start) < 0.55 and desc not in transitions_in:
                transitions_in.append(desc)
            if abs(t - end) < 0.55 and desc not in transitions_out:
                transitions_out.append(desc)

    if shot_events:
        avg_conf = sum(e.active.confidence for e in shot_events) / len(shot_events)
    elif overlapping_batches:
        avg_conf = sum(a.confidence for a in overlapping_batches) / len(overlapping_batches)
    else:
        avg_conf = 0.4 if analysis_level == "coarse_only" else 0.6

    if duration < _MIN_SHOT_SECONDS * 2:
        warnings.append(f"very short shot ({duration:.3f}s)")

    summary_parts = (
        subject_motions + camera_motions + post_transforms + text_animations + graphic_animations
    )
    visual_summary = "; ".join(summary_parts)[:500] if summary_parts else ""
    if not visual_summary and overlapping_batches:
        visual_summary = "; ".join(
            a.summary for a in overlapping_batches if a.summary
        )[:500]

    return ShotAnalysis(
        shot_id=shot_id,
        start=round(start, 6),
        end=round(end, 6),
        duration=round(duration, 6),
        visual_summary=visual_summary,
        subjects=subjects,
        visible_text=visible_text,
        subject_motions=subject_motions,
        camera_motions=camera_motions,
        post_transforms=post_transforms,
        text_animations=text_animations,
        graphic_animations=graphic_animations,
        transitions_in=transitions_in,
        transitions_out=transitions_out,
        evidence_ids=evidence_ids,
        confidence=round(avg_conf, 3),
        analysis_level=analysis_level,
        warnings=warnings,
        fine_coverage_ratio_within_shot=round(fine_coverage_ratio, 3),
    )


def save_shots(layout: OutputLayout, shots: list[ShotAnalysis]) -> None:
    cache = CacheStore(layout)
    cache.write_json(
        layout.final_analysis_dir / "shots.json",
        {"shots": [s.model_dump(mode="json") for s in shots]},
    )


__all__ = ["build_shot_timeline", "save_shots"]
