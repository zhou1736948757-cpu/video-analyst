"""Analysis coverage report (Epic 5).

区分：
- coarse / fine / terminal 覆盖
- failed ranges
- deferred ranges
- unexamined ranges
- 字幕审核状态
- provider mode (mock / real)
"""

from __future__ import annotations

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.fine.models import TimeRange
from video_analyst.final.models import AnalysisCoverage


def _covered_duration(ranges: list[TimeRange]) -> float:
    if not ranges:
        return 0.0
    sorted_r = sorted(ranges, key=lambda r: r.start)
    cur_start, cur_end = sorted_r[0].start, sorted_r[0].end
    total = 0.0
    for r in sorted_r[1:]:
        if r.start <= cur_end:
            cur_end = max(cur_end, r.end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = r.start, r.end
    total += cur_end - cur_start
    return total


def _merge_ranges(ranges: list[TimeRange]) -> list[TimeRange]:
    if not ranges:
        return []
    sorted_r = sorted(ranges, key=lambda r: r.start)
    merged: list[TimeRange] = [sorted_r[0]]
    for r in sorted_r[1:]:
        last = merged[-1]
        if r.start <= last.end:
            merged[-1] = TimeRange(start=last.start, end=max(last.end, r.end))
        else:
            merged.append(r)
    return merged


def build_coverage_report(layout: OutputLayout, *, video_duration: float) -> AnalysisCoverage:
    """构建 AnalysisCoverage。

    Inputs:
    - fine_video_analysis.json: analyzed_intervals + analysis_gaps
    - terminal_fine_analysis.json (optional): terminal 分析区间
    - media_info.json + subtitles.json: 时长、字幕审核状态
    """
    cache = CacheStore(layout)

    # 加载 fine 区间 / gap
    fine_analyzed: list[TimeRange] = []
    fine_gaps: list[TimeRange] = []
    if layout.fine_video_analysis_json.is_file():
        try:
            data = cache.read_json(layout.fine_video_analysis_json)
            for r in data.get("analyzed_intervals") or []:
                try:
                    fine_analyzed.append(
                        TimeRange(start=float(r["start"]), end=float(r["end"]))
                    )
                except Exception:
                    pass
            for r in data.get("analysis_gaps") or []:
                try:
                    fine_gaps.append(
                        TimeRange(start=float(r["start"]), end=float(r["end"]))
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # 加载 terminal_fine 区间
    terminal_analyzed: list[TimeRange] = []
    if layout.terminal_fine_analysis_json.is_file():
        try:
            data = cache.read_json(layout.terminal_fine_analysis_json)
            for r in data.get("analyzed_intervals") or []:
                try:
                    terminal_analyzed.append(
                        TimeRange(start=float(r["start"]), end=float(r["end"]))
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # 加载 deferred
    deferred: list[TimeRange] = []
    if layout.fine_second_refinement_plan_json.is_file():
        try:
            data = cache.read_json(layout.fine_second_refinement_plan_json)
            for iv in data.get("deferred_intervals") or []:
                try:
                    deferred.append(
                        TimeRange(start=float(iv["start"]), end=float(iv["end"]))
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # 字幕审核状态
    subtitle_reviewed = False
    subtitle_needs_review = False
    if layout.review_json.is_file():
        try:
            data = cache.read_json(layout.review_json)
            subtitle_reviewed = bool(data.get("reviewed"))
            subtitle_needs_review = bool(data.get("needs_review"))
        except Exception:
            pass

    # Vision provenance: product path is in-session agent only.
    # Never treat mock/external API as the default real analysis mode.
    providers_used: list[str] = ["in_session_agent"]
    models_used: list[str] = []
    real_api_used = False
    provider_mode = "in_session_agent"
    # Optional legacy manifest scan (does not re-enable external APIs)
    if layout.coarse_manifest_json.is_file():
        try:
            data = cache.read_json(layout.coarse_manifest_json)
            p = str(data.get("provider") or "")
            if p and p not in ("mock", "in_session_agent", ""):
                # Record but do not flip real_api for external keys
                if p not in providers_used:
                    providers_used.append(p)
            if data.get("model"):
                models_used.append(str(data["model"]))
        except Exception:
            pass
    if layout.fine_manifest_json.is_file():
        try:
            data = cache.read_json(layout.fine_manifest_json)
            p = str(data.get("provider") or "")
            m = str(data.get("model") or "")
            if p and p not in providers_used:
                providers_used.append(p)
            if m and m not in models_used:
                models_used.append(m)
        except Exception:
            pass

    # 计算覆盖比例
    coarse_analyzed_duration = video_duration
    fine_analyzed_dur = _covered_duration(fine_analyzed)
    terminal_analyzed_dur = _covered_duration(terminal_analyzed)

    coarse_cov = coarse_analyzed_duration / video_duration if video_duration > 0 else 0.0
    fine_cov = fine_analyzed_dur / video_duration if video_duration > 0 else 0.0
    terminal_cov = terminal_analyzed_dur / video_duration if video_duration > 0 else 0.0

    # 未精细分析区间 = 全片 - fine_analyzed - fine_gaps
    unexamined: list[TimeRange] = []
    if video_duration > 0:
        covered_with_gaps = _merge_ranges(fine_analyzed + fine_gaps)
        last_end = 0.0
        for r in covered_with_gaps:
            if r.start > last_end:
                unexamined.append(
                    TimeRange(start=last_end, end=min(r.start, video_duration))
                )
            last_end = max(last_end, r.end)
        if last_end < video_duration:
            unexamined.append(TimeRange(start=last_end, end=video_duration))

    warnings: list[str] = []
    if fine_cov < 0.5:
        warnings.append(f"fine coverage only {fine_cov:.1%} of video duration")
    if deferred:
        warnings.append(f"{len(deferred)} second-refinement intervals deferred")
    if not subtitle_reviewed and not subtitle_needs_review:
        warnings.append(
            "active subtitles not human-reviewed; treat as supporting context only"
        )
    if subtitle_needs_review:
        warnings.append("active subtitles flagged needs_review")
    # In-session agent analysis is real visual analysis without external API.
    warnings.append(
        "vision_execution_mode=in_session_agent; "
        "visual analysis written by current conversation Agent "
        "(external_api_used=false, api_key_used=false)"
    )

    return AnalysisCoverage(
        video_duration=round(video_duration, 6),
        coarse_analyzed_duration=round(coarse_analyzed_duration, 6),
        fine_analyzed_duration=round(fine_analyzed_dur, 6),
        terminal_analyzed_duration=round(terminal_analyzed_dur, 6),
        coarse_coverage_ratio=round(coarse_cov, 3),
        fine_coverage_ratio=round(fine_cov, 3),
        terminal_coverage_ratio=round(terminal_cov, 3),
        successful_ranges=fine_analyzed,
        failed_ranges=fine_gaps,
        deferred_ranges=deferred,
        unexamined_ranges=unexamined,
        subtitle_reviewed=subtitle_reviewed,
        subtitle_needs_review=subtitle_needs_review,
        real_vision_api_used=False,
        providers_used=providers_used,
        models_used=models_used,
        warnings=warnings,
    )


def save_coverage_report(layout: OutputLayout, coverage: AnalysisCoverage) -> None:
    cache = CacheStore(layout)
    cache.write_json(
        layout.final_analysis_coverage_json, coverage.model_dump(mode="json")
    )


__all__ = ["build_coverage_report", "save_coverage_report"]