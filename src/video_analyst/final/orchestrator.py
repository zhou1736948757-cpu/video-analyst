"""Epic 5 orchestrator: end-to-end final report generation.

Flow:
    Epic 1–4 outputs
    → load coarse + fine results
    → merge temporal events (round1 + terminal if executed)
    → build coverage
    → build shot timeline
    → build sections
    → build evidence index
    → copy evidence images
    → build FinalVideoAnalysis JSON (report.json)
    → render Markdown report (report.md)
    → render CSV (shot_table / event_table / unresolved_items)
    → write run_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig, FinalAnalysisConfig, VisionConfig
from video_analyst.env.detect import ensure_ready_for_media
from video_analyst.exceptions import (
    Epic5PipelineError,
    FinalEventMergeError,
    ProvisionalReportNotAllowedError,
    ReportBuildError,
    SectionAggregationError,
    ShotTimelineError,
    TerminalRefinementPlanValidationError,
)
from video_analyst.fine.batch_runner import load_fine_batches_index
from video_analyst.fine.mock import MockFineVisionProvider
from video_analyst.fine.models import FineBatchAnalysis, TimeRange
from video_analyst.final.aggregation import build_final_video_analysis
from video_analyst.final.coverage import build_coverage_report, save_coverage_report
from video_analyst.final.csv_report import write_csv_reports
from video_analyst.final.evidence import (
    build_evidence_index,
    copy_evidence_images,
    save_evidence_index,
)
from video_analyst.final.merge import (
    collect_unresolved_items,
    merge_question_answers,
    merge_temporal_events,
)
from video_analyst.final.models import FinalVideoAnalysis, ReportStatus
from video_analyst.final.report import (
    save_final_video_analysis_json,
    write_markdown_report,
)
from video_analyst.final.run_summary import build_run_summary, save_run_summary
from video_analyst.final.sections import build_sections
from video_analyst.final.shot_timeline import build_shot_timeline
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.models import FineVisualBatch
from video_analyst.refinement.round import prepare_refinement_round
from video_analyst.fine.terminal import (
    run_terminal_fine_analysis,
    validate_terminal_refinement_plan,
)

log = get_logger("final.orchestrator")


@dataclass
class Epic5Result:
    status: str  # final / provisional / mock_validation / failed
    analysis: FinalVideoAnalysis | None
    output_dir: Path
    terminal_refinement_required: bool = False
    terminal_refinement_executed: bool = False
    terminal_refinement_status: str = "not_required"  # not_required / completed / skipped
    manifest_path: Path | None = None
    report_md_path: Path | None = None
    report_json_path: Path | None = None
    evidence_index_path: Path | None = None
    run_summary_path: Path | None = None
    shot_table_path: Path | None = None
    event_table_path: Path | None = None
    unresolved_table_path: Path | None = None
    failed_stage: str | None = None
    warnings: list[str] = field(default_factory=list)


def _load_fine_analyses(layout: OutputLayout) -> dict[str, FineBatchAnalysis]:
    """Load agent-written FineBatchAnalysis JSON from analysis/fine/batches/.

    Product path: the conversation Agent writes structured fine analysis.
    Never invent empty stubs that drop text/graphic/transition fields.
    Falls back to interval bounds from fine_batches.json only when a batch
    analysis file is missing (still without inventing visual conclusions).
    """
    out: dict[str, FineBatchAnalysis] = {}
    cache = CacheStore(layout)

    # 1) Prefer agent-written analysis/fine/batches/*.json
    batches_dir = layout.fine_batches_out_dir
    if batches_dir.is_dir():
        for path in sorted(batches_dir.glob("*.json")):
            try:
                data = cache.read_json(path)
                analysis = FineBatchAnalysis.model_validate(data)
            except Exception as exc:
                log.warning("skip fine batch %s: %s", path.name, exc)
                continue
            out[analysis.fine_batch_id] = analysis

    if out:
        return out

    # 2) Fallback: interval shells from Epic 3 fine_batches index (no fake events)
    if not layout.refinement_fine_batches_json.is_file():
        return {}
    try:
        data = load_fine_batches_index(layout.refinement_fine_batches_json)
    except Exception:
        return {}
    for b in data.get("batches") or []:
        try:
            fb = FineVisualBatch.model_validate(b)
            out[fb.fine_batch_id] = FineBatchAnalysis(
                fine_batch_id=fb.fine_batch_id,
                interval_id=fb.interval_id,
                start=fb.interval.start,
                end=fb.interval.end,
                confidence=0.5,
                warnings=["fine analysis JSON missing; interval shell only"],
            )
        except Exception:
            continue
    return out


def _iter_event_payloads(data: Any) -> list:
    """Accept agent-written list or {events|temporal_events: [...]} dict."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.get("events") or data.get("temporal_events") or [])
    return []


def _load_round1_events(layout: OutputLayout) -> list:
    """从 fine temporal_events.json 读取 round1 事件列表（用于合并）。"""
    cache = CacheStore(layout)
    if not layout.fine_temporal_events_json.is_file():
        return []
    try:
        data = cache.read_json(layout.fine_temporal_events_json)
    except Exception:
        return []
    from video_analyst.fine.models import TemporalEvent

    out: list[TemporalEvent] = []
    for raw in _iter_event_payloads(data):
        try:
            out.append(TemporalEvent.model_validate(raw))
        except Exception:
            continue
    return out


def _load_round1_answers(layout: OutputLayout) -> list:
    """从 analysis/fine/batches/*.json 读取 Agent 写入的 question_answers。"""
    from video_analyst.fine.models import RefinementQuestionAnswer

    cache = CacheStore(layout)
    out: list[RefinementQuestionAnswer] = []
    seen_q: set[str] = set()

    # Primary: agent-written fine analysis batches
    if layout.fine_batches_out_dir.is_dir():
        for path in sorted(layout.fine_batches_out_dir.glob("*.json")):
            try:
                data = cache.read_json(path)
                analysis = FineBatchAnalysis.model_validate(data)
            except Exception:
                continue
            for qa in analysis.question_answers:
                key = qa.question.strip()
                if key in seen_q:
                    continue
                seen_q.add(key)
                out.append(qa)

    if out:
        return out

    # Fallback: fine_video_analysis.answered_questions
    if layout.fine_video_analysis_json.is_file():
        try:
            data = cache.read_json(layout.fine_video_analysis_json)
            for raw in data.get("answered_questions") or []:
                try:
                    qa = RefinementQuestionAnswer.model_validate(raw)
                except Exception:
                    continue
                key = qa.question.strip()
                if key in seen_q:
                    continue
                seen_q.add(key)
                out.append(qa)
        except Exception:
            pass
    return out


def _load_terminal_events(layout: OutputLayout) -> list:
    """从 terminal_fine/temporal_events.json 读取终局事件（如有）。"""
    if not layout.terminal_fine_temporal_events_json.is_file():
        return []
    cache = CacheStore(layout)
    try:
        data = cache.read_json(layout.terminal_fine_temporal_events_json)
    except Exception:
        return []
    from video_analyst.fine.models import TemporalEvent

    out: list[TemporalEvent] = []
    for raw in _iter_event_payloads(data):
        try:
            out.append(TemporalEvent.model_validate(raw))
        except Exception:
            continue
    return out


def _load_terminal_answers(layout: OutputLayout) -> list:
    """从 terminal_fine/batches/*.json 读取终局问题答案。"""
    from video_analyst.fine.models import RefinementQuestionAnswer

    cache = CacheStore(layout)
    out: list[RefinementQuestionAnswer] = []
    for fb_path in layout.terminal_fine_batches_out_dir.glob("fine_*.json"):
        if not fb_path.is_file():
            continue
        try:
            data = cache.read_json(fb_path)
        except Exception:
            continue
        try:
            analysis = FineBatchAnalysis.model_validate(data)
        except Exception:
            continue
        for qa in analysis.question_answers:
            qa_dict = qa.model_dump()
            qa_dict["answer_origin"] = "terminal_refinement"
            qa_dict["terminal_attempted"] = True
            try:
                out.append(RefinementQuestionAnswer.model_validate(qa_dict))
            except Exception:
                continue
    return out


def _get_video_duration(layout: OutputLayout) -> float:
    cache = CacheStore(layout)
    if layout.media_info_json.is_file():
        try:
            data = cache.read_json(layout.media_info_json)
            dur = float(data.get("duration") or 0.0)
            if dur > 0:
                return dur
        except Exception:
            pass
    # 兜底：fine_video_analysis.video_end
    if layout.fine_video_analysis_json.is_file():
        try:
            data = cache.read_json(layout.fine_video_analysis_json)
            dur = float(data.get("video_end") or 0.0)
            if dur > 0:
                return dur
        except Exception:
            pass
    return 0.0


def _get_video_metadata(layout: OutputLayout) -> dict[str, Any]:
    cache = CacheStore(layout)
    meta: dict[str, Any] = {}
    if layout.media_info_json.is_file():
        try:
            data = cache.read_json(layout.media_info_json)
            meta = dict(data)
            # 抽取核心字段
            streams = data.get("video_streams") or []
            if streams:
                meta["video"] = streams[0]
        except Exception:
            pass
    if layout.review_json.is_file():
        try:
            data = cache.read_json(layout.review_json)
            meta["subtitle_reviewed"] = bool(data.get("reviewed"))
            meta["subtitle_source"] = data.get("active")
        except Exception:
            pass
    return meta


def _make_provider(config: VisionConfig, requested_name: str) -> Any:
    """根据 provider name 构造 provider（带回退到 mock）。"""
    name = (requested_name or config.provider or "mock").lower()
    try:
        from video_analyst.fine.registry import create_fine_vision_provider

        return create_fine_vision_provider(provider=name, config=config)
    except Exception:
        # fallback
        return MockFineVisionProvider(scenario="normal")


def _run_terminal_round_if_needed(
    *,
    layout: OutputLayout,
    config: AppConfig,
    execute_terminal: bool,
    allow_provisional: bool,
) -> tuple[bool, bool, str, str]:
    """按配置决定是否执行终局 round。

    Returns: (required, executed, status, message)
    status: not_required / completed / skipped / failed
    """
    plan_path = layout.fine_second_refinement_plan_json
    if not plan_path.is_file():
        return (False, False, "not_required", "no second_refinement_plan")
    try:
        plan = validate_terminal_refinement_plan(plan_path)
    except TerminalRefinementPlanValidationError as exc:
        raise Epic5PipelineError(f"terminal plan invalid: {exc}") from exc

    intervals = plan.get("intervals") or []
    deferred = plan.get("deferred_intervals") or []
    if not intervals and not deferred:
        return (False, False, "not_required", "empty plan")

    required = True
    # In-session agent path: if agent already wrote terminal_fine/, treat as executed
    # without calling any vision provider.
    agent_terminal_done = (
        layout.terminal_fine_analysis_json.is_file()
        or (
            layout.terminal_fine_batches_out_dir.is_dir()
            and any(layout.terminal_fine_batches_out_dir.glob("*.json"))
        )
    )
    if agent_terminal_done:
        return (
            True,
            True,
            "completed",
            "terminal analysis present (in_session_agent; no vision API)",
        )

    if not execute_terminal:
        if not allow_provisional:
            raise ProvisionalReportNotAllowedError(
                "terminal refinement plan non-empty but "
                "execute_terminal=False and allow_provisional not set"
            )
        return (True, False, "skipped", "provisional report; terminal skipped")

    # Legacy path only: never invent mock vision results in the default product.
    # Prefer agent-written terminal analysis. If execute_terminal is forced and
    # no agent output exists, mark failed rather than calling external APIs.
    log.warning(
        "execute_terminal=True but no analysis/terminal_fine/ present; "
        "in-session agent must write terminal analysis. Not calling vision APIs."
    )
    return (
        required,
        False,
        "failed",
        "terminal analysis missing; agent must write analysis/terminal_fine/",
    )


def run_epic5(
    video_path: Path,
    output_dir: Path,
    config: AppConfig,
    *,
    user_request: str = "",
    execute_terminal: bool = True,
    allow_provisional: bool = False,
    refresh_report: bool = False,
    no_markdown: bool = False,
    no_csv: bool = False,
    no_evidence_copy: bool = False,
    allow_partial: bool = True,
) -> Epic5Result:
    """Epic 5 顶级入口：可选执行终局 round + 生成最终报告。

    严格不复制 Epic 3 / Epic 4 系统，全部复用。
    """
    output_dir = output_dir.expanduser().resolve()
    layout = OutputLayout.create(output_dir)

    # 0. 校验 fine_batches.json 必须存在（Epic 4 前置）
    if not layout.refinement_fine_batches_json.is_file():
        raise Epic5PipelineError(
            f"fine_batches.json missing at {layout.refinement_fine_batches_json}; "
            "run Epic 3 + Epic 4 first"
        )

    # 1. 执行终局 round（如需要）
    cfg = config.final_analysis
    if refresh_report:
        CacheStore(layout).clear_epic5_tree()
        layout = OutputLayout.create(output_dir)
    try:
        terminal_required, terminal_executed, terminal_status, terminal_msg = _run_terminal_round_if_needed(
            layout=layout,
            config=config,
            execute_terminal=execute_terminal,
            allow_provisional=allow_provisional,
        )
    except ProvisionalReportNotAllowedError as exc:
        raise Epic5PipelineError(str(exc)) from exc
    except Exception as exc:
        raise Epic5PipelineError(f"terminal round setup failed: {exc}") from exc

    # 2. 加载 round1 数据
    round1_answers = _load_round1_answers(layout)
    round1_events = _load_round1_events(layout)

    # 3. 加载 terminal 数据
    terminal_events = _load_terminal_events(layout) if terminal_executed else []
    terminal_answers = _load_terminal_answers(layout) if terminal_executed else []

    # 4. 合并
    try:
        merged_events = merge_temporal_events(
            round1_events=round1_events,
            terminal_events=terminal_events,
        )
        merged_answers = merge_question_answers(
            round1_answers=round1_answers,
            terminal_answers=terminal_answers,
        )
    except FinalEventMergeError as exc:
        raise Epic5PipelineError(f"merge failed: {exc}") from exc

    # 5. 构建 coverage
    video_duration = _get_video_duration(layout)
    coverage = build_coverage_report(layout, video_duration=video_duration)
    save_coverage_report(layout, coverage)

    # 6. 构建 shot timeline + sections
    fine_analyses = _load_fine_analyses(layout)
    # Prefer answers from loaded fine analyses when round1_answers empty
    if not round1_answers and fine_analyses:
        for a in fine_analyses.values():
            round1_answers.extend(list(a.question_answers or []))
        try:
            merged_answers = merge_question_answers(
                round1_answers=round1_answers,
                terminal_answers=terminal_answers,
            )
        except Exception:
            merged_answers = list(round1_answers)

    terminal_ranges: list[tuple[float, float]] = []
    if terminal_executed and layout.terminal_fine_analysis_json.is_file():
        try:
            tdata = CacheStore(layout).read_json(layout.terminal_fine_analysis_json)
            for iv in tdata.get("analyzed_intervals") or []:
                terminal_ranges.append((float(iv["start"]), float(iv["end"])))
        except Exception:
            pass
    if not terminal_ranges and terminal_executed:
        for a in fine_analyses.values():
            # terminal analyses live under terminal_fine; optional
            pass
    # also scan terminal fine batches
    if terminal_executed and layout.terminal_fine_batches_out_dir.is_dir():
        for path in layout.terminal_fine_batches_out_dir.glob("*.json"):
            try:
                td = CacheStore(layout).read_json(path)
                terminal_ranges.append((float(td["start"]), float(td["end"])))
            except Exception:
                continue

    try:
        shots = build_shot_timeline(
            layout=layout,
            video_duration=video_duration,
            fine_analyses=fine_analyses,
            merged_events=merged_events,
            terminal_ranges=terminal_ranges,
        )
    except ShotTimelineError as exc:
        raise Epic5PipelineError(f"shot timeline failed: {exc}") from exc

    try:
        sections = build_sections(
            layout=layout,
            shots=shots,
            merged_events=merged_events,
        )
    except SectionAggregationError as exc:
        raise Epic5PipelineError(f"sections failed: {exc}") from exc

    # 7. unresolved items
    deferred_intervals_raw = []
    if layout.fine_second_refinement_plan_json.is_file():
        try:
            plan = CacheStore(layout).read_json(layout.fine_second_refinement_plan_json)
            deferred_intervals_raw = plan.get("deferred_intervals") or []
        except Exception:
            pass
    coarse_gaps = list(coverage.failed_ranges) if hasattr(coverage, "failed_ranges") else []
    unresolved_items = collect_unresolved_items(
        merged_events=merged_events,
        round1_answers=round1_answers,
        merged_answers=merged_answers,
        deferred_intervals=deferred_intervals_raw,
        coarse_gaps=coarse_gaps,
        fine_gaps=list(coverage.failed_ranges or []),
    )

    # 8. evidence index + image copy
    evidence_index = build_evidence_index(
        layout=layout,
        fine_analyses=fine_analyses,
        merged_events=merged_events,
        shots=shots,
        terminal_origin=terminal_executed,
    )
    save_evidence_index(layout, evidence_index)
    if not no_evidence_copy:
        copy_evidence_images(
            index=evidence_index,
            images_dir=layout.final_evidence_images_dir,
            config=cfg,
        )

    # 9. 决定 report status（Epic 6：正交字段，不互相覆盖）
    from video_analyst.final.models import ReportRuntimeStatus

    # completion_status
    if terminal_required and not terminal_executed:
        completion_status = "provisional"
    else:
        completion_status = "final"
    # In-session agent is the only product vision mode
    provider_mode_rt = "in_session_agent"
    # terminal_refinement_status：completed 映射为 executed
    _terminal_map = {"completed": "executed", "completed_no_batches": "executed"}
    terminal_rt = _terminal_map.get(terminal_status, terminal_status)  # not_required / executed / skipped / failed
    # analysis_completeness
    if coverage.failed_ranges or unresolved_items:
        completeness = "partial"
    else:
        completeness = "complete"
    runtime_status = ReportRuntimeStatus(
        completion_status=completion_status,  # type: ignore[arg-type]
        provider_mode=provider_mode_rt,  # type: ignore[arg-type]
        vision_execution_mode="in_session_agent",
        terminal_refinement_status=terminal_rt,  # type: ignore[arg-type]
        analysis_completeness=completeness,  # type: ignore[arg-type]
        real_vision_analysis_used=True,
        external_api_used=False,
        api_key_used=False,
    )
    status: ReportStatus = runtime_status.to_legacy()

    # 10. aggregation
    video_metadata = _get_video_metadata(layout)
    provider_mode = "in_session_agent"
    analysis = build_final_video_analysis(
        layout=layout,
        shots=shots,
        sections=sections,
        coverage=coverage,
        merged_events=merged_events,
        answered_questions=merged_answers,
        unresolved_items=unresolved_items,
        video_metadata=video_metadata,
        analysis_request=user_request,
        status=status,
        terminal_refinement_executed=terminal_executed,
        fine_analyses=fine_analyses,
        provider_mode=provider_mode,
        real_vision_api_used=False,
        runtime_status=runtime_status,
    )

    # 11. 写入 report.json
    save_final_video_analysis_json(layout, analysis)

    # 12. 写入 CSV
    csv_paths = {}
    if not no_csv:
        csv_paths = write_csv_reports(
            layout, analysis=analysis, merged_events=merged_events
        )

    # 13. 渲染 Markdown（失败不应删除 JSON）
    md_path = None
    if not no_markdown:
        try:
            md_path = write_markdown_report(
                layout, analysis=analysis, cfg=cfg
            )
        except Exception as exc:
            log.warning("Markdown render failed; JSON preserved: %s", exc)

    # 14. run_summary.json
    summary = build_run_summary(
        layout=layout,
        provider_mode=provider_mode,
        real_api_used=False,
        terminal_refinement_required=terminal_required,
        terminal_refinement_executed=terminal_executed,
        terminal_refinement_status=terminal_status,
        status=status,
        coarse_batches_count=_count_coarse_batches(layout),
        fine_intervals_count=len(round1_events) if round1_events else 0,
        terminal_intervals_count=len(terminal_events),
        failed_ranges=coverage.failed_ranges,
        deferred_ranges=coverage.deferred_ranges,
        unresolved_count=len(unresolved_items),
        warnings=[terminal_msg],
    )
    save_run_summary(layout, summary)

    return Epic5Result(
        status=status,
        analysis=analysis,
        output_dir=output_dir,
        terminal_refinement_required=terminal_required,
        terminal_refinement_executed=terminal_executed,
        terminal_refinement_status=terminal_status,
        manifest_path=layout.final_run_summary_json,
        report_md_path=md_path,
        report_json_path=layout.final_report_json,
        evidence_index_path=layout.final_evidence_index_json,
        run_summary_path=layout.final_run_summary_json,
        shot_table_path=csv_paths.get("shot_table"),
        event_table_path=csv_paths.get("event_table"),
        unresolved_table_path=csv_paths.get("unresolved_table"),
        warnings=[terminal_msg] if terminal_status != "completed" else [],
    )


def _count_coarse_batches(layout: OutputLayout) -> int:
    cache = CacheStore(layout)
    if not layout.coarse_manifest_json.is_file():
        return 0
    try:
        data = cache.read_json(layout.coarse_manifest_json)
        return len(data.get("batches") or [])
    except Exception:
        return 0


__all__ = ["Epic5Result", "run_epic5"]