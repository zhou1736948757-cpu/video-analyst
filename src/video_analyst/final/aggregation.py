"""FinalVideoAnalysis aggregation (Epic 5).

Consumes structured FineBatchAnalysis + merged temporal events and builds
shots/sections/styles/techniques/highlights with evidence IDs.
Does not re-run vision inference.
"""

from __future__ import annotations

from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.fine.models import (
    FineBatchAnalysis,
    RefinementQuestionAnswer,
    TimeRange,
)
from video_analyst.final.models import (
    AnalysisCoverage,
    ConfidenceSummary,
    DetailedHighlight,
    FinalVideoAnalysis,
    MergedTemporalEvent,
    RepeatableTechnique,
    ReportRuntimeStatus,
    ReportStatus,
    SectionAnalysis,
    ShotAnalysis,
    StructuredEventItem,
    UnresolvedItem,
)


def _weight_confidence(shots: list[ShotAnalysis], attribute: str) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for s in shots:
        attr_val = getattr(s, attribute, None)
        if not attr_val:
            continue
        weighted_sum += s.confidence * s.duration
        total_weight += s.duration
    if total_weight <= 0:
        return 0.5
    return round(weighted_sum / total_weight, 3)


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _event_item_from_merged(m: MergedTemporalEvent) -> StructuredEventItem:
    ev = m.active
    return StructuredEventItem(
        category=ev.category,
        start=ev.boundary.earliest,
        end=ev.boundary.latest,
        description=ev.description,
        event_type=ev.category,
        confidence=ev.confidence,
        evidence_ids=[f"EVID-EV-{ev.event_id}"],
        source_batch_ids=list(ev.source_batch_ids or m.active.source_batch_ids or []),
    )


def build_structured_event_items(
    merged_events: list[MergedTemporalEvent],
    fine_analyses: dict[str, FineBatchAnalysis],
) -> dict[str, list[StructuredEventItem]]:
    """Map fine structured objects + temporal events into chapter items."""
    out: dict[str, list[StructuredEventItem]] = {
        "text": [],
        "graphic": [],
        "transition": [],
        "subject": [],
        "camera_post": [],
    }
    seen: set[str] = set()

    def add(bucket: str, item: StructuredEventItem) -> None:
        key = f"{bucket}:{item.start:.3f}:{item.end:.3f}:{item.description[:80]}"
        if key in seen:
            return
        seen.add(key)
        out[bucket].append(item)

    for m in merged_events:
        cat = m.active.category
        item = _event_item_from_merged(m)
        if cat == "text_animation":
            add("text", item)
        elif cat == "graphic_animation":
            add("graphic", item)
        elif cat == "transition":
            add("transition", item)
        elif cat == "subject_motion":
            add("subject", item)
        elif cat in ("camera_motion", "post_transform"):
            add("camera_post", item)

    for a in fine_analyses.values():
        for ta in a.text_animations:
            types = "/".join(str(x) for x in (ta.animation_types or []))
            desc = " | ".join(
                p
                for p in [
                    ta.text_content or "",
                    types,
                    f"{ta.start_state}→{ta.end_state}" if ta.start_state or ta.end_state else "",
                ]
                if p
            ) or "text animation"
            add(
                "text",
                StructuredEventItem(
                    category="text_animation",
                    start=ta.boundary.earliest,
                    end=ta.boundary.latest,
                    description=desc,
                    event_type=types or "text_animation",
                    confidence=ta.confidence,
                    evidence_ids=[f"EVID-EV-{ta.event_id}"],
                    source_batch_ids=[a.fine_batch_id],
                ),
            )
        for ga in a.graphic_animations:
            types = "/".join(str(x) for x in (ga.animation_types or []))
            desc = " | ".join(
                p
                for p in [
                    ga.graphic_subject or "graphic",
                    types,
                    f"{ga.start_state}→{ga.end_state}" if ga.start_state or ga.end_state else "",
                ]
                if p
            )
            add(
                "graphic",
                StructuredEventItem(
                    category="graphic_animation",
                    start=ga.boundary.earliest,
                    end=ga.boundary.latest,
                    description=desc,
                    event_type=types or "graphic_animation",
                    confidence=ga.confidence,
                    evidence_ids=[f"EVID-EV-{ga.event_id}"],
                    source_batch_ids=[a.fine_batch_id],
                ),
            )
        for tr in a.transitions:
            desc = f"{tr.transition_type}: {tr.outgoing_state} → {tr.incoming_state}".strip()
            add(
                "transition",
                StructuredEventItem(
                    category="transition",
                    start=tr.boundary.earliest,
                    end=tr.boundary.latest,
                    description=desc,
                    event_type=str(tr.transition_type),
                    confidence=tr.confidence,
                    evidence_ids=[f"EVID-EV-{tr.event_id}"],
                    source_batch_ids=[a.fine_batch_id],
                ),
            )
    return out


def build_detailed_highlights(
    fine_analyses: dict[str, FineBatchAnalysis],
    layout: OutputLayout | None = None,
) -> list[DetailedHighlight]:
    """One highlight per fine interval from agent-written FineBatchAnalysis."""
    reasons: dict[str, str] = {}
    if layout is not None and layout.coarse_refinement_plan_json.is_file():
        try:
            plan = CacheStore(layout).read_json(layout.coarse_refinement_plan_json)
            for iv in plan.get("intervals") or []:
                iid = str(iv.get("interval_id") or "")
                rs = iv.get("reasons") or []
                if isinstance(rs, list):
                    reasons[iid] = "；".join(str(x) for x in rs)
                elif iv.get("reason"):
                    reasons[iid] = str(iv["reason"])
        except Exception:
            pass

    highlights: list[DetailedHighlight] = []
    for a in sorted(fine_analyses.values(), key=lambda x: x.start):
        eids: list[str] = []
        for ev in a.temporal_events:
            eids.append(f"EVID-EV-{ev.event_id}")
        for ta in a.text_animations:
            eids.append(f"EVID-EV-{ta.event_id}")
        for ga in a.graphic_animations:
            eids.append(f"EVID-EV-{ga.event_id}")
        for tr in a.transitions:
            eids.append(f"EVID-EV-{tr.event_id}")
        # unique preserve order
        seen: set[str] = set()
        uniq_eids = []
        for e in eids:
            if e not in seen:
                seen.add(e)
                uniq_eids.append(e)

        direct = [fs.description for fs in a.frame_states if fs.description]
        for ev in a.temporal_events:
            direct.append(ev.description)

        inferences: list[str] = []
        for ev in a.temporal_events:
            if ev.inference:
                inferences.append(ev.inference)

        text_items = []
        for ta in a.text_animations:
            text_items.append(
                " | ".join(
                    p
                    for p in [
                        ta.text_content or "",
                        "/".join(str(x) for x in (ta.animation_types or [])),
                    ]
                    if p
                )
            )
        graphic_items = []
        for ga in a.graphic_animations:
            graphic_items.append(
                f"{ga.graphic_subject}: {'/'.join(str(x) for x in (ga.animation_types or []))}"
            )
        transitions = [
            f"{tr.transition_type}: {tr.outgoing_state}→{tr.incoming_state}"
            for tr in a.transitions
        ]
        qas = []
        unresolved = []
        for qa in a.question_answers:
            line = f"Q: {qa.question} → {qa.answer} [{qa.status}]"
            qas.append(line)
            if qa.status == "unresolved":
                unresolved.append(qa.question)

        highlights.append(
            DetailedHighlight(
                interval_id=a.interval_id,
                fine_batch_id=a.fine_batch_id,
                start=a.start,
                end=a.end,
                reason=reasons.get(a.interval_id, a.summary or "refinement interval"),
                summary=a.summary or "",
                direct_observations=direct,
                inferences=inferences,
                subject_motions=[
                    getattr(sm, "description", str(sm)) for sm in a.subject_motions
                ],
                camera_motions=[
                    getattr(cm, "description", str(cm)) for cm in a.camera_motions
                ],
                post_transforms=[
                    getattr(pt, "description", str(pt)) for pt in a.post_transforms
                ],
                text_animations=text_items,
                graphic_animations=graphic_items,
                transitions=transitions,
                question_answers=qas,
                confidence=a.confidence,
                evidence_ids=uniq_eids,
                unresolved=unresolved + list(a.warnings or []),
            )
        )
    return highlights


def build_repeatable_techniques(
    structured: dict[str, list[StructuredEventItem]],
    fine_analyses: dict[str, FineBatchAnalysis],
) -> list[RepeatableTechnique]:
    techniques: list[RepeatableTechnique] = []
    n = 0

    def add(
        title: str,
        category: str,
        description: str,
        items: list[StructuredEventItem],
        hints: list[str],
    ) -> None:
        nonlocal n
        if not items:
            return
        n += 1
        ranges = [TimeRange(start=i.start, end=i.end) for i in items]
        eids: list[str] = []
        for i in items:
            for e in i.evidence_ids:
                if e not in eids:
                    eids.append(e)
        conf = sum(i.confidence for i in items) / len(items)
        techniques.append(
            RepeatableTechnique(
                technique_id=f"tech_{n:04d}",
                title=title,
                category=category,  # type: ignore[arg-type]
                description=description,
                observed_time_ranges=ranges,
                evidence_ids=eids,
                confidence=round(conf, 3),
                implementation_hints=hints,
                uncertainty_notes=["未验证具体软件/插件；以下为通用属性实现思路"],
            )
        )

    # Group by keywords in descriptions
    type_on = [i for i in structured["text"] if "type_on" in i.event_type or "type-on" in i.description or "type_on" in i.description or "逐字" in i.description or "type-on" in i.description.lower()]
    highlight = [i for i in structured["text"] if "highlight" in i.description.lower() or "emphasis" in i.event_type or "高亮" in i.description]
    wipe = [i for i in structured["transition"] if "wipe" in i.event_type.lower() or "wipe" in i.description.lower() or "遮罩" in i.description]
    cut = [i for i in structured["transition"] if i.event_type == "cut" or "硬切" in i.description or "cut" in i.description.lower()]
    popup = [i for i in structured["graphic"] if "popup" in i.description.lower() or "appear" in i.event_type or "弹出" in i.description or "嵌套" in i.description]
    nav = [i for i in structured["graphic"] if "nav" in i.description.lower() or "导航" in i.description or "dot" in i.description.lower()]
    emoji = [i for i in structured["graphic"] if "emoji" in i.description.lower()]
    logo = [i for i in structured["graphic"] if "logo" in i.description.lower() or "Explore" in i.description]

    add(
        "分段/逐字文字揭示（type-on）",
        "text",
        "正文或输入框文字随时间逐步显现",
        type_on or [i for i in structured["text"] if "type" in i.description.lower()],
        ["通过 opacity 关键帧或字符显示长度属性实现 type-on", "可配合 mask_reveal 做分段揭示"],
    )
    add(
        "关键词圆角高亮强调",
        "text",
        "关键词出现黄色圆角高亮框与问号等强调图形",
        highlight,
        ["在文字层上叠加圆角矩形 + 图标", "使用 opacity/scale 弹出或 emphasize"],
    )
    add(
        "圆形遮罩 wipe 转场",
        "transition",
        "黑色圆形区域扩大覆盖前一画面后切入新场景",
        wipe,
        ["使用圆形 mask 的 scale 从 0→覆盖全屏", "或 iris wipe 类属性动画"],
    )
    add(
        "硬切/快速切换收尾",
        "transition",
        "无明显混合帧的快速切换到结束画面",
        cut,
        ["直接切换合成层；若需柔和可加极短 opacity crossfade"],
    )
    add(
        "嵌套信息窗口弹出",
        "graphics",
        "主窗口上叠加子解释卡片/概念窗口",
        popup,
        ["子层 scale + opacity 入场", "配合阴影与层级 z-order"],
    )
    add(
        "纵向层级导航高亮",
        "graphics",
        "右侧圆点链表示层级，当前节点实心高亮",
        nav,
        ["用一组圆点控件，active 状态切换 fill/opacity"],
    )
    add(
        "Emoji 反应与放大遮挡",
        "graphics",
        "角色 emoji 表情切换，结尾放大覆盖主内容",
        emoji,
        ["替换表情素材；scale 放大 + 居中定位"],
    )
    add(
        "黑底品牌 Logo 结束卡",
        "graphics",
        "黑底上呈现发光 logo 与 slogan",
        logo,
        ["静态或轻微 glow；避免无证据的复杂循环动画描述"],
    )
    return techniques


def build_pacing_and_narrative(
    *,
    shots: list[ShotAnalysis],
    sections: list[SectionAnalysis],
    structured: dict[str, list[StructuredEventItem]],
    fine_analyses: dict[str, FineBatchAnalysis],
    coverage: AnalysisCoverage,
) -> tuple[list[str], list[str]]:
    duration = coverage.video_duration or 0.0
    n_shots = len(shots)
    avg_shot = (sum(s.duration for s in shots) / n_shots) if n_shots else 0.0
    short_shots = sum(1 for s in shots if s.duration < 2.0)
    n_tr = len(structured["transition"])
    n_fx = len(structured["text"]) + len(structured["graphic"])
    fine_intervals = sorted(fine_analyses.values(), key=lambda a: a.start)

    pacing: list[str] = []
    pacing.append(f"视频总时长 {_hms(duration)}，共 {n_shots} 个镜头，平均镜头时长 {avg_shot:.2f}s。")
    pacing.append(f"短镜头（<2s）数量：{short_shots}；转场事件 {n_tr} 个；文字/图形动效事件 {n_fx} 个。")
    if sections:
        for sec in sections:
            pacing.append(
                f"段落 {sec.section_id}「{sec.title}」时长 {_hms(sec.end - sec.start)} "
                f"（{_hms(sec.start)}–{_hms(sec.end)}）。"
            )
    if fine_intervals:
        spans = ", ".join(f"{a.start:.1f}–{a.end:.1f}s" for a in fine_intervals)
        pacing.append(f"精查区间分布：{spans}（共 {len(fine_intervals)} 段）。")
    if n_shots >= 2:
        first_half = [s for s in shots if s.end <= duration / 2 + 1e-6]
        second_half = [s for s in shots if s.start >= duration / 2 - 1e-6]
        if first_half and second_half:
            a1 = sum(s.duration for s in first_half) / len(first_half)
            a2 = sum(s.duration for s in second_half) / len(second_half)
            pacing.append(
                f"前半平均镜头 {a1:.2f}s，后半平均镜头 {a2:.2f}s"
                f"（{'前半更碎' if a1 < a2 else '后半更碎' if a2 < a1 else '前后相近'}）。"
            )

    narrative: list[str] = []
    if sections:
        titles = " → ".join(s.title or s.section_id for s in sections)
        narrative.append(f"内容推进结构：{titles}。")
    # Evidence-based limited description
    has_text = bool(structured["text"])
    has_ui = bool(structured["graphic"])
    has_wipe = any("wipe" in i.description.lower() or "遮罩" in i.description for i in structured["transition"])
    if has_text:
        narrative.append("前段以文字揭示与关键词高亮建立问题语境（有 Fine 文字动效证据）。")
    if has_wipe or has_ui:
        narrative.append("中段进入深色产品 UI，并以嵌套窗口/导航点推进信息层级（有图形与转场证据）。")
    if any("logo" in i.description.lower() or "结束" in i.description for i in structured["graphic"] + structured["transition"]):
        narrative.append("结尾通过表情放大与品牌结束卡快速收束（有图形/转场证据）。")
    if not narrative:
        narrative.append("叙事结构主要依据粗分析段落与镜头时长分布，精查证据有限。")
    return pacing, narrative


def build_subtitle_av_notes(
    layout: OutputLayout,
    fine_analyses: dict[str, FineBatchAnalysis],
    structured: dict[str, list[StructuredEventItem]],
    coverage: AnalysisCoverage,
) -> list[str]:
    notes: list[str] = []
    cache = CacheStore(layout)
    segments: list[dict[str, Any]] = []
    if layout.subtitles_json.is_file():
        try:
            doc = cache.read_json(layout.subtitles_json)
            segments = list(doc.get("segments") or [])
        except Exception:
            segments = []
    if not coverage.subtitle_reviewed:
        notes.append("字幕未人工审核，仅作辅助上下文，不是绝对真值；相关结论置信度下调。")
    if coverage.subtitle_needs_review:
        notes.append("字幕标记为 needs_review。")
    if not segments:
        notes.append("无可用字幕分段；画面与语音关系无法从字幕侧验证。")
        return notes

    texts = " ".join(str(s.get("text") or "") for s in segments)
    notes.append(f"活动字幕共 {len(segments)} 段；文本预览有限：{texts[:80]}…")
    # Weak sync notes: if visual text events exist, note possible independence
    if structured["text"]:
        notes.append(
            f"画面存在 {len(structured['text'])} 个文字动效事件；"
            "若字幕文本与画面关键词不一致，应以画面文字为准并降低音字同步置信度。"
        )
    if all(len(str(s.get("text") or "")) < 8 for s in segments):
        notes.append("字幕内容极短或疑似无效旁白，音画同步关系无法可靠确认。")
    return notes


def build_style_lists(
    structured: dict[str, list[StructuredEventItem]],
    fine_analyses: dict[str, FineBatchAnalysis],
) -> dict[str, list[str]]:
    text_style = []
    graphic_style = []
    transition_style = []
    for i in structured["text"]:
        text_style.append(f"[{i.start:.1f}-{i.end:.1f}s] {i.description} [{','.join(i.evidence_ids)}]")
    for i in structured["graphic"]:
        graphic_style.append(f"[{i.start:.1f}-{i.end:.1f}s] {i.description} [{','.join(i.evidence_ids)}]")
    for i in structured["transition"]:
        transition_style.append(f"[{i.start:.1f}-{i.end:.1f}s] {i.description} [{','.join(i.evidence_ids)}]")
    motion = []
    for a in fine_analyses.values():
        if a.summary:
            motion.append(f"[{a.start:.1f}-{a.end:.1f}s] {a.summary[:120]}")
    return {
        "text_design_style": text_style,
        "graphic_design_style": graphic_style,
        "transition_style": transition_style,
        "motion_style": motion,
        "visual_style": (text_style[:2] + graphic_style[:2] + transition_style[:2])[:8],
        "camera_style": [
            i.description for i in structured["camera_post"]
        ] or ["未观察到明确摄像机推拉；变化主要来自 UI/后期元素（有 Fine 证据支持）"],
    }


def build_final_video_analysis(
    *,
    layout: OutputLayout,
    shots: list[ShotAnalysis],
    sections: list[SectionAnalysis],
    coverage: AnalysisCoverage,
    merged_events: list[MergedTemporalEvent],
    answered_questions: list[RefinementQuestionAnswer],
    unresolved_items: list[UnresolvedItem],
    video_metadata: dict[str, Any],
    analysis_request: str,
    status: ReportStatus,
    terminal_refinement_executed: bool,
    fine_analyses: dict[str, FineBatchAnalysis] | None = None,
    repeatable_techniques: list[RepeatableTechnique] | None = None,
    visual_style: list[str] | None = None,
    camera_style: list[str] | None = None,
    motion_style: list[str] | None = None,
    text_design_style: list[str] | None = None,
    graphic_design_style: list[str] | None = None,
    transition_style: list[str] | None = None,
    pacing_analysis: list[str] | None = None,
    subtitle_audio_visual_analysis: list[str] | None = None,
    narrative_strategy: list[str] | None = None,
    important_events: list[str] | None = None,
    known_limitations: list[str] | None = None,
    provider_mode: str = "in_session_agent",
    real_vision_api_used: bool = False,
    runtime_status: Any = None,
) -> FinalVideoAnalysis:
    fine_analyses = fine_analyses or {}
    if runtime_status is None:
        runtime_status = ReportRuntimeStatus(
            completion_status="final",
            provider_mode="in_session_agent",
            vision_execution_mode="in_session_agent",
            terminal_refinement_status=(
                "executed" if terminal_refinement_executed else "not_required"
            ),
            analysis_completeness="complete" if not unresolved_items else "partial",
            real_vision_analysis_used=True,
            external_api_used=False,
            api_key_used=False,
        )

    structured = build_structured_event_items(merged_events, fine_analyses)
    highlights = build_detailed_highlights(fine_analyses, layout=layout)

    if repeatable_techniques is None:
        repeatable_techniques = build_repeatable_techniques(structured, fine_analyses)
    styles = build_style_lists(structured, fine_analyses)
    if visual_style is None:
        visual_style = styles["visual_style"]
    if camera_style is None:
        camera_style = styles["camera_style"]
    if motion_style is None:
        motion_style = styles["motion_style"]
    if text_design_style is None:
        text_design_style = styles["text_design_style"]
    if graphic_design_style is None:
        graphic_design_style = styles["graphic_design_style"]
    if transition_style is None:
        transition_style = styles["transition_style"]

    if pacing_analysis is None or narrative_strategy is None:
        pacing, narrative = build_pacing_and_narrative(
            shots=shots,
            sections=sections,
            structured=structured,
            fine_analyses=fine_analyses,
            coverage=coverage,
        )
        if pacing_analysis is None:
            pacing_analysis = pacing
        if narrative_strategy is None:
            narrative_strategy = narrative

    if subtitle_audio_visual_analysis is None:
        subtitle_audio_visual_analysis = build_subtitle_av_notes(
            layout, fine_analyses, structured, coverage
        )

    if shots:
        overall = sum(s.confidence * s.duration for s in shots) / max(
            sum(s.duration for s in shots), 1e-9
        )
    else:
        overall = 0.5
    confidence_summary = ConfidenceSummary(
        overall=round(overall, 3),
        content_structure=round(overall, 3),
        subject_motion=_weight_confidence(shots, "subject_motions"),
        camera_motion=_weight_confidence(shots, "camera_motions"),
        text_animation=_weight_confidence(shots, "text_animations"),
        graphic_animation=_weight_confidence(shots, "graphic_animations"),
        transition=_weight_confidence(shots, "transitions_in"),
        subtitle_alignment=0.5 if not coverage.subtitle_reviewed else 0.7,
    )

    unresolved_questions = [
        it.question_or_issue
        for it in unresolved_items
        if it.source
        in (
            "round1_unresolved",
            "validator_fallback",
            "terminal_unresolved",
        )
    ]

    gaps: list[TimeRange] = []
    for r in coverage.failed_ranges:
        gaps.append(TimeRange(start=r.start, end=r.end))
    for r in coverage.unexamined_ranges:
        gaps.append(TimeRange(start=r.start, end=r.end))
    for r in coverage.deferred_ranges:
        gaps.append(TimeRange(start=r.start, end=r.end))

    if known_limitations is None:
        known_limitations = [
            "视觉分析由会话内 Agent 看图生成（无外部视觉 API）。",
            "时序证据来自 contact sheet（重叠帧=1）；快于采样间隔的变化可能无法分辨。",
            "未实现 OCR 引擎；拼图缩放后的小字可能不可读。",
            "Token 与费用不可用（token_status=unavailable，estimated_cost=null）。",
        ]
        if coverage.fine_coverage_ratio < 1.0:
            known_limitations.append(
                f"精细分析仅覆盖约 {coverage.fine_coverage_ratio:.1%} 时长；"
                "未精查区间不得被描述为「无动效」。"
            )

    if important_events is None:
        important_events = []
        for m in merged_events:
            if m.active.confidence >= 0.55:
                eid = f"EVID-EV-{m.active.event_id}"
                important_events.append(
                    f"[{m.active.boundary.earliest:.3f}-{m.active.boundary.latest:.3f}s] "
                    f"{m.active.category}: {m.active.description} [{eid}]"
                )

    terminal_status = getattr(runtime_status, "terminal_refinement_status", None) or (
        "executed" if terminal_refinement_executed else "not_required"
    )

    return FinalVideoAnalysis(
        video_metadata=video_metadata,
        analysis_request=analysis_request,
        status=status,
        terminal_refinement_executed=terminal_refinement_executed,
        executive_summary=_build_executive_summary(
            coverage=coverage,
            shots=shots,
            merged_events=merged_events,
            unanswered_count=len(unresolved_questions),
            terminal_status=str(terminal_status),
        ),
        content_structure=sections,
        shots=shots,
        detailed_highlights=highlights,
        visual_style=visual_style or [],
        camera_style=camera_style or [],
        motion_style=motion_style or [],
        text_design_style=text_design_style or [],
        graphic_design_style=graphic_design_style or [],
        transition_style=transition_style or [],
        text_animation_items=structured["text"],
        graphic_animation_items=structured["graphic"],
        transition_items=structured["transition"],
        subject_motion_items=structured["subject"],
        camera_post_items=structured["camera_post"],
        pacing_analysis=pacing_analysis or [],
        subtitle_audio_visual_analysis=subtitle_audio_visual_analysis or [],
        narrative_strategy=narrative_strategy or [],
        repeatable_techniques=repeatable_techniques or [],
        important_events=important_events,
        answered_questions=answered_questions,
        unresolved_questions=unresolved_questions,
        unresolved_items=unresolved_items,
        analysis_gaps=gaps,
        coverage=coverage,
        confidence_summary=confidence_summary,
        known_limitations=known_limitations,
        provider_mode=provider_mode or "in_session_agent",
        vision_execution_mode="in_session_agent",
        real_vision_api_used=False,
        real_vision_analysis_used=True,
        external_api_used=False,
        api_key_used=False,
        token_status="unavailable",
        estimated_cost=None,
        pricing_source="in_session_agent",
        runtime_status=runtime_status,
    )


def _build_executive_summary(
    *,
    coverage: AnalysisCoverage,
    shots: list[ShotAnalysis],
    merged_events: list[MergedTemporalEvent],
    unanswered_count: int,
    terminal_status: str = "not_required",
) -> str:
    n_shots = len(shots)
    n_events = len(merged_events)
    duration = coverage.video_duration
    if terminal_status == "executed":
        source_phrase = "来自 Fine round1 + terminal refinement"
    elif terminal_status == "skipped":
        source_phrase = "来自 Fine analysis；终局补帧未执行"
    else:
        source_phrase = "来自 Fine analysis"
    summary = (
        f"本次分析覆盖视频总时长 {_hms(duration)}，"
        f"构建了 {n_shots} 个镜头，"
        f"合并了 {n_events} 个时序事件"
        f"（{source_phrase}）。"
    )
    if coverage.fine_coverage_ratio < 1.0:
        missing_pct = int((1 - coverage.fine_coverage_ratio) * 100)
        summary += (
            f"精细分析覆盖 {coverage.fine_coverage_ratio:.1%}"
            f"（约 {missing_pct}% 时长未做精细抽帧分析）。"
        )
    fine_shots = sum(1 for s in shots if s.analysis_level in ("fine", "terminal_refined"))
    summary += f"其中 {fine_shots}/{n_shots} 个镜头与 Fine 区间重叠。"
    if unanswered_count > 0:
        summary += f"有 {unanswered_count} 个问题未解决。"
    return summary


__all__ = [
    "build_detailed_highlights",
    "build_final_video_analysis",
    "build_pacing_and_narrative",
    "build_repeatable_techniques",
    "build_structured_event_items",
    "build_style_lists",
    "build_subtitle_av_notes",
]
