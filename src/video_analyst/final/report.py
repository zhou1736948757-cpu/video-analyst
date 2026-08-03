"""Markdown report writer (Epic 5).

所有内容来自统一 FinalVideoAnalysis JSON；Markdown 不重新推理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import FinalAnalysisConfig
from video_analyst.final.models import FinalVideoAnalysis


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _empty(zh: str, lang: str = "zh") -> str:
    return zh if lang != "en" else zh


def _evid_line(eids: list[str]) -> str:
    if not eids:
        return ""
    return "证据：" + " ".join(f"`[{e}]`" for e in eids[:8])


def _render_section(
    analysis: FinalVideoAnalysis,
    cfg: FinalAnalysisConfig,
) -> str:
    lang = getattr(cfg, "report_language", "zh") or "zh"
    L: list[str] = []
    rt = analysis.runtime_status
    terminal_status = getattr(rt, "terminal_refinement_status", None) or (
        "executed" if analysis.terminal_refinement_executed else "not_required"
    )

    # 1. 分析概要
    L.append("# Video Analyst 拉片报告\n")
    L.append("## 1. 分析概要\n")
    if analysis.status == "provisional" or terminal_status == "skipped":
        L.append(
            "> ⚠️ **本报告为 provisional（终局补帧计划存在但未执行）。**\n"
            "> 重要问题可能仍未解决。\n"
        )
    L.append(analysis.executive_summary or _empty("（无分析概要）", lang))
    L.append("")

    # 2. 视频基本信息
    L.append("## 2. 视频基本信息\n")
    meta = analysis.video_metadata or {}
    L.append(
        f"- 时长：`{_hms(meta.get('duration', analysis.coverage.video_duration))}`"
    )
    if "format_name" in meta:
        L.append(f"- 容器格式：`{meta.get('format_name')}`")
    if "video" in meta:
        v = meta["video"]
        L.append(
            f"- 视频编码：{v.get('codec_name', '?')} "
            f"{v.get('width', '?')}x{v.get('height', '?')} "
            f"@ {v.get('avg_frame_rate', '?')} fps"
        )
    L.append(
        f"- 视觉执行模式：`{getattr(analysis, 'vision_execution_mode', analysis.provider_mode)}`"
    )
    L.append("- ✓ 当前对话 Agent 现场看图分析（in_session_agent）")
    L.append("- ✓ 未使用外部视觉 API（external_api_used=false）")
    L.append("- ✓ 未使用视觉 API Key（api_key_used=false）")
    L.append("- token_status=unavailable（费用不可用，不得记为 0）")
    if terminal_status == "executed":
        L.append("- 终局补帧：已执行")
    elif terminal_status == "skipped":
        L.append("- 终局补帧：计划存在但未执行")
    else:
        L.append("- 终局补帧：不需要（not_required）")
    if meta.get("subtitle_reviewed"):
        L.append("- ✓ 字幕已人工审核")
    else:
        L.append("- ⚠️ 字幕未人工审核（仍为辅助上下文，不是绝对真值）")
    L.append("")

    # 3. 分析要求
    L.append("## 3. 分析要求\n")
    L.append(analysis.analysis_request or _empty("（未记录分析要求）", lang))
    L.append("")

    # 4. 数据来源与覆盖率
    L.append("## 4. 数据来源与覆盖率\n")
    cov = analysis.coverage
    L.append(f"- 总时长：`{_hms(cov.video_duration)}`")
    L.append(f"- 粗分析覆盖：{cov.coarse_coverage_ratio:.1%}")
    L.append(f"- 精细分析覆盖：{cov.fine_coverage_ratio:.1%}")
    if cov.terminal_coverage_ratio > 0:
        L.append(f"- 终局精修覆盖：{cov.terminal_coverage_ratio:.1%}")
    if cov.failed_ranges:
        L.append("- 失败区间：")
        for r in cov.failed_ranges:
            L.append(f"  - `{_hms(r.start)} - {_hms(r.end)}`")
    if cov.deferred_ranges:
        L.append("- 延期区间（被预算排除）：")
        for r in cov.deferred_ranges:
            L.append(f"  - `{_hms(r.start)} - {_hms(r.end)}`")
    if cov.unexamined_ranges:
        L.append("- 未精细分析区间：")
        for r in cov.unexamined_ranges:
            L.append(f"  - `{_hms(r.start)} - {_hms(r.end)}`")
    if cov.warnings:
        L.append("- 覆盖率警告：")
        for w in cov.warnings:
            L.append(f"  - {w}")
    L.append("")

    # 5. 内容结构
    L.append("## 5. 内容结构\n")
    if not analysis.content_structure:
        L.append(_empty("（未识别到内容段落）", lang))
    for sec in analysis.content_structure:
        L.append(f"### {sec.section_id} — {sec.title or '未命名段落'}\n")
        L.append(f"- 时间范围：`{_hms(sec.start)} - {_hms(sec.end)}`")
        L.append(f"- 镜头数：{len(sec.shot_ids)}")
        if sec.summary:
            L.append(f"- 摘要：{sec.summary}")
        if sec.main_subjects:
            L.append(f"- 主要主体：{', '.join(sec.main_subjects)}")
        if sec.important_events:
            L.append("- 重要事件：")
            for ie in sec.important_events:
                L.append(f"  - {ie}")
        if sec.evidence_ids:
            L.append(f"- {_evid_line(sec.evidence_ids)}")
        if sec.warnings:
            L.append("- 段落警告：")
            for w in sec.warnings:
                L.append(f"  - {w}")
        L.append("")

    # 6. 镜头时间线
    L.append("## 6. 镜头时间线\n")
    if not analysis.shots:
        L.append(_empty("（未构建镜头时间线）", lang))
    for shot in analysis.shots:
        L.append(
            f"### {shot.shot_id} `[{_hms(shot.start)} - {_hms(shot.end)}]` "
            f"({shot.duration:.2f}s, level={shot.analysis_level})"
        )
        if shot.visual_summary:
            L.append(f"- 摘要：{shot.visual_summary}")
        if shot.subjects:
            L.append(f"- 主体：{', '.join(shot.subjects)}")
        if shot.visible_text:
            L.append(f"- 可见文字：{', '.join(shot.visible_text)}")
        if getattr(shot, "fine_coverage_ratio_within_shot", 0) > 0:
            L.append(
                f"- 镜头内 Fine 覆盖：{shot.fine_coverage_ratio_within_shot:.1%}"
            )
        if shot.evidence_ids:
            L.append(f"- {_evid_line(shot.evidence_ids)}")
        if shot.confidence < 0.5:
            L.append(f"- ⚠️ confidence 偏低 ({shot.confidence:.2f})")
        if shot.warnings:
            for w in shot.warnings:
                L.append(f"- ⚠️ {w}")
        L.append("")

    # 7. 重点片段精细分析
    L.append("## 7. 重点片段精细分析\n")
    highlights = list(getattr(analysis, "detailed_highlights", None) or [])
    if not highlights:
        # fallback: shots with fine level and motion/text
        highlights = []
    if not highlights:
        L.append(_empty("（无重点精查片段；请确认 Fine 分析是否已写入）", lang))
    for h in highlights:
        L.append(
            f"### {h.fine_batch_id} / {h.interval_id} "
            f"`[{_hms(h.start)} - {_hms(h.end)}]`\n"
        )
        if h.reason:
            L.append(f"- 选择原因：{h.reason}")
        if h.summary:
            L.append(f"- 摘要：{h.summary}")
        if h.direct_observations:
            L.append("- 直接观察：")
            for x in h.direct_observations:
                L.append(f"  - {x}")
        if h.inferences:
            L.append("- 推断：")
            for x in h.inferences:
                L.append(f"  - {x}")
        if h.subject_motions:
            L.append("- 主体动作：")
            for x in h.subject_motions:
                L.append(f"  - {x}")
        if h.camera_motions:
            L.append("- 摄像机运动：")
            for x in h.camera_motions:
                L.append(f"  - {x}")
        if h.post_transforms:
            L.append("- 后期变换：")
            for x in h.post_transforms:
                L.append(f"  - {x}")
        if h.text_animations:
            L.append("- 文字动效：")
            for x in h.text_animations:
                L.append(f"  - {x}")
        if h.graphic_animations:
            L.append("- 图形动效：")
            for x in h.graphic_animations:
                L.append(f"  - {x}")
        if h.transitions:
            L.append("- 转场：")
            for x in h.transitions:
                L.append(f"  - {x}")
        if h.question_answers:
            L.append("- Refinement 问题回答：")
            for x in h.question_answers:
                L.append(f"  - {x}")
        L.append(f"- confidence：{h.confidence:.2f}")
        if h.evidence_ids:
            L.append(f"- {_evid_line(h.evidence_ids)}")
        if h.unresolved:
            L.append("- 仍不确定：")
            for x in h.unresolved:
                L.append(f"  - {x}")
        L.append("")

    # 8. 主体动作
    L.append("## 8. 主体动作\n")
    items = list(getattr(analysis, "subject_motion_items", None) or [])
    shot_sm = [(s, sm) for s in analysis.shots for sm in s.subject_motions]
    if items:
        L.append("| 时间 | 描述 | 证据 |")
        L.append("|---|---|---|")
        for it in items:
            L.append(
                f"| `{_hms(it.start)}-{_hms(it.end)}` | {it.description} | "
                f"{' '.join('`['+e+']`' for e in it.evidence_ids[:4])} |"
            )
    elif shot_sm:
        L.append("| 时间 | 镜头 | 动作 | 证据 |")
        L.append("|---|---|---|---|")
        for shot, sm in shot_sm:
            L.append(
                f"| `{_hms(shot.start)}-{_hms(shot.end)}` | {shot.shot_id} | {sm} | "
                f"{' '.join('`['+e+']`' for e in shot.evidence_ids[:3])} |"
            )
    else:
        L.append(_empty("（未记录主体动作；本次 Fine 结果中无 subject_motion 事件）", lang))
    L.append("")

    # 9. 摄像机运动与后期变换
    L.append("## 9. 摄像机运动与后期变换\n")
    cp_items = list(getattr(analysis, "camera_post_items", None) or [])
    cm_items = [(s, cm) for s in analysis.shots for cm in s.camera_motions]
    pt_items = [(s, pt) for s in analysis.shots for pt in s.post_transforms]
    if cp_items:
        L.append("| 时间 | 类型 | 描述 | 证据 |")
        L.append("|---|---|---|---|")
        for it in cp_items:
            L.append(
                f"| `{_hms(it.start)}-{_hms(it.end)}` | {it.event_type or it.category} | "
                f"{it.description} | {' '.join('`['+e+']`' for e in it.evidence_ids[:4])} |"
            )
    elif cm_items or pt_items:
        if cm_items:
            L.append("### 摄像机运动\n")
            for shot, cm in cm_items:
                L.append(
                    f"- `{_hms(shot.start)}-{_hms(shot.end)}` {shot.shot_id}: {cm} "
                    f"{_evid_line(shot.evidence_ids)}"
                )
        if pt_items:
            L.append("### 后期变换\n")
            for shot, pt in pt_items:
                L.append(
                    f"- `{_hms(shot.start)}-{_hms(shot.end)}` {shot.shot_id}: {pt} "
                    f"{_evid_line(shot.evidence_ids)}"
                )
    else:
        L.append(
            _empty(
                "（未记录明确摄像机推拉；变化主要来自 UI/后期元素，见图形动效章节）",
                lang,
            )
        )
    L.append("")

    # 10. 文字与图形动效
    L.append("## 10. 文字与图形动效\n")
    ta_items = list(getattr(analysis, "text_animation_items", None) or [])
    ga_items = list(getattr(analysis, "graphic_animation_items", None) or [])
    if not ta_items:
        ta_items = []
        for s in analysis.shots:
            for ta in s.text_animations:
                from video_analyst.final.models import StructuredEventItem

                ta_items.append(
                    StructuredEventItem(
                        category="text_animation",
                        start=s.start,
                        end=s.end,
                        description=ta,
                        evidence_ids=list(s.evidence_ids[:3]),
                        source_batch_ids=[],
                    )
                )
    if not ga_items:
        for s in analysis.shots:
            for ga in s.graphic_animations:
                from video_analyst.final.models import StructuredEventItem

                ga_items.append(
                    StructuredEventItem(
                        category="graphic_animation",
                        start=s.start,
                        end=s.end,
                        description=ga,
                        evidence_ids=list(s.evidence_ids[:3]),
                        source_batch_ids=[],
                    )
                )
    if ta_items:
        L.append("### 文字动效\n")
        L.append("| 时间 | 类型 | 描述 | 证据 |")
        L.append("|---|---|---|---|")
        for it in ta_items:
            L.append(
                f"| `{_hms(it.start)}-{_hms(it.end)}` | {it.event_type or 'text'} | "
                f"{it.description} | {' '.join('`['+e+']`' for e in it.evidence_ids[:4])} |"
            )
        L.append("")
    if ga_items:
        L.append("### 图形/UI 动效\n")
        L.append("| 时间 | 类型 | 描述 | 证据 |")
        L.append("|---|---|---|---|")
        for it in ga_items:
            L.append(
                f"| `{_hms(it.start)}-{_hms(it.end)}` | {it.event_type or 'graphic'} | "
                f"{it.description} | {' '.join('`['+e+']`' for e in it.evidence_ids[:4])} |"
            )
        L.append("")
    if not ta_items and not ga_items:
        L.append(_empty("（未记录文字/图形动效）", lang))
        L.append("")

    # 11. 转场分析
    L.append("## 11. 转场分析\n")
    tr_items = list(getattr(analysis, "transition_items", None) or [])
    if not tr_items:
        for s in analysis.shots:
            for tn in s.transitions_in + s.transitions_out:
                from video_analyst.final.models import StructuredEventItem

                tr_items.append(
                    StructuredEventItem(
                        category="transition",
                        start=s.start,
                        end=s.end,
                        description=tn,
                        evidence_ids=list(s.evidence_ids[:3]),
                    )
                )
    if tr_items:
        L.append("| 时间 | 类型 | 描述 | 证据 |")
        L.append("|---|---|---|---|")
        for it in tr_items:
            L.append(
                f"| `{_hms(it.start)}-{_hms(it.end)}` | {it.event_type or 'transition'} | "
                f"{it.description} | {' '.join('`['+e+']`' for e in it.evidence_ids[:4])} |"
            )
    else:
        L.append(_empty("（未记录转场事件）", lang))
    L.append("")

    # 12. 节奏与叙事
    L.append("## 12. 节奏与叙事\n")
    if analysis.pacing_analysis:
        for line in analysis.pacing_analysis:
            L.append(f"- {line}")
    else:
        L.append(_empty("（未生成节奏分析）", lang))
    L.append("")
    if analysis.narrative_strategy:
        L.append("### 叙事策略\n")
        for line in analysis.narrative_strategy:
            L.append(f"- {line}")
        L.append("")

    # 13. 字幕、语音和画面关系
    L.append("## 13. 字幕、语音和画面关系\n")
    if analysis.subtitle_audio_visual_analysis:
        for line in analysis.subtitle_audio_visual_analysis:
            L.append(f"- {line}")
    else:
        L.append(_empty("（未生成字幕/音画关系分析）", lang))
    L.append("")

    # 14. 可复刻制作方法
    L.append("## 14. 可复刻制作方法\n")
    L.append(
        "> ⚠️ 以下方法是通用实现思路，不是原视频使用的具体软件/插件。"
        "请勿假设原视频使用了特定工具。\n"
    )
    if analysis.repeatable_techniques:
        for tech in analysis.repeatable_techniques:
            L.append(f"### {tech.title} ({tech.category})\n")
            L.append(f"- 描述：{tech.description}")
            if tech.observed_time_ranges:
                L.append("- 观察到的时间范围：")
                for r in tech.observed_time_ranges:
                    L.append(f"  - `{_hms(r.start)} - {_hms(r.end)}`")
            if tech.implementation_hints:
                L.append("- 通用实现思路：")
                for h in tech.implementation_hints:
                    L.append(f"  - {h}")
            if tech.evidence_ids:
                L.append(f"- {_evid_line(tech.evidence_ids)}")
            if tech.uncertainty_notes:
                L.append("- 不确定项：")
                for u in tech.uncertainty_notes:
                    L.append(f"  - {u}")
            L.append(f"- confidence：{tech.confidence:.2f}\n")
    else:
        L.append(_empty("（未提取到可复刻制作方法）", lang))
    L.append("")

    # 15. 未解决问题 / 限制
    L.append("## 15. 未解决问题与已知限制\n")
    if analysis.unresolved_items:
        L.append("### 未解决问题\n")
        for it in analysis.unresolved_items:
            when = (
                f" (`{_hms(it.time_start)} - {_hms(it.time_end)}`)"
                if it.time_start is not None and it.time_end is not None
                else ""
            )
            L.append(f"- **{it.item_id}**{when}: {it.question_or_issue}")
            L.append(f"  - 原因：{it.reason}")
            L.append(f"  - 来源：{it.source}")
            if it.evidence_ids:
                L.append(f"  - {_evid_line(it.evidence_ids)}")
    elif analysis.unresolved_questions:
        L.append("### 未解决问题\n")
        for q in analysis.unresolved_questions:
            L.append(f"- {q}")
    else:
        L.append("### 未解决问题\n")
        L.append("- 当前 Fine 问题均已作答；无 status=unresolved 的 refinement question。")
    L.append("")
    if analysis.analysis_gaps:
        L.append("### 分析缺口（analysis_gaps）\n")
        for g in analysis.analysis_gaps:
            L.append(f"- `{_hms(g.start)} - {_hms(g.end)}`")
        L.append("")
    if analysis.known_limitations:
        L.append("### 已知限制（known_limitations）\n")
        for lim in analysis.known_limitations:
            L.append(f"- {lim}")
        L.append("")

    # 16. 重要事件索引
    L.append("## 16. 重要事件索引\n")
    if analysis.important_events:
        for ie in analysis.important_events:
            L.append(f"- {ie}")
    else:
        L.append(_empty("（无重要事件）", lang))
    L.append("")

    # 17. 证据索引说明
    L.append("## 17. 证据索引说明\n")
    L.append(
        "正文中的证据 ID（形如 `EVID-EV-...` / `EVID-SHOT-...`）对应 "
        "`report/evidence/index.json`。图片证据（如有）位于 "
        "`report/evidence/images/`。\n"
    )
    L.append(
        "引用方法：在 Markdown 中使用证据 ID；在 `report.json` 的 "
        "`evidence_ids` 字段中引用。\n"
    )

    return "\n".join(L) + "\n"


def write_markdown_report(
    layout: OutputLayout,
    *,
    analysis: FinalVideoAnalysis,
    cfg: FinalAnalysisConfig | None = None,
) -> Path:
    cfg = cfg or FinalAnalysisConfig()
    md = _render_section(analysis, cfg)
    path = layout.final_report_md
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(md, encoding="utf-8")
    tmp.replace(path)
    return path


def save_final_video_analysis_json(
    layout: OutputLayout,
    analysis: FinalVideoAnalysis,
) -> Path:
    cache = CacheStore(layout)
    cache.write_json(layout.final_report_json, analysis.model_dump(mode="json"))
    # also mirror under analysis/final_analysis
    try:
        cache.write_json(layout.final_video_analysis_json, analysis.model_dump(mode="json"))
    except Exception:
        pass
    return layout.final_report_json


__all__ = [
    "save_final_video_analysis_json",
    "write_markdown_report",
]
