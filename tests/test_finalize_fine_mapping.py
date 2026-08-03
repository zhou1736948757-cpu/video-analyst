"""Regression: Fine analysis must flow into Final report chapters."""

from __future__ import annotations

import json
from pathlib import Path

from video_analyst.cache.store import OutputLayout
from video_analyst.config import AppConfig
from video_analyst.final.orchestrator import run_epic5
from video_analyst.fine.models import (
    FineBatchAnalysis,
    FineEvidenceReference,
    FineVideoAnalysis,
    RefinementQuestionAnswer,
    TemporalBoundary,
    TemporalEvent,
    TextAnimationAnalysis,
    GraphicAnimationAnalysis,
    TransitionAnalysis,
    TimeRange,
)
from video_analyst.coarse_analysis.models import VideoStructure, VideoSection


def _write_agent_fine_fixture(root: Path) -> None:
    layout = OutputLayout.create(root)
    # media duration 60s
    (layout.media_info_json).write_text(
        json.dumps({"duration": 60.0, "format_name": "mp4", "video_streams": []}),
        encoding="utf-8",
    )
    # coarse structure 3 sections
    structure = VideoStructure(
        video_start=0.0,
        video_end=60.0,
        overall_summary="fixture video",
        sections=[
            VideoSection(
                section_id="sec_01",
                start=0.0,
                end=20.0,
                title="intro",
                summary="intro",
                confidence=0.7,
            ),
            VideoSection(
                section_id="sec_02",
                start=20.0,
                end=45.0,
                title="demo",
                summary="demo",
                confidence=0.7,
            ),
            VideoSection(
                section_id="sec_03",
                start=45.0,
                end=60.0,
                title="outro",
                summary="outro",
                confidence=0.7,
            ),
        ],
    )
    layout.coarse_structure_json.write_text(
        structure.model_dump_json(indent=2), encoding="utf-8"
    )
    # refinement fine_batches index (required by orchestrator gate)
    layout.refinement_fine_batches_json.parent.mkdir(parents=True, exist_ok=True)
    batches = []
    intervals = [
        (2.0, 7.0, "interval_0001"),
        (17.0, 20.5, "interval_0002"),
        (27.0, 33.0, "interval_0003"),
        (43.0, 49.0, "interval_0004"),
        (51.0, 56.0, "interval_0005"),
    ]
    for i, (s, e, iid) in enumerate(intervals, start=1):
        fb_id = f"fine_interval_{i:04d}"
        batches.append(
            {
                "fine_batch_id": fb_id,
                "interval_id": iid,
                "interval": {
                    "interval_id": iid,
                    "start": s,
                    "end": e,
                    "requested_fps": 10,
                    "priority": "high",
                },
                "frames": [{"frame_id": f"{fb_id}_f0", "timestamp_seconds": s}],
                "contact_sheets": [
                    {
                        "sheet_id": f"{iid}__sheet_0001",
                        "path": str(layout.refinement_contact_sheets_dir / iid / "sheet_0001.jpg"),
                        "frame_ids": [f"{fb_id}_f0"],
                        "timestamps": [s, e],
                    }
                ],
                "detail_frames": [],
                "refinement_questions": [f"Q{i}?"],
            }
        )
    layout.refinement_fine_batches_json.write_text(
        json.dumps({"schema_version": "1.0", "batches": batches}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5 fine batch analyses with 9 temporal events total
    def te(eid, cat, s, e, desc):
        return TemporalEvent(
            event_id=eid,
            category=cat,
            boundary=TemporalBoundary(earliest=s, latest=e, precision="bounded"),
            description=desc,
            evidence=FineEvidenceReference(frame_ids=[f"{eid}_f"], timestamps=[s]),
            confidence=0.8,
            source_batch_ids=[],
        )

    analyses = [
        FineBatchAnalysis(
            fine_batch_id="fine_interval_0001",
            interval_id="interval_0001",
            start=2.0,
            end=7.0,
            summary="type-on and highlight",
            temporal_events=[
                te("e1", "text_animation", 2.0, 3.5, "正文 type-on"),
                te("e2", "text_animation", 3.5, 5.5, "关键词高亮"),
            ],
            text_animations=[
                TextAnimationAnalysis(
                    event_id="e1",
                    text_content="正文",
                    animation_types=["type_on"],
                    boundary=TemporalBoundary(earliest=2.0, latest=3.5),
                    confidence=0.8,
                ),
                TextAnimationAnalysis(
                    event_id="e2",
                    text_content="关键词",
                    animation_types=["emphasis", "appear"],
                    boundary=TemporalBoundary(earliest=3.5, latest=5.5),
                    confidence=0.8,
                ),
            ],
            question_answers=[
                RefinementQuestionAnswer(
                    question="Q1?",
                    answer="type-on",
                    status="answered",
                    confidence=0.8,
                    evidence=FineEvidenceReference(frame_ids=["e1_f"], timestamps=[2.0]),
                )
            ],
            confidence=0.8,
        ),
        FineBatchAnalysis(
            fine_batch_id="fine_interval_0002",
            interval_id="interval_0002",
            start=17.0,
            end=20.5,
            summary="wipe + type-on search",
            temporal_events=[
                te("e3", "transition", 17.0, 17.5, "圆形 wipe"),
                te("e4", "text_animation", 19.5, 20.5, "搜索框 type-on"),
            ],
            text_animations=[
                TextAnimationAnalysis(
                    event_id="e4",
                    text_content="搜索",
                    animation_types=["type_on"],
                    boundary=TemporalBoundary(earliest=19.5, latest=20.5),
                    confidence=0.8,
                )
            ],
            transitions=[
                TransitionAnalysis(
                    event_id="e3",
                    transition_type="wipe",
                    outgoing_state="white",
                    incoming_state="dark ui",
                    boundary=TemporalBoundary(earliest=17.0, latest=17.5),
                    confidence=0.85,
                )
            ],
            question_answers=[
                RefinementQuestionAnswer(
                    question="Q2?",
                    answer="wipe",
                    status="answered",
                    confidence=0.8,
                    evidence=FineEvidenceReference(frame_ids=["e3_f"], timestamps=[17.0]),
                )
            ],
            confidence=0.85,
        ),
        FineBatchAnalysis(
            fine_batch_id="fine_interval_0003",
            interval_id="interval_0003",
            start=27.0,
            end=33.0,
            summary="popup + nav",
            temporal_events=[
                te("e5", "graphic_animation", 28.0, 30.0, "嵌套子窗口弹出"),
                te("e6", "graphic_animation", 28.5, 32.0, "导航点高亮"),
            ],
            graphic_animations=[
                GraphicAnimationAnalysis(
                    event_id="e5",
                    graphic_subject="nested popup",
                    animation_types=["appear", "scale"],
                    boundary=TemporalBoundary(earliest=28.0, latest=30.0),
                    confidence=0.8,
                ),
                GraphicAnimationAnalysis(
                    event_id="e6",
                    graphic_subject="nav dots",
                    animation_types=["appear"],
                    boundary=TemporalBoundary(earliest=28.5, latest=32.0),
                    confidence=0.75,
                ),
            ],
            question_answers=[
                RefinementQuestionAnswer(
                    question="Q3?",
                    answer="popup",
                    status="answered",
                    confidence=0.75,
                    evidence=FineEvidenceReference(frame_ids=["e5_f"], timestamps=[28.0]),
                )
            ],
            confidence=0.8,
        ),
        FineBatchAnalysis(
            fine_batch_id="fine_interval_0004",
            interval_id="interval_0004",
            start=43.0,
            end=49.0,
            summary="concept stack",
            temporal_events=[te("e7", "graphic_animation", 44.0, 48.0, "概念窗口叠加")],
            graphic_animations=[
                GraphicAnimationAnalysis(
                    event_id="e7",
                    graphic_subject="concept stack",
                    animation_types=["appear"],
                    boundary=TemporalBoundary(earliest=44.0, latest=48.0),
                    confidence=0.8,
                )
            ],
            question_answers=[
                RefinementQuestionAnswer(
                    question="Q4?",
                    answer="stack",
                    status="answered",
                    confidence=0.75,
                    evidence=FineEvidenceReference(frame_ids=["e7_f"], timestamps=[44.0]),
                )
            ],
            confidence=0.78,
        ),
        FineBatchAnalysis(
            fine_batch_id="fine_interval_0005",
            interval_id="interval_0005",
            start=51.0,
            end=56.0,
            summary="emoji + logo cut",
            temporal_events=[
                te("e8", "graphic_animation", 51.5, 53.0, "emoji 放大"),
                te("e9", "transition", 53.0, 54.5, "logo 结束卡硬切"),
            ],
            graphic_animations=[
                GraphicAnimationAnalysis(
                    event_id="e8",
                    graphic_subject="emoji",
                    animation_types=["scale"],
                    boundary=TemporalBoundary(earliest=51.5, latest=53.0),
                    confidence=0.8,
                )
            ],
            transitions=[
                TransitionAnalysis(
                    event_id="e9",
                    transition_type="cut",
                    outgoing_state="ui",
                    incoming_state="logo",
                    boundary=TemporalBoundary(earliest=53.0, latest=54.5),
                    confidence=0.75,
                )
            ],
            question_answers=[
                RefinementQuestionAnswer(
                    question="Q5?",
                    answer="cut",
                    status="answered",
                    confidence=0.7,
                    evidence=FineEvidenceReference(frame_ids=["e9_f"], timestamps=[53.0]),
                )
            ],
            confidence=0.8,
        ),
    ]

    all_events = []
    for a in analyses:
        path = layout.fine_batches_out_dir / f"{a.fine_batch_id}.json"
        path.write_text(a.model_dump_json(indent=2), encoding="utf-8")
        all_events.extend(a.temporal_events)

    fva = FineVideoAnalysis(
        video_start=0.0,
        video_end=60.0,
        analyzed_intervals=[TimeRange(start=a.start, end=a.end) for a in analyses],
        temporal_events=all_events,
        second_refinement_required=False,
        confidence=0.8,
    )
    layout.fine_video_analysis_json.write_text(fva.model_dump_json(indent=2), encoding="utf-8")
    layout.fine_temporal_events_json.write_text(
        json.dumps([e.model_dump(mode="json") for e in all_events], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    layout.fine_second_refinement_plan_json.write_text(
        json.dumps(
            {
                "intervals": [],
                "terminal_refinement_status": "not_required",
                "max_additional_rounds": 1,
                "terminal_after_this_round": True,
            }
        ),
        encoding="utf-8",
    )
    # subtitles
    layout.subtitles_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "source_video": "fixture.mp4",
                "segments": [
                    {"index": 1, "start": 0.0, "end": 2.0, "text": "测试字幕"}
                ],
            }
        ),
        encoding="utf-8",
    )


def test_finalize_consumes_fine_structured_fields(tmp_path: Path):
    _write_agent_fine_fixture(tmp_path)
    result = run_epic5(
        video_path=tmp_path / "fixture.mp4",
        output_dir=tmp_path,
        config=AppConfig(),
        user_request="fixture",
        execute_terminal=False,
        allow_provisional=True,
    )
    assert result.analysis is not None
    a = result.analysis
    assert a.provider_mode == "in_session_agent"
    assert a.runtime_status.terminal_refinement_status == "not_required"

    # Fine events entered final
    assert len(a.important_events) >= 5
    assert a.detailed_highlights and len(a.detailed_highlights) == 5
    assert a.text_animation_items
    assert a.graphic_animation_items
    assert a.transition_items
    assert a.repeatable_techniques

    # shot levels and full coverage
    assert a.shots
    assert abs(a.shots[0].start - 0.0) < 1e-3
    assert abs(a.shots[-1].end - 60.0) < 1e-2
    assert abs(sum(s.duration for s in a.shots) - 60.0) < 0.1
    assert any(s.analysis_level == "fine" for s in a.shots)
    assert not all(s.analysis_level == "coarse_only" for s in a.shots)

    # evidence
    assert any(s.evidence_ids for s in a.shots)
    assert any(h.evidence_ids for h in a.detailed_highlights)

    # markdown
    md = Path(result.report_md_path).read_text(encoding="utf-8")
    assert "来自 Fine analysis" in a.executive_summary
    assert "终局精修" not in a.executive_summary or "不需要" in md
    assert "来自 round1 + 终局精修" not in md
    assert "EVID-" in md
    assert "_No detailed analytical highlights._" not in md
    assert "_No text / graphic animation recorded._" not in md
    assert "文字动效" in md or "文字动画" in md
    assert "转场" in md
    assert "可复刻" in md

    # consistency script
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_report_consistency",
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "validate_report_consistency.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    ok, errors, warnings = mod.validate(tmp_path)
    assert ok, errors
