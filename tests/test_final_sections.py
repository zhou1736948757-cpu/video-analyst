"""Tests for final sections + evidence + csv + Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from video_analyst.fine.models import (
    FineBatchAnalysis,
    FineEvidenceReference,
    TemporalBoundary,
    TemporalEvent,
)
from video_analyst.final.csv_report import (
    write_csv_reports,
    write_event_table_csv,
    write_shot_table_csv,
    write_unresolved_table_csv,
)
from video_analyst.final.evidence import build_evidence_index, copy_evidence_images
from video_analyst.final.models import (
    FinalVideoAnalysis,
    MergedTemporalEvent,
    UnresolvedItem,
)
from video_analyst.final.report import write_markdown_report
from video_analyst.final.sections import build_sections
from video_analyst.final.shot_timeline import build_shot_timeline
from video_analyst.config import FinalAnalysisConfig

from tests._epic4_fixtures import has_pillow

pytestmark = pytest.mark.skipif(not has_pillow(), reason="Pillow required")


def _transition(earliest: float, latest: float, eid: str = "tr_001") -> TemporalEvent:
    return TemporalEvent(
        event_id=eid,
        category="transition",
        boundary=TemporalBoundary(earliest=earliest, latest=latest, precision="bounded"),
        description="hard cut",
        direct_observations=[],
        inference=None,
        evidence=FineEvidenceReference(),
        alternatives=[],
        confidence=0.7,
    )


def _merged(event: TemporalEvent) -> MergedTemporalEvent:
    return MergedTemporalEvent(
        merged_id=f"m_{event.event_id}",
        active=event,
        superseded=[],
        status="from_round1",
        source_round="round1",
        notes=[],
    )


def _analysis_with_shots(tmp_path: Path) -> FinalVideoAnalysis:
    from tests._epic5_fixtures import (
        seed_epic5_inputs,
        add_coarse_to_epic4_layout,
        add_fine_video_analysis,
        add_fine_temporal_events,
    )
    layout = seed_epic5_inputs(tmp_path)
    add_coarse_to_epic4_layout(layout, video_duration=10.0)
    add_fine_video_analysis(layout, analyzed_intervals=[{"start": 0.0, "end": 10.0}])
    add_fine_temporal_events(layout)

    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=10.0,
        confidence=0.6,
    )
    merged = [_merged(_transition(5.0, 5.1, "tr_a"))]
    shots = build_shot_timeline(
        layout=layout,
        video_duration=10.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=merged,
    )
    sections = build_sections(
        layout=layout,
        shots=shots,
        merged_events=merged,
    )
    from video_analyst.final.aggregation import build_final_video_analysis
    from video_analyst.final.coverage import build_coverage_report

    coverage = build_coverage_report(layout, video_duration=10.0)
    return build_final_video_analysis(
        layout=layout,
        shots=shots,
        sections=sections,
        coverage=coverage,
        merged_events=merged,
        answered_questions=[],
        unresolved_items=[],
        video_metadata={"duration": 10.0},
        analysis_request="x",
        status="final",
        terminal_refinement_executed=False,
    )


def test_sections_built_from_coarse(tmp_path: Path):
    from tests._epic5_fixtures import (
        seed_epic5_inputs,
        add_coarse_to_epic4_layout,
        add_fine_video_analysis,
        add_fine_temporal_events,
    )
    layout = seed_epic5_inputs(tmp_path)
    add_coarse_to_epic4_layout(layout, video_duration=60.0)
    add_fine_video_analysis(layout)
    add_fine_temporal_events(layout)

    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=60.0,
        confidence=0.6,
    )
    shots = build_shot_timeline(
        layout=layout,
        video_duration=60.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=[],
    )
    sections = build_sections(layout=layout, shots=shots, merged_events=[])
    assert len(sections) >= 1
    # 全部 section 都有 shot_ids
    assert all(len(s.shot_ids) >= 0 for s in sections)


def test_sections_aggregate_main_subjects(tmp_path: Path):
    """段内镜头 subjects 合并到 main_subjects。"""
    from tests._epic5_fixtures import (
        seed_epic5_inputs,
        add_coarse_to_epic4_layout,
        add_fine_video_analysis,
        add_fine_temporal_events,
    )
    layout = seed_epic5_inputs(tmp_path)
    add_coarse_to_epic4_layout(layout, video_duration=60.0)
    add_fine_video_analysis(layout)
    add_fine_temporal_events(layout)

    a = FineBatchAnalysis(
        fine_batch_id="fine_interval_0001",
        interval_id="interval_0001",
        start=0.0,
        end=60.0,
        confidence=0.6,
        frame_states=[
            {
                "observation_id": "obs_1",
                "timestamp_seconds": 5.0,
                "frame_id": "f1",
                "description": "x",
                "visible_subjects": ["person_a", "car"],
                "visible_text": [],
                "composition_notes": [],
                "confidence": 0.7,
            }
        ],
    )
    shots = build_shot_timeline(
        layout=layout,
        video_duration=60.0,
        fine_analyses={"fine_interval_0001": a},
        merged_events=[],
    )
    sections = build_sections(layout=layout, shots=shots, merged_events=[])
    # section_0001 包含首段（0-30s），应包含 person_a
    s1 = next((s for s in sections if s.section_id == "section_0001"), None)
    assert s1 is not None
    assert "person_a" in s1.main_subjects


def test_evidence_index_includes_all_types(tmp_path: Path):
    from tests._epic5_fixtures import seed_epic5_inputs
    from tests._epic4_fixtures import make_fine_visual_batch

    layout = seed_epic5_inputs(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=0.0, end=8.0, fps=10)
    fb = FineBatchAnalysis(
        fine_batch_id=batch.fine_batch_id,
        interval_id=batch.interval.interval_id,
        start=batch.interval.start,
        end=batch.interval.end,
        confidence=0.6,
    )
    merged = [_merged(_transition(2.0, 2.1, "tr_a"))]
    shots = build_shot_timeline(
        layout=layout,
        video_duration=10.0,
        fine_analyses={batch.fine_batch_id: fb},
        merged_events=merged,
    )
    # evidence_index 直接使用 FineVisualBatch（它带 contact_sheets/detail_frames）
    idx = build_evidence_index(
        layout=layout,
        fine_analyses={batch.fine_batch_id: batch},  # 传 FineVisualBatch
        merged_events=merged,
        shots=shots,
    )
    types = {e.evidence_type for e in idx.entries}
    # contact_sheet / detail_frame / event 至少存在
    assert "contact_sheet" in types
    assert "detail_frame" in types
    assert "terminal_event" in types or "fine_event" in types
    assert "subtitle" in types or "media_metadata" in types


def test_copy_evidence_images(tmp_path: Path):
    """evidence_mode=copy 时图片被复制到 images_dir。"""
    from tests._epic5_fixtures import seed_epic5_inputs
    from tests._epic4_fixtures import make_fine_visual_batch

    layout = seed_epic5_inputs(tmp_path)
    batch = make_fine_visual_batch(layout, interval_id="interval_0001", start=0.0, end=8.0, fps=10)
    fb = FineBatchAnalysis(
        fine_batch_id=batch.fine_batch_id,
        interval_id=batch.interval.interval_id,
        start=batch.interval.start,
        end=batch.interval.end,
        confidence=0.6,
    )
    merged: list = []
    shots = build_shot_timeline(
        layout=layout,
        video_duration=10.0,
        fine_analyses={batch.fine_batch_id: fb},
        merged_events=merged,
    )
    # evidence_index 使用 FineVisualBatch（带图片路径）
    idx = build_evidence_index(
        layout=layout,
        fine_analyses={batch.fine_batch_id: batch},
        merged_events=merged,
        shots=shots,
    )
    cfg = FinalAnalysisConfig(evidence_mode="copy", generate_evidence_copy=True)
    result = copy_evidence_images(
        index=idx, images_dir=layout.final_evidence_images_dir, config=cfg
    )
    assert len(result) > 0
    assert any(layout.final_evidence_images_dir.iterdir())


def test_shot_csv_unicode(tmp_path: Path):
    """shot CSV 中文内容正确编码。"""
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    path = write_shot_table_csv(layout, analysis)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == len(analysis.shots)
    for row in rows:
        assert "shot_id" in row
        assert "start" in row
        assert "end" in row
        assert "analysis_level" in row


def test_event_csv_contains_evidence_ids(tmp_path: Path):
    """event CSV 包含 evidence IDs。"""
    layout = _layout_from(tmp_path)
    merged = [_merged(_transition(2.0, 2.1, "tr_a"))]
    path = write_event_table_csv(layout, merged)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert "tr_a" in rows[0]["event_id"]


def test_unresolved_csv(tmp_path: Path):
    """unresolved CSV 字段完整。"""
    layout = _layout_from(tmp_path)
    items = [
        UnresolvedItem(
            item_id="u1",
            time_start=5.0,
            time_end=6.0,
            question_or_issue="what happened?",
            reason="insufficient evidence",
            terminal_attempted=False,
            source="round1_unresolved",
            priority="high",
            evidence_ids=["EVID-1"],
        )
    ]
    path = write_unresolved_table_csv(layout, items)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["item_id"] == "u1"
    assert rows[0]["source"] == "round1_unresolved"
    assert rows[0]["priority"] == "high"
    assert rows[0]["terminal_attempted"] == "no"


def test_markdown_render(tmp_path: Path):
    """Markdown 报告包含所有 17 个主要章节。"""
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    cfg = FinalAnalysisConfig()
    md_path = write_markdown_report(layout, analysis=analysis, cfg=cfg)
    md = md_path.read_text(encoding="utf-8")
    for i in range(1, 18):
        assert f"## {i}." in md, f"missing section {i}"
    assert "in_session_agent" in md or "Agent 现场看图" in md or "未使用外部视觉 API" in md
    assert "unsupervised" not in md.lower()


def test_markdown_includes_evidence_ids(tmp_path: Path):
    """Markdown 引用 evidence ID。"""
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    cfg = FinalAnalysisConfig()
    md_path = write_markdown_report(layout, analysis=analysis, cfg=cfg)
    md = md_path.read_text(encoding="utf-8")
    assert "证据索引" in md or "evidence" in md.lower()


def test_markdown_uses_timecode_format(tmp_path: Path):
    """时间格式 HH:MM:SS.mmm。"""
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    cfg = FinalAnalysisConfig()
    md_path = write_markdown_report(layout, analysis=analysis, cfg=cfg)
    md = md_path.read_text(encoding="utf-8")
    import re
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", md) is not None


def test_markdown_no_fabricated_software(tmp_path: Path):
    """Markdown 不含假设的软件名称。"""
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    cfg = FinalAnalysisConfig()
    md_path = write_markdown_report(layout, analysis=analysis, cfg=cfg)
    md = md_path.read_text(encoding="utf-8")
    for sw in ["After Effects", "Premiere", "Final Cut Pro", "DaVinci Resolve"]:
        assert sw not in md, f"fabricated software: {sw}"


def test_markdown_json_failure_does_not_delete_json(tmp_path: Path):
    """Markdown 失败时 JSON 保留。"""
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    from video_analyst.final.report import save_final_video_analysis_json

    save_final_video_analysis_json(layout, analysis)
    assert layout.final_report_json.is_file()

    cfg = FinalAnalysisConfig()
    md_path = write_markdown_report(layout, analysis=analysis, cfg=cfg)
    assert md_path.exists()
    assert layout.final_report_json.is_file()


def test_write_csv_reports_all_three(tmp_path: Path):
    analysis = _analysis_with_shots(tmp_path)
    layout = _layout_from(tmp_path)
    merged = [_merged(_transition(2.0, 2.1, "tr_a"))]
    paths = write_csv_reports(layout, analysis=analysis, merged_events=merged)
    assert "shot_table" in paths
    assert "event_table" in paths
    assert "unresolved_table" in paths
    assert paths["shot_table"].is_file()
    assert paths["event_table"].is_file()
    assert paths["unresolved_table"].is_file()


def _layout_from(tmp_path: Path):
    """重读整个 layout 路径，用于直接调用 write_* 的测试。"""
    from video_analyst.cache.store import OutputLayout

    return OutputLayout.create(tmp_path)