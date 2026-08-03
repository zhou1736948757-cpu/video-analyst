"""Final report subsystem (Epic 5).

依赖 Epic 1–4 的结构化输出：
- Epic 1: subtitles.json, review.json, media_info.json
- Epic 2: coarse/video_structure.json, coarse/refinement_plan.json
- Epic 3: refinement/visual_inputs/fine_batches.json
- Epic 4: fine/fine_video_analysis.json, fine/temporal_events.json,
           fine/second_refinement_plan.json
- (可选) Epic 4 终局补帧: terminal_fine/*

生成：
- analysis/final_analysis/* （merged JSONs）
- report/* （report.md, report.json, shot_table.csv, event_table.csv, unresolved_items.csv, run_summary.json）
- report/evidence/index.json + report/evidence/images/* （证据索引 + 复制/硬链接）
"""

from video_analyst.final.aggregation import build_final_video_analysis
from video_analyst.final.coverage import build_coverage_report
from video_analyst.final.csv_report import write_csv_reports
from video_analyst.final.evidence import (
    EvidenceEntry,
    EvidenceIndex,
    build_evidence_index,
    copy_evidence_images,
)
from video_analyst.final.merge import (
    merge_question_answers,
    merge_temporal_events,
)
from video_analyst.final.models import (
    AnalysisCoverage,
    FINAL_SCHEMA_VERSION,
    FinalVideoAnalysis,
    MergedTemporalEvent,
    MergedTemporalEventStatus,
    RepeatableTechnique,
    ReportEvidenceReference,
    ReportEvidenceType,
    SectionAnalysis,
    ShotAnalysis,
    ShotAnalysisLevel,
    UnresolvedItem,
)
from video_analyst.final.report import write_markdown_report
from video_analyst.final.run_summary import build_run_summary
from video_analyst.final.sections import build_sections
from video_analyst.final.shot_timeline import build_shot_timeline

__all__ = [
    "AnalysisCoverage",
    "EvidenceEntry",
    "EvidenceIndex",
    "FINAL_SCHEMA_VERSION",
    "FinalVideoAnalysis",
    "MergedTemporalEvent",
    "MergedTemporalEventStatus",
    "RepeatableTechnique",
    "ReportEvidenceReference",
    "ReportEvidenceType",
    "SectionAnalysis",
    "ShotAnalysis",
    "ShotAnalysisLevel",
    "UnresolvedItem",
    "build_coverage_report",
    "build_evidence_index",
    "build_final_video_analysis",
    "build_run_summary",
    "build_sections",
    "build_shot_timeline",
    "copy_evidence_images",
    "merge_question_answers",
    "merge_temporal_events",
    "write_csv_reports",
    "write_markdown_report",
]