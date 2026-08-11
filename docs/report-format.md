# Report Format

Video Analyst generates a structured final report from Epic 5. All outputs are
derived from a single `FinalVideoAnalysis` JSON — Markdown and CSV never
re-derive facts independently.

The legacy `status` and `provider_mode` fields remain in the schema for
backward-compatible reports. The current runtime provenance is always
`vision_execution_mode=in_session_agent`, with `external_api_used=false` and
`api_key_used=false`.

## Output files

```text
report/
├── report.md              # human-readable Markdown
├── report.json            # machine-readable FinalVideoAnalysis
├── shot_table.csv         # one row per shot
├── event_table.csv        # one row per merged temporal event
├── unresolved_items.csv   # one row per unresolved item
├── run_summary.json       # run metadata (status, provider, cost)
└── evidence/
    ├── index.json         # evidence index (all evidence IDs)
    └── images/            # copied/hardlinked evidence images
```

## report.json (FinalVideoAnalysis)

Top-level fields:

| Field | Type | Description |
|---|---|---|
| `schema_version` | str | `1.0` |
| `video_metadata` | dict | duration, codec, resolution, subtitle status |
| `analysis_request` | str | user-supplied analysis request |
| `generated_at` | datetime | UTC timestamp |
| `status` | str | legacy enum (`final` / `provisional` / `mock_validation`) |
| `runtime_status` | object | **orthogonal** status (see below) |
| `terminal_refinement_executed` | bool | whether terminal round ran |
| `executive_summary` | str | auto-derived from structured data |
| `content_structure` | list[SectionAnalysis] | sections from coarse structure |
| `shots` | list[ShotAnalysis] | shot-level timeline |
| `visual_style` / `camera_style` / ... | list[str] | style observations |
| `repeatable_techniques` | list[RepeatableTechnique] | reproducible methods |
| `important_events` | list[str] | high-confidence events |
| `answered_questions` | list[RefinementQuestionAnswer] | all refinement Q&A |
| `unresolved_questions` | list[str] | unanswered questions |
| `unresolved_items` | list[UnresolvedItem] | structured unresolved entries |
| `analysis_gaps` | list[TimeRange] | failed / unexamined / deferred ranges |
| `coverage` | AnalysisCoverage | coarse/fine/terminal coverage ratios |
| `confidence_summary` | ConfidenceSummary | per-dimension confidence (0–1) |
| `known_limitations` | list[str] | limitations (mock, no OCR, etc.) |
| `provider_mode` | str | `mock` / `real` |
| `real_vision_api_used` | bool | based on actual call records |

### Runtime status (orthogonal)

Epic 6 split the single `status` enum into four independent fields:

```json
{
  "completion_status": "provisional",       // final | provisional | failed
  "provider_mode": "mock",                   // mock | real_api | mixed
  "terminal_refinement_status": "skipped",   // not_required | executed | skipped | failed
  "analysis_completeness": "partial"         // complete | partial
}
```

These never overlap. Example: "Mock provider + terminal skipped + partial" is
expressible, unlike the legacy single enum.

### Confidence summary

Confidence is **not** a simple average. Per-dimension values are
duration-weighted over shots that have the relevant attribute:

```json
{
  "overall": 0.71,
  "content_structure": 0.85,
  "subject_motion": 0.68,
  "camera_motion": 0.55,
  "text_animation": 0.79,
  "graphic_animation": 0.5,
  "transition": 0.74,
  "subtitle_alignment": 0.5
}
```

`subtitle_alignment` is 0.7 if subtitles are human-reviewed, 0.5 otherwise.

## report.md

17 sections:

1. 分析概要 (executive summary + status warnings)
2. 视频基本信息 (metadata)
3. 分析要求 (user request)
4. 数据来源与覆盖率 (coverage)
5. 内容结构 (sections)
6. 镜头时间线 (shots)
7. 重点片段精细分析 (high-confidence shots)
8. 主体动作 (subject motion table)
9. 摄像机运动与后期变换 (camera + post-transform tables)
10. 文字与图形动效 (text + graphic tables)
11. 转场分析 (transition table)
12. 节奏与叙事 (pacing + narrative)
13. 字幕、语音和画面关系 (subtitle/A/V)
14. 可复刻制作方法 (repeatable techniques — no software assumptions)
15. 未解决问题 (unresolved items)
16. 分析缺口与局限 (gaps + limitations)
17. 证据索引 (evidence index reference)

All times use `HH:MM:SS.mmm`. Status warnings (mock / provisional / unreviewed
subtitles) are displayed prominently.

## shot_table.csv

Columns: `shot_id, start, end, duration, section_id, visual_summary, subjects,
camera_motions, subject_motions, text_animations, graphic_animations,
transition_in, transition_out, narrative_function, confidence, analysis_level,
evidence_ids, warnings`.

Multi-value fields use `|` as separator.

## event_table.csv

Columns: `event_id, category, start_earliest, start_latest, precision,
description, alternatives, confidence, source_round, evidence_ids, status`.

`source_round` is `round1` or `terminal`. `status` is
`from_round1` / `confirmed_by_terminal_refinement` /
`revised_by_terminal_refinement` / `rejected_by_terminal_refinement`.

## unresolved_items.csv

Columns: `item_id, time_start, time_end, question_or_issue, reason,
terminal_attempted, source, priority, evidence_ids`.

`source` is `round1_unresolved` / `validator_fallback` / `terminal_unresolved`
/ `deferred_terminal` / `coarse_gap` / `fine_gap`.

## Evidence index

`report/evidence/index.json` lists every evidence entry with a unique
`evidence_id` (e.g. `EVID-CS-fine_interval_0001-sheet_0001`). Markdown and CSV
reference these IDs so every conclusion is traceable.

Image files are copied (default) / hardlinked / symlinked into
`report/evidence/images/` according to `evidence_mode` config.
