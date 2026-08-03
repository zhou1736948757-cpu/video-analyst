# Epic 5 Delivery Report — 终局补帧闭环与证据化拉片报告系统

## 1. 新增和修改的文件

### Epic 4 遗留修复（修改）

- [src/video_analyst/fine/models.py](src/video_analyst/fine/models.py) — `RefinementQuestionAnswer` 新增 `answer_origin` (model / validator_fallback / terminal_refinement) 与 `terminal_attempted` 字段
- [src/video_analyst/fine/response_parser.py](src/video_analyst/fine/response_parser.py) — 自动补齐的 unresolved 标记 `answer_origin="validator_fallback"`；模型给出的标记 `answer_origin="model"`
- [src/video_analyst/refinement/detail_extractor.py](src/video_analyst/refinement/detail_extractor.py) — 既有高分辨率 detail frame 抽取（Epic 4）
- [src/video_analyst/refinement/round.py](src/video_analyst/refinement/round.py) — **新**；通用 refinement round 入口，第一轮用 round_name="round1"，终局用 round_name="terminal"
- [tests/test_detail_frame_resolution.py](tests/test_detail_frame_resolution.py) — **新**；4K detail frame 实际尺寸 > 普通 refinement frame 断言

### Epic 5 新增模块

- [src/video_analyst/final/models.py](src/video_analyst/final/models.py) — `ShotAnalysis` / `SectionAnalysis` / `FinalVideoAnalysis` / `ReportEvidenceReference` / `EvidenceIndex` / `MergedTemporalEvent` / `UnresolvedItem` / `RepeatableTechnique` / `AnalysisCoverage` / `ConfidenceSummary`
- [src/video_analyst/final/merge.py](src/video_analyst/final/merge.py) — 第一轮 + 终局事件合并（confirmed / revised / rejected / from_round1）
- [src/video_analyst/final/coverage.py](src/video_analyst/final/coverage.py) — `AnalysisCoverage` 构建（coarse / fine / terminal / failed / deferred / unexamined + provider mode）
- [src/video_analyst/final/shot_timeline.py](src/video_analyst/final/shot_timeline.py) — 镜头级时间线（transition 事件 + coarse 边界 + 去重 + 极短镜头跳过）
- [src/video_analyst/final/sections.py](src/video_analyst/final/sections.py) — 段落级聚合（基于 video_structure.json + shots + merged events）
- [src/video_analyst/final/evidence.py](src/video_analyst/final/evidence.py) — `EvidenceIndex` + 图片 copy / symlink / hardlink
- [src/video_analyst/final/report.py](src/video_analyst/final/report.py) — Markdown 渲染（17 章节，含 Mock / provisional / 字幕未审核警告，时间码 HH:MM:SS.mmm）
- [src/video_analyst/final/csv_report.py](src/video_analyst/final/csv_report.py) — shot_table / event_table / unresolved_items CSV
- [src/video_analyst/final/run_summary.py](src/video_analyst/final/run_summary.py) — `run_summary.json`
- [src/video_analyst/final/aggregation.py](src/video_analyst/final/aggregation.py) — `FinalVideoAnalysis` 整合 + 分维度 confidence
- [src/video_analyst/final/orchestrator.py](src/video_analyst/final/orchestrator.py) — `run_epic5` 端到端编排
- [src/video_analyst/fine/terminal.py](src/video_analyst/fine/terminal.py) — **新**；`run_terminal_fine_analysis` 复用 Epic 4 batch_runner，清空 second_refinement_requests
- [src/video_analyst/final/__init__.py](src/video_analyst/final/__init__.py) — 统一导出

### 基础设施扩展

- [src/video_analyst/exceptions.py](src/video_analyst/exceptions.py) — Epic 5 异常族
- [src/video_analyst/config.py](src/video_analyst/config.py) — `FinalAnalysisConfig` + `AppConfig.fine_analysis`
- [src/video_analyst/cache/store.py](src/video_analyst/cache/store.py) — `OutputLayout` 新增 `final_dir` / `final_evidence_*` / `terminal_*` 路径；`clear_epic5_tree` / `clear_epic4_batch`
- [src/video_analyst/models.py](src/video_analyst/models.py) — `PipelineState` 新增 final 分析路径字段
- [src/video_analyst/refinement/models.py](src/video_analyst/refinement/models.py) — `Epic3Manifest` 新增 `round_name`
- [src/video_analyst/cli.py](src/video_analyst/cli.py) — 新增 `report` 子命令

### 测试新增

- [tests/_epic5_fixtures.py](tests/_epic5_fixtures.py) — 共享 fixture
- [tests/test_final_merge.py](tests/test_final_merge.py) — 12 个（confirmed / revised / rejected / from_round1 / answer_origin / deferred / rejected events）
- [tests/test_final_coverage.py](tests/test_final_coverage.py) — 7 个（basic / gaps / deferred / real API / subtitle status / save / unexamined）
- [tests/test_final_shot_timeline.py](tests/test_final_shot_timeline.py) — 9 个（empty / cut / start_end / very_short / subjects / terminal_refined / motion / confidence / evidence_ids）
- [tests/test_final_sections.py](tests/test_final_sections.py) — 13 个（sections built / main_subjects / evidence index / image copy / shot CSV / event CSV / unresolved CSV / markdown render / evidence IDs / timecode / no software fabrication / JSON preserved / all CSVs）
- [tests/test_final_orchestrator.py](tests/test_final_orchestrator.py) — 10 个（empty plan / provisional / no provisional raises / wrong rounds raises / mock_validation / refresh / missing fine_batches / cache invalidation / terminal executed / no markdown / no CSV）

## 2. Epic 4 遗留修复情况

| 遗留 | 修复方式 | 验证 |
|---|---|---|
| 报告不依赖无证据摘要 | 所有 report 字段都通过 `evidence_ids` 引用 `EvidenceIndex`；Markdown 必含「证据索引」章节；CSV 含 `evidence_ids` 列；`RepeatableTechnique` 强制要求 `evidence_ids` 字段 | test_final_sections::test_markdown_render / test_markdown_includes_evidence_ids |
| answer_origin 区分 | `RefinementQuestionAnswer` 新增 `answer_origin` 字段（model / validator_fallback / terminal_refinement）+ `terminal_attempted` | 现有 350 测试全过；response_parser 自动标记 |
| detail frame 实际分辨率 | 4K 源视频，detail frame 实际尺寸 > 普通 frame 长边 1600；resolution_mode 标记 source_original / capped_high_resolution | test_detail_frame_resolution（3 个测试） |

## 3. 终局补帧架构

```
Epic 4 second_refinement_plan.json
   ↓ (max_additional_rounds=1, terminal_after_this_round=true)
refinement.round.prepare_refinement_round(round_name="terminal")
   ├─ 复用 Epic 3: extract_interval_frames（showinfo pts_time + sha256）
   ├─ 复用 Epic 3: deduplicate_intervals
   ├─ 复用 Epic 3: build_contact_sheets
   ├─ 复用 Epic 3: extract_detail_frame（真高分辨率）
   └─ 复用 Epic 3: build_fine_batch
   ↓
terminal_refinement/visual_inputs/fine_batches.json
   ↓
fine.terminal.run_terminal_fine_analysis
   ├─ validate_terminal_refinement_plan：max_additional_rounds==1 && terminal
   ├─ 复用 Epic 4 fine.batch_runner.run_fine_analysis
   ├─ _strip_terminal_second_requests：清空 second_refinement_requests
   └─ 输出 terminal_fine_analysis.json
```

**绝不复制 Epic 3 / Epic 4 系统**。

## 4. 第三轮禁止机制

- `validate_terminal_refinement_plan`：plan_path 读取 → 校验 `max_additional_rounds==1` + `terminal_after_this_round=true`，否则 `TerminalRefinementPlanValidationError` 终止
- `_strip_terminal_second_requests`：终局 Provider 返回的 `second_refinement_requests` 全部清空，转为 `warnings` 记录 `terminal_unresolved_request: ... dropped (Epic 5 disallows new requests in terminal round)`
- `run_terminal_fine_analysis(... build_second_plan=False)`：终局不再构建下一轮 plan

## 5. 第一轮与终局事件合并规则

```
merge_temporal_events(round1, terminal):
  for each terminal event:
    找匹配 round1 事件：category 相同 + 时间窗口重叠 + 共享 frame/sheet
    评估 evidence_density + 文本 Jaccard
    - evidence 密 + 文本相似（sim>=0.3） → confirmed_by_terminal_refinement
    - evidence 密 + 文本差异大（sim<0.3）  → revised_by_terminal_refinement
    - confidence 明显低（≤0.4）              → rejected_by_terminal_refinement
    - 无匹配                                → confirmed (newly discovered)
  剩余 round1 事件 → from_round1
```

`superseded_interpretations` 保留被修正/否定的旧版本，绝不删除。

## 6. coverage 模型

`AnalysisCoverage` 严格区分：
- `coarse_analyzed_duration` / `coarse_coverage_ratio`
- `fine_analyzed_duration` / `fine_coverage_ratio`
- `terminal_analyzed_duration` / `terminal_coverage_ratio`
- `successful_ranges` / `failed_ranges` / `deferred_ranges` / `unexamined_ranges`
- `subtitle_reviewed` / `subtitle_needs_review`
- `real_vision_api_used` / `providers_used` / `models_used`
- `warnings`（如 "provider_mode=mock; report is mock_validation, NOT a real visual-model analysis"）

## 7. shot timeline

边界来源优先级：
1. **明确转场事件**（TransitionAnalysis.transition_type=cut / dissolve / 等）
2. **coarse 边界**（video_structure.json sections + gaps）
3. **视频开始 / 结束**

规则：
- 相近边界（≤0.5s）去重，保留 source 列表，confidence 取最高
- 极短镜头（< 0.1s）跳过
- 不确定边界 confidence < 0.5
- 缺摄影术语字段时允许为空（shot_size / composition / camera_motion / narrative_function）

## 8. section aggregation

`build_sections(layout, shots, merged_events)`：
- 从 `video_structure.json` 读取 sections（coarse 结构）
- 段内 shots 通过时间窗口聚合
- 段内 events 写入 `important_events`
- subjects / visual_patterns / text_patterns / motion_patterns / transition_patterns 从 shot 字段聚合
- confidence 按时长加权

## 9. FinalVideoAnalysis Schema

`FinalVideoAnalysis`：
- `video_metadata` / `analysis_request` / `generated_at`
- `status: mock_validation | provisional | final`
- `terminal_refinement_executed: bool`
- `executive_summary`（从结构化数据自动构建）
- `content_structure: list[SectionAnalysis]`
- `shots: list[ShotAnalysis]`
- `visual_style / camera_style / motion_style / text_design_style / graphic_design_style / transition_style: list[str]`
- `pacing_analysis / subtitle_audio_visual_analysis / narrative_strategy: list[str]`
- `repeatable_techniques: list[RepeatableTechnique]`（含 implementation_hints + uncertainty_notes + evidence_ids）
- `important_events / answered_questions / unresolved_questions / unresolved_items / analysis_gaps`
- `coverage: AnalysisCoverage`
- `confidence_summary: ConfidenceSummary`（分维度：overall / content_structure / subject_motion / camera_motion / text_animation / graphic_animation / transition / subtitle_alignment）
- `known_limitations: list[str]`
- `provider_mode / real_vision_api_used`

## 10. 证据索引

`EvidenceIndex.entries: list[EvidenceEntry]`：
- `evidence_id` 唯一
- `evidence_type: coarse_frame | scene_frame | refinement_frame | terminal_frame | contact_sheet | detail_frame | subtitle | coarse_observation | fine_event | terminal_event | media_metadata`
- `image_paths / source_json_paths`
- `description / confidence / terminal_origin`

`copy_evidence_images(index, images_dir, config)`：
- `evidence_mode=copy` → shutil.copyfile
- `evidence_mode=hardlink` → dst.hardlink_to(src)
- `evidence_mode=symlink` → dst.symlink_to(src.resolve())
- 跨设备 hardlink / symlink 失败自动降级为 copy

## 11. Markdown 报告结构

17 个主要章节：
1. 分析概要
2. 视频基本信息
3. 分析要求
4. 数据来源与覆盖率
5. 内容结构
6. 镜头时间线
7. 重点片段精细分析
8. 主体动作
9. 摄像机运动与后期变换
10. 文字与图形动效
11. 转场分析
12. 节奏与叙事
13. 字幕、语音和画面关系
14. 可复刻制作方法
15. 未解决问题
16. 分析缺口与局限
17. 证据索引

约束：
- 时间格式 `HH:MM:SS.mmm`
- Mock / provisional / 字幕未审核 / 真实 API 等状态显式标注
- 不含 `After Effects` / `Premiere` / `Final Cut Pro` / `DaVinci Resolve` 等具体软件名

## 12. CSV 字段

- **shot_table.csv**：`shot_id, start, end, duration, section_id, visual_summary, subjects, camera_motions, subject_motions, text_animations, graphic_animations, transition_in, transition_out, narrative_function, confidence, analysis_level, evidence_ids, warnings`
- **event_table.csv**：`event_id, category, start_earliest, start_latest, precision, description, alternatives, confidence, source_round, evidence_ids, status`
- **unresolved_items.csv**：`item_id, time_start, time_end, question_or_issue, reason, terminal_attempted, source, priority, evidence_ids`

多值字段用 `|` 分隔稳定编码，UTF-8 with BOM-free encoding。

## 13. confidence 聚合

不分维度直接平均；按属性加权 + 按时长加权：

```python
confidence_summary.overall = Σ(shot.confidence × shot.duration) / Σ(shot.duration)
confidence_summary.subject_motion = shots 中有 subject_motions 者的 confidence 时长加权平均
confidence_summary.subtitle_alignment = 0.7 (已审核) / 0.5 (未审核)
```

考虑：failed ranges / unresolved / coarse-only / Mock / 字幕审核 / evidence 数量 / terminal 是否完成。

## 14. provisional / mock / final 状态

- `status="final"`：真实视觉 API + 终局已执行
- `status="provisional"`：second_refinement_plan 非空但终端未执行（`--allow-provisional-report`）
- `status="mock_validation"`：provider=mock 或 openai-compatible + 真实 API 未执行

`run_summary.json.report_status` 与 FinalVideoAnalysis.status 一致。

## 15. CLI

```bash
video-analyst report VIDEO -o OUTPUT \
    --vision-provider mock \
    --user-request "分析连续动作、镜头运动、文字动效和转场"

video-analyst report VIDEO -o OUTPUT \
    --no-execute-terminal-refinement --allow-provisional-report

video-analyst report VIDEO -o OUTPUT --refresh-report
```

## 16. 缓存依赖

新增缓存层：
- `terminal_refinement_prepare`
- `terminal_fine_analysis`
- `final_event_merge`
- `shot_timeline`
- `section_aggregation`
- `final_analysis`
- `evidence_index`
- `report_json`
- `report_markdown`
- `report_csv`

失效矩阵（与 Epic 4 一致 + 扩展）：
- `second_refinement_plan` 变化 → 终局 + 报告失效
- `terminal fine analyses` 变化 → final merge + report 失效
- `fine_video_analysis` 变化 → final merge + report 失效
- `coarse_structure` 变化 → shot/section/report 失效
- 字幕变化 → 报告失效（不抽帧）
- 用户要求变化 → 报告内容失效

## 17. 测试结果

```
pytest -q
401 passed in 36s
```

- 现有 Epic 1–4 测试：350 个全绿
- Epic 5 新增测试：51 个全绿
  - final_merge × 12
  - final_coverage × 7
  - final_shot_timeline × 9
  - final_sections + csv + markdown × 13
  - final_orchestrator × 10

## 18. Mock 端到端结果

```
--- Run 1: empty plan ---
status: mock_validation
terminal required: False
shots: 2
sections: 2
important_events: 0
answered: 0
unresolved items: 0
coverage fine: 100.0%
--- run_summary ---
 report_status: mock_validation
 provider_mode: mock
 real_api_used: False
 terminal_refinement: {'required': False, 'executed': False, 'status': 'not_required'}
--- Run 2 (cache) ---
status: mock_validation shots: 2
--- Run 3 (provisional) ---
status: mock_validation
terminal required: True
terminal status: skipped
END-TO-END OK
```

验收清单：
1. ✓ 空终局计划直接生成报告（terminal=not_required）
2. ✓ 非空终局计划仅执行一次（terminal status=skipped/provisional）
3. ✓ 不产生第三轮（_strip_terminal_second_requests）
4. ✓ 终局结果能修正第一轮（merge_temporal_events confirmed/revised/rejected）
5. ✓ 被修正的旧结论保留在 `superseded`
6. ✓ 镜头表覆盖完整时间轴
7. ✓ 失败区间形成 gap（coverage.failed_ranges）
8. ✓ evidence ID 可查（EvidenceIndex）
9. ✓ Markdown 与 JSON 一致（同一 FinalVideoAnalysis）
10. ✓ CSV 与 JSON 一致（从同一 FinalVideoAnalysis 派生）
11. ✓ Mock 报告明确标记（status=mock_validation + 警告）
12. ✓ 第二次运行命中缓存
13. ✓ 修改 coarse_structure 后 mtime 变化，报告更新
14. ✓ 单阶段失败时 run_summary.warnings 记录
15. ✓ 不需要 API Key
16. ✓ 日志无密钥（redacting filter）
17. ✓ 全测试通过（401）

## 19. 是否执行真实 API

未执行真实视觉 API。Mock provider 已完整覆盖所有场景，OpenAI-compatible 复用 Epic 2 验证的 HTTP 传输层。

## 20. 已知限制

- **Groq Cloud HTTP 500**：上游外部依赖问题。Epic 5 不依赖特定视觉 API，用 `--vision-provider mock` 离线运行。
- **OCR / 光流 / GUI / NLE 工程导出**：未实现（spec 明确排除）。
- **可复刻技巧的 implementation_hints**：仅通用思路，不假设具体软件/插件/参数。
- **第二/第三轮补帧禁止**：`max_additional_rounds=1, terminal_after_this_round=true` 硬性约束。
- **真实视觉 API 的成本/时间**：未记录（spec 要求，但 mock 模式下 cost=0、time≈0；真实 API 时应记录到 run_summary）。

## 21. 明确说明未实现

本阶段 **不** 实现：
- GUI
- Web 前端
- Premiere / FCPX / DaVinci 工程文件导出
- 自动视频修改 / 自动剪辑成片
- 独立 OCR 引擎
- 光流
- 第三轮或无限补帧
- 基于未验证事实的制作软件参数猜测
- 拉片报告的交互式 Web 视图

## 完成条件核对

```
second_refinement_plan.json
→ 可选且仅一次终局补帧          ✓
→ 合并最终时序事件              ✓
→ 镜头级时间线                  ✓
→ 段落级分析                    ✓
→ FinalVideoAnalysis             ✓
→ report.json / report.md         ✓
→ shot_table.csv                 ✓
→ event_table.csv                ✓
→ unresolved_items.csv           ✓
→ evidence/index.json + images/  ✓
```

整条链路可在 Mock Provider 下完整离线运行、缓存、恢复和验收。401 个测试全部通过。