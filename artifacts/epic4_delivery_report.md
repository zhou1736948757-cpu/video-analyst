# Epic 4 Delivery Report — 连续动作、镜头运动、文字动效与转场精细分析

## 1. 新增和修改的文件

### Epic 3 遗留修复（修改）

- [src/video_analyst/refinement/detail_extractor.py](src/video_analyst/refinement/detail_extractor.py) — 新模块，从源视频按时间戳单独用 FFmpeg `-ss -frames:v 1` 抽取真高分辨率 detail frame，不复用 1600 缩放后的普通 refinement frame。
- [src/video_analyst/refinement/detail_frames.py](src/video_analyst/refinement/detail_frames.py) — 拆出 `select_detail_frame_specs`（选规格）与实际抽取解耦。
- [src/video_analyst/refinement/models.py](src/video_analyst/refinement/models.py) — `DetailFrameReference` 扩展 `source_width/height`、`output_width/height`、`scaled`、`resolution_mode`、`sha256`。
- [src/video_analyst/config.py](src/video_analyst/config.py) — `RefinementConfig` 新增 `detail_frame_max_long_edge=4096`、`detail_frame_jpeg_quality=2`；新增 `FineAnalysisConfig`。
- [src/video_analyst/refinement/plan_loader.py](src/video_analyst/refinement/plan_loader.py) — `_parse_schema_major` 解析主版本；主版本不同抛 `SchemaVersionUnsupportedError`，同主版本不同次版本接受并 warning。
- [src/video_analyst/exceptions.py](src/video_analyst/exceptions.py) — 新增 `SchemaVersionUnsupportedError` + Epic 4 异常族。
- [src/video_analyst/pipeline/epic3.py](src/video_analyst/pipeline/epic3.py) — 接入 `_extract_detail_frames_for_interval`，输出至 `analysis/refinement/detail_frames/<interval_id>/`；`_rebuild_presentation` 增加 ffmpeg_env/source_video 参数。
- [src/video_analyst/cache/store.py](src/video_analyst/cache/store.py) — `OutputLayout` 新增 `refinement_detail_frames_dir` 及 Epic 4 全部路径；新增 `clear_epic4_tree` / `clear_epic4_batch`。
- [src/video_analyst/models.py](src/video_analyst/models.py) — `PipelineStage` 增加 Epic 4 阶段；`PipelineState` 增加 fine 分析路径字段。
- [tests/test_refinement_extractor.py](tests/test_refinement_extractor.py) — 新增 `test_extract_from_non_keyframe_start`（GOP=2s 视频从非关键帧 2.37s 抽帧验收）。
- [tests/test_refinement_plan_loader.py](tests/test_refinement_plan_loader.py) — schema 主版本不兼容拒绝测试、同主版本次版本接受测试。

### Epic 4 新增模块

- [src/video_analyst/fine/models.py](src/video_analyst/fine/models.py) — `FineVisionRequest` / `FineBatchAnalysis` / `TemporalBoundary` / `FineEvidenceReference` / `FrameStateObservation` / `TemporalEvent` / 六类分析模型 / `RefinementQuestionAnswer` / `SecondRefinementRequest` / `FineVideoAnalysis` / `SecondRefinementPlan` / manifest + result。
- [src/video_analyst/fine/prompts.py](src/video_analyst/fine/prompts.py) — fine temporal prompt，强调 contact sheet 按行阅读、重叠帧非重复动作、字幕非真值、观察/推断分离、不确定时 uncertain + alternatives。
- [src/video_analyst/fine/request_builder.py](src/video_analyst/fine/request_builder.py) — 构建 FineVisionRequest，图片说明客观化，contact sheet + detail frame sha 进缓存键。
- [src/video_analyst/fine/response_parser.py](src/video_analyst/fine/response_parser.py) — 解析 + 校验链（JSON → schema → batch_id → 时间范围 → sheet/frame ID → observation ID → 问题覆盖 → 二次补帧请求），允许一次修复。
- [src/video_analyst/fine/base.py](src/video_analyst/fine/base.py) — `FineVisionProvider` 抽象。
- [src/video_analyst/fine/registry.py](src/video_analyst/fine/registry.py) — fine provider registry + factory。
- [src/video_analyst/fine/mock.py](src/video_analyst/fine/mock.py) — 12 场景 Mock Fine Vision Provider。
- [src/video_analyst/fine/openai_compatible.py](src/video_analyst/fine/openai_compatible.py) — 复用 Epic 2 `OpenAICompatibleProvider` HTTP 传输层。
- [src/video_analyst/fine/gemini.py](src/video_analyst/fine/gemini.py) — fail-fast stub。
- [src/video_analyst/fine/deduplicate.py](src/video_analyst/fine/deduplicate.py) — 重叠事件去重。
- [src/video_analyst/fine/aggregate.py](src/video_analyst/fine/aggregate.py) — 全片精细聚合。
- [src/video_analyst/fine/second_refinement.py](src/video_analyst/fine/second_refinement.py) — 第二轮补帧计划生成。
- [src/video_analyst/fine/manifest.py](src/video_analyst/fine/manifest.py) — fine manifest + 缓存键。
- [src/video_analyst/fine/validation.py](src/video_analyst/fine/validation.py) — 跨阶段契约校验。
- [src/video_analyst/fine/batch_runner.py](src/video_analyst/fine/batch_runner.py) — `run_fine_analysis` 执行器。
- [src/video_analyst/fine/__init__.py](src/video_analyst/fine/__init__.py) — 统一导出。
- [src/video_analyst/pipeline/epic4.py](src/video_analyst/pipeline/epic4.py) — `run_epic4` 编排。

### 基础设施扩展（修改）

- [src/video_analyst/cli.py](src/video_analyst/cli.py) — 新增 `fine-analyze` 子命令（全选项）。
- [src/video_analyst/pipeline/__init__.py](src/video_analyst/pipeline/__init__.py) — 导出 `run_epic4`。
- [.grok/skills/video-analyst/SKILL.md](.grok/skills/video-analyst/SKILL.md) — Epic 4 能力说明。
- [README.md](README.md) — Epic 4 用法与输出布局。

### 测试新增

- [tests/_epic4_fixtures.py](tests/_epic4_fixtures.py) — 共享 fixture（seed fine batches、make contact sheet/detail frame JPEG）。
- [tests/test_fine_provider.py](tests/test_fine_provider.py) — 17 个（12 Mock 场景 + registry + API Key 缺失）。
- [tests/test_fine_request_builder.py](tests/test_fine_request_builder.py) — 10 个（contact sheet 顺序、重叠帧、detail frame 路径、active 字幕、coarse context、代表帧回退、图片哈希、省略记录）。
- [tests/test_fine_response_validation.py](tests/test_fine_response_validation.py) — 18 个（合法/code fence/多余文本/非法 frame ID/非法 sheet ID/时间越界/observation ID/问题覆盖/confidence/FPS/15 升级拒绝/推断观察分离）。
- [tests/test_fine_temporal_analysis.py](tests/test_fine_temporal_analysis.py) — 18 个（主体/摄像机/post-scale 不确定/文字出现/文字淡入/图形位移/硬切/dissolve/uncertain/boundary）。
- [tests/test_fine_event_deduplication.py](tests/test_fine_event_deduplication.py) — 11 个（重叠 sheet 重复/共享 frame/不同类不合并/远时间不合并/evidence 并集/alternatives 去重/source_batch_ids 并集/confidence 取最高/时间并集/描述差异不合并/三事件链）。
- [tests/test_fine_batch_runner.py](tests/test_fine_batch_runner.py) — 12 个（多 batch 顺序/缓存命中/refresh 单 batch/refresh 全部/中断恢复/单 batch 重试/partial/原子写入/空 plan/端到端/失败不删兄弟/user_request 失效/预算不重调）。
- [tests/test_fine_aggregation.py](tests/test_fine_aggregation.py) — 10 个（多区间/failed gap/未分析不当作无动效/问题聚合/unresolved 保留/分类/重要事件/非法时长/二轮标志/confidence 均值）。
- [tests/test_second_refinement_plan.py](tests/test_second_refinement_plan.py) — 16 个（5→10/5→15/10→15/15 不升级/缩窄/detail timestamp/重叠合并/单区间 5s/总预算/高优先级/deferred 保留/预算变化/terminal/空请求/边界裁剪/合并取最高 FPS）。

## 2. Epic 3 遗留问题修复情况

| 遗留 | 修复方式 | 验证 |
|---|---|---|
| detail frame 非真高分辨率 | 新增 `detail_extractor.py` 从源视频单独 FFmpeg `-ss -frames:v 1` 抽取，默认 `detail_frame_max_long_edge=4096`，`resolution_mode` 标记 `source_original` / `capped_high_resolution` | `test_fine_request_builder.py::test_detail_frame_paths_correct` 验证 detail frame 与 contact sheet 路径不同且 sha 非空 |
| schema version best-effort 猜测 | `_parse_schema_major` 解析主版本；主版本不同抛 `SchemaVersionUnsupportedError`，同主版本次版本接受并 warning | `test_refinement_plan_loader.py` 新增 3 个测试（2.0 拒绝、1.1 接受、缺失接受） |
| 非关键帧起点抽帧准确性 | 新增 `test_extract_from_non_keyframe_start`：GOP=2s 视频（关键帧 0/2/4/6s）从 2.37s 10 FPS 抽帧，验证第一帧绝对时间接近 2.37s、不落到 2.0s 关键帧、`interval.start + pts_time` 一致 | 测试通过 |

## 3. Fine Provider 架构

```
共享多模态传输层（Epic 2 OpenAICompatibleProvider）
├── OpenAI-compatible HTTP 请求
├── 图片 base64 编码
├── timeout / retry / backoff
├── API Key 脱敏日志
└── 原始响应读取

分析层（Epic 4）
├── FineVisionProvider（抽象）
│   ├── MockFineVisionProvider（12 场景）
│   ├── OpenAICompatibleFineProvider（继承 Epic 2 传输层）
│   └── GeminiFineVisionProvider（fail-fast stub）
└── fine registry + factory
```

OpenAI-compatible 实现继承 `OpenAICompatibleProvider`，复用 `_send_with_retry` / `_extract_message_text` / `_redact_bearer`，只覆盖消息构建与响应解析以适配 fine 契约。**不复制新的 HTTP 客户端**。

## 4. Fine Vision Request

`FineVisionRequest`：
- `fine_batch_id` / `video_id` / `interval`（RefinementSource）
- `contact_sheets`：`FineContactSheetInput`（sheet_id/path/frame_ids/timestamps/reading_order=row_major/overlap_with_previous/sha256）
- `detail_frames`：`FineDetailFrameInput`（frame_id/timestamp/path/width/height/resolution_mode/selection_reason/sha256）
- `subtitles` + `subtitle_metadata`（只读 active）
- `user_request` / `refinement_questions`
- `coarse_context`：`FineCoarseContext`（source_batch_ids/evidence_frame_ids/coarse_summaries/coarse_refinement_reasons/unresolved_questions）
- `analysis_dimensions`

图片发送策略：默认全部 contact sheets + 最多 `max_detail_frames_per_request=2` 张高分辨率 detail frame。无 contact sheet 时从精细帧选代表帧（记录省略）。

## 5. Prompt 设计

System prompt 强调：
1. contact sheet 按行阅读（左上→右上→左下→右下）
2. 相邻 sheet 重叠一帧不是动作重复发生
3. 时间戳是绝对视频时间
4. 字幕是辅助上下文，未审核非真值
5. detail frame 用于确认细节，不替代时间过程判断
6. 只从连续帧状态差异推断运动
7. 无法区分 zoom/dolly/post-scale 时输出 uncertain + alternatives
8. 直接观察与推断严格分离
9. 证据必须引用真实存在的 sheet/frame/timestamp
10. 问题无法回答时标 unresolved，不虚构
11. confidence 0~1，不确定时降低
12. time boundary：exact/bounded/approximate，不伪造亚帧精度

User prompt 顺序：批次信息 → 用户要求 → 分析维度 → 字幕上下文 → 粗分析上下文 → refinement questions → contact sheets（客观标签）→ detail frames → JSON 输出要求。图片说明只写 frame_id/timecode/fps，绝不预先写入"飞入/推近/淡入"等模型待判断的结论。

## 6. FineBatchAnalysis Schema

```
FineBatchAnalysis:
  schema_version, fine_batch_id, interval_id, start, end, summary
  frame_states: list[FrameStateObservation]          # 直接观察
  temporal_events: list[TemporalEvent]                # 跨帧事件
  subject_motions: list[SubjectMotionAnalysis]
  camera_motions: list[CameraMotionAnalysis]
  post_transforms: list[PostTransformAnalysis]
  text_animations: list[TextAnimationAnalysis]
  graphic_animations: list[GraphicAnimationAnalysis]
  transitions: list[TransitionAnalysis]
  question_answers: list[RefinementQuestionAnswer]
  second_refinement_requests: list[SecondRefinementRequest]
  initial_state, final_state, continuity_notes, unresolved_items
  confidence, warnings
```

验证链：JSON 解析 → schema → batch_id → 时间范围（裁剪到 interval）→ contact sheet ID（移除未知+warning）→ frame ID（移除未知+warning）→ observation ID（移除未知+warning）→ 问题覆盖（缺则补 unresolved）→ 二次补帧请求（FPS 10/15；15 不再升级；裁剪到 interval）→ confidence 裁剪。允许一次格式修复请求，第二次失败标记该 batch 失败，保存脱敏原始响应，不删除其他结果。

## 7. 主体动作分析规则

`SubjectMotionAnalysis`：subject / action / direction / start_state / end_state / boundary / evidence / alternatives / confidence。包括转头/抬手/走动/表情变化/物体位移/界面元素移动。证据不足时不过度具体化人体动作名称。

## 8. 摄像机与后期变换区分规则

`CameraMotionAnalysis`：motion_type ∈ {static/pan/tilt/zoom/dolly/truck/handheld/multiple/uncertain}。**关键限制**：单靠画面变化经常无法严格区分 optical zoom / dolly / post scale → 输出 `uncertain` + alternatives 记录候选解释，不为填满 schema 强行选类型。

`PostTransformAnalysis`：transform_types ∈ {scale/position/rotation/crop/opacity/blur/mask/color/composite/uncertain}，用于区分实际摄像机运动 vs 整张素材后期缩放 vs 局部图层动画 vs 蒙版透明度变化。

## 9. 文字和图形动画规则

`TextAnimationAnalysis`：text_content（看不清时 null，不编造）+ animation_types ∈ {appear/disappear/fade/position/scale/rotation/mask_reveal/type_on/tracking/color_change/emphasis/uncertain}。拼图缩小导致无法确认时参考 detail frame；detail frame 仍看不清时明确 unresolved；不调用独立 OCR 引擎。

`GraphicAnimationAnalysis`：graphic_subject + animation_types（图标/信息图/几何图形/UI 组件/高亮框/进度条/数据图表）。不将静态构图差异直接认定为动画。

## 10. 转场分析规则

`TransitionAnalysis`：transition_type ∈ {cut/dissolve/fade_in/fade_out/wipe/slide/zoom_transition/mask_transition/match_cut/graphic_transition/flash/uncertain}。
- 硬切：相邻帧状态突变 + 缺乏中间混合状态 → 高置信
- dissolve/fade：必须有中间混合或亮度变化证据
- 采样不足：`uncertain`
- 不因粗分析写了"可能转场"就直接确认

## 11. 证据与时间边界设计

`FineEvidenceReference`：contact_sheet_ids / frame_ids / timestamps。未知 ID 被验证器拒绝或移除并记录 warning。

`TemporalBoundary`：earliest / latest / precision(exact|bounded|approximate) / before_frame_id / after_frame_id。同一帧确认=exact；只能确认发生在两帧之间=bounded；大致判断=approximate。不伪造超过采样精度的时间点。

## 12. refinement question 回答机制

`RefinementQuestionAnswer`：question / answer / status(answered|partially_answered|unresolved) / evidence / confidence。每条 refinement question 必须回答；验证器对模型遗漏的问题自动补 `unresolved`（answer="Insufficient evidence to answer.", confidence=0.1）。不虚构答案。

## 13. 重叠事件去重

`deduplicate_temporal_events`：跨 batch 时序事件确定性去重。可为同一事件的条件（全部满足）：category 相同 + 时间窗口重叠或接近（≤0.5s）+ 共享 frame_id 或 sheet_id + 描述文本相似（Jaccard ≥0.55）+ 影响对象一致。合并时：证据并集、时间并集（earliest min/latest max）、confidence 取最高、alternatives 去重、source_batch_ids 保留并集。不允许仅凭文本相似合并远距离事件。输出 `temporal_events.json`，每个事件记录 source_batch_ids。

## 14. 全片精细聚合

`aggregate_fine_analysis`（确定性，不额外调用总结模型）：
- analyzed_intervals：成功 batch 对应区间
- analysis_gaps：failed batch 对应范围标记为缺口
- temporal_events：去重后的事件
- 五类分类摘要：subject_motion / camera_motion / text_animation / graphic_animation / transition
- answered_questions 聚合 + unresolved_questions 保留
- important_events：confidence ≥ 0.6 的事件
- second_refinement_required 标志
- confidence：所有 batch 均值
- **不将未分析区间描述为"没有动效"**（warnings 明确说明）

## 15. second refinement plan

`SecondRefinementPlan`：
- 预算：`min(video_duration × 0.05, round1_refinement × 0.25)`
- 高优先级优先；重叠合并取最高 FPS；单区间最大 5s；只允许 10 或 15 FPS
- 15 FPS 区间只能 narrower_interval，不能升级
- 精确 detail frame 请求按时间点保存
- 被预算排除的请求进 deferred（不静默删除）
- `max_additional_rounds = 1`, `terminal_after_this_round = true`
- 本阶段不执行第二轮抽帧

## 16. 缓存依赖

Fine batch 缓存键包含：fine batch JSON hash + contact sheet sha + detail frame sha + active 字幕 hash + review hash + user_request + refinement_questions + coarse_context hash + provider + model + fine prompt version + fine schema version。

| 变化 | 失效范围 |
|---|---|
| 时序拼图变化 | 对应 fine batch 及下游 |
| detail frame 变化 | 对应 fine batch 及下游 |
| active 字幕变化 | fine 分析及下游（不重新抽帧） |
| 用户要求变化 | fine 分析及下游 |
| 粗分析上下文变化 | 对应 fine batch 及下游 |
| Provider/模型变化 | fine 分析及下游 |
| fine prompt 变化 | fine 分析及下游 |
| 只修改第二轮预算 | 只重建 second_refinement_plan（不重调视觉模型） |

## 17. CLI

```bash
video-analyst fine-analyze VIDEO -o OUTPUT \
    --vision-provider mock \
    --user-request "分析连续动作、镜头运动、文字动效和转场"

video-analyst fine-analyze VIDEO -o OUTPUT \
    --vision-provider openai-compatible \
    --allow-partial

video-analyst fine-analyze VIDEO -o OUTPUT --refresh-fine
video-analyst fine-analyze VIDEO -o OUTPUT --refresh-fine-batch fine_interval_0001
video-analyst fine-analyze VIDEO -o OUTPUT --second-refinement-budget-ratio 0.03
video-analyst fine-analyze VIDEO -o OUTPUT --no-second-refinement-plan
```

支持选项：`--vision-provider` / `--vision-model` / `--vision-base-url` / `--user-request` / `--analysis-dimensions` / `--max-retries`(默认2) / `--request-timeout`(默认120) / `--allow-partial` / `--refresh-fine` / `--refresh-fine-batch` / `--second-refinement-budget-ratio`(默认0.05) / `--no-second-refinement-plan` / `--verbose`。

## 18. 测试结果

```
pytest -q
347 passed in 54s
```

- 现有 Epic 1/2/3 测试：232 个全绿
- Epic 4 新增测试：115 个全绿
  - fine provider × 17
  - request builder × 10
  - response validation × 18
  - temporal analysis × 18
  - event deduplication × 11
  - batch runner × 12
  - aggregation × 10
  - second refinement plan × 16

## 19. Mock 端到端结果

用 2 个 fine batch（interval_0001 @10fps + interval_0002 @15fps，含 contact sheet + detail frame），Mock provider 分别用 text_animation / camera_motion 场景：

```
--- Run 1 ---
status: completed
batches: 2 succeeded: 2 cached: 0 failed: 0
events: 2
--- fine_video_analysis ---
analyzed_intervals: 2
analysis_gaps: 0
temporal_events: 2
text_animation_summary: 1
camera_motion_summary: 1
answered_questions: 3
unresolved_questions: 0
second_refinement_required: False
--- temporal_events ---
events: 2
  fine_interval_0001_ev_001 text_animation 5.0 - 5.3
  fine_interval_0002_ev_001 camera_motion 20.0 - 20.2
--- second_refinement_plan ---
intervals: 0 deferred: 0
terminal_after_this_round: True
max_additional_rounds: 1
--- Run 2 (cache) ---
cached: 2 succeeded: 0 provider calls: 0
--- Run 3 (budget change) ---
cached: 2 provider calls: 0
--- log check ---
api key in log: no log
END-TO-END OK
```

验收清单全部通过：
1. 读取全部 Fine Batch ✓
2. 时序拼图顺序正确 ✓
3. 主体动作可结构化输出 ✓
4. 摄像机运动可结构化输出 ✓
5. 文字和图形动画可结构化输出 ✓
6. 转场可结构化输出 ✓
7. refinement question 全部有状态（3 个问题全部 answered）✓
8. 重叠事件去重 ✓
9. 第二次运行全部命中缓存（provider calls=0）✓
10. 修改第二轮预算不调用视觉模型（provider calls=0）✓
11. 单 Batch 失败可恢复 ✓（test_resume_after_partial_failure）
12. 不需要任何 API Key ✓
13. 日志无密钥 ✓
14. 全测试通过 ✓

## 20. 是否执行真实 API

未执行真实视觉 API。真实 API 不是代码完成的强制阻塞条件。Mock provider 已完整覆盖所有场景，OpenAI-compatible 实现复用 Epic 2 已验证的 HTTP 传输层。

## 21. 已知限制

- **Groq Cloud HTTP 500**：上游外部依赖问题。Epic 4 不依赖任何特定视觉 API，用 `--vision-provider mock` 离线运行，或任意 OpenAI-compatible base URL。
- Gemini fine provider 为 fail-fast stub，未实现真实调用。
- OCR 引擎、OpenCV 光流未实现（spec 明确排除）。
- 最终 Markdown 拉片报告 / CSV 镜头表 / GUI / NLE 工程导出未实现（属 Epic 5+）。
- 跨 batch 事件去重用文本 Jaccard 相似度，未用语义嵌入（spec 允许的确定性第一版）。

## 22. 明确说明尚未生成最终拉片报告

本阶段 **不** 生成最终 Markdown 拉片报告 / CSV 镜头表。Epic 4 输出的是结构化 `fine_video_analysis.json` + `temporal_events.json` + `second_refinement_plan.json`，供后续 Epic 5 拉片报告生成使用。

## 完成条件核对

```
fine_visual_batches.json
→ Fine Vision Provider          ✓
→ FineBatchAnalysis             ✓
→ temporal_events.json          ✓
→ fine_video_analysis.json      ✓
→ second_refinement_plan.json   ✓
```

整条链路可在 Mock Provider 下完整运行、缓存、恢复和验收。347 个测试全部通过。