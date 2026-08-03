# Epic 3 Delivery Report — 自适应精细抽帧与时序拼图输入层

## 1. 新增和修改的文件

### Epic 2 技术债修复（修改）

- [src/video_analyst/util/hashing.py](src/video_analyst/util/hashing.py) — 新增 `ImageFingerprint`（relative_path + size_bytes + sha256 + mtime_ns 预筛选）和 `image_fast_key` / `image_fingerprint_cache_key`。
- [src/video_analyst/coarse_analysis/batch_runner.py](src/video_analyst/coarse_analysis/batch_runner.py) — `_frame_hash_for_cache` 改为内容感知（frame_id, path, size, **sha256**）；新增 `_previous_context_signature` 并纳入 `_build_cache_key`；`manifest.batches` 写回时按 `batch_id` 排序，修复 second-run 因批次顺序翻转变为 `cached` 时上下文错位的回归。
- [src/video_analyst/vision/__init__.py](src/video_analyst/vision/__init__.py) — 显式 re-export `response_parser` 模块，消除运行时导入遗漏。

### Epic 3 新增模块

- [src/video_analyst/refinement/models.py](src/video_analyst/refinement/models.py) — `RefinementSource` / `RefinementFrame` / `TemporalContactSheet` / `DetailFrameReference` / `IntervalArtifact` / `FineVisualBatch` / `FineBatchesIndex` / `Epic3Manifest` / `Epic3Result`。
- [src/video_analyst/refinement/plan_loader.py](src/video_analyst/refinement/plan_loader.py) — 读 + 校验 `refinement_plan.json`。
- [src/video_analyst/refinement/timestamp_parser.py](src/video_analyst/refinement/timestamp_parser.py) — 解析 FFmpeg `showinfo` 的 `pts_time`，文件/记录配对，实际 FPS 推断。
- [src/video_analyst/refinement/extractor.py](src/video_analyst/refinement/extractor.py) — 5/10/15 FPS 精细抽帧。
- [src/video_analyst/refinement/deduplicate.py](src/video_analyst/refinement/deduplicate.py) — 区间内 + 跨区间确定性去重。
- [src/video_analyst/refinement/contact_sheet.py](src/video_analyst/refinement/contact_sheet.py) — 2×N 时序拼图（Pillow，仅客观标签）。
- [src/video_analyst/refinement/detail_frames.py](src/video_analyst/refinement/detail_frames.py) — 首末帧选择。
- [src/video_analyst/refinement/batch_builder.py](src/video_analyst/refinement/batch_builder.py) — 构建 FineVisualBatch + 全局 `fine_batches.json`。
- [src/video_analyst/refinement/manifest.py](src/video_analyst/refinement/manifest.py) — extraction / presentation 双签名、manifest 读写、interval artifact 持久化。
- [src/video_analyst/refinement/validation.py](src/video_analyst/refinement/validation.py) — 跨阶段契约校验。
- [src/video_analyst/refinement/__init__.py](src/video_analyst/refinement/__init__.py) — 统一导出。
- [src/video_analyst/pipeline/epic3.py](src/video_analyst/pipeline/epic3.py) — `run_epic3` 编排。

### 基础设施扩展（修改）

- [src/video_analyst/exceptions.py](src/video_analyst/exceptions.py) — Epic 3 异常族。
- [src/video_analyst/config.py](src/video_analyst/config.py) — 新增 `RefinementConfig` 并加入 `AppConfig` + `fingerprint()`。
- [src/video_analyst/cache/store.py](src/video_analyst/cache/store.py) — `OutputLayout` 增加 refinement 全部路径；`CacheStore` 增加 `clear_epic3_tree` / `clear_epic3_interval`。
- [src/video_analyst/models.py](src/video_analyst/models.py) — `PipelineStage` 增加 Epic 3 阶段；`PipelineState` 增加 `refinement_manifest_path` / `fine_batches_path`。
- [src/video_analyst/cli.py](src/video_analyst/cli.py) — 新增 `refine-prepare` 子命令（全选项）。
- [.grok/skills/video-analyst/SKILL.md](.grok/skills/video-analyst/SKILL.md) — Epic 3 能力说明。
- [README.md](README.md) — Epic 3 用法与输出布局。

### 测试新增

- [tests/_epic3_fixtures.py](tests/_epic3_fixtures.py) — 共享 fixture（seed layout、make_test_video、make_refinement_plan_json）。
- [tests/test_refinement_plan_loader.py](tests/test_refinement_plan_loader.py)
- [tests/test_refinement_timestamp_parser.py](tests/test_refinement_timestamp_parser.py)
- [tests/test_refinement_extractor.py](tests/test_refinement_extractor.py)
- [tests/test_refinement_deduplicate.py](tests/test_refinement_deduplicate.py)
- [tests/test_temporal_contact_sheet.py](tests/test_temporal_contact_sheet.py)
- [tests/test_detail_frame_selection.py](tests/test_detail_frame_selection.py)
- [tests/test_fine_batch_builder.py](tests/test_fine_batch_builder.py)
- [tests/test_epic3_cache.py](tests/test_epic3_cache.py)
- [tests/test_epic3_pipeline.py](tests/test_epic3_pipeline.py)
- [tests/test_epic3_resume.py](tests/test_epic3_resume.py)

## 2. Epic 2 技术债修复情况

| 技术债 | 修复方式 | 验证 |
|---|---|---|
| 图片缓存签名只含文件大小 | `_frame_hash_for_cache` 改为 `(frame_id, path, size_bytes, sha256)`；新增 `ImageFingerprint` 提供 mtime 预筛选 + 内容哈希权威校验 | 现有 135 测试 + 新测试全绿 |
| previous_context 未进缓存键 | 新增 `_previous_context_signature`（仅 5 个字段：previous_batch_id / previous_summary / last_known_subjects / last_known_scene / unresolved_questions）纳入 `_build_cache_key`；同时修复 `manifest.batches` 写回排序，避免 second-run 因顺序翻转变为 cached 时上下文错位 | `test_second_run_hits_cache_no_reinvocation`、`test_pipeline_resume_uses_cache` 通过 |
| response_parser 模块确认 | 模块真实存在于版本控制；Mock + OpenAI-compatible 均经 `parse_response_payload` / `validate_batch_response`；在 `vision/__init__.py` 显式 re-export 消除运行时导入遗漏 | 导入测试通过 |

## 3. refinement plan 输入契约

`load_refinement_plan(layout)` 读取 `analysis/coarse/refinement_plan.json`，校验：

- `schema_version` 兼容（不兼容时 best-effort 加载 + warning）
- Pydantic 结构校验失败 → `RefinementPlanValidationError`
- interval_id 唯一
- `start >= 0`，`end > start`，`end <= video_duration`（容差 1ms）
- `requested_fps ∈ {5, 10, 15}`
- `deferred_intervals` 保留在 plan 对象供诊断，但 **不参与处理**
- 空 plan 是合法状态，返回 `[]`，pipeline 产出 `completed_no_intervals`

## 4. FFmpeg 精细抽帧命令

每个 interval 独立抽取，`-ss` 在 `-i` 之前做 fast seek，`-t duration` 限定输出，`fps=N,scale=...,showinfo` 解析每帧 `pts_time`：

```
ffmpeg -y -nostdin -loglevel info \
  -ss <start> -i <input> -t <duration> \
  -vf "fps=<5|10|15>,scale='if(gt(iw\,ih)\,min(<long_edge>\,iw)\,-2)':'if(gt(iw\,ih)\,-2\,min(<long_edge>\,ih))',showinfo" \
  -fps_mode vfr -q:v <2..31> \
  <out>/frame%08d.<jpg|png>
```

- 长边默认 1600px，保持比例，不拉伸
- `q:v` 默认 2（高质量）
- 绝对时间 = `interval.start + pts_time`
- FFmpeg 在 `-t` 边界可能 muxer 丢弃最后一帧但 showinfo 仍上报 → 容差为 `len(entries) == len(files) + 1` 时裁掉尾部 entry；其他不一致一律 `RefinementFrameMismatchError`，**绝不从文件名推算时间戳**

## 5. 时间戳解析方法

`parse_showinfo_stderr` 用 `pts_time:(-?[0-9.]+)` 提取每条 showinfo 行，保持 stderr 顺序；`pair_with_files` 按索引与输出 JPG 一一配对，数量不符即 `RefinementTimestampParseError`。`compute_actual_fps` 由首末 pts_time 推断实际 FPS。

## 6. 帧去重规则

`deduplicate_intervals` 两阶段确定性去重：

1. **区间内**：`|Δt| ≤ threshold(默认0.005s) AND sha256 相同` → 丢弃较晚记录，保留最早
2. **跨区间**：同样判据 → 后到者 alias 到 canonical frame_id（物理文件保留一份，两个 interval 都保留引用）

- 同时间不同 sha → 保留两者
- 远时间相同 sha → 保留两者
- 不使用感知哈希
- 所有去重动作计入 `DeduplicationReport.stats`

## 7. 时序拼图规则

`build_contact_sheets`：

- 默认 `frames_per_sheet=4`，`sheet_overlap_frames=1`，`sheet_columns=2` → 2×2
- 阅读顺序：左上 → 右上 → 左下 → 右下
- step = `frames_per_sheet - overlap_frames`
- 相邻 sheet 重叠 1 帧（`overlap_with_previous` 记录）
- 不足 4 帧：用真实剩余帧，空格中性留白，**不复制末帧冒充新时间点**
- 每格标签仅含：`frame_id` / `timecode (HH:MM:SS.mmm)` / `requested_fps`
- 整图仅含：`interval_id  TIME ->`
- **禁止** fade-in / pan / transition / 缩放等任何语义标签
- Pillow 实现，不依赖 ImageMagick；字体缺失回退默认
- 长边默认 ≤ 4096，保持原比例 letterbox

## 8. detail frame 选择规则

`select_detail_frames`（确定性，无模型调用）：

- 选首帧（reason=`interval_start`）+ 末帧（reason=`interval_end`）
- 单帧区间（`end - start ≤ 1ms`）→ 只选首帧，reason=`single_frame_interval`
- 上限 `detail_max_per_interval`（默认 2）
- 保留原始分辨率，不缩放

## 9. fine batch 数据结构

`FineVisualBatch`（`fine_<interval_id>.json`）：

```json
{
  "schema_version": "1.0",
  "fine_batch_id": "fine_interval_0001",
  "interval": { "interval_id", "start", "end", "requested_fps", "priority", "reasons", "questions", "source_batch_ids", "evidence_frame_ids" },
  "frames": [ { "frame_id", "timestamp_seconds", "timecode", "relative_timestamp_seconds", "requested_fps", "path", "width", "height", "size_bytes", "sha256", "source_video_sha256", "is_detail" } ],
  "contact_sheets": [ { "sheet_id", "frame_ids", "timestamps", "path", "columns", "rows", "width", "height", "overlap_with_previous", "sha256" } ],
  "detail_frames": [ { "frame_id", "timestamp_seconds", "path", "selection_reason" } ],
  "subtitles": [ { "index", "start", "end", "text", "language" } ],
  "subtitle_metadata": { "active_source", "reviewed", "needs_review", "asr_model", "calibration_model" },
  "user_request": "",
  "refinement_questions": [],
  "coarse_context": { "source_batch_ids", "evidence_frame_ids", "coarse_analysis_paths" }
}
```

- 字幕只读 `subtitles/subtitles.json` + `subtitles/review.json`，**不读** `subtitles_asr.*` / `subtitles_llm.*`
- 默认上下文窗口：前后各 1.0s
- Epic 4 只能读 fine batch，不重新解析 FFmpeg 日志

## 10. 缓存依赖

双签名：

- **Extraction signature**：source_video_sha + interval(start/end/FPS) + ffmpeg_version + max_frame_long_edge + image_format + jpeg_quality + timestamp_parser_version + schema_version
- **Presentation signature**：frame sha256 集合 + frames_per_sheet + sheet_overlap_frames + sheet_columns + contact_sheet_max_edge + label_format_version + sheets_enabled + subtitle_hash + subtitle_context_before/after + user_request + refinement_questions + schema_version

失效矩阵（单源真相：`manifest.json`）：

| 变化 | 失效范围 |
|---|---|
| 源视频 | 全部 Epic 3 |
| interval start/end/FPS | 该 interval 抽帧及下游 |
| 拼图布局 | 仅拼图 + fine batch，不抽帧 |
| active 字幕版本 | 仅字幕映射 + fine batch，不抽帧 |
| user_request / questions | 仅 fine batch，不抽帧 |
| detail frame 规则 | 仅 detail frames + fine batch |
| deferred intervals | 不影响已处理 intervals |

## 11. CLI 使用方式

```bash
# 完整运行
video-analyst refine-prepare VIDEO -o OUTPUT

# 重跑所有 Epic 3 区间
video-analyst refine-prepare VIDEO -o OUTPUT --refresh-refinement

# 重跑指定区间（可重复）
video-analyst refine-prepare VIDEO -o OUTPUT \
    --refresh-interval interval_0002 --refresh-interval interval_0003

# 仅改拼图布局（不重抽帧）
video-analyst refine-prepare VIDEO -o OUTPUT --frames-per-sheet 4 --sheet-overlap 1

# 跳过拼图
video-analyst refine-prepare VIDEO -o OUTPUT --no-contact-sheets

# 完整选项
video-analyst refine-prepare VIDEO -o OUTPUT \
    --frames-per-sheet 4 --sheet-overlap 1 --contact-sheet-columns 2 \
    --max-frame-long-edge 1600 --contact-sheet-max-edge 4096 \
    --image-format jpg --jpeg-quality 2 \
    --subtitle-context-before 1.0 --subtitle-context-after 1.0 \
    --allow-partial --verbose
```

## 12. 测试结果

```
pytest -q
229 passed in 24s
```

- 现有 Epic 1/2 测试：135 个全绿
- Epic 3 新增测试：94 个全绿
  - plan_loader × 10
  - timestamp_parser × 11
  - extractor × 15（5/10/15 FPS、CFR/VFR、横竖屏、中英文路径、视频首尾、sha256、长边、实际 FPS）
  - deduplicate × 10（同时间同 sha、同时间不同 sha、跨区间 alias、合法连续帧不误删）
  - contact_sheet × 9（1/3/4/5/10 帧、2×2 顺序、重叠 1 帧、不足留白、无语义标签）
  - detail_frames × 6（首末帧、单帧区间、空、上限、reason、path 保留）
  - fine_batch_builder × 6（active 字幕、review metadata、coarse_context）
  - epic3_cache × 5（layout、clear tree、clear interval、state、stages）
  - epic3_pipeline × 12（全成功、空 plan、缓存命中、user_request 不抽帧、布局不抽帧、refresh all、refresh 单区间、partial 失败、manifest 原子、pipeline_state 更新）
  - epic3_resume × 6（resume 后恢复、成功区间不受后续失败影响、refresh 单区间保留其他、refresh all 重抽、manifest 状态记录、空 plan 写空 index）

## 13. 离线端到端验收结果

用 lavfi 生成 8s 测试视频，seed 3 个 interval（5/10/15 FPS），全离线运行：

```
--- Run 1 ---
status: completed
intervals: 3 succeeded: 3 cached: 0 failed: 0
frames: 45 sheets: 15
--- fine_interval_0002 ---
fps: 10
frames: 20
contact_sheets: 7
detail_frames: 2
subtitles: 1
questions: ['what happens between 4.0s and 6.0s?']
coarse_context keys: ['source_batch_ids', 'evidence_frame_ids', 'coarse_analysis_paths']
--- Run 2 (cache) ---
cached: 3 succeeded: 0
frame mtime unchanged: True
--- Run 3 (user_request change) ---
frame mtime unchanged: True
--- Run 4 (refresh interval_0002) ---
succeeded: 1 cached: 2
END-TO-END OK
```

验收清单全部通过：

1. 多个 interval 正确抽帧 ✓
2. 每个 interval 使用计划指定 FPS ✓
3. 时间戳精确（单调递增、不越界、间隔接近目标 FPS）✓
4. 拼图顺序正确 ✓
5. 相邻拼图重叠一帧 ✓
6. fine batch 包含问题和粗分析来源 ✓
7. 第二次运行全部命中缓存（frame mtime 不变）✓
8. 修改 user_request 不重新调用 FFmpeg ✓
9. 单 interval refresh 后可恢复 ✓
10. 不需要任何 API Key ✓

## 14. 失败恢复结果

- 单 interval 失败 → 其他成功 interval 的 frames / contact_sheets / fine_batch 保留在磁盘
- 第二次运行（无失败注入）→ 失败 interval 重试成功，已成功 interval 命中缓存
- `manifest.json` 每个 interval 独立记录 status（ok / cached / failed）
- `--allow-partial=false` 时单 interval 失败立即抛 `Epic3PipelineError`
- 失败 interval **不** 生成伪造的空成功结果

## 15. 已知限制

- **Groq Cloud HTTP 500**：上游外部依赖问题。Epic 3 不依赖任何视觉 API，不受影响。Epic 2 用 `--vision-provider mock` 离线运行。
- FFmpeg `-t` 边界 muxer 行为：showinfo 可能比实际输出文件多 1 条，已确定性裁剪；其他不一致硬失败。
- detail frame 选择为首末帧确定性规则，未实现"场景变化附近 / 文件细节较高"等扩展（接口已预留）。
- contact sheet 字体依赖系统 TTF；缺失时回退 Pillow 默认字体（仍可用，仅排版略差）。
- 跨区间去重只保留物理文件一份 + alias 引用；未实现感知哈希（spec 明确不强制）。

## 16. 明确未实现

本阶段 **不** 实现：

- 精细视觉模型 API 调用（Epic 4）
- 动作类型判断 / 摄像机运动判断 / 转场类型判断
- 文字动画语义分析 / OCR 引擎 / 光流
- 二次补帧模型决策
- 最终拉片报告 / GUI / NLE 工程导出

## 完成条件核对

```
refinement_plan.json
→ 精确 5 / 10 / 15 FPS 连续帧        ✓
→ temporal contact sheets            ✓
→ detail frames                       ✓
→ fine_visual_batches.json           ✓
```

整条链路可在 Mock Provider 下完整运行、缓存、恢复和验收。229 个测试全部通过。