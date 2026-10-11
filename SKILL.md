---
name: video-analyst
description: >
  Agent-assisted local video analysis. Local FFmpeg/Python scripts prepare
  media, ASR, coarse frames, adaptive refinement frames, and final reports.
  The current conversation Agent must actually open and inspect images, then
  write coarse / fine / terminal analysis JSON using existing schemas.
  No external vision API, mock vision provider, or agent-CLI provider.
  Use for /video-analyst, prepare, refine-prepare, terminal-prepare, finalize,
  inspect-prepare, import-url, status, validate-coarse, validate-fine,
  validate-terminal, subtitles, review, doctor, setup, or mixed JP/ZH/EN
  transcription.
---

# Video Analyst (in-session agent workflow)

## Product positioning

Video Analyst is **not** an automatic vision-API program.
It is a **Skill for the current coding Agent**:

```text
本地脚本粗抽帧
→ 当前对话 Agent 真正查看图片
→ 当前 Agent 写 coarse analysis 和 refinement plan
→ 本地脚本按照 plan 详细抽帧
→ 当前 Agent 真正查看精细帧
→ 当前 Agent 写 fine analysis
→ 本地脚本聚合并生成最终报告
```

### Local scripts own

* Video precheck, ffprobe, audio extract
* ASR + subtitle calibration/activation
* 1 FPS coarse frames + scene-change frames + coarse contact sheets
* Low-resolution grayscale pixel near-duplicate filtering for coarse inputs,
  with first/last/scene/pinned retention and an audit mapping
* Adaptive 5/10/15 FPS extraction, contact sheets, high-res detail frames
* Deterministic absolute-time inspection inputs on explicit Agent request
* Explicit one-public-URL `yt-dlp` import with subtitle selection and source manifest
* Timestamp calculation, cache, schema/evidence validation
* Event merge, shot/section aggregation
* Final Markdown / JSON / CSV reports

### Current conversation Agent owns

* **Actually opening images** (coarse frames, contact sheets, fine sheets, detail frames)
* Understanding picture content, structure, shots, text/graphic motion
* Writing `CoarseBatchAnalysis`, `VideoStructure`, `RefinementPlan`
* Writing `FineBatchAnalysis`, temporal events, fine aggregation
* Answering refinement questions; optional one terminal refinement
* Never inventing motion from filenames/timestamps alone

```text
脚本不做视觉推理。
当前对话 Agent 不负责媒体抽帧。
```

## Critical rules for agents

1. **You must open and view images** with the image read tool. Do not claim inspection from filenames, dimensions, timestamps, or subtitles alone.
2. **Default multilingual ASR ON** — do not force `--language` on mixed JP/ZH/EN.
3. LLM calibrate is **not** translation/polish.
4. Downstream reads only active `subtitles/subtitles.json` + `review.json`.
5. Subtitles are supporting context, not absolute ground truth.
6. **No vision API Key / Base URL / vision model website.**
7. **Never call mock or external vision providers** to fill analysis.
8. Write analysis JSON with **existing bundled JSON schemas only**.
9. Evidence must cite real `frame_id` / `timestamp` / `batch_id` / image paths that exist.
10. Refinement plan: short intervals, lowest sufficient FPS, never full-video 15 FPS.
11. Terminal refinement at most once; third round forbidden.
12. When unknown, mark unresolved — do not invent schema filler.
13. Record provenance: `vision_execution_mode=in_session_agent`, `external_api_used=false`, `api_key_used=false`, `token_status=unavailable`, `estimated_cost=null`.

## Pipeline stages

```text
Stage 1 prepare
Stage 2 coarse-agent-analysis   ← YOU look at images
Stage 3 refine-prepare
Stage 4 fine-agent-analysis     ← YOU look at images
Stage 5 finalize
```

### Stage 1 — prepare

```bash
video-analyst prepare VIDEO -o OUTPUT
```

Produces agent-readable inputs under:

```text
OUTPUT/analysis/coarse/inputs/
  manifest.json
  batches/
  frames/
  contact_sheets/
```

Then:

```text
Preparation complete.
The current Agent must now inspect coarse images and write coarse analysis.
```

### Stage 2 — Agent coarse analysis

**Open and view:**

* `analysis/coarse/inputs/manifest.json`
* `analysis/coarse/inputs/frames/*`
* `analysis/coarse/inputs/contact_sheets/*`
* `subtitles/subtitles.json`

Per batch analyze content, subjects, scene, structure, shot changes, motion,
possible camera/post transforms, text/graphic elements, transitions, and what
needs higher FPS. Write:

```text
analysis/coarse/batches/<batch_id>.json   # CoarseBatchAnalysis
analysis/coarse/video_structure.json
analysis/coarse/refinement_plan.json
```

Then: `video-analyst validate-coarse OUTPUT`

### Refinement plan FPS

| FPS | When |
|---|---|
| 5 | Normal continuous motion / slow text-graphic change |
| 10 | Faster motion / enter-exit / scale-pan-mask / transitions |
| 15 | Flash, micro transitions, fast UI/text, unresolved 10 FPS edges |

Each interval needs: `interval_id`, `start`, `end`, `fps`/`requested_fps`,
`priority`, `reason(s)`, `refinement_questions`/`questions`,
`source_batch_ids`, `evidence_frame_ids`.

### Stage 3 — refine-prepare

```bash
video-analyst refine-prepare VIDEO -o OUTPUT
```

### Stage 4 — Agent fine analysis

**Open and view:**

* `analysis/refinement/visual_inputs/fine_batches.json`
* `analysis/refinement/contact_sheets/`
* `analysis/refinement/detail_frames/`
* coarse outputs + active subtitles

Write:

```text
analysis/fine/batches/<batch_id>.json
analysis/fine/temporal_events.json
analysis/fine/fine_video_analysis.json
analysis/fine/second_refinement_plan.json
```

If terminal not needed: empty intervals + `terminal_refinement_status=not_required`.

Then: `video-analyst validate-fine OUTPUT`

### Terminal (once only)

```bash
video-analyst terminal-prepare VIDEO -o OUTPUT
# Agent inspects terminal frames → write analysis/terminal_fine/
video-analyst validate-terminal OUTPUT
```

### Stage 5 — finalize

```bash
video-analyst finalize OUTPUT
```

Deterministic only — no re-vision.

## Installation

```bash
# 包 + 入口（零第三方依赖：stdlib + ffmpeg/ffprobe/yt-dlp；PIL 仅可选增强）
ls ./video_analyst/          # python 包
./bin/video-analyst --help                          # CLI 入口（PATH 内）

# 检查环境 / ASR 通道 / Groq key 状态
video-analyst doctor
```

* 命令全集（已实现，全部可用）：`prepare / refine-prepare / terminal-prepare / finalize /
  validate-{coarse,fine,terminal} / inspect-prepare / import-url / status /
  doctor / clean / run / setup-key`（实现于 `video_analyst/` 包，stdlib-only）。
* ASR：本地 `whisper/whisper.cpp/faster-whisper` → 缺省 Groq Whisper
  （`whisper-large-v3`）；两者都没有时 `asr=unavailable`（prepare 如实跳过并记录）。
* 证据校验：`validate-*` 自写 stdlib JSON-schema 校验器
  （`video_analyst/schemas/*.json`），硬规则 9/10/12 语义检查。

## 密钥与外部服务（Keys & external services）

* 唯一外部服务：**Groq Whisper**（云端 ASR，`whisper-large-v3`）。
  key 只存本机 `~/.config/video-analyst/config.json`（chmod 600，写入后校验权限），
  优先级：环境变量 `GROQ_API_KEY` > config 文件。
* 交互协议（当 Skill 检测到缺 key 时，当前 Agent 按此引导使用者）：
  1. **检测**：`video-analyst doctor` 输出 `groq_api_key: MISSING`，并附固定指引：
     `需要 GROQ_API_KEY（Groq Whisper 云端 ASR）→ 申请：https://console.groq.com/keys → 把 key 发给当前 Agent，Agent 会帮你写入 ~/.config/video-analyst/config.json`
  2. **告知**：向使用者说明 key 名（GROQ_API_KEY）、用途（Groq Whisper 云端 ASR）、申请地址（https://console.groq.com/keys）。
  3. **使用者把 key 发给当前 Agent**；Agent 执行 `video-analyst setup-key groq <KEY>`（或 `--stdin`）写入 config（chmod 600）。
  4. **复查**：`video-analyst doctor` 显示 `groq_api_key: present`（仅 presence，永不打印值）。
* 安全声明：key 值仅存于本机 config（600），**绝不写入分析 JSON / manifest / 报告 / 字幕 / state.json / 日志**；
  `doctor`/`status` 只显示 `present|MISSING`；key 不随分析产物或记忆外传。
* 离线说明：无 key 或无本地 whisper 时，`prepare` 跳过 ASR 并把 `asr=unavailable`
  与原因写入 manifest provenance；绝不假转录。

## CLI summary（全部已实现，可用）

```bash
video-analyst import-url PUBLIC_URL -o LOCAL_IMPORT_DIR
video-analyst prepare VIDEO -o OUTPUT
video-analyst inspect-prepare VIDEO -o OUTPUT --at TIME [--at TIME ...]
video-analyst inspect-prepare VIDEO -o OUTPUT --start START --end END --fps 5|10|15
video-analyst refine-prepare VIDEO -o OUTPUT
video-analyst terminal-prepare VIDEO -o OUTPUT
video-analyst finalize OUTPUT
video-analyst status OUTPUT
video-analyst validate-coarse OUTPUT
video-analyst validate-fine OUTPUT
video-analyst validate-terminal OUTPUT
video-analyst doctor
video-analyst clean OUTPUT --all-generated
video-analyst run VIDEO -o OUTPUT   # local stages only; pauses for agent vision
video-analyst setup-key groq <KEY>  # key 协议：写入 ~/.config/video-analyst/config.json (600)
```

### Optional targeted inspection

After reading the active subtitle document, the current Agent may decide that
semantic wording such as “看这里” or “注意” calls for a closer look and then
run `inspect-prepare`. This is an Agent decision, not a script heuristic:
scripts never parse subtitles to infer semantic cues. `--at`, `--start`, and
`--end` accept absolute seconds, `MM:SS`, or `HH:MM:SS`. `--at` values are
pinned, so they are never removed by pixel
near-duplicate filtering. Inspection artifacts live under
`analysis/inspection/`, do not change the workflow state, and are not consumed
automatically by `finalize`.

### Explicit URL import

`import-url` accepts one public HTTP(S) URL, invokes `yt-dlp` with
`--no-playlist` and `--ignore-config`, passes no cookie arguments, and writes a
`source_manifest.json` next to the downloaded files. Subtitle selection is
`all` by default or the explicit `zh.*,ja.*,en.*` option. The main `prepare`
command remains local-file-only; URL import is never implicit.

## Status machine

```text
not_started
preparing
waiting_for_coarse_agent_analysis
coarse_analysis_complete
preparing_refinement
waiting_for_fine_agent_analysis
fine_analysis_complete
waiting_for_terminal_agent_analysis
terminal_analysis_complete
ready_to_finalize
completed
failed
```

## Agent inspection report requirement

When finishing a real analysis, report the IDs you **actually opened**:

* coarse frame IDs
* coarse contact sheet IDs
* fine contact sheet IDs
* detail frame IDs
* terminal evidence (if any)

Do not only claim “已查看”.

## Removed product modes

```text
mock | openai-compatible | anthropic-compatible | agent-cli | interactive-agent
API Key / Base URL / vision model website / provider-check / pricing budget
task-pack ZIP / agent-export-import / manual JSON paste / cookie scraping
```

The corresponding runtime modes were intentionally removed from the product.
Reusable pieces kept: JSON schemas, response parsers, evidence validators,
JSON repair helpers, cache, aggregators, report generators, ASR (incl. Groq Whisper).
