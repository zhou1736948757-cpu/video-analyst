---
name: video-analyst
description: >
  Agent-assisted local video analysis. Local FFmpeg/Python scripts prepare
  media, ASR, coarse frames, adaptive refinement frames, and final reports.
  The current conversation Agent must actually open and inspect images, then
  write coarse / fine / terminal analysis JSON using existing schemas.
  No external vision API, mock vision provider, or agent-CLI provider.
  Use for /video-analyst, prepare, refine-prepare, terminal-prepare, finalize,
  status, validate-coarse, validate-fine, validate-terminal, subtitles, review,
  doctor, setup, or mixed JP/ZH/EN transcription.
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
* Adaptive 5/10/15 FPS extraction, contact sheets, high-res detail frames
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
8. Write analysis JSON with **existing Pydantic schemas only**.
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
video-analyst finalize VIDEO -o OUTPUT
```

Deterministic only — no re-vision.

## CLI summary

```bash
video-analyst prepare VIDEO -o OUTPUT
video-analyst refine-prepare VIDEO -o OUTPUT
video-analyst terminal-prepare VIDEO -o OUTPUT
video-analyst finalize VIDEO -o OUTPUT
video-analyst status OUTPUT
video-analyst validate-coarse OUTPUT
video-analyst validate-fine OUTPUT
video-analyst validate-terminal OUTPUT
video-analyst doctor
video-analyst clean OUTPUT --all-generated
video-analyst run VIDEO -o OUTPUT   # local stages only; pauses for agent vision
```

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
Reusable pieces kept: Pydantic schemas, response parsers, evidence validators,
JSON repair helpers, cache, aggregators, report generators, ASR (incl. Groq Whisper).
