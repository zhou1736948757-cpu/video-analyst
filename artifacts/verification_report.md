THR-360 Verification Report — Epic 1 Broker-Controlled Foundation
================================================================
Date: 2026-07-21
Agent: workbuddy (6df25bf7)
Epic: THR-360 — workbuddy Broker-controlled foundation implementation

## Acceptance Criteria

### AC1: Every scene frame contract prevents `time:null` and preserves traceable timecode

- `src/video_analyst/phase2/prepare.py:extract_scene_frames()`:
  - ffmpeg filter chain now includes `showinfo` (was `select=gt(scene,T)` only)
  - `re.findall(r"pts_time:([0-9.]+)", stderr)` extracts exact frame PTS timestamps
  - Each frame dict carries `time: round(t, 3)` — never None
  - Fallback: if showinfo parsing yields fewer timestamps than frames, estimates from index
  - Resume path: companion `_timestamps.json` persisted alongside frames, read on cache-hit
- `src/video_analyst/phase2/prepare.py:build_visual_batches()`:
  - Scene frames now distributed by `time` across all batch windows (was: first-batch-only extras)
  - Fixed-frame + scene-frame filtering both use the same `t <= time < end` window logic

### AC2: Provider output normalizes before cache and downstream subtitle artifacts

- `src/video_analyst/asr/groq_whisper.py:normalize_provider_response()`:
  - Guarantees every segment has valid float start/end (never None, never negative, end >= start)
  - Strips segment text, drops empty segments, re-indexes IDs sequentially
  - Supplies `_engine` / `_model` markers if raw response lacks them
  - Creates synthetic segment from payload text when segments list is empty
- Applied at every cache-write boundary in `src/video_analyst/asr/chunked.py`:
  - `transcribe_chunked_groq()` single-shot path
  - `transcribe_chunked_groq()` multi-chunk path (per-chunk raw response)
  - `_persist_single_shot()` before any cache/store writes
- Downstream artifacts (manifest, asr_raw, groq_response, ASRChunkRecord) receive normalized data

### AC3: Focused tests pass with no network/provider call

- `tests/test_scene_timecode.py` — 8 tests, all passing:
  - 6 normalization contract tests: null times, empty segments, missing segments, provider markers, end>=start, text stripping
  - 2 scene timecode tests: extract_scene_frames with mocked ffmpeg (parses showinfo), build_visual_batches distributes scene frames by timecode
- Full suite: 51 passed, 0 failed (8 new + 43 existing)

### AC4: Evidence sufficient for separately gated Groq validation

- Broker record at `artifacts/broker_record.json` (run_id: thr360_epic1_broker_foundation)
- Changed paths:
  - `src/video_analyst/asr/groq_whisper.py` — normalize_provider_response()
  - `src/video_analyst/asr/chunked.py` — normalize before cache writes
  - `src/video_analyst/phase2/prepare.py` — showinfo pts_time, resume timestamps, batch distribution
  - `tests/test_scene_timecode.py` — 8 focused tests

## Test Command

```
cd /Users/mac/antigravity/default_work && .venv/bin/python -m pytest tests/ -v
```

Output: 51 passed in 1.64s — all tests PASSED.

## Changed-Path Summary

| File | Change |
|---|---|
| `src/video_analyst/asr/groq_whisper.py` | Added `normalize_provider_response()`, `import re` |
| `src/video_analyst/asr/chunked.py` | `normalize_provider_response()` at cache-write boundaries |
| `src/video_analyst/phase2/prepare.py` | `showinfo` pts_time in `extract_scene_frames()`, `_timestamps.json` resume, scene frame batch placement |
| `tests/test_scene_timecode.py` | 8 focused tests for normalization and timecode contracts |
