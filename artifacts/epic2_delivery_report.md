# Epic 2 Delivery Report

## 1. New / modified files

### New modules (Epic 2)

```text
src/video_analyst/vision/
├── __init__.py
├── base.py                # VisionProvider ABC + repair hook
├── models.py              # SCHEMA_VERSION, VisionBatchRequest, CoarseBatchAnalysis, Observation, Inference, RefinementRequest …
├── prompts.py             # PROMPT_VERSION, build_system_prompt/build_user_prompt/build_repair_user_prompt
├── registry.py            # register_provider / create_vision_provider (mock, openai-compatible, gemini)
├── mock.py                # MockVisionProvider with 7 scenarios (normal/empty/malformed/timeout/rate_limit/partial_failure/needs_refinement)
├── openai_compatible.py   # OpenAICompatibleProvider (POST /chat/completions, retry/backoff/timeout, redaction)
├── gemini.py              # explicit fail-fast stub (until future Epic)
└── request_builder.py     # build_request_for_batch / build_all_batch_requests (reads only active subs + review.json)

src/video_analyst/coarse_analysis/
├── __init__.py
├── models.py              # CoarseAnalysisManifest, CoarseAnalysisResult, VideoSection, VideoStructure, VideoGap, RefinementInterval, RefinementPlan
├── validation.py          # cross-stage contract checks
├── batch_runner.py        # run_coarse_analysis, manifest load/save, cache key, raw+analysis persistence, single-batch repair path
├── aggregate.py           # deterministic aggregate_structure + save_structure (gaps merged, failed batches ≠ empty)
└── refinement.py          # build_refinement_plan: pad → merge → split → budget, priority & FPS ranking

src/video_analyst/pipeline/
├── __init__.py            # re-exports run_phase1
├── phase1.py              # (renamed from pipeline.py for clean package layout)
└── epic2.py               # run_epic2: batch runner → aggregate → refinement plan → manifest → pipeline state

tests/_epic2_fixtures.py   # shared fixtures (seed_layout, make_*_json, make_config)
```

### New test files (10 files / 84 tests)

```text
tests/test_vision_provider_base.py         3 tests
tests/test_mock_vision_provider.py        10 tests
tests/test_openai_compatible_provider.py   8 tests
tests/test_vision_request_builder.py       9 tests
tests/test_coarse_response_validation.py  12 tests
tests/test_coarse_batch_runner.py         11 tests
tests/test_coarse_aggregation.py           8 tests
tests/test_refinement_plan.py             12 tests
tests/test_epic2_cache.py                  4 tests
tests/test_epic2_pipeline.py               7 tests
```

### Modified files (extensions only — no parallel systems)

```text
src/video_analyst/models.py        # added 5 PipelineStage entries + 3 PipelineState fields
src/video_analyst/exceptions.py    # added 9 Epic 2 exceptions
src/video_analyst/config.py        # VisionConfig + CoarseAnalysisConfig already present
src/video_analyst/cache/store.py   # OutputLayout: analysis_dir, coarse_*, clear_epic2_tree()
src/video_analyst/cli.py           # added video-analyst coarse-analyze subcommand
src/video_analyst/__init__.py      # (unchanged)
.grok/skills/video-analyst/SKILL.md
README.md
```

The `pipeline.py` script was converted into a `pipeline/` package so Epic 2
orchestration can live alongside Phase 1 (`pipeline/phase1.py` + `pipeline/epic2.py`).

### Test totals

```text
Phase 1 (existing):    51 tests
Epic 2 (new):         84 tests
TOTAL:               135 tests
```

---

## 2. Vision Provider architecture

```text
                ┌─────────────────────────┐
                │  VisionProvider (ABC)   │
                │  provider_name / model  │
                │  analyze_batch()         │
                └────────────┬────────────┘
                             │
       ┌─────────────┬───────┼───────────────────┐
       │             │       │                   │
┌──────▼──────┐ ┌────▼─────┐ ┌▼─────────────┐
│   Mock       │ │ OpenAI-  │ │   Gemini     │
│  Provider    │ │Compat    │ │   stub       │
│              │ │ Provider │ │ (fail-fast)  │
└──────────────┘ └────┬─────┘ └──────────────┘
                      │ uses httpx (mocked in tests)
                      ▼
              POST {BASE_URL}/chat/completions
              Authorization: Bearer $VIDEO_ANALYST_VISION_API_KEY
              response_format: {"type": "json_object"}
              temperature: 0.0
              max_tokens: 4096
```

### Provider contract

Every provider must:

1. Build the request from `VisionBatchRequest` (frame paths → API image format, plus prompts via `vision.prompts`).
2. Send the request with `request_timeout` (env or VisionConfig).
3. Retry `429` and `5xx` with exponential backoff up to `max_retries`.
4. NEVER retry `4xx` (other than `429`) or parse errors twice.
5. Convert raw response into the **unified** `CoarseBatchAnalysis` shape via `vision.response_parser`.
6. Raise domain exceptions: `VisionTimeoutError`, `VisionRateLimitError`, `VisionRequestError`, `VisionResponseParseError`, `VisionResponseValidationError`, `VisionProviderUnavailableError`.
7. NEVER log API keys (the `logging.RedactingFilter` is applied globally; providers only include text strings).

### Supported Providers

| Provider name        | Status               | Notes |
|----------------------|----------------------|-------|
| `mock`               | ✅ implemented       | 7 scenarios, deterministic, used for offline verification |
| `openai-compatible`  | ✅ implemented       | POST `/chat/completions`, base URL + key + model from env |
| `gemini`             | 🟡 interface only    | Fail-fast `VisionProviderUnavailableError`; ready for future Epic |

`create_vision_provider(name, vision_config)` always returns a `VisionProvider`.

---

## 3. Vision request data structure

```python
class VisionBatchRequest(BaseModel):
    schema_version: str = "1.0"
    batch_id: str                          # e.g. "batch_0001"
    video_id: str
    batch_start: float                     # inclusive
    batch_end: float                       # exclusive
    frames: list[VisionFrame]              # 1 FPS + scene-protection frames
    subtitles: list[SubtitleContext]       # active subs overlapping the window
    subtitle_metadata: SubtitleMetadata     # active_source/reviewed/needs_review/asr_model/calibration_model
    user_request: str
    analysis_dimensions: list[str]         # default: structure, subjects, camera, motion, text, transition, refinement_needs
    previous_context: PreviousBatchContext | None
```

`VisionFrame`: `frame_id`, `timestamp_seconds`, `timecode`, `path: Path`, `sources: list[str]`, `scene_score: float | None`

`SubtitleMetadata`:
- `active_source: "asr" | "llm"` (always from review.json or active subtitles.json — never `subtitles_asr.json` / `subtitles_llm.json`).
- `reviewed`, `needs_review`, `asr_model`, `calibration_model`.

Request builder always reads `phase2/visual_batches.json` + `subtitles/review.json` (or active `subtitles/subtitles.json` fallback). It NEVER reads `subtitles_asr.*` or `subtitles_llm.*` directly.

---

## 4. Coarse analysis output schema

```text
schema_version: "1.0"            # stable contract
batch_id, start, end             # echo back for traceability
summary                          # short natural-language description
direct_observations              # things literally visible in the frames
inferred_interpretations         # guesses explicitly labeled as such
visible_text                     # on-screen text (titles, UI)
subjects, scene_changes,         # category-tagged observations
camera_observations, motion_observations,
possible_transitions,
subtitle_visual_relationship,
important_moments
refinement_requests              # 5/10/15 FPS requests with priority + reason + evidence
confidence                       # float 0..1
warnings                         # parsed + auto-emitted
```

Rules enforced by `vision.response_parser`:
- `confidence` clamped to `[0, 1]` with warning.
- `time_range` clamped to batch bounds; unknown frame IDs dropped with warning.
- `requested_fps` snapped to one of `{5, 10, 15}`.
- Empty / whitespace `reason` raises `VisionResponseValidationError`.
- `Observation` and `Inference` live in separate fields; never mixed.
- `inferred_interpretations` requires `description + confidence + alternatives + supporting_observation_ids`.

---

## 5. Batch execution and recovery

The batch runner (`coarse_analysis.batch_runner.run_coarse_analysis`) implements:

1. **Iterate** `phase2/visual_batches.json` in time order.
2. **For each batch**: build request → compute cache key → call provider (or hit cache) → parse → validate → atomic-persist `batches/batch_XXXX.json` and `raw/batch_XXXX_response.json` → update manifest → proceed.
3. **Retry policy**: `VisionTimeoutError`, `VisionRateLimitError`, `VisionRequestError` retry up to `vision_cfg.max_retries` with exponential backoff (`retry_backoff_base ** attempt`).
4. **Repair policy**: `VisionResponseParseError` and `VisionResponseValidationError` trigger **one** `provider.build_repair_payload → provider.parse_repair_response → re-validate` cycle. If repair also fails the batch is marked `failed` and the run continues.
5. **Cache key** includes: `batch_id`, frame IDs, frame file signature (size only — no mtime), `selected/omitted/preserved_scene_frame_ids`, subtitle signature (content hash of `subtitles/subtitles.json` + `subtitles/review.json`), `user_request`, `analysis_dimensions`, `provider_name`, `model`, `prompt_version`, `schema_version`. *Exclude `previous_context`* for cache stability across runs.
6. **Resume**: a previously-successful batch with the same cache key is reused without a provider call. The provider is only invoked on missing/changed batches.
7. **Refresh controls**:
   - `--refresh-vision` wipes `analysis/coarse/` and rebuilds from scratch.
   - `--refresh-batch batch_XXXX` reruns only that batch.
   - Changing only `--refinement-budget-ratio` does NOT rerun the vision model (covered by `test_pipeline_budget_change_does_not_rerun_vision`).
8. **Partial success**: controlled by `coarse_analysis.allow_partial`. Default `false` raises `VisionBatchAnalysisError` if any batch failed; with `true` the run continues and the aggregation step emits `VideoGap` records for the failed ranges.

---

## 6. Full-video aggregation logic

`aggregate_structure` (`coarse_analysis.aggregate`) is fully deterministic and LLM-free (the spec marks the optional model summarization as disabled by default):

1. Walk batches in time order (driven by `batches_meta` from the VisualBatches JSON).
2. For each **successful** batch with a sufficiently word-similar subject/scene fingerprint, **merge** into the previous section if boundaries are continuous.
3. For each **failed** batch, emit a `VideoGap(start, end, failed_batch_ids=[...])`. Adjacent gaps are merged.
4. Build a single `overall_summary` like `"Composed of N sections between Xs and Ys. K gap(s) covering Zs of content."`.
5. Populate `recurring_subjects` and `recurring_visual_patterns` (word-frequency over `subjects` + `scene_changes` + `camera_observations`).
6. Add `subtitle_visual_relationship` notes from each batch.
7. `unresolved_questions` always lists failed batches by ID; failed batches NEVER become "empty" sections.

`VideoStructure` saved to `analysis/coarse/video_structure.json` with `schema_version = "1.0"`.

---

## 7. Refinement plan generation rules

`build_refinement_plan` (`coarse_analysis.refinement`) enforces exactly the spec:

1. Start with every `RefinementRequest` from every successful batch.
2. **Pad** each interval by `cfg.refinement_pad_seconds` (default `0.5`) on both ends.
3. **Merge** intervals that overlap OR are within `pad` seconds of each other. Merged interval keeps the union of reasons/questions/evidence. `requested_fps` becomes `max(current, incoming)`. `priority` becomes `max(current, incoming)` using rank `{low:1, medium:2, high:3}`.
4. **Split** any interval longer than `cfg.refinement_max_interval_seconds` (default `15`) into chunks.
5. **Clip** to `[0, video_duration]`.
6. **Apply budget** = `video_duration * cfg.refinement_budget_ratio` (default `0.20`). Greedy, sorted by priority desc, then FPS desc, then start time. Anything that doesn't fit becomes a `deferred_intervals` entry — **never silently dropped**.
7. `estimated_frame_count = max(0, round((end-start) * requested_fps))`.
8. `total_requested_duration` only sums accepted intervals.

Output saved to `analysis/coarse/refinement_plan.json`.

---

## 8. Cache dependencies

| Change in input                                   | Invalidates            |
|---------------------------------------------------|------------------------|
| 1 FPS / scene frames                              | batch + structure + plan (only affected batches) |
| active subtitle source/reviewed flag/text         | batch + structure + plan |
| `user_request`                                    | batch only |
| vision provider / model / prompt version          | batch only |
| `schema_version` / `coarse-vision-v1` prompt bump | batch only |
| only `refinement_budget_ratio`                    | plan only — vision cache untouched |
| only `analysis_dimensions`                        | batch only (rare) |

Each batch has its own `cache_key` stored in `manifest.json` →
`BatchRunRecord.cache_key`. A second run with the same cache key reuses
the on-disk `batches/batch_XXXX.json` and skips the provider entirely
(covered by `test_second_run_hits_cache_no_reinvocation`).

---

## 9. CLI usage

```bash
# Mock (offline) — recommended first run
video-analyst coarse-analyze VIDEO -o OUT \
    --vision-provider mock \
    --user-request "分析镜头节奏、字幕动效和转场"

# OpenAI-compatible with API key from env
export VIDEO_ANALYST_VISION_API_KEY="..."
export VIDEO_ANALYST_VISION_BASE_URL="https://api.openai.com/v1"
export VIDEO_ANALYST_VISION_MODEL="gpt-4o-mini"
video-analyst coarse-analyze VIDEO -o OUT --vision-provider openai-compatible

# Refresh controls
video-analyst coarse-analyze VIDEO -o OUT --refresh-vision
video-analyst coarse-analyze VIDEO -o OUT \
    --refresh-batch batch_0001 --refresh-batch batch_0002

# Plan-only rebuild (no vision re-call)
video-analyst coarse-analyze VIDEO -o OUT --refinement-budget-ratio 0.30

# Safety: allow partial success; verbose logging
video-analyst coarse-analyze VIDEO -o OUT --allow-partial --verbose
```

All flags from the spec are supported: `--vision-provider`, `--vision-model`,
`--vision-base-url`, `--user-request`, `--analysis-dimensions` (comma),
`--max-retries`, `--request-timeout`, `--allow-partial / --no-allow-partial`,
`--refresh-vision`, `--refresh-batch` (repeatable),
`--no-aggregate-model / --allow-aggregate-model`,
`--refinement-budget-ratio`, `--verbose`.

---

## 10. Test results

```text
$ python -m pytest -q
........................................................................ [ 53%]
...............................................................          [100%]
135 passed in 0.95s
```

Per-file breakdown:

```text
  3  tests/test_asr_pipeline.py                  (Phase 1)
  1  tests/test_cache.py                         (Phase 1 + Epic2 layout assertions)
  2  tests/test_chunk_plan.py                    (Phase 1)
  8  tests/test_coarse_aggregation.py            (Epic 2)
 11  tests/test_coarse_batch_runner.py           (Epic 2)
 12  tests/test_coarse_response_validation.py    (Epic 2)
  4  tests/test_env_detect.py                    (Phase 1)
  4  tests/test_epic2_cache.py                   (Epic 2)
  7  tests/test_epic2_pipeline.py                (Epic 2)
  7  tests/test_llm_calibrate.py                 (Phase 1)
  3  tests/test_merge_chunks.py                  (Phase 1)
  5  tests/test_merge_rules.py                   (Phase 1)
  9  tests/test_mixed_language_fixture.py        (Phase 1)
 10  tests/test_mock_vision_provider.py          (Epic 2)
  8  tests/test_openai_compatible_provider.py    (Epic 2)
  2  tests/test_phase2_mapping.py                (Phase 1 / Phase 2 prep)
  4  tests/test_probe_audio.py                   (Phase 1)
 12  tests/test_refinement_plan.py               (Epic 2)
  8  tests/test_scene_timecode.py                (Phase 1)
  2  tests/test_srt.py                           (Phase 1)
  1  tests/test_subtitle_convert.py              (Phase 1)
  3  tests/test_vision_provider_base.py          (Epic 2)
  9  tests/test_vision_request_builder.py        (Epic 2)
```

### Required Epic 2 test coverage (spec → coverage)

#### Provider

| Spec requirement                              | Test |
|-----------------------------------------------|------|
| Mock normal results                           | `test_mock_normal_returns_structured_analysis` |
| Mock timeout                                  | `test_mock_timeout_raises` |
| Mock invalid JSON                             | `test_mock_malformed_raises_parse_error` |
| Mock partial failure                          | `test_mock_partial_failure_clamps_invalid_values` |
| OpenAI-compatible request parameters          | `test_sends_messages_with_images` |
| API Key missing                               | `test_missing_api_key_raises` |
| Log redaction                                 | `test_log_redacts_bearer_token` |
| Provider registry                             | `test_registry_can_resolve_mock` |

#### Request Builder

| Spec requirement                              | Test |
|-----------------------------------------------|------|
| Reads active subtitles                        | `test_reads_only_active_subtitles_for_metadata` |
| Does NOT read asr/llm files directly          | `test_does_not_read_asr_or_llm_files_directly` |
| Frames sorted by time                         | `test_frames_sorted_by_time` |
| Scene-protection frames preserved             | `test_frames_sorted_by_time` |
| Image over-cap explicit `omitted_frame_ids`   | `test_frame_cap_drops_excess_with_explicit_omitted_list` |
| Subtitle metadata passed                      | `test_subtitle_metadata_reviewed_and_needs_review` |
| User request change invalidates cache hash    | `test_user_request_changes_cache_hash` |

#### Response validation

| Spec requirement                              | Test |
|-----------------------------------------------|------|
| Valid JSON                                    | `test_parse_markdown_fenced_json`, `test_parse_json_with_prose_around`, `test_validate_accepts_good_payload` |
| Markdown-wrapped JSON                         | `test_parse_markdown_fenced_json` |
| Time exceeds batch                            | `test_validate_clamps_out_of_range_time_range` |
| Evidence frame missing                        | `test_validate_drops_unknown_frame_ids` |
| confidence out of range                       | `test_validate_clamps_out_of_range_time_range` |
| FPS not 5/10/15                               | `test_validate_snap_invalid_fps` |
| Observation / Inference mixed                 | `test_validate_keeps_observations_and_inferences_separate` |
| Item missing fields                           | `test_validate_fixes_missing_items_with_warning` |

#### Batch runner

| Spec requirement                              | Test |
|-----------------------------------------------|------|
| Multi-batch in order                          | `test_processes_all_batches_in_order` |
| Resume after partial failure                  | `test_single_batch_failure_continues_when_partial` |
| Single-batch retry                            | `test_retry_then_succeed` |
| allow_partial                                 | `test_single_batch_failure_continues_when_partial`, `test_no_partial_raises_on_any_failure` |
| Cache hit on second run                       | `test_second_run_hits_cache_no_reinvocation`, `test_pipeline_resume_uses_cache` |
| refresh single batch                          | `test_refresh_batch_reruns_only_named` |
| refresh all                                   | `test_refresh_vision_clears_all`, `test_pipeline_refresh_vision` |
| Atomic write per batch                        | `test_atomic_write_each_batch` |

#### Aggregation

| Spec requirement                              | Test |
|-----------------------------------------------|------|
| Normal multi-batch                            | `test_aggregate_produces_sections_per_batch` |
| Adjacent scenes merged                        | `test_similar_consecutive_sections_merge` |
| Failed batches → gaps                         | `test_failed_batches_emit_gaps_not_empty_sections` |
| Failed batches ≠ empty content                | `test_no_sections_for_all_failed` |

#### Refinement plan

| Spec requirement                              | Test |
|-----------------------------------------------|------|
| Overlapping intervals merged                   | `test_overlapping_intervals_merge` |
| Take highest FPS                              | `test_fps_uses_max_of_overlaps` |
| Pad before/after                              | `test_close_intervals_within_pad_merge`, `test_intervals_padded_then_merged` |
| Video bound clipping                          | `test_video_bounds_clip` |
| Budget limit                                  | `test_budget_excludes_low_priority_overruns` |
| High priority first                           | `test_priority_picks_high_over_medium` |
| Deferred requests preserved                   | `test_budget_excludes_low_priority_overruns` |
| Budget change ≠ vision rerun                  | `test_budget_change_does_not_invalidate_vision_cache`, `test_pipeline_budget_change_does_not_rerun_vision` |

---

## 11. Mock end-to-end acceptance

`video-analyst coarse-analyze /tmp/v.mp4 -o /tmp/e2e_run --vision-provider mock --user-request "分析镜头节奏、字幕动效和转场" --allow-partial`

Result of `Pass 1` (first end-to-end run):

```text
Total batches:    3
Succeeded:        3
Failed:           0
```

Output tree:

```text
/tmp/e2e_run/analysis/coarse/
├── manifest.json              ← counts + per-batch records + cache keys
├── video_structure.json       ← sections / gaps / recurring subjects
├── refinement_plan.json       ← intervals / deferred_intervals
├── batches/
│   ├── batch_0001.json
│   ├── batch_0002.json
│   └── batch_0003.json
└── raw/
    ├── batch_0001_response.json
    ├── batch_0002_response.json
    └── batch_0003_response.json
```

Additional e2e probes (in `tests/test_epic2_pipeline.py` + ad-hoc shell):

| Scenario                                            | Result |
|-----------------------------------------------------|--------|
| Pass 1 — full successful run                        | success=3 failed=0 |
| Pass 2 — second run with same config (cache hit)    | success=3 failed=0, no provider call, manifest cache_keys unchanged |
| Pass 3 — `user_request` changed → rerun             | cache_keys differ |
| Pass 4 — only `refinement-budget-ratio` changed     | batch file mtime unchanged ⇒ vision NOT re-called |
| Pass 5 — single batch failure + `--allow-partial`   | succeeded=2 failed=1, gap emitted for failed range |
| Pass 6 — refinement plan with `needs_refinement` mock and budget=1.0 | 3 intervals emitted (high priority, 15 fps) |

Log redaction check: `test_log_redacts_bearer_token` asserts that
`sk-test` does not appear in any log record when the provider emits an
auth error. Confirmed.

---

## 12. Real API execution

Not executed in this delivery — Epic 2 explicitly does not require it,
and the OpenAI-compatible integration uses `httpx` with mocked calls
in `tests/test_openai_compatible_provider.py` (8 tests covering: request
shape, image encoding, 429/5xx/timeout retry/backoff, 4xx no-retry, log
redaction, and one-repair-cycle for parse failures).

Running the real provider only requires:

```bash
export VIDEO_ANALYST_VISION_API_KEY=...
export VIDEO_ANALYST_VISION_BASE_URL=https://api.openai.com/v1
export VIDEO_ANALYST_VISION_MODEL=gpt-4o-mini
video-analyst coarse-analyze VIDEO -o OUT --vision-provider openai-compatible
```

The Groq Cloud HTTP 500 issue mentioned in the spec is **recorded but not
addressed** here, since Epic 2 does not bind to Groq.

---

## 13. Known limits

1. **Groq Cloud HTTP 500** — recorded as upstream issue. Epic 2 is decoupled from Groq.
2. **Vision Gem** — placeholder (`GeminiVisionProvider`) raises
   `VisionProviderUnavailableError`. Real implementation requires
   `google-generativeai` client + upload-and-poll logic; tracked for the
   next Epic.
3. **Time mosaic / temporal contact sheet** — Epic 2 sends only the raw
   coarse frames. Contact-sheet templates can be plugged into the
   OpenAI-compatible provider later without changing the rest of the
   contract (request schema already includes `frame_id` and `sources`).
4. **5 / 10 / 15 FPS refinement** — out of scope for this Epic. The
   `RefinementPlan` lists exactly what needs to be re-extracted.
5. **OCR, optical flow, fine action analysis, final 拉片 report** —
   explicitly out of scope (Phase 3+).
6. **GUI / NLE export** — out of scope.
7. **Auto cloud upload of source video** — out of scope.

---

## 14. Out-of-scope confirmation

This delivery **does NOT** implement:

- ❌ Actual 5 / 10 / 15 FPS frame extraction
- ❌ Temporal contact-sheet compositing
- ❌ Fine action / optical-flow analysis
- ❌ OCR engine
- ❌ Final 拉片 report
- ❌ GUI
- ❌ NLE (Premiere / FCPX / DaVinci) project export
- ❌ Automatic upload of source video

The Epic 2 chain implemented and verified end-to-end with the Mock
provider is:

```text
visual_batches
  + active subtitles (from review.json)
  + subtitle source / review flags
  + user request
        │
        ▼
build VisionBatchRequest (request_builder.py)
        │
        ▼
VisionProvider.analyze_batch (mock / openai-compatible / gemini-stub)
        │
        ▼
parse + validate (response_parser.py, deterministic + one repair pass)
        │
        ▼
atomic write batches/batch_XXXX.json + raw/batch_XXXX_response.json
        │
        ▼
aggregate → video_structure.json (deterministic, no extra LLM call)
        │
        ▼
refinement plan → refinement_plan.json (pad, merge, FPS, priority, budget)
```

All steps are cacheable, resumable, retryable, and reusable under the
mock provider without any external API calls.
