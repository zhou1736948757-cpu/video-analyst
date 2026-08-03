THR-347 Verification Report — Epic 1 路由合规基础层独立验证
================================================================
Date: 2026-07-21
Agent: workbuddy (6df25bf7)
Upstream: THR-360 — Broker-controlled foundation (completed)
Scope: Groq cloud path only (per final user instruction)

## Verification Summary

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Non-null timestamp/timecode | **PASS** | `asr/groq_whisper.py:normalize_provider_response()` clamps null/negative to valid floats; end ≥ start invariant; synthetic segment fallback for missing segments |
| 2 | Normalized provider outputs | **PASS** | `_engine` set to `ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value` ("groq-whisper-large-v3-turbo"); `_model` set from payload or constant |
| 3 | Cache isolation | **PASS** | Dedicated Groq paths via `OutputLayout` (`asr_groq_raw_dir`, `groq_response_json`); all manifest/records carry `provider: "groq"` |
| 4 | GROQ_API_KEY in .env only | **PASS** | `.env` is sole location; no hardcoded keys in source; all reads via `os.environ.get()` |
| 5 | No-network/no-real-provider boundary | **PASS** | All 51 tests use mocks (`groq_available`, `transcribe_chunked_groq`, `run_command`); no real API/ffmpeg calls at test time |
| 6 | Full test suite passing | **PASS** | 51/51 passed in 1.41s — including 8 new normalization/timecode tests + 3 ASR pipeline tests |

## Detailed Evidence

### 1. Non-null timestamp/timecode

**Source** — `src/video_analyst/asr/groq_whisper.py:31-72`:
```python
s["start"] = max(float(s.get("start") or 0.0), 0.0)          # ↓ lines 59-61
raw_end = float(s.get("end") or s.get("start") or 0.0)
s["end"] = max(raw_end, s["start"])
```
- Null/negative start → clamped to 0.0
- end < start → end clamped to start
- Empty text segments dropped (skip non-empty invariant)
- Missing segments list → synthetic segment from payload text + duration

**Tests** — `tests/test_scene_timecode.py`:
| Test | What it verifies |
|---|---|
| `test_normalize_segments_get_valid_times` | null/negative start/end → valid floats, start ≥ 0, end ≥ start |
| `test_normalize_end_never_less_than_start` | end clamped to start when end < start |
| `test_normalize_text_stripped` | leading/trailing whitespace removed |
| `test_normalize_drops_empty_segments` | blank text segments removed, re-indexed sequentially |
| `test_normalize_creates_segments_from_text_when_missing` | synthetic segment from text+duration when segments missing |

### 2. Normalized provider outputs

**Source** — `src/video_analyst/asr/groq_whisper.py:41-44`:
```python
if "_engine" not in result:
    result["_engine"] = ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value  # "groq-whisper-large-v3-turbo"
if "_model" not in result:
    result["_model"] = str(result.get("model", GROQ_MODEL))
```

**Enum definition** — `src/video_analyst/models.py:17-19`:
```python
class ASREngine(str, Enum):
    GROQ_WHISPER_LARGE_V3_TURBO = "groq-whisper-large-v3-turbo"
    LOCAL_WHISPER_SMALL = "local-whisper-small"
```

**Test** — `test_normalize_supplies_provider_markers`: asserts `_engine == ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value`.

### 3. Cache isolation

**Source** — `src/video_analyst/cache/store.py:56-58` — `OutputLayout` creates provider-specific paths:
- `asr_groq_raw_dir = asr_raw_dir / "groq_responses"`  — per-chunk raw Groq responses
- `groq_response_json = raw_dir / "groq_response.json"` — merged Groq payload
- `asr_raw_json = cache_dir / "asr_raw.json"` — legacy flat cache (Groq-only in current regime)

**All Groq writes in** `chunked.py` pass through these dedicated paths. Every record carries `provider: "groq"` (e.g. `ASRChunkRecord.provider`, `ASRManifest.provider`).

**Note**: `cache_dir / "asr_raw.json"` is a provider-agnostic path name. No cross-provider collision currently exists since only the Groq path writes to it. If a future change re-enables parallel provider writing, this path would need partitioning.

### 4. GROQ_API_KEY in .env only

`.env` file at project root is the sole location:
```
GROQ_API_KEY=gsk_*********************************************
```
No hardcoded keys in any `src/` file. All reads via `os.environ.get("GROQ_API_KEY")` across 5 files:
- `groq_whisper.py:76,88` — `groq_available()`, `transcribe_with_groq()`
- `pipeline.py:89` — error message
- `cli.py:161` — CLI help text
- `calibrate.py:59,139,214` — LLM calibration
- `env/detect.py:164` — env detection
- `local_whisper.py:52` — error message

### 5. No-network/no-real-provider boundary

All 51 tests run with mocked external dependencies:

**`tests/test_asr_pipeline.py`** — 3 tests, all mock both `groq_available` and `transcribe_chunked_groq`:
- `test_transcribe_prefers_groq` — mock available + mock transcribe → returns payload
- `test_transcribe_raises_when_groq_unavailable` — mock unavailable → ASRError("GROQ_API_KEY")
- `test_transcribe_raises_when_groq_fails` — mock available + mock raises → ASRError("Groq ASR failed")

No real provider calls, no media uploads, no network connections at runtime.

## Source Paths Summary

| File | Role |
|---|---|
| `src/video_analyst/asr/groq_whisper.py:31-72` | `normalize_provider_response()` — timecode normalization contract |
| `src/video_analyst/asr/groq_whisper.py:75-163` | `groq_available()`, `transcribe_with_groq()` — Groq API client |
| `src/video_analyst/asr/chunked.py:72,104,152,305` | `normalize_provider_response()` at every cache-write boundary |
| `src/video_analyst/asr/pipeline.py:55-105` | `transcribe_audio()` — routing logic (Groq primary) |
| `src/video_analyst/cache/store.py:56-58` | `OutputLayout` — Groq-specific cache paths |
| `src/video_analyst/models.py:17-19` | `ASREngine` enum definition |
| `tests/test_asr_pipeline.py` | 3 ASR pipeline tests (all mocked) |
| `tests/test_scene_timecode.py` | 8 normalization + timecode tests |
| `.env` | `GROQ_API_KEY` (sole location) |

## Exceptions / Observations

| Observation | Impact |
|---|---|
| `pipeline.py` still imports `local_whisper_installed` and `transcribe_with_local_whisper` (lines 11-14) and retains fallback path (lines 73-86) | Dead code when Groq is available; not an issue for Groq-only regime |
| `cache_dir / "asr_raw.json"` is provider-agnostic in name | No collision risk in current single-provider execution; would need partitioning if parallel providers return |

## Test Command

```
cd /Users/mac/antigravity/default_work && .venv/bin/python -m pytest tests/ -v
```
Result: 51 passed in 1.41s.

## Disposition

All acceptance criteria **PASS**. No deficiencies identified. Upstream THR-360 broker-controlled foundation provides sufficient evidence for Groq cloud compliance.
