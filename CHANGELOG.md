# Changelog

All notable changes to Video Analyst are documented here.
Format based on Keep a Changelog; versions follow semantic versioning.

## [0.6.0] — 2026-07-22 — Epic 6: Productization

### Added
- **Telemetry subsystem** (`src/video_analyst/telemetry/`): unified recording of
  all Provider calls — call count, input images, tokens, estimated cost, duration,
  retries, cache hits, error classification. Outputs `calls.jsonl`, `summary.json`,
  `errors.jsonl`. API Key / Authorization / base64 images never recorded.
- **Provider Capability Probe** (`video-analyst provider-check`): sends a minimal
  test request (no user video) to verify API key, endpoint, model, vision support,
  JSON mode, and usage readability.
- **Dry-run + budget guards** (`--dry-run`, `--max-estimated-cost`, `--max-api-calls`,
  `--confirm-api-call`): estimates pending API calls and cost before any network
  request; blocks real API invocation when budget exceeded or cost unknown.
- **Orthogonal report status** (`ReportRuntimeStatus`): `completion_status`,
  `provider_mode`, `terminal_refinement_status`, `analysis_completeness` are now
  independent fields instead of a single overlapping enum.
- **Product CLI**: `init` (generates `.env.example` + config template, no keys),
  `run` (one-shot Phase 1 → Epic 5 with `--through`/`--resume`/`--force`/`--dry-run`),
  `clean` (removes generated outputs, never source video), `doctor --json`.
- **Release readiness model** (`src/video_analyst/release.py`): assesses
  `development` / `release_candidate` / `production_ready` / `blocked_external`.
- **BYOK security hardening**: redacting filter covers `sk-`/`gsk_`/`Bearer`;
  telemetry redaction module; output directory scanned for secrets in tests.

### Changed
- Coarse and fine batch runners now route every Provider call through the
  global `TelemetryRecorder`.
- `FinalVideoAnalysis` now carries `runtime_status` (orthogonal) alongside the
  legacy `status` enum for backward compatibility.
- `README.md` restructured with full CLI reference and output layout.

## [0.5.0] — 2026-07-22 — Epic 5: Final Report

### Added
- Terminal refinement round (`refinement/round.py`, `fine/terminal.py`) reusing
  Epic 3 + Epic 4; third round forbidden (`max_additional_rounds=1`,
  `terminal_after_this_round=true`).
- Event merge (`final/merge.py`): `confirmed` / `revised` / `rejected` /
  `from_round1` with `superseded` preservation.
- Shot timeline, section aggregation, evidence index, Markdown / JSON / CSV
  report generation, `run_summary.json`.
- `answer_origin` field (`model` / `validator_fallback` / `terminal_refinement`)
  on `RefinementQuestionAnswer`.
- True high-resolution detail frame extraction (independent FFmpeg `-ss -frames:v 1`).

## [0.4.0] — 2026-07-22 — Epic 4: Fine Temporal Vision

### Added
- `FineVisionProvider` abstraction + Mock (12 scenarios) + OpenAI-compatible
  (reuses Epic 2 transport) + Gemini stub.
- Six analysis categories: subject motion, camera motion, post-transform, text
  animation, graphic animation, transition.
- `TemporalBoundary` (exact / bounded / approximate), `FineEvidenceReference`,
  direct-observation vs inference separation.
- Cross-batch event deduplication, second refinement plan generation.

## [0.3.0] — 2026-07-21 — Epic 3: Adaptive Refinement

### Added
- 5/10/15 FPS fine-frame extraction via FFmpeg `showinfo` pts_time parsing.
- Deterministic intra- + cross-interval deduplication by `|Δt| ≤ 5ms AND sha256`.
- Objective 2×N temporal contact sheets (frame_id / timecode / fps only).
- `FineVisualBatch` contracts for Epic 4.

### Fixed
- Schema version compatibility tightened: major version mismatch now raises
  `SchemaVersionUnsupportedError` instead of best-effort load.

## [0.2.0] — 2026-07-21 — Epic 2: Vision Coarse Analysis

### Added
- `VisionProvider` abstraction (mock / openai-compatible / gemini stub).
- Per-batch coarse analysis with retry/cache/repair, `video_structure.json`,
  `refinement_plan.json`.

## [0.1.0] — 2026-07-20 — Phase 1 + Phase 2 Prep

### Added
- Environment check, FFmpeg setup, media probe, audio extract.
- Multilingual Whisper ASR (silence-chunked, same-language merge).
- LLM subtitle calibration + validation, dual ASR/LLM versions, `review.json`.
- Phase 2 prep: 1 FPS frames, scene frames, `visual_batches.json`.
