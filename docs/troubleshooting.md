# Troubleshooting

## Common issues

### `FFmpeg: missing` from `video-analyst doctor`

FFmpeg is a system dependency, not a Python package.

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Or run `video-analyst setup` (auto-detects OS, installs after confirmation)

### `Groq API Key: not configured`

Only needed for cloud Whisper ASR. Options:
1. Set `GROQ_API_KEY` in `.env` or environment.
2. Use local Whisper: `pip install 'video-analyst[local-whisper]'` and set
   `VIDEO_ANALYST_FORCE_LOCAL_WHISPER=1`.
3. Skip ASR entirely (Phase 1 will hard-fail if no backend is available — never
   fakes success).

### `Missing API key for vision provider`

- For `openai-compatible`: set `VIDEO_ANALYST_VISION_API_KEY`.
- Mock provider needs no key — use `--vision-provider mock` for offline runs.

### `vision API 429` (rate limit)

- The transport layer retries 429 with exponential backoff (default 2 retries).
- Increase: `--max-retries 4`.
- Reduce images per batch: lower `batch_max_frames` in config.

### `vision API 500 / 502 / 503`

- Retried automatically (5xx is retryable).
- If persistent, the provider is down — use `--vision-provider mock` or wait.

### `vision response not valid JSON`

- The response parser strips markdown fences and surrounding prose, then
  attempts one deterministic repair request.
- If repair fails, the batch is marked failed (not retried indefinitely).
- Check `analysis/fine/raw/<batch>_response.json` for the raw response.

### `cannot estimate cost ... --confirm-api-call not set`

- The pricing snapshot doesn't cover your model.
- Either add the model to a pricing YAML, or pass `--confirm-api-call` to
  proceed with unknown cost.

### `TerminalRefinementPlanValidationError`

- `second_refinement_plan.json` has `max_additional_rounds != 1` or
  `terminal_after_this_round != true`.
- Epic 5 forbids a third round. Regenerate the plan from Epic 4.

### `ProvisionalReportNotAllowedError`

- `second_refinement_plan` is non-empty but you passed
  `--no-execute-terminal-refinement` without `--allow-provisional-report`.
- Either execute the terminal round, or explicitly allow a provisional report.

### Report says `mock_validation` but I used a real provider

- `real_vision_api_used` is based on actual call records
  (`network_request_sent=true` AND `response_received=true`), not on the
  provider name.
- If all calls hit cache, no network request was sent → still `mock_validation`.
- Check `output/telemetry/summary.json` for `network_calls_sent`.

### `python: FAIL` from doctor

- Video Analyst requires Python 3.11+. Check `python --version`.
- Python 3.14 is experimental (local Whisper deps may not support it yet).

## Cache & resume

### Pipeline didn't re-run after I changed subtitles

- Active subtitle changes invalidate Phase 2 mapping + visual batches + all
  downstream analysis, but do NOT re-extract frames.
- Use `--refresh-asr` to force ASR + subtitle rebuild.

### Ctrl+C during a long run

- Per-batch / per-interval results are written atomically as they complete.
- Re-run the same command (default `--resume`) to continue from the last
  completed batch.
- Use `--force` (or `--no-resume`) to ignore cache and re-run everything.

### Cache corruption

- `video-analyst clean OUT --cache-only` removes `cache/` only.
- `video-analyst clean OUT --all-generated` removes all generated outputs
  (analysis, report, cache, logs, telemetry) — never the source video.

## Performance

### Long video (20+ minutes) is slow

- Mock provider: wall-time scales with frame count, not video length.
- Real API: cost scales with batch count × images per batch. Use `--dry-run`
  to estimate before running.
- Terminal refinement is capped at `min(video × 5%, round1 × 25%)` of video
  duration; the third round is forbidden.

### Disk space

- Fine frames + contact sheets + detail frames can be large.
- Pre-check: `du -sh OUT/analysis/refinement/`.
- Clean: `video-analyst clean OUT --analysis-only`.

## Getting help

1. Run `video-analyst doctor --json` and include the output (no secrets).
2. Check `output/telemetry/summary.json` for call/cost/error stats.
3. Check `output/logs/run.log` (secrets are redacted automatically).
