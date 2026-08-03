# Providers

Video Analyst supports three vision providers and two ASR backends. All are
Bring-Your-Own-Key (BYOK) — no keys ship with the package.

## Vision providers

### `mock` (default, no API key)

- **No network calls**. Deterministic offline scenarios for testing.
- Use for: development, CI, offline verification, no-cost runs.
- Capabilities: supports images / JSON mode (simulated); `returns_usage=False`.

```bash
video-analyst coarse-analyze VIDEO -o OUT --vision-provider mock
video-analyst fine-analyze VIDEO -o OUT --vision-provider mock
video-analyst report VIDEO -o OUT --vision-provider mock
```

### `openai-compatible` (real API, BYOK)

- Any OpenAI-compatible `/chat/completions` endpoint.
- Sends base64 contact sheets + detail frames + prompt (never the source video).
- Reuses Epic 2's hardened HTTP transport: retry/backoff, timeout, 429/5xx handling, key redaction.

```bash
export VIDEO_ANALYST_VISION_API_KEY=sk-...
export VIDEO_ANALYST_VISION_BASE_URL=https://api.openai.com/v1
export VIDEO_ANALYST_VISION_MODEL=gpt-4o-mini

video-analyst provider-check --vision-provider openai-compatible
video-analyst report VIDEO -o OUT --vision-provider openai-compatible \
  --max-api-calls 20 --max-estimated-cost 1.00 --confirm-api-call
```

Supported models (price snapshot may be outdated; verify with your provider):
- `gpt-4o-mini` (default, cheapest)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-4-vision-preview`

Relay / proxy endpoints: if `base_url` points to a non-official host, that
operator can inspect your images. Use relays you trust. See `docs/privacy.md`.

### `gemini` (stub, not implemented)

- Constructing this provider raises `FineVisionProviderUnavailableError`.
- Fail-fast: the package never silently falls back to mock when you ask for Gemini.

## ASR backends

### Groq Whisper (cloud, optional)

- Sends extracted audio to Groq for transcription.
- Requires `GROQ_API_KEY`.
- Default cloud model: `whisper-large-v3-turbo`.

### Local Whisper (offline, optional)

- No audio leaves the machine.
- Install: `pip install 'video-analyst[local-whisper]'`.
- Model: `small` by default.
- Set `VIDEO_ANALYST_FORCE_LOCAL_WHISPER=1` to skip Groq even if a key is present.

## Provider check

Before a real run, verify the vision provider:

```bash
video-analyst provider-check --vision-provider openai-compatible --json
```

Checks: API key configured, endpoint reachable, model exists, vision supported,
JSON mode supported, usage readable, latency. Does **not** upload user video —
uses a generated 64×64 test image.

## Telemetry

Every Provider call is recorded (no keys, no base64 images):

```text
output/telemetry/
  calls.jsonl    one JSON per call
  summary.json   aggregate: count, tokens, cost, retries, cache hits
  errors.jsonl   failed calls only
```

`real_vision_api_used` in `summary.json` is `true` only if a vision-stage call
had `network_request_sent=true` AND `response_received=true`. Mock never sets
this.

## Pricing

Cost estimates use a built-in conservative snapshot (`builtin-v1`,
`2026-07-22`). To override, pass a YAML pricing file:

```yaml
version: user-v1
snapshot_date: "2026-07-22"
source: user
models:
  openai-compatible:gpt-4o-mini:
    input_per_million_tokens: 0.15
    output_per_million_tokens: 0.6
    currency: USD
```

When pricing is unavailable, `estimated_cost=null` and the pipeline refuses to
invoke a real API unless `--confirm-api-call` is set.
