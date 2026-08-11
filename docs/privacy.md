# Privacy & Data Handling

Video Analyst processes video locally. The current product path keeps source
video, frames, contact sheets, analysis JSON, and reports local; it does not
send frames to an external vision API.

## Data that stays on your machine

- **Source video file**: never uploaded. Only read locally by FFmpeg.
- **Extracted audio** (`audio/audio.mp3`): stays local unless you use cloud
  Whisper ASR (Groq).
- **Extracted frames** (`phase2/`, `analysis/refinement/frames/`): local only.
- **Contact sheets** and **detail frames**: local only.
- **Subtitles** (ASR / LLM / active): local JSON/SRT files.
- **All analysis JSON** (coarse, fine, terminal, final): local only.
- **Reports** (Markdown / JSON / CSV): local only.
- **Telemetry** (`telemetry/calls.jsonl` etc.): local only, never transmitted.
- **Cache** (`cache/pipeline_state.json`): local only.

## Optional external ASR

Only the optional Groq ASR backend can receive derived audio when explicitly
configured:

### Groq (ASR, optional)

- **Sent**: the extracted audio file (`audio.mp3` or chunks).
- **Purpose**: speech-to-text transcription.
- **Avoid**: use local Whisper (`pip install 'video-analyst[local-whisper]'`)
  for a fully offline ASR path.

There is no vision-provider upload path in the product contract. The current
conversation Agent opens the prepared images in-session and writes the existing
analysis schemas.

## Deleting local data

To remove all generated outputs without touching the source video:

```bash
video-analyst clean OUTPUT --all-generated
```

Subdirectories removed: `analysis/`, `report/`, `phase2/`, `audio/`, `asr/`,
`subtitles/`, `cache/`, `logs/`, `raw/`, `telemetry/`.

For finer control:
- `--analysis-only`: removes `analysis/` only.
- `--cache-only`: removes `cache/` only.

If you use Groq ASR, provider-side audio retention is governed by Groq's data
policy and cannot be cleared by Video Analyst.

## API key safety

- Keys are read from environment variables only.
- Keys are never written to the output directory, config YAML, telemetry, logs,
  or reports.
- A redacting log filter strips `sk-...`, `gsk_...`, and `Authorization: Bearer`
  patterns from all log output.
- If you suspect a key leaked, run `video-analyst clean OUTPUT --all-generated`
  and rotate the key at the provider immediately.
