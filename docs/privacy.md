# Privacy & Data Handling

Video Analyst processes video locally but may send derived data (frames,
contact sheets, audio, subtitles) to external API providers when configured.
This document explains exactly what data stays local and what is transmitted.

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

## Data sent to external providers

What gets transmitted depends on which providers you configure:

### Groq (ASR, optional)

- **Sent**: the extracted audio file (`audio.mp3` or chunks).
- **Purpose**: speech-to-text transcription.
- **Avoid**: use local Whisper (`pip install 'video-analyst[local-whisper]'`)
  for a fully offline ASR path.

### OpenAI-compatible vision provider (optional)

- **Sent**: base64-encoded contact sheet images, detail frame images, and the
  text prompt (including subtitle context and refinement questions).
- **NOT sent**: the source video file, the audio, or any frame not in the
  current batch.
- **Purpose**: coarse and fine visual analysis.

### Gemini (stub)

- Not implemented. Constructing the Gemini provider raises
  `FineVisionProviderUnavailableError`.

## Relay / proxy endpoints

If you set `VIDEO_ANALYST_VISION_BASE_URL` to a non-official host (a relay or
proxy), that operator can inspect, log, or cache your images and prompts. Use
relays you control or trust. For sensitive content, use `--vision-provider mock`
(no network) or a self-hosted vision model.

## BYOK does not mean "no provider retention"

Bring-Your-Own-Key means Video Analyst never sees or stores your key. It does
**not** prevent the provider from retaining the images/audio you send them
according to that provider's data policy. Review your provider's policy for:
- Training-data usage
- Retention period
- Logging and human review

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

**Provider-side retention** (images/audio already sent to a cloud API) is
governed by that provider's data policy and cannot be cleared by Video Analyst.

## API key safety

- Keys are read from environment variables only.
- Keys are never written to the output directory, config YAML, telemetry, logs,
  or reports.
- A redacting log filter strips `sk-...`, `gsk_...`, and `Authorization: Bearer`
  patterns from all log output.
- If you suspect a key leaked, run `video-analyst clean OUTPUT --all-generated`
  and rotate the key at the provider immediately.
