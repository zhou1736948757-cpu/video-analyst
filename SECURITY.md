# Security Policy

## Supported Versions

Video Analyst is currently in active development. Security fixes apply to the
latest `main` branch.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security problems.
2. Email the maintainer with a description and reproduction steps.
3. You will receive an acknowledgment within 72 hours.
4. A fix or mitigation will be coordinated before public disclosure.

## API Key Safety (BYOK)

Video Analyst is Bring-Your-Own-Key (BYOK):

- **No built-in keys**: The package never ships with developer API keys.
- **No key uploads**: User keys are never transmitted to any Video Analyst
  server — they stay on the user's machine.
- **No keys in outputs**: API keys are never written to the output directory,
  config files, telemetry, logs, or reports.
- **Redacting log filter**: All logs pass through a redacting filter that
  strips `sk-...`, `gsk_...`, and `Authorization: Bearer ...` patterns.
- **Telemetry redaction**: The telemetry recorder redacts secrets before
  writing `calls.jsonl` / `summary.json` / `errors.jsonl`.

If you suspect a key has leaked into an output directory, run:

```bash
video-analyst clean OUTPUT --all-generated
```

and rotate the key at the provider immediately.

## Third-Party Provider Risks

- **OpenAI-compatible providers**: Any base URL you configure receives your
  images (as base64), prompts, and subtitles. Choose providers you trust.
- **Relay / proxy endpoints**: If `base_url` points to a non-official host,
  that operator can inspect your images. **Use relays you trust.** Video
  Analyst does not block non-official `base_url` values — you may continue to
  use a relay if you have chosen it deliberately, but you are responsible for
  understanding what it does with your data. For sensitive content, prefer
  `--vision-provider mock` (no network) or run a self-hosted vision model.
- **Groq (ASR)**: Audio is sent to Groq for transcription if `GROQ_API_KEY`
  is set.
- **Local Whisper**: Use `pip install 'video-analyst[local-whisper]'` for a
  fully offline ASR path (no audio leaves the machine).

## Do Not Upload Sensitive Video

Video Analyst processes whatever video you give it. If your video contains
sensitive, confidential, or personally identifiable content, do not use a
cloud vision provider — use `--vision-provider mock` (no network) or run
the full pipeline locally with local Whisper and a self-hosted vision model.

## Deleting Data

To remove all generated outputs (frames, contact sheets, analysis, reports,
telemetry, cache) without touching the source video:

```bash
video-analyst clean OUTPUT --all-generated
```

Provider-side retention (images/audio already sent to a cloud API) is
governed by that provider's data policy and cannot be cleared by Video
Analyst.
