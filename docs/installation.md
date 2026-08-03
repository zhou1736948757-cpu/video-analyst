# Installation

## Prerequisites

- **Python 3.11+** (3.11, 3.12, 3.13 supported; 3.14 experimental)
- **FFmpeg + ffprobe** (system dependency; install via your OS package manager)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Or run `video-analyst setup` after install
- **Pillow** (auto-installed with the package; for contact-sheet + image probing)

## Install from source

```bash
git clone <repo-url> video-analyst
cd video-analyst
pip install -e .
```

For development / tests:

```bash
pip install -e '.[dev]'
```

For local Whisper ASR (no cloud dependency):

```bash
pip install 'video-analyst[local-whisper]'
```

## Install from wheel

```bash
python -m build
pip install dist/video_analyst-0.6.0-py3-none-any.whl
```

## Verify install

```bash
video-analyst doctor
video-analyst --help
video-analyst provider-check --vision-provider mock
```

`doctor` should report `ready` if Python + FFmpeg are present.

## Configure providers (optional, BYOK)

Video Analyst is Bring-Your-Own-Key. No keys ship with the package.

```bash
video-analyst init
# Writes .env.example + ~/.config/video-analyst/config.yaml (no keys)
```

Copy `.env.example` to `.env` and fill in:

```bash
GROQ_API_KEY=...                      # optional, for cloud Whisper ASR
VIDEO_ANALYST_VISION_API_KEY=...      # optional, for OpenAI-compatible vision
VIDEO_ANALYST_VISION_BASE_URL=https://api.openai.com/v1
VIDEO_ANALYST_VISION_MODEL=gpt-4o-mini
```

Then verify the vision provider:

```bash
video-analyst provider-check --vision-provider openai-compatible
```

## Python version support

| Version | Status |
|---|---|
| 3.11 | supported |
| 3.12 | supported |
| 3.13 | supported |
| 3.14 | experimental (local Whisper deps may lag) |
