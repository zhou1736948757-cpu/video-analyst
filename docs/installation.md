# Installation

## Prerequisites

- **Python 3.11+** (3.11, 3.12, 3.13 supported; 3.14 experimental)
- **FFmpeg + ffprobe** (downloaded by the explicit setup command by default)
- **yt-dlp** (downloaded by the explicit setup command by default for public URL import)
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

## Install shared media tools

The Python package install is side-effect free. Run `setup` when you want the
Skill to provision missing media tools; it downloads/installs FFmpeg + ffprobe
and yt-dlp by default, after confirmation unless `--yes` is supplied.

```bash
video-analyst setup --yes
```

Skip either tool explicitly:

```bash
video-analyst setup --yes --no-ffmpeg
video-analyst setup --yes --no-yt-dlp
video-analyst setup --yes --no-ffmpeg --no-yt-dlp
```

FFmpeg uses the platform's trusted package manager when available. yt-dlp is
downloaded from its official release endpoint into the shared tools directory.
No browser session or cookie store is read by setup. Set
`VIDEO_ANALYST_TOOLS_DIR` to choose another shared tools root.

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
video-analyst doctor --json
```

`doctor` reports the local-file pipeline as `ready` when Python + FFmpeg are
present, and reports `url_import_ready` separately for yt-dlp.

## Optional ASR configuration

No vision API key is used by the product path. No keys ship with the package.

```bash
video-analyst init
# Writes .env.example + ~/.config/video-analyst/config.yaml (no keys)
```

Copy `.env.example` to `.env` and fill in:

```bash
GROQ_API_KEY=...                      # optional, for cloud Whisper ASR
```

## Python version support

| Version | Status |
|---|---|
| 3.11 | supported |
| 3.12 | supported |
| 3.13 | supported |
| 3.14 | experimental (local Whisper deps may lag) |
