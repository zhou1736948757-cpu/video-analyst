# Video Analyst Skill

Agent-assisted local video analysis: prepare visual evidence with Python and
FFmpeg, inspect it with an image-capable agent in the current conversation, then
validate and aggregate the agent's analysis.

The Python scripts do not perform visual reasoning. An agent must actually open
frames and contact sheets before writing evidence-backed analysis. No external
vision API or mock vision provider is used.

## Features

- Coarse sampling at 1 FPS, scene-change candidates, near-duplicate filtering,
  frame manifests, and contact sheets.
- Targeted refinement at 5, 10, or 15 FPS, with at most one terminal refinement.
- Explicit timestamp and interval inspection.
- JSON schemas, evidence checks, persisted workflow state, and Markdown / JSON /
  CSV report aggregation.
- Explicit single-URL import through yt-dlp, with subtitles and a source manifest.
- Optional ASR using a local Whisper executable or Groq Whisper. **If a Groq key
  is configured and no local engine is detected, preparation can upload audio to
  Groq.** Without either engine, ASR is recorded as unavailable.

## Requirements

- Python 3.10+ (the core package uses the standard library).
- FFmpeg and ffprobe on PATH.
- yt-dlp for URL imports.
- Pillow is optional; FFmpeg provides the contact-sheet fallback.
- A current-conversation agent with image-reading tools for visual analysis.

## Install and use

Clone the repository into your agent's skill directory, or clone elsewhere and
register/link the checkout according to your agent's skill discovery rules.
The skill directory must contain both `SKILL.md` and `video_analyst/`.

```bash
git clone https://github.com/zhou1736948757-cpu/video-analyst.git
cd video-analyst
./bin/video-analyst --help
./bin/video-analyst doctor
```

To expose the command, add this checkout's `bin` directory to PATH. The launcher
resolves the checkout path and does not depend on a particular user's home path.
On macOS, FFmpeg and yt-dlp can be installed with Homebrew:

```bash
brew install ffmpeg yt-dlp
```

Read [SKILL.md](SKILL.md) for the agent workflow and evidence requirements.
The CLI's `--help` output is authoritative for command arguments.

## Analysis workflow

```bash
video-analyst prepare /path/to/video.mp4 -o /path/to/analysis
# Agent opens coarse images, writes coarse analysis and a refinement plan.
video-analyst validate-coarse /path/to/analysis
video-analyst refine-prepare /path/to/video.mp4 -o /path/to/analysis
# Agent opens refinement images and writes fine analysis.
video-analyst validate-fine /path/to/analysis
# If required: one terminal refinement and agent inspection.
video-analyst finalize /path/to/analysis
```

`run` pauses at the agent inspection step. It does not automatically complete a
visual analysis. Schema files live under `video_analyst/schemas/`.

## URL import

```bash
video-analyst import-url 'https://www.bilibili.com/video/BV...' -o /path/to/import
```

Import accepts one HTTP(S) URL and invokes yt-dlp with `--no-playlist` and
`--ignore-config`, without cookie arguments. It is not a batch downloader.
Platform availability and quality depend on yt-dlp and the source platform.

## Credentials and private data

Keep videos, audio, analysis outputs, credentials, and user configuration outside
this repository. Groq credentials are read from `GROQ_API_KEY` or
`~/.config/video-analyst/config.json`; the latter uses mode 600. Prefer entering
credentials locally with `video-analyst setup-key groq --stdin` rather than
pasting them into a conversation or command argument.

## Validation and limitations

This publication preserves the existing local Python implementation. Publishing
it does not establish that every pipeline branch is production-ready. There is
no existing automated test suite in this snapshot. See
[VALIDATION.md](VALIDATION.md) for checks performed for publication.

ASR branches and a full agent-authored coarse/fine/final report are not certified
by the publication checks. Whisper executable variants may need different CLI
arguments. Use active subtitle outputs only after checking their content.

## License

MIT. See [LICENSE](LICENSE). The existing repository license is retained.
