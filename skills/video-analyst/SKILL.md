---
name: video-analyst
description: >
  Agent-assisted local video analysis. Deterministic local scripts prepare
  media, subtitles, coarse frames, refinement frames, and reports; the current
  conversation Agent actually opens the images and writes existing schemas.
---

# Video Analyst

Video Analyst is a Skill for an in-session coding Agent. It is not an automatic
vision-API program. The local scripts prepare deterministic visual inputs and
validate/aggregate Agent-authored JSON; the current conversation Agent owns
visual interpretation.

## Product contract

The only supported analysis sequence is:

    prepare
    -> current Agent opens coarse images and writes coarse JSON
    -> refine-prepare
    -> current Agent opens fine images and writes fine JSON
    -> finalize

Scripts never infer scenes, actions, motion, transitions, or semantic moments.
Subtitles and transcript cues are supporting context only. An Agent may request
an explicit inspection timestamp with inspect-prepare, but no script may infer
that request from transcript text.

The current Agent must:

1. Open and inspect every listed coarse frame and coarse contact sheet.
2. Align observations to the manifest timestamp and cite real frame IDs.
3. Write the existing Pydantic coarse schemas and refinement plan.
4. Open every listed fine contact sheet and detail frame after refine-prepare.
5. Write the existing Pydantic fine/terminal schemas and real EVID-* evidence.
6. Mark unresolved questions rather than inventing schema filler.

The runtime provenance is always:

    vision_execution_mode=in_session_agent
    external_api_used=false
    api_key_used=false
    token_status=unavailable
    estimated_cost=null

Do not restore or call mock, OpenAI-compatible, Anthropic-compatible,
agent-CLI, or other external vision providers. Do not send frames to a vision
API. Preserve existing schema and final-report compatibility.

## Install and shared tools

Install the Python package first:

    pip install -e .

Then explicitly provision shared media tools:

    video-analyst setup --yes

Setup installs missing FFmpeg + ffprobe and the official yt-dlp executable by
default. Either can be skipped:

    video-analyst setup --yes --no-ffmpeg
    video-analyst setup --yes --no-yt-dlp
    video-analyst setup --yes --no-ffmpeg --no-yt-dlp

Package installation itself is side-effect free. Setup uses trusted platform
package managers for FFmpeg where available and the official yt-dlp release
asset for yt-dlp. It never reads browser sessions or cookie stores. An
explicit cookie file is accepted only by import-url when the caller supplies
one.

Run:

    video-analyst doctor --json

The local-file pipeline needs Python and FFmpeg. yt-dlp is needed only for the
explicit public-URL import layer.

## Commands

    video-analyst import-url PUBLIC_URL -o OUTPUT
    video-analyst prepare VIDEO -o OUTPUT
    video-analyst inspect-prepare VIDEO -o OUTPUT --at 12.5 --at 01:03
    video-analyst inspect-prepare VIDEO -o OUTPUT --start 00:10 --end 00:20 --fps 5
    video-analyst refine-prepare VIDEO -o OUTPUT
    video-analyst terminal-prepare VIDEO -o OUTPUT
    video-analyst finalize VIDEO -o OUTPUT
    video-analyst status OUTPUT
    video-analyst validate-coarse OUTPUT
    video-analyst validate-fine OUTPUT
    video-analyst validate-terminal OUTPUT

import-url accepts one public HTTP(S) URL, defaults to no-playlist, and
supports ordinary public Bilibili single-video URLs. Subtitle choices include
zh-Hans, zh-Hant, zh, ja, en, and all. URL import writes local source files
and a source manifest; the core prepare command remains local-file-only.

## Stage 1: prepare

prepare produces local media, active subtitles, coarse frames, scene-change
frames, optional pinned frames, contact sheets, and a coarse input manifest.
Coarse near-duplicate filtering is pixel-only, deterministic, and enabled by
default. It compares low-resolution grayscale thumbnails against the previous
kept frame. The first and last candidates, scene-change frames, and explicit
pinned timestamps are retained. The manifest records counts, threshold,
algorithm version, retained reasons, and dropped-to-kept audit mappings.

The Agent must open all listed images before writing:

    analysis/coarse/inputs/batches/
    analysis/coarse/inputs/frames/
    analysis/coarse/inputs/contact_sheets/
    subtitles/subtitles.json

Then write existing-schema files:

    analysis/coarse/batches/<batch_id>.json
    analysis/coarse/video_structure.json
    analysis/coarse/refinement_plan.json

Validate before refining:

    video-analyst validate-coarse OUTPUT

## Stage 2: refine-prepare

refine-prepare reads only the validated Agent refinement plan. Use short
intervals and the lowest sufficient FPS:

    5 FPS  normal continuous motion
    10 FPS faster motion, text enter/exit, scale/pan/mask, transitions
    15 FPS flash, micro-transition, fast UI/text, unresolved 10 FPS edge

Do not sample an entire video at 15 FPS.

The Agent must open every fine contact sheet and detail frame, then write:

    analysis/fine/batches/<batch_id>.json
    analysis/fine/temporal_events.json
    analysis/fine/fine_video_analysis.json
    analysis/fine/second_refinement_plan.json

Validate before finalizing:

    video-analyst validate-fine OUTPUT

## Explicit inspection

inspect-prepare and its focus-prepare alias create an independent sidecar under
analysis/inspection. Repeated --at values and --start/--end accept SS, MM:SS,
or HH:MM:SS. FPS is limited to 5, 10, or 15. Bounds use absolute source-video
timestamps. Pinned frames are immune to dedupe.

The sidecar contains directly openable absolute paths, frame IDs, timestamps,
contact sheets, and dropped-candidate audit data. It never changes coarse/fine
workflow state, never writes analysis JSON, and is never consumed by finalize.

## Finalize and evidence

finalize is deterministic. It consumes only validated existing Pydantic
coarse/fine/terminal schemas, real frame IDs and real EVID-* evidence. It never
consumes unvalidated free text or performs new visual inference.

Report all frame IDs and contact sheets that the Agent actually opened. Never
claim image inspection from filenames, timestamps, dimensions, or subtitles
alone.

## Privacy and safety

Source videos, extracted frames, analysis JSON, and reports stay local in the
current product path. Optional Groq ASR may receive derived audio only when
explicitly configured. No vision API key, base URL, browser session, or
implicit cookie store is used.

Never rerun real-video extraction or overwrite coarse/fine JSON merely to
validate documentation. Use mocked subprocess/network boundaries in tests.
