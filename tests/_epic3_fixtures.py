"""Shared fixtures for Epic 3 (refinement) tests.

Provides:
- ``seed_refinement_layout``: layout with review.json + active subtitles +
  phase2 batches + media_info.json + a refinement_plan.json
- ``make_test_video``: generates a small CFR or VFR MP4 from PIL using
  ffmpeg (no real video API required)
- ``make_refinement_plan``: helper to write a refinement_plan.json with
  active + deferred intervals
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.coarse_analysis.models import RefinementInterval, RefinementPlan


def _write_jpeg(path: Path, color: tuple[int, int, int] = (200, 100, 50), w: int = 64, h: int = 36) -> None:
    """Write a real JPEG using Pillow. Returns a small file with size > 0."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow required for Epic3 fixtures") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (w, h), color)
    im.save(path, format="JPEG", quality=70)


def seed_refinement_layout(
    tmp_path: Path,
    *,
    duration: float = 60.0,
    batch_seconds: float = 20.0,
    refinement_intervals: list[dict[str, Any]] | None = None,
    active_source: str = "llm",
    reviewed: bool = False,
    needs_review: bool = False,
) -> OutputLayout:
    """Seed a layout that has the inputs Epic 3 needs.

    Writes: media_info.json, subtitles/subtitles.json, review.json,
    phase2/visual_batches.json, phase2/subtitle_mapping.json,
    analysis/coarse/refinement_plan.json.
    """
    layout = OutputLayout.create(tmp_path)
    # media_info
    layout.media_info_json.write_text(
        json.dumps(
            {
                "format_name": "mp4",
                "duration": duration,
                "size": 1,
                "bit_rate": 1,
                "video_streams": [
                    {
                        "index": 0,
                        "codec_name": "h264",
                        "width": 320,
                        "height": 240,
                        "avg_frame_rate": "30/1",
                        "r_frame_rate": "30/1",
                        "duration": duration,
                    }
                ],
                "audio_streams": [],
                "raw": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Subtitles (active)
    layout.subtitles_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "source_video": "/tmp/v.mp4",
                "audio_path": "/tmp/v.mp3",
                "language": "mixed",
                "engine": "groq-whisper-large-v3-turbo",
                "duration": duration,
                "segments": [
                    {
                        "index": 1,
                        "start": max(0.0, duration * 0.1),
                        "end": max(0.0, duration * 0.1) + 5.0,
                        "text": "ja",
                        "language": "Japanese",
                        "is_code_switched": False,
                        "source_segment_ids": ["raw_seg_0001"],
                        "source": active_source,
                        "merge_reason": None,
                        "calibration_status": None,
                    },
                    {
                        "index": 2,
                        "start": max(0.0, duration * 0.4),
                        "end": max(0.0, duration * 0.4) + 5.0,
                        "text": "zh",
                        "language": "Chinese",
                        "is_code_switched": False,
                        "source_segment_ids": ["raw_seg_0002"],
                        "source": active_source,
                        "merge_reason": None,
                        "calibration_status": None,
                    },
                ],
                "metadata": {},
                "origin": active_source,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layout.subtitles_srt.write_text("1\n00:00:01,000 --> 00:00:05,000\nja\n", encoding="utf-8")
    # Review package
    layout.review_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "version": "1.0",
                "source_video": "/tmp/v.mp4",
                "active": active_source,
                "reviewed": reviewed,
                "needs_review": needs_review,
                "asr": {
                    "version": "1.0",
                    "source_video": "/tmp/v.mp4",
                    "audio_path": "/tmp/v.mp3",
                    "language": "mixed",
                    "engine": "groq-whisper-large-v3-turbo",
                    "duration": duration,
                    "segments": [
                        {
                            "index": 1,
                            "start": max(0.0, duration * 0.1),
                            "end": max(0.0, duration * 0.1) + 5.0,
                            "text": "ja",
                            "language": "Japanese",
                            "is_code_switched": False,
                            "source_segment_ids": ["raw_seg_0001"],
                            "source": "asr",
                            "merge_reason": None,
                            "calibration_status": None,
                        }
                    ],
                    "metadata": {},
                    "origin": "asr",
                },
                "llm": None,
                "diffs": [],
                "asr_meta": {
                    "provider": "groq",
                    "model": "whisper-large-v3-turbo",
                    "multilingual": True,
                    "prompt_version": "asr-multilingual-v1",
                },
                "llm_meta": {
                    "enabled": True,
                    "provider": "groq",
                    "model": "llama-3.3-70b-versatile",
                    "prompt_version": "subtitle-calibration-v1",
                    "status": "completed",
                    "error": None,
                },
                "source_hashes": {"video": None, "audio": None, "asr_subtitles": None, "config": None},
                "summary": {"total_entries": 0, "unchanged": 0, "accepted": 0, "rejected": 0, "needs_review": 0},
                "llm_model": None,
                "llm_error": None,
                "created_at": "2026-07-21T00:00:00Z",
                "updated_at": "2026-07-21T00:00:00Z",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Visual batches
    n_batches = max(1, int(duration // batch_seconds) + (1 if duration % batch_seconds else 0))
    batches: list[dict] = []
    for i in range(n_batches):
        start = round(i * batch_seconds, 3)
        end = round(min(duration, (i + 1) * batch_seconds), 3)
        batches.append(
            {
                "batch_id": f"batch_{i + 1:04d}",
                "start": start,
                "end": end,
                "frames": [
                    {"path": f"phase2/frames_1fps/frame_{int(start * 1000):08d}.jpg",
                     "time": start, "kind": "fixed_fps", "fps": 1.0}
                ],
                "subtitles": [],
                "subtitle_metadata": {
                    "active_source": active_source,
                    "reviewed": reviewed,
                    "needs_review": needs_review,
                    "asr_model": "whisper-large-v3-turbo",
                    "calibration_model": "llama-3.3-70b-versatile",
                },
                "user_request": "",
                "analysis_instruction": {
                    "treat_subtitles_as_supporting_context": True,
                    "subtitle_is_not_absolute_ground_truth": True,
                },
            }
        )
    layout.phase2_batches_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "batches": batches,
                "subtitle_metadata": {
                    "active_source": active_source,
                    "reviewed": reviewed,
                    "needs_review": needs_review,
                    "asr_model": "whisper-large-v3-turbo",
                    "calibration_model": "llama-3.3-70b-versatile",
                },
                "note": "Epic3 fixture",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layout.phase2_mapping_json.write_text(
        json.dumps({"schema_version": "1.0", "active_source": active_source, "windows": []}),
        encoding="utf-8",
    )

    # Refinement plan
    if refinement_intervals is None:
        refinement_intervals = [
            {
                "interval_id": "interval_0001",
                "start": 5.0,
                "end": 10.0,
                "requested_fps": 10,
                "priority": "high",
            },
            {
                "interval_id": "interval_0002",
                "start": 20.0,
                "end": 24.0,
                "requested_fps": 15,
                "priority": "medium",
            },
        ]
    intervals: list[RefinementInterval] = []
    for iv in refinement_intervals:
        intervals.append(
            RefinementInterval(
                interval_id=iv["interval_id"],
                start=iv["start"],
                end=iv["end"],
                requested_fps=int(iv["requested_fps"]),
                priority=iv.get("priority", "medium"),
                reasons=iv.get("reasons", [f"need higher FPS around {iv['start']}s"]),
                questions=iv.get("questions", [f"what happens between {iv['start']}s and {iv['end']}s?"]),
                source_batch_ids=iv.get("source_batch_ids", ["batch_0001"]),
                evidence_frame_ids=iv.get("evidence_frame_ids", []),
            )
        )
    deferred: list[RefinementInterval] = []
    for d in refinement_intervals if False else []:
        # No automatic deferred creation here; tests can append manually.
        pass
    plan = RefinementPlan(
        video_duration=duration,
        total_requested_duration=sum(iv.end - iv.start for iv in intervals),
        budget_duration=0.0,
        intervals=intervals,
        deferred_intervals=deferred,
        generation_warnings=[],
    )
    layout.coarse_refinement_plan_json.write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    return layout


def make_test_video(
    out_path: Path,
    *,
    duration: float = 6.0,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    vfr: bool = False,
    label: str = "test",
) -> Path:
    """Generate a tiny test MP4 using FFmpeg + lavfi (no internet).

    Produces a CFR video by default; pass ``vfr=True`` for VFR.
    The ``drawtext`` filter is optional — we only add it if ffmpeg has it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        out_path.unlink()
    vf = f"testsrc=size={width}x{height}:rate={fps}:duration={duration}"
    if vfr:
        # vary fps mid-stream: emit 2× duration seconds, then keep last frame
        vf = f"{vf},setpts=N/(30*TB)"

    # Probe drawtext support without crashing tests that lack it.
    drawtext_supported = False
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
            check=False,
        )
        drawtext_supported = "drawtext" in (probe.stdout or "")
    except Exception:
        drawtext_supported = False

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", vf,
    ]
    if drawtext_supported:
        cmd.extend(
            [
                "-vf",
                f"drawtext=text='{label}':x=8:y=8:fontsize=18:fontcolor=white",
            ]
        )
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(out_path),
        ]
    )
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"test video generation failed (rc={completed.returncode}): "
            f"{(completed.stderr or '')[-400:]}"
        )
    return out_path


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        shell=False,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return 0.0
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_refinement_plan_json(
    path: Path,
    *,
    duration: float,
    intervals: list[dict[str, Any]],
    deferred_intervals: list[dict[str, Any]] | None = None,
) -> Path:
    """Helper: write a custom refinement_plan.json for tests."""
    ivs: list[RefinementInterval] = []
    for iv in intervals:
        ivs.append(
            RefinementInterval(
                interval_id=iv["interval_id"],
                start=float(iv["start"]),
                end=float(iv["end"]),
                requested_fps=int(iv["requested_fps"]),
                priority=iv.get("priority", "medium"),
                reasons=iv.get("reasons", []),
                questions=iv.get("questions", []),
                source_batch_ids=iv.get("source_batch_ids", []),
                evidence_frame_ids=iv.get("evidence_frame_ids", []),
            )
        )
    deferred: list[RefinementInterval] = []
    for iv in deferred_intervals or []:
        deferred.append(
            RefinementInterval(
                interval_id=iv["interval_id"],
                start=float(iv["start"]),
                end=float(iv["end"]),
                requested_fps=int(iv["requested_fps"]),
                priority=iv.get("priority", "low"),
                reasons=iv.get("reasons", []),
                questions=iv.get("questions", []),
                source_batch_ids=iv.get("source_batch_ids", []),
                evidence_frame_ids=iv.get("evidence_frame_ids", []),
            )
        )
    plan = RefinementPlan(
        video_duration=duration,
        total_requested_duration=sum(iv.end - iv.start for iv in ivs),
        budget_duration=0.0,
        intervals=ivs,
        deferred_intervals=deferred,
        generation_warnings=[],
    )
    path.write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


__all__ = [
    "ffprobe_duration",
    "has_ffmpeg",
    "has_pillow",
    "make_refinement_plan_json",
    "make_test_video",
    "seed_refinement_layout",
]