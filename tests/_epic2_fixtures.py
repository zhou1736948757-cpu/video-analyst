"""Shared fixtures for Epic 2 (vision coarse analysis) tests.

These helpers prepare a real OutputLayout that contains:
- subtitles/subtitles.json (active)
- subtitles/review.json
- phase2/visual_batches.json
- a handful of dummy JPG frames under phase2/frames_1fps / frames_scene

All tests use the real filesystem (tmp_path) and never call external APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

from video_analyst.cache.store import OutputLayout
from video_analyst.config import AppConfig


def write_jpg(path: Path, size_bytes: int = 256) -> None:
    """Write a deterministic placeholder JPEG-like byte blob.

    We never feed these bytes to a vision model — Mock Provider only uses
    the frame metadata. Real OpenAI-compatible provider tests inject
    httpx mocks, never read the bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # JPEG magic header (FFD8FF) + padding. Real enough that file-detection works.
    pad = bytes([0xAB & (i & 0xFF) for i in range(max(1, size_bytes - 3))])
    path.write_bytes(b"\xff\xd8\xff" + pad)


def make_review_json(
    *,
    active: str = "llm",
    reviewed: bool = False,
    needs_review: bool = False,
    asr_model: str = "whisper-large-v3-turbo",
    calibration_model: str = "llama-3.3-70b-versatile",
) -> dict:
    return {
        "schema_version": "1.0",
        "version": "1.0",
        "source_video": "/tmp/v.mp4",
        "active": active,
        "reviewed": reviewed,
        "needs_review": needs_review,
        "asr": {
            "version": "1.0",
            "source_video": "/tmp/v.mp4",
            "audio_path": "/tmp/v.mp3",
            "language": "mixed",
            "engine": "groq-whisper-large-v3-turbo",
            "duration": 40.0,
            "segments": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "ja",
                    "language": "Japanese",
                    "is_code_switched": False,
                    "source_segment_ids": ["raw_seg_0001"],
                    "source": "asr",
                    "merge_reason": None,
                    "calibration_status": None,
                },
                {
                    "index": 2,
                    "start": 10.0,
                    "end": 15.0,
                    "text": "zh",
                    "language": "Chinese",
                    "is_code_switched": False,
                    "source_segment_ids": ["raw_seg_0002"],
                    "source": "asr",
                    "merge_reason": None,
                    "calibration_status": None,
                },
                {
                    "index": 3,
                    "start": 25.0,
                    "end": 30.0,
                    "text": "en",
                    "language": "English",
                    "is_code_switched": False,
                    "source_segment_ids": ["raw_seg_0003"],
                    "source": "asr",
                    "merge_reason": None,
                    "calibration_status": None,
                },
            ],
            "metadata": {},
            "origin": "asr",
        },
        "llm": None,
        "diffs": [],
        "asr_meta": {
            "provider": "groq",
            "model": asr_model,
            "multilingual": True,
            "prompt_version": "asr-multilingual-v1",
        },
        "llm_meta": {
            "enabled": True,
            "provider": "groq",
            "model": calibration_model,
            "prompt_version": "subtitle-calibration-v1",
            "status": "completed",
            "error": None,
        },
        "source_hashes": {
            "video": None,
            "audio": None,
            "asr_subtitles": None,
            "config": None,
        },
        "summary": {
            "total_entries": 0,
            "unchanged": 0,
            "accepted": 0,
            "rejected": 0,
            "needs_review": 0,
        },
        "llm_model": None,
        "llm_error": None,
        "created_at": "2026-07-21T00:00:00Z",
        "updated_at": "2026-07-21T00:00:00Z",
    }


def make_active_subtitles_json(*, origin: str = "llm") -> dict:
    return {
        "version": "1.0",
        "source_video": "/tmp/v.mp4",
        "audio_path": "/tmp/v.mp3",
        "language": "mixed",
        "engine": "groq-whisper-large-v3-turbo",
        "duration": 40.0,
        "segments": [
            {
                "index": 1,
                "start": 0.0,
                "end": 5.0,
                "text": "ja",
                "language": "Japanese",
                "is_code_switched": False,
                "source_segment_ids": ["raw_seg_0001"],
                "source": origin,
                "merge_reason": None,
                "calibration_status": None,
            },
            {
                "index": 2,
                "start": 10.0,
                "end": 15.0,
                "text": "zh",
                "language": "Chinese",
                "is_code_switched": False,
                "source_segment_ids": ["raw_seg_0002"],
                "source": origin,
                "merge_reason": None,
                "calibration_status": None,
            },
            {
                "index": 3,
                "start": 25.0,
                "end": 30.0,
                "text": "en",
                "language": "English",
                "is_code_switched": False,
                "source_segment_ids": ["raw_seg_0003"],
                "source": origin,
                "merge_reason": None,
                "calibration_status": None,
            },
        ],
        "metadata": {},
        "origin": origin,
    }


def make_visual_batches(
    *,
    duration: float = 40.0,
    batch_seconds: float = 20.0,
    active_source: str = "llm",
    reviewed: bool = False,
    needs_review: bool = False,
) -> dict:
    """Build a visual_batches.json-like payload with placeholder frames."""
    n_batches = max(1, int(duration // batch_seconds) + (1 if duration % batch_seconds else 0))
    batches: list[dict] = []
    for i in range(n_batches):
        start = round(i * batch_seconds, 3)
        end = round(min(duration, (i + 1) * batch_seconds), 3)
        # Put 4 fixed frames and 1 scene frame per batch (well under cap)
        frames = []
        for j in range(4):
            t = round(start + (j + 1) * 2.0, 3)
            if t >= end:
                continue
            frames.append(
                {
                    "path": f"phase2/frames_1fps/frame_{int(t * 1000):08d}.jpg",
                    "time": t,
                    "kind": "fixed_fps",
                    "fps": 1.0,
                }
            )
        # add a scene frame if room
        scene_t = round(start + 5.0, 3)
        if scene_t < end:
            frames.append(
                {
                    "path": f"phase2/frames_scene/scene_{i:06d}.jpg",
                    "time": scene_t,
                    "kind": "scene",
                    "index": i,
                }
            )
        # subtitles in window
        subs = []
        for s in [
            (1, 0.0, 5.0, "ja"),
            (2, 10.0, 15.0, "zh"),
            (3, 25.0, 30.0, "en"),
        ]:
            idx, ss, ee, txt = s
            if ee < start or ss > end:
                continue
            subs.append(
                {
                    "index": idx,
                    "start": ss,
                    "end": ee,
                    "text": txt,
                    "language": txt,
                    "is_code_switched": False,
                    "source_segment_ids": [],
                    "source": active_source,
                    "merge_reason": None,
                    "calibration_status": None,
                }
            )
        batches.append(
            {
                "batch_id": f"batch_{i + 1:04d}",
                "start": start,
                "end": end,
                "frames": frames,
                "subtitles": subs,
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
    return {
        "schema_version": "1.0",
        "batches": batches,
        "subtitle_metadata": {
            "active_source": active_source,
            "reviewed": reviewed,
            "needs_review": needs_review,
            "asr_model": "whisper-large-v3-turbo",
            "calibration_model": "llama-3.3-70b-versatile",
        },
        "note": "test fixture",
    }


def seed_layout(
    tmp_path: Path,
    *,
    duration: float = 40.0,
    batch_seconds: float = 20.0,
    active_source: str = "llm",
    reviewed: bool = False,
    needs_review: bool = False,
    write_frames: bool = True,
) -> OutputLayout:
    """Create an OutputLayout with full Phase 2 prep + subtitles."""
    layout = OutputLayout.create(tmp_path)
    # Save active subtitles + review
    layout.subtitles_json.write_text(
        json.dumps(make_active_subtitles_json(origin=active_source), ensure_ascii=False),
        encoding="utf-8",
    )
    layout.subtitles_srt.write_text("1\n00:00:00,000 --> 00:00:05,000\nja\n", encoding="utf-8")
    layout.review_json.write_text(
        json.dumps(
            make_review_json(
                active=active_source, reviewed=reviewed, needs_review=needs_review
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Visual batches
    layout.phase2_batches_json.write_text(
        json.dumps(
            make_visual_batches(
                duration=duration,
                batch_seconds=batch_seconds,
                active_source=active_source,
                reviewed=reviewed,
                needs_review=needs_review,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Subtitle mapping
    layout.phase2_mapping_json.write_text(
        json.dumps(
            {"schema_version": "1.0", "active_source": active_source, "windows": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Frames (real files so file-hash caches change correctly)
    if write_frames:
        for f in layout.phase2_fixed_frames_dir.glob("frame_*.jpg"):
            f.unlink()
        for f in layout.phase2_scene_frames_dir.glob("scene_*.jpg"):
            f.unlink()
        n_batches = max(1, int(duration // batch_seconds) + (1 if duration % batch_seconds else 0))
        for i in range(n_batches):
            for j in range(4):
                t = round(i * batch_seconds + (j + 1) * 2.0, 3)
                if t >= duration:
                    continue
                write_jpg(
                    layout.phase2_fixed_frames_dir / f"frame_{int(t * 1000):08d}.jpg",
                    size_bytes=512,
                )
            scene_t = round(i * batch_seconds + 5.0, 3)
            if scene_t < duration:
                write_jpg(
                    layout.phase2_scene_frames_dir / f"scene_{i:06d}.jpg",
                    size_bytes=512,
                )
    return layout


def make_config(**overrides) -> AppConfig:
    cfg = AppConfig()
    for k, v in overrides.items():
        if hasattr(cfg.vision, k):
            setattr(cfg.vision, k, v)
        elif hasattr(cfg.coarse_analysis, k):
            setattr(cfg.coarse_analysis, k, v)
    return cfg