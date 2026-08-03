"""Shared fixtures for Epic 4 (fine temporal vision) tests.

提供：
- ``seed_fine_batches``: 生成 fine_batches.json + 必要的 coarse/refinement 输入
- ``make_fine_request``: 构造一个最小合法 FineVisionRequest
- ``make_contact_sheet_jpeg``: 用 Pillow 生成真实 contact sheet JPEG
- ``make_detail_frame_jpeg``: 生成真实高分辨率 detail frame JPEG
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import OutputLayout
from video_analyst.refinement.models import (
    DetailFrameReference,
    FineBatchesIndex,
    FineVisualBatch,
    RefinementSource,
    TemporalContactSheet,
)
from video_analyst.vision.models import SubtitleContext, SubtitleMetadata


def has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def make_contact_sheet_jpeg(
    path: Path,
    *,
    w: int = 320,
    h: int = 240,
    color: tuple[int, int, int] = (180, 120, 60),
) -> Path:
    """生成一张真实 contact sheet JPEG（2x2 格子）用于测试。"""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), color)
    draw = ImageDraw.Draw(img)
    # 画 2x2 分隔线
    draw.line([(w // 2, 0), (w // 2, h)], fill=(255, 255, 255), width=1)
    draw.line([(0, h // 2), (w, h // 2)], fill=(255, 255, 255), width=1)
    img.save(path, format="JPEG", quality=85)
    return path


def make_detail_frame_jpeg(
    path: Path,
    *,
    w: int = 1280,
    h: int = 720,
    color: tuple[int, int, int] = (60, 120, 180),
) -> Path:
    """生成一张高分辨率 detail frame JPEG 用于测试。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), color)
    img.save(path, format="JPEG", quality=85)
    return path


def make_refinement_source(
    *,
    interval_id: str = "interval_0001",
    start: float = 5.0,
    end: float = 8.0,
    fps: int = 10,
    priority: str = "high",
    questions: list[str] | None = None,
) -> RefinementSource:
    return RefinementSource(
        interval_id=interval_id,
        start=start,
        end=end,
        requested_fps=fps,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        reasons=[f"need higher FPS around {start}s"],
        questions=questions or [f"what happens between {start}s and {end}s?"],
        source_batch_ids=["batch_0001"],
        evidence_frame_ids=[],
    )


def make_fine_visual_batch(
    layout: OutputLayout,
    *,
    interval_id: str = "interval_0001",
    start: float = 5.0,
    end: float = 8.0,
    fps: int = 10,
    priority: str = "high",
    questions: list[str] | None = None,
    sheet_count: int = 1,
    detail_count: int = 2,
    subtitles: list[SubtitleContext] | None = None,
    active_source: str = "llm",
    user_request: str = "",
) -> FineVisualBatch:
    """构造一个 FineVisualBatch，附带真实 contact sheet 和 detail frame 文件。"""
    interval = make_refinement_source(
        interval_id=interval_id,
        start=start,
        end=end,
        fps=fps,
        priority=priority,
        questions=questions,
    )
    # contact sheets
    contact_sheets: list[TemporalContactSheet] = []
    frames_per_sheet = 4
    for s_idx in range(sheet_count):
        sheet_id = f"{interval_id}__sheet_{s_idx + 1:04d}"
        sheet_path = layout.refinement_contact_sheets_dir / interval_id / f"sheet_{s_idx + 1:04d}.jpg"
        make_contact_sheet_jpeg(sheet_path)
        from video_analyst.util.hashing import sha256_file

        sha = sha256_file(sheet_path)
        # 4 帧时间戳，间隔 1/fps
        ts_list = [round(start + s_idx * (frames_per_sheet - 1) / fps + i / fps, 6) for i in range(frames_per_sheet)]
        # 限制不超过 end
        ts_list = [min(t, end) for t in ts_list]
        frame_ids = [f"{interval_id}__f{int(t * 1000):08d}" for t in ts_list]
        overlap = []
        if s_idx > 0:
            overlap = [frame_ids[0]]  # 与上一张共享第一帧
        contact_sheets.append(
            TemporalContactSheet(
                sheet_id=sheet_id,
                interval_id=interval_id,
                frame_ids=frame_ids,
                timestamps=ts_list,
                path=sheet_path,
                columns=2,
                rows=2,
                width=320,
                height=240,
                overlap_with_previous=overlap,
                sha256=sha,
            )
        )

    # detail frames（真高分辨率）
    detail_frames: list[DetailFrameReference] = []
    all_ts = []
    for cs in contact_sheets:
        all_ts.extend(cs.timestamps)
    all_ts = sorted(set(all_ts))
    if all_ts:
        detail_ts = [all_ts[0], all_ts[-1]][:detail_count]
    else:
        detail_ts = [start, end][:detail_count]
    for i, t in enumerate(detail_ts):
        df_path = layout.refinement_detail_frames_dir / interval_id / f"detail_{i:02d}.jpg"
        make_detail_frame_jpeg(df_path, w=1280, h=720)
        from video_analyst.util.hashing import sha256_file

        sha = sha256_file(df_path)
        reason = "interval_start" if i == 0 else "interval_end"
        detail_frames.append(
            DetailFrameReference(
                frame_id=f"{interval_id}__detail_{i:02d}",
                timestamp_seconds=float(t),
                path=df_path,
                selection_reason=reason,
                source_width=1280,
                source_height=720,
                output_width=1280,
                output_height=720,
                scaled=False,
                resolution_mode="source_original",
                sha256=sha,
            )
        )

    # subtitles
    if subtitles is None:
        subtitles = [
            SubtitleContext(
                index=1,
                start=max(0.0, start - 0.5),
                end=min(end, start + 1.0),
                text="ja",
                language="Japanese",
            )
        ]

    meta = SubtitleMetadata(
        active_source=active_source,  # type: ignore[arg-type]
        reviewed=False,
        needs_review=False,
        asr_model="whisper-large-v3-turbo",
        calibration_model="llama-3.3-70b-versatile",
    )

    return FineVisualBatch(
        fine_batch_id=f"fine_{interval_id}",
        interval=interval,
        frames=[],
        contact_sheets=contact_sheets,
        detail_frames=detail_frames,
        subtitles=subtitles,
        subtitle_metadata=meta,
        user_request=user_request,
        refinement_questions=list(questions or interval.questions),
        coarse_context={
            "source_batch_ids": ["batch_0001"],
            "evidence_frame_ids": [],
            "coarse_analysis_paths": {},
        },
    )


def seed_fine_batches(
    tmp_path: Path,
    *,
    duration: float = 60.0,
    intervals: list[dict[str, Any]] | None = None,
    active_source: str = "llm",
    user_request: str = "",
) -> OutputLayout:
    """生成完整 Epic 4 输入：media_info + subtitles + review + coarse +
    refinement fine_batches.json。
    """
    layout = OutputLayout.create(tmp_path)
    # media_info
    layout.media_info_json.write_text(
        json.dumps({"format_name": "mp4", "duration": duration, "video_streams": [{"index": 0, "codec_name": "h264", "width": 1280, "height": 720, "avg_frame_rate": "30/1", "duration": duration}], "audio_streams": [], "raw": {}}),
        encoding="utf-8",
    )
    # subtitles (active)
    layout.subtitles_json.write_text(
        json.dumps({"version": "1.0", "source_video": "/tmp/v.mp4", "duration": duration, "segments": [{"index": 1, "start": 4.5, "end": 9.0, "text": "ja", "language": "Japanese", "is_code_switched": False, "source_segment_ids": [], "source": active_source, "merge_reason": None, "calibration_status": None}], "metadata": {}, "origin": active_source}),
        encoding="utf-8",
    )
    layout.subtitles_srt.write_text("1\n00:00:04,500 --> 00:00:09,000\nja\n", encoding="utf-8")
    # review
    layout.review_json.write_text(
        json.dumps({"schema_version": "1.0", "version": "1.0", "source_video": "/tmp/v.mp4", "active": active_source, "reviewed": False, "needs_review": False, "asr": {"version": "1.0", "source_video": "/tmp/v.mp4", "duration": duration, "segments": [], "metadata": {}, "origin": "asr"}, "llm": None, "diffs": [], "asr_meta": {"provider": "groq", "model": "whisper-large-v3-turbo", "multilingual": True, "prompt_version": "asr-multilingual-v1"}, "llm_meta": {"enabled": True, "provider": "groq", "model": "llama-3.3-70b-versatile", "prompt_version": "subtitle-calibration-v1", "status": "completed", "error": None}, "source_hashes": {}, "summary": {"total_entries": 0, "unchanged": 0, "accepted": 0, "rejected": 0, "needs_review": 0}, "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z"}),
        encoding="utf-8",
    )
    # coarse manifest + structure + refinement plan
    layout.coarse_manifest_json.write_text(json.dumps({"schema_version": "1.0", "video_id": "v", "batches": [], "success_count": 0, "failed_count": 0}), encoding="utf-8")
    layout.coarse_structure_json.write_text(
        json.dumps({"schema_version": "1.0", "video_start": 0.0, "video_end": duration, "overall_summary": "coarse summary", "sections": [{"section_id": "section_0001", "start": 0.0, "end": duration, "title": "Section", "summary": "coarse section summary", "source_batch_ids": ["batch_0001"], "important_moments": [], "confidence": 0.6}], "recurring_subjects": [], "recurring_visual_patterns": [], "subtitle_visual_notes": [], "unresolved_questions": ["coarse unresolved"], "failed_batches": [], "gaps": []}),
        encoding="utf-8",
    )
    # refinement plan（提供 round1 时长信息）
    if intervals is None:
        intervals = [
            {"interval_id": "interval_0001", "start": 5.0, "end": 8.0, "requested_fps": 10, "priority": "high"},
            {"interval_id": "interval_0002", "start": 20.0, "end": 23.0, "requested_fps": 15, "priority": "medium"},
        ]
    plan_intervals = []
    for iv in intervals:
        plan_intervals.append({"interval_id": iv["interval_id"], "start": iv["start"], "end": iv["end"], "requested_fps": iv["requested_fps"], "priority": iv.get("priority", "medium"), "reasons": [f"need higher FPS around {iv['start']}s"], "questions": [f"what happens between {iv['start']}s and {iv['end']}s?"], "source_batch_ids": ["batch_0001"], "evidence_frame_ids": [], "estimated_frame_count": 0})
    layout.coarse_refinement_plan_json.write_text(
        json.dumps({"schema_version": "1.0", "video_duration": duration, "total_requested_duration": sum(iv["end"] - iv["start"] for iv in intervals), "budget_duration": duration * 0.2, "intervals": plan_intervals, "deferred_intervals": [], "generation_warnings": []}),
        encoding="utf-8",
    )

    # fine batches
    batches: list[FineVisualBatch] = []
    for iv in intervals:
        b = make_fine_visual_batch(
            layout,
            interval_id=iv["interval_id"],
            start=iv["start"],
            end=iv["end"],
            fps=iv["requested_fps"],
            priority=iv.get("priority", "medium"),
            questions=iv.get("questions", [f"what happens between {iv['start']}s and {iv['end']}s?"]),
            active_source=active_source,
            user_request=user_request,
        )
        batches.append(b)
    index = FineBatchesIndex(video_id="v", total_batches=len(batches), batches=batches)
    from video_analyst.cache.store import CacheStore

    CacheStore(layout).write_json(layout.refinement_fine_batches_json, index.model_dump(mode="json"))
    # 同时持久化每个 fine batch 文件（Epic 3 约定）
    for b in batches:
        CacheStore(layout).write_json(
            layout.refinement_batches_dir / f"{b.fine_batch_id}.json",
            b.model_dump(mode="json"),
        )
    return layout


__all__ = [
    "has_pillow",
    "make_contact_sheet_jpeg",
    "make_detail_frame_jpeg",
    "make_fine_visual_batch",
    "make_refinement_source",
    "seed_fine_batches",
]
