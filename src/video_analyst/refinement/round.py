"""通用 refinement round 入口（Epic 5）。

复用 Epic 3 抽帧 + 接触拼图 + detail frame 系统，支持任意 round 名称。
第一轮调用 round_name="round1"（Epic 3 pipeline 默认行为），终局调用
round_name="terminal"，输出至 analysis/terminal_refinement/，不覆盖
analysis/refinement/。

终局 round 会复用：
- extract_interval_frames（带 showinfo pts_time + sha256）
- deduplicate_intervals
- build_contact_sheets
- extract_detail_frame（高分辨率）
- build_fine_batch
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import RefinementConfig
from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import (
    RefinementFrameExtractionError,
    TerminalRefinementExecutionError,
)
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.batch_builder import (
    build_fine_batch,
    persist_fine_batch,
)
from video_analyst.refinement.contact_sheet import build_contact_sheets
from video_analyst.refinement.deduplicate import deduplicate_intervals
from video_analyst.refinement.detail_extractor import (
    extract_detail_frame,
    to_detail_frame_reference,
)
from video_analyst.refinement.detail_frames import select_detail_frame_specs
from video_analyst.refinement.extractor import extract_interval_frames
from video_analyst.refinement.models import (
    Epic3Manifest,
    Epic3Result,
    IntervalArtifact,
    IntervalManifestEntry,
    RefinementSource,
)
from video_analyst.util.hashing import sha256_file, sha256_json

log = get_logger("refinement.round")

ROUND_ROUND1 = "round1"
ROUND_TERMINAL = "terminal"


@dataclass
class RefinementRoundResult:
    """一轮 refinement 的最小返回结构（round1 与 terminal 共用）。"""

    round_name: str
    intervals_processed: int
    intervals_failed: int
    total_frames: int
    fine_batches_path: Path
    manifest_path: Path
    failed_interval_ids: list[str]
    warnings: list[str]


def _round_dir(layout: OutputLayout, round_name: str) -> dict[str, Path]:
    """返回指定 round 的所有路径。round1 用 refinement/*，terminal 用 terminal_refinement/*。

    返回字段：
    - frames_dir
    - contact_sheets_dir
    - detail_frames_dir
    - intervals_dir
    - visual_inputs_dir
    - batches_dir
    - manifest_path
    - fine_batches_json_path
    """
    if round_name == ROUND_ROUND1:
        return {
            "frames_dir": layout.refinement_frames_dir,
            "contact_sheets_dir": layout.refinement_contact_sheets_dir,
            "detail_frames_dir": layout.refinement_detail_frames_dir,
            "intervals_dir": layout.refinement_intervals_dir,
            "visual_inputs_dir": layout.refinement_visual_inputs_dir,
            "batches_dir": layout.refinement_batches_dir,
            "manifest_path": layout.refinement_manifest_json,
            "fine_batches_json_path": layout.refinement_fine_batches_json,
        }
    # terminal
    return {
        "frames_dir": layout.terminal_refinement_frames_dir,
        "contact_sheets_dir": layout.terminal_refinement_contact_sheets_dir,
        "detail_frames_dir": layout.terminal_refinement_detail_frames_dir,
        "intervals_dir": layout.terminal_refinement_intervals_dir,
        "visual_inputs_dir": layout.terminal_refinement_dir / "visual_inputs",
        "batches_dir": layout.terminal_refinement_batches_dir,
        "manifest_path": layout.terminal_refinement_manifest_json,
        "fine_batches_json_path": layout.terminal_refinement_fine_batches_json,
    }


def prepare_refinement_round(
    *,
    video_path: Path,
    output_dir: Path,
    intervals: list[RefinementSource],
    round_name: str,
    config: RefinementConfig,
    ffmpeg_env: FFmpegEnvironment,
    source_video_sha: str = "",
    questions_by_interval: dict[str, list[str]] | None = None,
    user_request: str = "",
) -> RefinementRoundResult:
    """通用 refinement 入口：对每个 interval 执行：

    1. extract_interval_frames（showinfo pts_time + sha256）
    2. deduplicate_intervals（intra + cross）
    3. build_contact_sheets
    4. extract_detail_frame（真高分辨率）
    5. build_fine_batch

    复用 Epic 3 全套系统；不创建并行 pipeline。

    注意：本函数为「精细帧 + fine batch 构建」通用入口。Ep ic 3 完整 pipeline
    （包含 presentation、manifest 持久化等）仍由 `pipeline/epic3.run_epic3` 负责，
    但本入口可供 Epic 5 终局 round 复用相同的抽帧 / 接触拼图 / detail 逻辑。
    """
    if not intervals:
        log.info("prepare_refinement_round: empty intervals for round=%s", round_name)
        layout = OutputLayout.create(output_dir)
        dirs = _round_dir(layout, round_name)
        # 写空 manifest
        manifest = Epic3Manifest(round_name=round_name)
        CacheStore(layout).write_json(dirs["manifest_path"], manifest.model_dump(mode="json"))
        return RefinementRoundResult(
            round_name=round_name,
            intervals_processed=0,
            intervals_failed=0,
            total_frames=0,
            fine_batches_path=dirs["fine_batches_json_path"],
            manifest_path=dirs["manifest_path"],
            failed_interval_ids=[],
            warnings=["no intervals to process"],
        )

    layout = OutputLayout.create(output_dir)
    dirs = _round_dir(layout, round_name)
    # Ensure dirs exist
    for d in dirs.values():
        if d.suffix == "":
            d.mkdir(parents=True, exist_ok=True)

    manifest = Epic3Manifest(round_name=round_name)
    total_frames = 0
    failed_ids: list[str] = []
    warnings: list[str] = []
    batches_out: list[dict[str, Any]] = []

    for iv in intervals:
        try:
            entry, _canonical, batch = _process_interval_for_round(
                layout=layout,
                dirs=dirs,
                interval=iv,
                config=config,
                ffmpeg_env=ffmpeg_env,
                source_video=video_path,
                questions_by_interval=questions_by_interval or {},
                user_request=user_request,
            )
            manifest.intervals.append(entry)
            manifest.intervals = sorted(manifest.intervals, key=lambda e: e.interval_id)
            manifest.recompute_counts()
            manifest.total_frames = sum(e.frame_count for e in manifest.intervals)
            CacheStore(layout).write_json(
                dirs["manifest_path"], manifest.model_dump(mode="json")
            )
            total_frames += entry.frame_count
            batches_out.append(batch.model_dump(mode="json"))
        except Exception as exc:
            log.error("round=%s interval=%s failed: %s", round_name, iv.interval_id, exc)
            failed_ids.append(iv.interval_id)
            warnings.append(f"{iv.interval_id}: {exc}")

    # 写 fine_batches.json
    CacheStore(layout).write_json(
        dirs["fine_batches_json_path"],
        {
            "schema_version": "1.0",
            "round_name": round_name,
            "video_id": "video",
            "total_batches": len(batches_out),
            "batches": batches_out,
        },
    )

    return RefinementRoundResult(
        round_name=round_name,
        intervals_processed=len(intervals) - len(failed_ids),
        intervals_failed=len(failed_ids),
        total_frames=total_frames,
        fine_batches_path=dirs["fine_batches_json_path"],
        manifest_path=dirs["manifest_path"],
        failed_interval_ids=failed_ids,
        warnings=warnings,
    )


def _process_interval_for_round(
    *,
    layout: OutputLayout,
    dirs: dict[str, Path],
    interval: RefinementSource,
    config: RefinementConfig,
    ffmpeg_env: FFmpegEnvironment,
    source_video: Path,
    questions_by_interval: dict[str, list[str]],
    user_request: str,
) -> tuple[IntervalManifestEntry, list, Any]:
    """单个 interval 的精细抽取 + 接触拼图 + detail + fine batch 构建。

    复用 Epic 3 抽帧 + 接触拼图 + detail_extractor。
    """
    interval_id = interval.interval_id
    frames_dir = dirs["frames_dir"] / interval_id
    sheets_dir = dirs["contact_sheets_dir"] / interval_id
    detail_dir = dirs["detail_frames_dir"] / interval_id
    intervals_dir = dirs["intervals_dir"]
    visual_inputs_dir = dirs["visual_inputs_dir"]
    batches_dir = dirs["batches_dir"]

    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    # 1. 抽帧
    try:
        extracted = extract_interval_frames(
            ffmpeg_env=ffmpeg_env,
            source_video=source_video,
            interval=interval,
            out_dir=frames_dir,
            cfg=config,
        )
    except RefinementFrameExtractionError as exc:
        raise TerminalRefinementExecutionError(
            f"round extraction failed for {interval_id}: {exc}"
        ) from exc

    # 2. dedup
    interval_fps_map = {interval_id: int(interval.requested_fps)}
    dedup_report = deduplicate_intervals(
        {interval_id: extracted},
        time_threshold_seconds=config.dedup_time_threshold_seconds,
        requested_fps_by_interval=interval_fps_map,
        source_video_sha="",
    )
    canonical = dedup_report.canonical_frames

    # 3. 接触拼图
    contact_sheets: list = []
    if config.sheets_enabled and not config.no_contact_sheets:
        sheet_frames: list[dict] = []
        for fr in canonical:
            sheet_frames.append(
                {
                    "frame_id": fr.frame_id,
                    "timestamp_seconds": float(fr.timestamp_seconds),
                    "path": fr.path,
                    "requested_fps": int(interval.requested_fps),
                }
            )
        contact_sheets = build_contact_sheets(
            interval_id=interval_id,
            frames=sheet_frames,
            out_dir=sheets_dir,
            cfg=config,
        )

    # 4. detail frames（真高分辨率）
    detail_refs: list = []
    specs = select_detail_frame_specs(
        canonical,
        interval_id=interval_id,
        interval_start=interval.start,
        interval_end=interval.end,
        cfg=config,
    )
    for spec in specs:
        try:
            ext = extract_detail_frame(
                ffmpeg_env=ffmpeg_env,
                source_video=source_video,
                frame_id=spec.frame_id,
                timestamp_seconds=spec.timestamp_seconds,
                out_dir=detail_dir,
                cfg=config,
                selection_reason=spec.selection_reason,
            )
            detail_refs.append(
                to_detail_frame_reference(ext, selection_reason=spec.selection_reason)
            )
        except Exception as exc:
            log.warning(
                "detail frame %s extraction failed (fall back): %s",
                spec.frame_id,
                exc,
            )

    # 5. fine batch 构建（不含字幕/coarse context，由调用方注入）
    questions = questions_by_interval.get(interval_id, interval.questions)
    batch = build_fine_batch(
        interval=interval,
        frames=canonical,
        contact_sheets=contact_sheets,
        detail_frames=detail_refs,
        layout=layout,
        cfg=config,
        user_request=user_request,
        refinement_questions=list(questions),
        coarse_analysis_paths={},
    )

    # 持久化
    fb_path = batches_dir / f"{batch.fine_batch_id}.json"
    CacheStore(layout).write_json(fb_path, batch.model_dump(mode="json"))

    # interval artifact
    artifact = IntervalArtifact(
        interval_id=interval_id,
        status="ok",
        interval=interval,
        frames=batch.frames,
        contact_sheets=contact_sheets,
        detail_frames=detail_refs,
        extraction_signature="",
        presentation_signature="",
    )
    iv_path = intervals_dir / f"{interval_id}.json"
    CacheStore(layout).write_json(iv_path, artifact.model_dump(mode="json"))

    entry = IntervalManifestEntry(
        interval_id=interval_id,
        status="ok",
        attempts=1,
        frame_count=len(canonical),
        contact_sheet_count=len(contact_sheets),
        detail_frame_count=len(detail_refs),
    )

    return entry, canonical, batch


__all__ = [
    "ROUND_ROUND1",
    "ROUND_TERMINAL",
    "RefinementRoundResult",
    "prepare_refinement_round",
]