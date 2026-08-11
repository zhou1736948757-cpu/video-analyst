"""Stage 1 — local prepare (no vision inference).

Input video precheck → ffprobe → audio → ASR → active subtitles
→ 1 FPS coarse frames → scene-change frames → coarse batches
→ coarse contact sheets → agent-readable inputs under
analysis/coarse/inputs/
"""

from __future__ import annotations

import json
import copy
import shutil
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import (
    AppConfig,
    DEFAULT_CONFIG,
    Phase2Config,
    RefinementConfig,
)
from video_analyst.env.detect import ensure_ready_for_media
from video_analyst.exceptions import Phase2PrepareError, VideoAnalystError
from video_analyst.logging.setup import get_logger, setup_logging
from video_analyst.pipeline.phase1 import run_phase1
from video_analyst.pipeline.video_precheck import validate_video_input
from video_analyst.refinement.contact_sheet import build_contact_sheets
from video_analyst.phase2.deduplicate import DedupeResult, deduplicate_frames
from video_analyst.util.hashing import sha256_file
from video_analyst.workflow.state import (
    WorkflowState,
    save_workflow_state,
)

log = get_logger("workflow.prepare")

COARSE_INPUTS_SCHEMA = "1.0"


def _rel(root: Path, path: Path) -> str:
    """Relative path from output root. Does not follow symlinks for the leaf."""
    root_abs = root.expanduser()
    if not root_abs.is_absolute():
        root_abs = root_abs.resolve()
    path_abs = path if path.is_absolute() else (root_abs / path)
    try:
        # Avoid Path.resolve() on the leaf so symlink destinations keep the
        # packaged relative path under analysis/coarse/inputs/frames/.
        return str(path_abs.relative_to(root_abs))
    except ValueError:
        try:
            return str(path_abs.resolve().relative_to(root_abs.resolve()))
        except ValueError:
            return str(path)


def _hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def package_coarse_agent_inputs(
    layout: OutputLayout,
    *,
    video_id: str | None = None,
    sheet_cfg: RefinementConfig | None = None,
    phase2_cfg: Phase2Config | None = None,
    pinned_timestamps: list[float] | None = None,
) -> dict[str, Any]:
    """Build analysis/coarse/inputs/ from phase2 visual batches.

    Creates:
      inputs/manifest.json
      inputs/batches/<batch_id>.json
      inputs/frames/  (linked/copied)
      inputs/contact_sheets/
    """
    root = layout.root
    if not layout.phase2_batches_json.is_file():
        raise Phase2PrepareError(
            f"visual_batches.json missing at {layout.phase2_batches_json}"
        )

    store = CacheStore(layout)
    data = store.read_json(layout.phase2_batches_json)
    batches = data.get("batches") or []
    if not batches:
        raise Phase2PrepareError("visual_batches.json has no batches")

    # Duration / video id
    duration = 0.0
    if layout.media_info_json.is_file():
        try:
            mi = store.read_json(layout.media_info_json)
            duration = float(mi.get("duration") or 0)
        except Exception:
            duration = 0.0
    if duration <= 0 and batches:
        duration = max(float(b.get("end") or 0) for b in batches)

    vid = video_id or root.name
    cfg = sheet_cfg or RefinementConfig(
        frames_per_sheet=6,
        sheet_overlap_frames=1,
        sheet_columns=3,
        contact_sheet_max_edge=2400,
    )

    inputs_dir = layout.coarse_dir / "inputs"
    frames_dir = inputs_dir / "frames"
    sheets_dir = inputs_dir / "contact_sheets"
    batch_inputs_dir = inputs_dir / "batches"
    for d in (inputs_dir, frames_dir, sheets_dir, batch_inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # metadata/
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if layout.media_info_json.is_file():
        dest = metadata_dir / "media_info.json"
        if not dest.exists():
            shutil.copy2(layout.media_info_json, dest)

    phase2_cfg = phase2_cfg or Phase2Config()
    all_frames: list[dict[str, Any]] = []
    all_sheets: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    scene_meta: list[dict[str, Any]] = []

    # Gather all valid source candidates before packaging.  Running one global
    # chronological pass means a scene frame is protected even when it is
    # visually identical to a fixed-FPS frame in an adjacent batch.
    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for batch in batches:
        batch_id = str(batch["batch_id"])
        for fr in batch.get("frames") or []:
            src_path = Path(fr.get("path") or "")
            if not src_path.is_file():
                alt = root / str(fr.get("path") or "")
                if alt.is_file():
                    src_path = alt
            if not src_path.is_file():
                log.warning("frame missing for %s: %s", batch_id, fr.get("path"))
                continue
            item = dict(fr)
            item["path"] = str(src_path)
            item["batch_id"] = batch_id
            item["candidate_index"] = candidate_index
            candidates.append(item)
            candidate_index += 1

    dedupe: DedupeResult = deduplicate_frames(
        candidates,
        threshold=phase2_cfg.coarse_dedupe_threshold,
        enabled=phase2_cfg.coarse_dedupe_enabled,
        pinned_timestamps=pinned_timestamps or [],
        algorithm_version=phase2_cfg.coarse_dedupe_algorithm_version,
        thumbnail_size=phase2_cfg.coarse_dedupe_thumbnail_size,
    )
    kept_indexes = {int(fr["candidate_index"]) for fr in dedupe.kept}
    candidates_by_batch: dict[str, list[dict[str, Any]]] = {}
    for fr in candidates:
        if int(fr["candidate_index"]) in kept_indexes:
            candidates_by_batch.setdefault(str(fr["batch_id"]), []).append(fr)

    # Map the input candidate index to the public frame id assigned below. This
    # lets the manifest link every dropped candidate to the exact kept frame.
    candidate_frame_ids: dict[int, str] = {}

    for batch in batches:
        batch_id = str(batch["batch_id"])
        b_start = float(batch.get("start") or 0)
        b_end = float(batch.get("end") or 0)
        raw_frames = candidates_by_batch.get(batch_id, [])
        subs = batch.get("subtitles") or []

        packaged_frames: list[dict[str, Any]] = []
        sheet_frame_defs: list[dict[str, Any]] = []

        for i, fr in enumerate(raw_frames):
            src_path = Path(fr.get("path") or "")
            if not src_path.is_file():
                # try relative to root
                alt = root / str(fr.get("path") or "")
                if alt.is_file():
                    src_path = alt
                else:
                    log.warning("frame missing for %s: %s", batch_id, fr.get("path"))
                    continue

            ts = float(fr.get("time") if fr.get("time") is not None else fr.get("timestamp") or 0)
            kind = str(fr.get("kind") or "fixed_fps")
            frame_id = f"{batch_id}_f{i:04d}"
            if kind == "scene":
                frame_id = f"{batch_id}_scene_{i:04d}"
                scene_meta.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": ts,
                        "batch_id": batch_id,
                        "scene_score": fr.get("scene_score"),
                    }
                )

            ext = src_path.suffix or ".jpg"
            dest_name = f"{frame_id}{ext}"
            dest_path = frames_dir / dest_name
            _copy_or_link(src_path, dest_path)
            rel_image = _rel(root, dest_path)
            try:
                sha = sha256_file(dest_path)
            except Exception:
                sha = ""

            entry = {
                "frame_id": frame_id,
                "batch_id": batch_id,
                "timestamp": round(ts, 6),
                "timecode": _hms(ts),
                "image_path": rel_image,
                "kind": kind,
                "sha256": sha,
                "sources": [kind],
                "retained_reason": list(fr.get("retained_reason") or ["pixel_change"]),
            }
            packaged_frames.append(entry)
            all_frames.append(entry)
            candidate_frame_ids[int(fr["candidate_index"])] = frame_id
            sheet_frame_defs.append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": ts,
                    "path": str(dest_path),
                    "requested_fps": 1,
                }
            )

        # Contact sheet(s) for this batch
        contact_sheet_paths: list[str] = []
        if sheet_frame_defs:
            batch_sheet_dir = sheets_dir / batch_id
            batch_sheet_dir.mkdir(parents=True, exist_ok=True)
            try:
                sheets = build_contact_sheets(
                    interval_id=batch_id,
                    frames=sheet_frame_defs,
                    out_dir=batch_sheet_dir,
                    cfg=cfg,
                )
                for sh in sheets:
                    rel_sheet = _rel(root, Path(sh.path))
                    contact_sheet_paths.append(rel_sheet)
                    all_sheets.append(
                        {
                            "sheet_id": sh.sheet_id,
                            "batch_id": batch_id,
                            "contact_sheet_path": rel_sheet,
                            "frame_ids": list(sh.frame_ids),
                            "timestamps": list(sh.timestamps),
                            "sha256": sh.sha256,
                        }
                    )
            except Exception as exc:
                log.warning("contact sheet failed for %s: %s", batch_id, exc)

        primary_sheet = contact_sheet_paths[0] if contact_sheet_paths else None
        batch_input = {
            "schema_version": COARSE_INPUTS_SCHEMA,
            "batch_id": batch_id,
            "start": b_start,
            "end": b_end,
            "frames": packaged_frames,
            "contact_sheets": contact_sheet_paths,
            "contact_sheet_path": primary_sheet,
            "subtitles": subs,
            "subtitle_metadata": batch.get("subtitle_metadata") or data.get("subtitle_metadata"),
            "user_request": batch.get("user_request") or "",
            "instruction": {
                "agent_must_inspect_images": True,
                "agent_must_open_every_listed_image": True,
                "timestamp_alignment_is_required": True,
                "do_not_infer_from_filenames_alone": True,
                "subtitles_are_supporting_context_only": True,
                "semantic_cues_require_agent_requested_pins": True,
            },
        }
        batch_path = batch_inputs_dir / f"{batch_id}.json"
        store.write_json(batch_path, batch_input)
        batch_summaries.append(
            {
                "batch_id": batch_id,
                "start": b_start,
                "end": b_end,
                "frame_count": len(packaged_frames),
                "contact_sheet_paths": contact_sheet_paths,
                "batch_input_path": _rel(root, batch_path),
            }
        )

    retained_frames: list[dict[str, Any]] = []
    for frame in dedupe.kept:
        idx = int(frame["candidate_index"])
        retained_frames.append(
            {
                "candidate_index": idx,
                "frame_id": candidate_frame_ids.get(idx),
                "batch_id": str(frame.get("batch_id") or ""),
                "timestamp": round(float(frame.get("timestamp") or frame.get("time") or 0), 6),
                "kind": str(frame.get("kind") or "candidate"),
                "retained_reason": list(frame.get("retained_reason") or []),
                "path": str(frame.get("path") or ""),
            }
        )
    dropped_candidates: list[dict[str, Any]] = []
    for drop in dedupe.dropped:
        row = dict(drop)
        kept_index = row.get("kept_candidate_index")
        row["kept_frame_id"] = candidate_frame_ids.get(int(kept_index)) if kept_index is not None else None
        dropped_candidates.append(row)
    dedupe_payload = {
        "enabled": dedupe.enabled,
        "algorithm_version": dedupe.algorithm_version,
        "thumbnail_size": dedupe.thumbnail_size,
        "threshold": dedupe.threshold,
        "candidate_count": dedupe.candidate_count,
        "kept_count": dedupe.kept_count,
        "dropped_count": dedupe.dropped_count,
        "retained_frames": retained_frames,
        "dropped_candidates": dropped_candidates,
        "dropped_to_kept": [
            {
                "dropped_candidate_index": d.get("candidate_index"),
                "dropped_timestamp": d.get("timestamp"),
                "kept_candidate_index": d.get("kept_candidate_index"),
                "kept_frame_id": d.get("kept_frame_id"),
                "delta": d.get("delta"),
            }
            for d in dropped_candidates
        ],
    }

    manifest = {
        "schema_version": COARSE_INPUTS_SCHEMA,
        "video_id": vid,
        "duration": duration,
        "vision_execution_mode": "in_session_agent",
        "external_api_used": False,
        "api_key_used": False,
        "note": (
            "Local prepare only. The current conversation Agent must inspect "
            "images and write coarse analysis JSON. Scripts do not perform "
            "visual inference."
        ),
        "batches": batch_summaries,
        "frames": all_frames,
        "contact_sheets": all_sheets,
        "scene_change_metadata": scene_meta,
        "dedupe": dedupe_payload,
        # Flat aliases make the preparation audit easy to inspect without
        # requiring a consumer to know the nested key name.
        "candidate_count": dedupe.candidate_count,
        "kept_count": dedupe.kept_count,
        "dropped_count": dedupe.dropped_count,
        "dedupe_threshold": dedupe.threshold,
        "dedupe_algorithm_version": dedupe.algorithm_version,
        "paths": {
            "frames": _rel(root, frames_dir),
            "contact_sheets": _rel(root, sheets_dir),
            "batches": _rel(root, batch_inputs_dir),
            "subtitles": _rel(root, layout.subtitles_json),
            "phase2_batches": _rel(root, layout.phase2_batches_json),
        },
        "agent_write_targets": {
            "batch_analyses": "analysis/coarse/batches/<batch_id>.json",
            "video_structure": "analysis/coarse/video_structure.json",
            "refinement_plan": "analysis/coarse/refinement_plan.json",
        },
    }
    manifest_path = inputs_dir / "manifest.json"
    store.write_json(manifest_path, manifest)
    log.info(
        "Coarse agent inputs packaged: %d batches, %d frames, %d sheets → %s",
        len(batch_summaries),
        len(all_frames),
        len(all_sheets),
        manifest_path,
    )
    return manifest


def run_prepare(
    video_path: Path,
    output_dir: Path,
    *,
    language: str | None = None,
    resume: bool = True,
    refresh_asr: bool = False,
    multilingual: bool = True,
    llm_calibrate: bool = True,
    user_request: str = "",
    config: AppConfig | None = None,
    pinned_timestamps: list[float] | None = None,
) -> dict[str, Any]:
    """Stage 1 end-to-end: media + ASR + phase2 + agent input packaging."""
    # Never mutate the process-global default when a CLI override is supplied.
    cfg = copy.deepcopy(config or DEFAULT_CONFIG)
    resolved_video = validate_video_input(video_path)
    output_dir = output_dir.expanduser().resolve()

    layout = OutputLayout.create(output_dir)
    setup_logging(layout.run_log)

    state = WorkflowState(
        status="preparing",
        video_path=str(resolved_video),
        output_dir=str(output_dir),
        vision_execution_mode="in_session_agent",
        real_vision_analysis_used=True,
        external_api_used=False,
        api_key_used=False,
    )
    save_workflow_state(output_dir, state)

    # Ensure tools
    ensure_ready_for_media(input_path=resolved_video, output_dir=output_dir)

    # Phase 1 + Phase 2 prep (no vision API)
    run_phase1(
        video_path=resolved_video,
        output_dir=output_dir,
        language=language,
        resume=resume,
        refresh_asr=refresh_asr,
        multilingual=multilingual,
        llm_calibrate=llm_calibrate,
        run_phase2_prep=True,
        user_request=user_request,
        config=cfg,
        pinned_timestamps=pinned_timestamps,
    )

    # Package agent inputs (frames + contact sheets + manifest)
    manifest = package_coarse_agent_inputs(
        layout,
        video_id=resolved_video.stem,
        sheet_cfg=cfg.refinement,
        phase2_cfg=cfg.phase2,
        pinned_timestamps=pinned_timestamps,
    )

    state.set_status(
        "waiting_for_coarse_agent_analysis",
        message="Preparation complete; awaiting in-session agent coarse analysis",
    )
    save_workflow_state(output_dir, state)

    return {
        "status": "waiting_for_coarse_agent_analysis",
        "video": str(resolved_video),
        "output": str(output_dir),
        "manifest_path": str(layout.coarse_dir / "inputs" / "manifest.json"),
        "manifest": manifest,
        "message": (
            "Preparation complete.\n"
            "The current Agent must now inspect coarse images and write coarse analysis."
        ),
    }


__all__ = ["package_coarse_agent_inputs", "run_prepare"]
