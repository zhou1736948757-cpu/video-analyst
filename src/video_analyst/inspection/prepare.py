"""Prepare deterministic, agent-readable inspection frames.

Inspection is intentionally a sidecar capability.  It writes only under
``OUTPUT/analysis/inspection`` and never reads or mutates coarse/fine analysis
state.  The current conversation Agent decides what the images mean; this
module only extracts pixels, records absolute timestamps, and builds objective
contact sheets.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from video_analyst.cache.store import CacheStore
from video_analyst.env.detect import FFmpegEnvironment, detect_ffmpeg_environment
from video_analyst.exceptions import InspectionPrepareError
from video_analyst.phase2.deduplicate import (
    ALGORITHM_VERSION,
    DEFAULT_THRESHOLD,
    DedupeResult,
    deduplicate_frames,
)
from video_analyst.pipeline.video_precheck import validate_video_input
from video_analyst.refinement.contact_sheet import build_contact_sheets
from video_analyst.config import RefinementConfig
from video_analyst.media.probe import probe_media
from video_analyst.process import run_command
from video_analyst.util.hashing import sha256_file

INSPECTION_SCHEMA_VERSION = "1.0"
INSPECTION_FPS_LIMITS = (5, 10, 15)
_PTS_TIME_RE = re.compile(r"pts_time:(-?[0-9.]+)")
_TIMESTAMP_TOLERANCE = 1e-3


@dataclass
class InspectionPrepareResult:
    """Small, JSON-friendly result returned by the CLI workflow."""

    status: str
    output: Path
    manifest_path: Path
    frame_count: int
    candidate_count: int
    dropped_count: int
    start: float
    end: float
    fps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": str(self.output),
            "manifest_path": str(self.manifest_path),
            "frame_count": self.frame_count,
            "candidate_count": self.candidate_count,
            "dropped_count": self.dropped_count,
            "start": self.start,
            "end": self.end,
            "fps": self.fps,
        }


def parse_timestamp(value: str | float | int | None) -> float:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS`` into non-negative seconds.

    Colon forms are strict about minute/second fields.  This prevents a typo
    such as ``1:90`` from silently becoming a different source timestamp.
    """

    if value is None:
        raise ValueError("timestamp is required")
    if isinstance(value, bool):
        raise ValueError(f"invalid timestamp: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"timestamp must be finite and non-negative: {value!r}")
        return seconds
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp cannot be empty")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds_part = float(parts[1])
            if minutes < 0 or not 0 <= seconds_part < 60:
                raise ValueError
            seconds = minutes * 60 + seconds_part
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_part = float(parts[2])
            if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds_part < 60:
                raise ValueError
            seconds = hours * 3600 + minutes * 60 + seconds_part
        else:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot parse timestamp {value!r}; expected SS, MM:SS, or HH:MM:SS"
        ) from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"timestamp must be finite and non-negative: {value!r}")
    return seconds


def _timecode(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _scale_filter(max_long_edge: int = 1600) -> str:
    edge = max(256, int(max_long_edge))
    return (
        f"scale='if(gt(iw,ih),min({edge},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({edge},ih))'"
    )


def _parse_showinfo_timestamps(stderr: str) -> list[float]:
    return [float(match.group(1)) for match in _PTS_TIME_RE.finditer(stderr or "")]


def _extract_interval_candidates(
    *,
    video_path: Path,
    out_dir: Path,
    env: FFmpegEnvironment,
    start: float,
    end: float,
    fps: int,
) -> list[dict[str, Any]]:
    """Extract interval frames and pair files with showinfo timestamps."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("candidate_*.jpg"):
        path.unlink()
    pattern = out_dir / "candidate_%06d.jpg"
    duration = max(0.001, end - start)
    result = run_command(
        [
            env.ffmpeg_path,
            "-y",
            "-nostdin",
            "-loglevel",
            "info",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.6f}",
            "-vf",
            f"fps={fps},{_scale_filter()},showinfo",
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            str(pattern),
        ],
        timeout=600,
    )
    if not result.ok:
        raise InspectionPrepareError(
            f"inspection frame extraction failed: {(result.stderr or '')[-500:]}"
        )

    files = sorted(out_dir.glob("candidate_*.jpg"))
    if not files:
        raise InspectionPrepareError(
            f"inspection range {start:.3f}..{end:.3f}s produced no frames"
        )
    relative_timestamps = _parse_showinfo_timestamps(result.stderr)
    # As in the refinement extractor, FFmpeg can print one trailing showinfo
    # record for a frame not muxed at the exact -t boundary.  It is safe to
    # trim only that known one-record case; otherwise fail rather than guess.
    if len(relative_timestamps) == len(files) + 1:
        relative_timestamps = relative_timestamps[:-1]
    if len(relative_timestamps) != len(files):
        raise InspectionPrepareError(
            "inspection frame/timestamp count mismatch; cannot produce citable "
            f"absolute timestamps ({len(files)} files, {len(relative_timestamps)} pts)"
        )

    out: list[dict[str, Any]] = []
    for index, (path, relative) in enumerate(zip(files, relative_timestamps)):
        timestamp = start + relative
        # Source absolute timestamps are the evidence contract.  A decoder can
        # emit a frame just outside a seek boundary; reject it instead of
        # silently relabeling the frame.
        if timestamp < start - _TIMESTAMP_TOLERANCE or timestamp > end + _TIMESTAMP_TOLERANCE:
            raise InspectionPrepareError(
                f"frame timestamp {timestamp:.6f}s is outside requested "
                f"range {start:.6f}..{end:.6f}s"
            )
        out.append(
            {
                "candidate_index": index,
                "path": str(path),
                "time": round(min(max(timestamp, start), end), 6),
                "kind": "inspection_fps",
                "requested_fps": fps,
                "pinned": False,
            }
        )
    return out


def _extract_pinned_candidates(
    *,
    video_path: Path,
    out_dir: Path,
    env: FFmpegEnvironment,
    timestamps: list[float],
    fps: int,
) -> list[dict[str, Any]]:
    """Extract one directly citable frame per user-supplied timestamp."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("at_*.jpg"):
        path.unlink()
    out: list[dict[str, Any]] = []
    for index, timestamp in enumerate(sorted(set(round(t, 6) for t in timestamps))):
        path = out_dir / f"at_{index:06d}.jpg"
        result = run_command(
            [
                env.ffmpeg_path,
                "-y",
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                _scale_filter(),
                "-q:v",
                "3",
                str(path),
            ],
            timeout=120,
        )
        if not result.ok or not path.is_file():
            raise InspectionPrepareError(
                f"inspection pinned frame extraction failed at {timestamp:.3f}s: "
                f"{(result.stderr or '')[-300:]}"
            )
        out.append(
            {
                "candidate_index": index,
                "path": str(path),
                "time": timestamp,
                "kind": "user_pinned",
                "requested_fps": fps,
                "pinned": True,
            }
        )
    return out


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _validate_scope(
    *,
    duration: float,
    start: str | float | int | None,
    end: str | float | int | None,
    at: Iterable[str | float | int],
) -> tuple[float, float, list[float]]:
    try:
        start_value = 0.0 if start is None else parse_timestamp(start)
        end_value = duration if end is None else parse_timestamp(end)
    except ValueError as exc:
        raise InspectionPrepareError(str(exc)) from exc
    if start_value > duration + _TIMESTAMP_TOLERANCE:
        raise InspectionPrepareError(
            f"--start {start_value:.3f}s is outside source duration {duration:.3f}s"
        )
    if end_value > duration + _TIMESTAMP_TOLERANCE:
        raise InspectionPrepareError(
            f"--end {end_value:.3f}s is outside source duration {duration:.3f}s"
        )
    if end_value <= start_value:
        raise InspectionPrepareError(
            f"--end ({end_value:.3f}s) must be greater than --start ({start_value:.3f}s)"
        )
    pins: list[float] = []
    for value in at:
        try:
            timestamp = parse_timestamp(value)
        except ValueError as exc:
            raise InspectionPrepareError(str(exc)) from exc
        if timestamp < start_value - _TIMESTAMP_TOLERANCE or timestamp > end_value + _TIMESTAMP_TOLERANCE:
            raise InspectionPrepareError(
                f"--at {timestamp:.3f}s is outside requested bounds "
                f"{start_value:.3f}..{end_value:.3f}s"
            )
        if timestamp > duration + _TIMESTAMP_TOLERANCE:
            raise InspectionPrepareError(
                f"--at {timestamp:.3f}s is outside source duration {duration:.3f}s"
            )
        pins.append(round(min(max(timestamp, start_value), end_value), 6))
    return start_value, end_value, sorted(set(pins))


def _drop_files_for_audit(
    *,
    result: DedupeResult,
    candidates: list[dict[str, Any]],
    dropped_dir: Path,
) -> None:
    """Move dropped candidates under an audit-only directory, never delete."""

    dropped_dir.mkdir(parents=True, exist_ok=True)
    by_index = {int(candidate["candidate_index"]): candidate for candidate in candidates}
    for row in result.dropped:
        candidate = by_index.get(int(row["candidate_index"]))
        if not candidate:
            continue
        source = Path(str(candidate.get("path") or ""))
        if not source.is_file():
            continue
        target = dropped_dir / source.name
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        row["path"] = str(target)


def prepare_inspection(
    video_path: Path,
    output_dir: Path,
    *,
    start: str | float | int | None = None,
    end: str | float | int | None = None,
    at: Iterable[str | float | int] = (),
    fps: int = 5,
    dedupe: bool = True,
    dedupe_threshold: float = DEFAULT_THRESHOLD,
    ffmpeg_env: FFmpegEnvironment | None = None,
) -> InspectionPrepareResult:
    """Prepare a sidecar visual inspection package without changing pipeline state."""

    try:
        fps_int = int(fps)
    except (TypeError, ValueError) as exc:
        raise InspectionPrepareError("--fps must be one of 5, 10, or 15") from exc
    if fps_int not in INSPECTION_FPS_LIMITS or float(fps) != float(fps_int):
        raise InspectionPrepareError(
            f"--fps must be one of {', '.join(str(v) for v in INSPECTION_FPS_LIMITS)}"
        )
    try:
        resolved = validate_video_input(video_path, env=ffmpeg_env)
        env = ffmpeg_env or detect_ffmpeg_environment()
        info = probe_media(resolved, env)
        duration = float(info.duration or 0.0)
    except InspectionPrepareError:
        raise
    except Exception as exc:
        raise InspectionPrepareError(f"Could not inspect source video: {exc}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise InspectionPrepareError("source video has no positive duration")

    start_value, end_value, pins = _validate_scope(
        duration=duration,
        start=start,
        end=end,
        at=at,
    )

    root = Path(output_dir).expanduser().resolve()
    inspection_dir = root / "analysis" / "inspection"
    frames_dir = inspection_dir / "frames"
    sheets_dir = inspection_dir / "contact_sheets"
    dropped_dir = inspection_dir / "dropped_candidates"
    # Only clear this command's own sidecar outputs.  Coarse/fine directories,
    # workflow_state.json, subtitles, and source media are never touched.
    for path in (frames_dir, sheets_dir, dropped_dir):
        if path.is_dir():
            shutil.rmtree(path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    regular = _extract_interval_candidates(
        video_path=resolved,
        out_dir=frames_dir,
        env=env,
        start=start_value,
        end=end_value,
        fps=fps_int,
    )
    # Candidate indexes are reassigned once across both streams; no filename
    # or subprocess ordering is used to identify a frame in the manifest.
    pinned = _extract_pinned_candidates(
        video_path=resolved,
        out_dir=frames_dir,
        env=env,
        timestamps=pins,
        fps=fps_int,
    )
    candidates = [*regular, *pinned]
    for index, candidate in enumerate(candidates):
        candidate["candidate_index"] = index

    result = deduplicate_frames(
        candidates,
        threshold=dedupe_threshold,
        enabled=dedupe,
        pinned_timestamps=pins,
        algorithm_version=ALGORITHM_VERSION,
    )
    _drop_files_for_audit(result=result, candidates=candidates, dropped_dir=dropped_dir)

    frame_id_by_candidate: dict[int, str] = {}
    frames: list[dict[str, Any]] = []
    at_counter = 0
    ordinary_counter = 0
    for kept in result.kept:
        candidate_index = int(kept["candidate_index"])
        kind = str(kept.get("kind") or "inspection_fps")
        if kind == "user_pinned":
            frame_id = f"inspection_at_{at_counter:04d}"
            at_counter += 1
        else:
            frame_id = f"inspection_f{ordinary_counter:06d}"
            ordinary_counter += 1
        frame_id_by_candidate[candidate_index] = frame_id
        path = Path(str(kept["path"]))
        if not path.is_file():
            raise InspectionPrepareError(f"kept frame disappeared during preparation: {path}")
        timestamp = float(kept.get("timestamp") or kept.get("time") or 0.0)
        frames.append(
            {
                "frame_id": frame_id,
                "candidate_index": candidate_index,
                "timestamp": round(timestamp, 6),
                "timecode": _timecode(timestamp),
                "path": str(path),
                "image_path": _rel(root, path),
                "kind": kind,
                "pinned": bool(kept.get("pinned")) or kind == "user_pinned",
                "requested_fps": fps_int,
                "retained_reason": list(kept.get("retained_reason") or []),
                "sha256": sha256_file(path),
            }
        )
    frames.sort(key=lambda frame: (float(frame["timestamp"]), int(frame["candidate_index"])))

    # Convert the low-level audit records to public frame IDs after IDs are
    # assigned.  Dropped paths remain citable under dropped_candidates/.
    retained_audit = [
        {
            "candidate_index": int(kept["candidate_index"]),
            "frame_id": frame_id_by_candidate.get(int(kept["candidate_index"])),
            "timestamp": round(float(kept.get("timestamp") or kept.get("time") or 0), 6),
            "kind": str(kept.get("kind") or "inspection_fps"),
            "path": str(kept.get("path") or ""),
            "retained_reason": list(kept.get("retained_reason") or []),
        }
        for kept in result.kept
    ]
    dropped_audit: list[dict[str, Any]] = []
    for dropped in result.dropped:
        row = dict(dropped)
        kept_index = row.get("kept_candidate_index")
        row["kept_frame_id"] = (
            frame_id_by_candidate.get(int(kept_index)) if kept_index is not None else None
        )
        row["image_path"] = _rel(root, Path(str(row.get("path") or "")))
        dropped_audit.append(row)

    contact_sheets: list[dict[str, Any]] = []
    if frames:
        sheet_cfg = RefinementConfig(
            frames_per_sheet=4,
            sheet_overlap_frames=1,
            sheet_columns=2,
            contact_sheet_max_edge=4096,
        )
        sheet_defs = [
            {
                "frame_id": frame["frame_id"],
                "timestamp_seconds": frame["timestamp"],
                "path": frame["path"],
                "requested_fps": fps_int,
            }
            for frame in frames
        ]
        try:
            sheets = build_contact_sheets(
                interval_id="inspection",
                frames=sheet_defs,
                out_dir=sheets_dir,
                cfg=sheet_cfg,
            )
        except Exception as exc:
            raise InspectionPrepareError(f"inspection contact sheet failed: {exc}") from exc
        for sheet in sheets:
            sheet_path = Path(sheet.path)
            contact_sheets.append(
                {
                    "sheet_id": sheet.sheet_id,
                    "path": str(sheet_path),
                    "contact_sheet_path": _rel(root, sheet_path),
                    "frame_ids": list(sheet.frame_ids),
                    "timestamps": list(sheet.timestamps),
                    "sha256": sheet.sha256,
                }
            )

    dedupe_payload = {
        "enabled": result.enabled,
        "algorithm_version": result.algorithm_version,
        "thumbnail_size": result.thumbnail_size,
        "threshold": result.threshold,
        "candidate_count": result.candidate_count,
        "kept_count": result.kept_count,
        "dropped_count": result.dropped_count,
        "retained_frames": retained_audit,
        "dropped_candidates": dropped_audit,
        "dropped_to_kept": [
            {
                "dropped_candidate_index": row.get("candidate_index"),
                "dropped_timestamp": row.get("timestamp"),
                "kept_candidate_index": row.get("kept_candidate_index"),
                "kept_frame_id": row.get("kept_frame_id"),
                "delta": row.get("delta"),
            }
            for row in dropped_audit
        ],
    }
    manifest = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "source_video": str(resolved),
        "source_video_path": str(resolved),
        "source_duration": round(duration, 6),
        "duration": round(duration, 6),
        "scope": {
            "start": round(start_value, 6),
            "end": round(end_value, 6),
            "start_timecode": _timecode(start_value),
            "end_timecode": _timecode(end_value),
        },
        "start": round(start_value, 6),
        "end": round(end_value, 6),
        "fps": fps_int,
        "fps_limits": list(INSPECTION_FPS_LIMITS),
        "pinned_timestamps": pins,
        "frames": frames,
        "contact_sheets": contact_sheets,
        "dedupe": dedupe_payload,
        "candidate_count": result.candidate_count,
        "kept_count": result.kept_count,
        "dropped_count": result.dropped_count,
        "vision_execution_mode": "in_session_agent",
        "external_api_used": False,
        "api_key_used": False,
        "note": (
            "Inspection preparation only. The current conversation Agent must "
            "open every listed image and may request additional --at timestamps; "
            "no script performs visual or transcript-cue inference."
        ),
        "instruction": {
            "agent_must_open_every_listed_image": True,
            "timestamp_alignment_is_required": True,
            "subtitles_are_supporting_context_only": True,
            "semantic_cues_require_explicit_agent_requested_at": True,
            "does_not_change_coarse_or_fine_state": True,
        },
        "paths": {
            "frames": _rel(root, frames_dir),
            "dropped_candidates": _rel(root, dropped_dir),
            "contact_sheets": _rel(root, sheets_dir),
            "manifest": _rel(root, inspection_dir / "manifest.json"),
        },
    }
    manifest_path = inspection_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return InspectionPrepareResult(
        status="inspection_prepared",
        output=inspection_dir,
        manifest_path=manifest_path,
        frame_count=len(frames),
        candidate_count=result.candidate_count,
        dropped_count=result.dropped_count,
        start=start_value,
        end=end_value,
        fps=fps_int,
    )


run_inspection_prepare = prepare_inspection


__all__ = [
    "INSPECTION_FPS_LIMITS",
    "INSPECTION_SCHEMA_VERSION",
    "InspectionPrepareResult",
    "parse_timestamp",
    "prepare_inspection",
    "run_inspection_prepare",
]
