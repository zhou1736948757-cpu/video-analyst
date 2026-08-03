"""Audio chunk planning for multilingual ASR."""

from __future__ import annotations

from dataclasses import dataclass

from video_analyst.config import ChunkingConfig
from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import AudioChunkingError
from video_analyst.logging.setup import get_logger
from video_analyst.media.silence import TimeRange, detect_speech_ranges, _probe_duration
from pathlib import Path

log = get_logger("media.chunk_plan")


@dataclass(frozen=True)
class AudioChunkPlan:
    chunk_id: str
    absolute_start: float
    absolute_end: float
    overlap_before: float
    overlap_after: float

    @property
    def duration(self) -> float:
        return max(0.0, self.absolute_end - self.absolute_start)


def plan_audio_chunks(
    audio_path: Path,
    env: FFmpegEnvironment,
    *,
    duration: float | None = None,
    cfg: ChunkingConfig | None = None,
    video_duration: float | None = None,
) -> list[AudioChunkPlan]:
    """
    Build ASR chunk plan from silence candidates + max length + min length rules.

    Silence points are cut candidates only — not final subtitle boundaries.
    """
    cfg = cfg or ChunkingConfig()
    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise AudioChunkingError(f"Audio not found: {path}")

    if duration is None:
        duration = _probe_duration(path, env)
    # never exceed video duration when provided
    if video_duration is not None and video_duration > 0:
        duration = min(duration, video_duration)

    if duration <= 0:
        raise AudioChunkingError(f"Invalid audio duration: {duration}")

    speech = detect_speech_ranges(
        path,
        env,
        duration=duration,
        noise_db=cfg.silence_threshold_db,
        min_silence=cfg.silence_duration,
        min_speech=max(0.2, cfg.minimum_seconds * 0.25),
        pad=0.05,
    )

    # Merge ultra-short speech islands into neighbors before chunking
    speech = _merge_short_ranges(speech, minimum=cfg.minimum_seconds)

    # Split long ranges by maximum_seconds with overlap
    windows: list[TimeRange] = []
    for r in speech:
        windows.extend(
            _split_with_overlap(
                r,
                max_seconds=cfg.maximum_seconds,
                overlap=cfg.overlap_seconds,
            )
        )

    if not windows:
        windows = [TimeRange(0.0, duration)]

    # Ensure coverage of full duration is not required (silence-only tails ok),
    # but clamp to duration and drop empties.
    plans: list[AudioChunkPlan] = []
    n = len(windows)
    for i, w in enumerate(windows):
        start = max(0.0, min(w.start, duration))
        end = max(start, min(w.end, duration))
        if end - start < 0.15:
            continue
        # expand tiny windows toward minimum by borrowing neighbors later
        overlap_before = cfg.overlap_seconds if i > 0 else 0.0
        overlap_after = cfg.overlap_seconds if i < n - 1 else 0.0
        # apply overlap expansion within bounds
        abs_start = max(0.0, start - (overlap_before if i > 0 else 0.0) * 0.0)
        # keep planned core; overlap is metadata for merge; audio cut uses expanded window
        cut_start = max(0.0, start - (cfg.overlap_seconds if i > 0 else 0.0) * 0.5)
        cut_end = min(duration, end + (cfg.overlap_seconds if i < n - 1 else 0.0) * 0.5)
        if cut_end <= cut_start:
            continue
        plans.append(
            AudioChunkPlan(
                chunk_id=f"chunk_{len(plans)+1:04d}",
                absolute_start=round(cut_start, 3),
                absolute_end=round(cut_end, 3),
                overlap_before=round(cut_start - start if i > 0 else 0.0, 3),
                overlap_after=round(cut_end - end if i < n - 1 else 0.0, 3),
            )
        )

    # Merge consecutive plans shorter than minimum_seconds
    plans = _merge_short_plans(plans, minimum=cfg.minimum_seconds, duration=duration)

    # Re-id
    plans = [
        AudioChunkPlan(
            chunk_id=f"chunk_{i+1:04d}",
            absolute_start=p.absolute_start,
            absolute_end=p.absolute_end,
            overlap_before=p.overlap_before,
            overlap_after=p.overlap_after,
        )
        for i, p in enumerate(plans)
    ]

    if not plans:
        plans = [
            AudioChunkPlan(
                chunk_id="chunk_0001",
                absolute_start=0.0,
                absolute_end=round(duration, 3),
                overlap_before=0.0,
                overlap_after=0.0,
            )
        ]

    log.info(
        "Chunk plan: %d chunks (duration=%.2fs min=%.1f max=%.1f)",
        len(plans),
        duration,
        cfg.minimum_seconds,
        cfg.maximum_seconds,
    )
    return plans


def _merge_short_ranges(ranges: list[TimeRange], *, minimum: float) -> list[TimeRange]:
    if not ranges:
        return ranges
    out: list[TimeRange] = [ranges[0]]
    for r in ranges[1:]:
        prev = out[-1]
        if prev.duration < minimum or r.duration < minimum:
            out[-1] = TimeRange(start=prev.start, end=max(prev.end, r.end))
        elif r.start - prev.end < 0.25:
            out[-1] = TimeRange(start=prev.start, end=max(prev.end, r.end))
        else:
            out.append(r)
    return out


def _split_with_overlap(
    r: TimeRange,
    *,
    max_seconds: float,
    overlap: float,
) -> list[TimeRange]:
    if r.duration <= max_seconds:
        return [r]
    out: list[TimeRange] = []
    t = r.start
    step = max(1.0, max_seconds - overlap)
    while t < r.end - 1e-3:
        end = min(r.end, t + max_seconds)
        out.append(TimeRange(start=t, end=end))
        if end >= r.end:
            break
        t = t + step
    return out


def _merge_short_plans(
    plans: list[AudioChunkPlan],
    *,
    minimum: float,
    duration: float,
) -> list[AudioChunkPlan]:
    if not plans:
        return plans
    out: list[AudioChunkPlan] = []
    for p in plans:
        if out and p.duration < minimum:
            prev = out[-1]
            out[-1] = AudioChunkPlan(
                chunk_id=prev.chunk_id,
                absolute_start=prev.absolute_start,
                absolute_end=min(duration, max(prev.absolute_end, p.absolute_end)),
                overlap_before=prev.overlap_before,
                overlap_after=0.0,
            )
        elif out and out[-1].duration < minimum:
            prev = out[-1]
            out[-1] = AudioChunkPlan(
                chunk_id=prev.chunk_id,
                absolute_start=prev.absolute_start,
                absolute_end=min(duration, max(prev.absolute_end, p.absolute_end)),
                overlap_before=prev.overlap_before,
                overlap_after=p.overlap_after,
            )
        else:
            out.append(p)
    return out
