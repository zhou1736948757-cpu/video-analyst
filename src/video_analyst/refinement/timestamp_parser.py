"""Parse `pts_time` from FFmpeg `showinfo` filter output.

The extractor invokes FFmpeg with `-vf ...,showinfo` so each output frame
emits a line like:

    [Parsed_showinfo_1 @ 0x...] n:0 pts:240000 pts_time:2.400000 pos:...

The output JPEG filename is unrelated; we DO NOT trust the `frame_NNNNNN`
index to derive time. Instead we parse `pts_time` per emit and pair it
with the matching output JPEG via FFmpeg's mtime ordering.

Returned list has one entry per showinfo emit, in the order FFmpeg printed
them. The caller is responsible for matching each emit to a physical file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from video_analyst.exceptions import RefinementTimestampParseError

_PTS_TIME_RE = re.compile(r"pts_time:(-?[0-9.]+)")
_PTS_RE = re.compile(r"\bpts:(-?[0-9]+)")


@dataclass(frozen=True)
class ShowinfoEntry:
    """One showinfo line."""

    pts: int | None
    pts_time: float

    @property
    def has_pts(self) -> bool:
        return self.pts is not None


def parse_showinfo_stderr(stderr: str) -> list[ShowinfoEntry]:
    """Extract one ShowinfoEntry per emit, in stderr order.

    Empty stderr is treated as a hard error: if FFmpeg produced no stderr
    at all, the showinfo filter likely never ran. The caller (extractor)
    is responsible for deciding whether an empty frame set is legitimate
    (zero-duration interval) by catching this error when no files exist.
    """
    if not stderr:
        raise RefinementTimestampParseError(
            "empty FFmpeg stderr; cannot parse showinfo pts_time"
        )
    out: list[ShowinfoEntry] = []
    for line in stderr.splitlines():
        if "pts_time:" not in line:
            continue
        m_t = _PTS_TIME_RE.search(line)
        if m_t is None:
            continue
        try:
            pts_time = float(m_t.group(1))
        except (TypeError, ValueError) as exc:
            raise RefinementTimestampParseError(
                f"could not parse pts_time from line: {line!r}"
            ) from exc
        m_pts = _PTS_RE.search(line)
        pts_val: int | None = None
        if m_pts is not None:
            try:
                pts_val = int(m_pts.group(1))
            except (TypeError, ValueError):
                pts_val = None
        out.append(ShowinfoEntry(pts=pts_val, pts_time=pts_time))
    if not out:
        raise RefinementTimestampParseError(
            "no `pts_time:` lines found in FFmpeg stderr; "
            "is the showinfo filter present in the command?"
        )
    return out


def pair_with_files(
    entries: list[ShowinfoEntry],
    files: list,  # list[Path]
) -> list[tuple[ShowinfoEntry, object]]:
    """Pair showinfo entries with output JPGs by order.

    FFmpeg prints showinfo lines in the same order it writes output files
    (for the -vsync / -fps_mode vfr pipeline we use). So we pair by
    index. If counts disagree, we raise — never guess.
    """
    if len(entries) != len(files):
        raise RefinementTimestampParseError(
            f"showinfo entries ({len(entries)}) and output files "
            f"({len(files)}) count mismatch — cannot pair safely"
        )
    return list(zip(entries, files))


def compute_actual_fps(entries: Iterable[ShowinfoEntry]) -> float | None:
    """Compute effective FPS from the parsed timestamps.

    Returns ``None`` if there are fewer than 2 frames.
    """
    items = list(entries)
    if len(items) < 2:
        return None
    first = items[0].pts_time
    last = items[-1].pts_time
    if last <= first:
        return None
    return (len(items) - 1) / (last - first)


def expected_interval_seconds(fps: int, duration: float) -> float:
    """The theoretical sampling period for a given FPS."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return 1.0 / float(fps)


__all__ = [
    "ShowinfoEntry",
    "compute_actual_fps",
    "expected_interval_seconds",
    "pair_with_files",
    "parse_showinfo_stderr",
]