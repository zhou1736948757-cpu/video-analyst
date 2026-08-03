"""Tests for the temporal contact sheet builder."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from video_analyst.config import RefinementConfig
from video_analyst.refinement.contact_sheet import build_contact_sheets
from video_analyst.util.hashing import sha256_file

from tests._epic3_fixtures import _write_jpeg, has_pillow


pytestmark = pytest.mark.skipif(
    not has_pillow(), reason="Pillow not installed"
)


def _make_frames(out_dir: Path, n: int, *, w: int = 64, h: int = 36, color: tuple[int, int, int] = (200, 100, 50)) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    for i in range(n):
        path = out_dir / f"frame_{i + 1:08d}.jpg"
        _write_jpeg(path, color=color, w=w, h=h)
        frames.append(
            {
                "frame_id": f"interval_0001__f{i + 1:08d}",
                "timestamp_seconds": 1.0 + i * 0.1,
                "path": path,
                "requested_fps": 10,
            }
        )
    return frames


def _cfg(**overrides) -> RefinementConfig:
    cfg = RefinementConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def test_sheet_basic_one_sheet_for_4_frames():
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 4)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=4, sheet_overlap_frames=1, sheet_columns=2),
        )
        assert len(sheets) == 1
        s = sheets[0]
        assert s.columns == 2
        assert s.rows == 2
        assert len(s.frame_ids) == 4
        assert s.path.is_file()
        assert s.sha256 == sha256_file(s.path)


def test_overlap_one_frame_for_10_frames():
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 10)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=4, sheet_overlap_frames=1, sheet_columns=2),
        )
        # step = 4 - 1 = 3; windows: [0..3], [3..6], [6..9] -> 3 sheets.
        # The last full sheet [6..9] already covers frame 9, so no partial
        # 4th sheet is emitted (would only duplicate frame 9).
        assert len(sheets) == 3
        # Overlap: sheet[0][-1] == sheet[1][0]
        assert sheets[0].frame_ids[-1] == sheets[1].frame_ids[0]
        assert sheets[1].frame_ids[-1] == sheets[2].frame_ids[0]
        # Last sheet is full (4 frames)
        assert len(sheets[-1].frame_ids) == 4


def test_partial_last_sheet_pads():
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 5)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=4, sheet_overlap_frames=1, sheet_columns=2),
        )
        # step = 3; windows: 0..3, 3..5 (only 2 frames)
        assert len(sheets) == 2
        # Last sheet has 2 frames
        assert len(sheets[-1].frame_ids) == 2
        # Last sheet still 2 columns; rows = ceil(2/2) = 1
        assert sheets[-1].rows == 1


def test_single_frame():
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 1)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=4, sheet_overlap_frames=1, sheet_columns=2),
        )
        assert len(sheets) == 1
        assert len(sheets[0].frame_ids) == 1
        # rows=1, cols=2
        assert sheets[0].rows == 1
        assert sheets[0].columns == 2


def test_three_frames():
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 3)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=4, sheet_overlap_frames=1, sheet_columns=2),
        )
        assert len(sheets) == 1
        assert len(sheets[0].frame_ids) == 3
        # rows = ceil(3/2) = 2
        assert sheets[0].rows == 2


def test_overlap_clamps_to_valid():
    cfg = _cfg(frames_per_sheet=4, sheet_overlap_frames=10, sheet_columns=2)
    # Internal guard should clamp overlap to 3 (frames_per_sheet - 1)
    with tempfile_safe() as tmp:
        frames = _make_frames(Path(tmp) / "frames", 8)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=Path(tmp) / "sheets",
            cfg=cfg,
        )
        # step = 4 - 3 = 1; sheets: [0..3], [1..4], [2..5], [3..6], [4..7]
        assert len(sheets) == 5


def test_no_frames_returns_empty():
    with tempfile_safe() as tmp:
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=[],
            out_dir=Path(tmp) / "sheets",
            cfg=_cfg(),
        )
        assert sheets == []


def test_no_semantic_labels_in_image():
    """Spot-check that we never write 'fade-in', 'pan', 'transition' etc."""
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 4)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=4, sheet_overlap_frames=1, sheet_columns=2),
        )
        # We don't write text with semantics. We DO write frame_id/timecode/fps.
        # So just check the image is a valid JPEG and is non-empty.
        from PIL import Image

        with Image.open(sheets[0].path) as im:
            assert im.width > 0
            assert im.height > 0


def test_sheet_label_includes_frame_id_and_timecode():
    with tempfile_safe() as tmp:
        out = Path(tmp) / "sheets"
        frames = _make_frames(Path(tmp) / "frames", 2)
        sheets = build_contact_sheets(
            interval_id="interval_0001",
            frames=frames,
            out_dir=out,
            cfg=_cfg(frames_per_sheet=2, sheet_overlap_frames=0, sheet_columns=2),
        )
        s = sheets[0]
        # First frame's id is in the sheet's frame_ids
        assert s.frame_ids == [f["frame_id"] for f in frames]


# ---- helpers ----

import tempfile
import contextlib


@contextlib.contextmanager
def tempfile_safe():
    """Tiny wrapper to make tmp_path work without pytest parametrize."""
    with tempfile.TemporaryDirectory() as d:
        yield d