"""Deterministic 2×N temporal contact sheets using Pillow.

Constraints:
- Each cell is the SAME size; images are letterboxed onto neutral gray.
- Per-cell label shows ONLY:
    frame_id  /  timecode  /  requested_fps
- Whole-sheet header shows ONLY interval_id.
- NO semantic labels (fade-in, pan, transition, etc.).
- Overlap of 1 frame between consecutive sheets (configurable).

If frames < frames_per_sheet, the LAST sheet is built with the remaining
frames and the remaining cells are neutral padding; we never duplicate a
frame to fill a sheet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from video_analyst.config import RefinementConfig
from video_analyst.exceptions import TemporalContactSheetError
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.models import (
    TemporalContactSheet as TemporalContactSheetModel,
)
from video_analyst.util.hashing import sha256_file

log = get_logger("refinement.contact_sheet")

NEUTRAL_BG = (24, 24, 24)
NEUTRAL_FG = (220, 220, 220)
PADDING = 8


@dataclass
class ContactSheetInputs:
    interval_id: str
    frames: list[dict]  # [{frame_id, timestamp_seconds, path, requested_fps}]
    sheet_id_start: int = 1


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise TemporalContactSheetError(
            "Pillow is required for temporal contact sheets: pip install Pillow"
        ) from exc
    return Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        from PIL import ImageFont  # type: ignore

        # Prefer a fixed-pitch TTF if available; fall back to default
        for candidate in (
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/Library/Fonts/Arial.ttf",
        ):
            p = Path(candidate)
            if p.is_file():
                return ImageFont.truetype(str(p), size=size)
        return ImageFont.load_default()
    except Exception:
        from PIL import ImageFont  # type: ignore

        return ImageFont.load_default()


def _letterbox(
    pil_image,
    canvas_w: int,
    canvas_h: int,
    pad: int,
    bg: tuple[int, int, int],
):
    """Resize ``pil_image`` (preserving aspect) to fit within (canvas_w - 2*pad)
    x (canvas_h - 2*pad) and paste centered onto a fresh bg canvas.
    """
    from PIL import Image  # type: ignore

    inner_w = max(1, canvas_w - 2 * pad)
    inner_h = max(1, canvas_h - 2 * pad)
    src_w, src_h = pil_image.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (canvas_w, canvas_h), bg)
    scale = min(inner_w / src_w, inner_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    fitted = pil_image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    canvas.paste(fitted, (off_x, off_y))
    return canvas


def _hms_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _build_sheet(
    *,
    sheet_id: str,
    interval_id: str,
    cell_frames: list[dict],
    out_path: Path,
    cfg: RefinementConfig,
) -> TemporalContactSheetModel:
    """Build one contact sheet."""
    Image, ImageDraw, _ = _require_pillow()

    columns = max(1, int(cfg.sheet_columns))
    frames_per_sheet = max(1, int(cfg.frames_per_sheet))
    rows = max(1, math.ceil(len(cell_frames) / columns))
    # Cell pixel size: distribute total canvas to fit long_edge budget
    long_edge = max(256, int(cfg.contact_sheet_max_edge))
    # Pick a sensible cell size: aim for total width ≈ long_edge (caps rows*cols grid)
    if columns >= 2:
        cell_w = long_edge // columns
    else:
        cell_w = long_edge // 2
    # Keep cells roughly 16:9 (or 9:16 if portrait)
    cell_h = max(64, int(cell_w * 9 / 16))
    sheet_w = columns * cell_w + (columns + 1) * PADDING
    header_h = 36
    sheet_h = header_h + rows * cell_h + (rows + 1) * PADDING

    canvas = Image.new("RGB", (sheet_w, sheet_h), NEUTRAL_BG)
    draw = ImageDraw.Draw(canvas)
    # Header (interval_id only)
    try:
        font_header = _font(18)
    except Exception:
        font_header = None
    draw.text(
        (PADDING, PADDING // 2),
        f"{interval_id}  TIME ->",
        fill=NEUTRAL_FG,
        font=font_header,
    )

    # Per-cell label fonts
    try:
        font_cell = _font(12)
    except Exception:
        font_cell = None

    for i, fdef in enumerate(cell_frames):
        r = i // columns
        c = i % columns
        x = PADDING + c * (cell_w + PADDING)
        y = header_h + PADDING + r * (cell_h + PADDING)
        cell_canvas = Image.new("RGB", (cell_w, cell_h), NEUTRAL_BG)
        try:
            with Image.open(str(fdef["path"])) as img:
                img.load()
                fitted = _letterbox(img, cell_w, cell_h, PADDING // 2, NEUTRAL_BG)
        except Exception as exc:
            log.warning("failed to load frame %s for sheet: %s", fdef.get("path"), exc)
            fitted = cell_canvas
        # Paste image
        canvas.paste(fitted, (x, y))
        # Label (frame_id / timecode / fps) at bottom of cell
        label = (
            f"{fdef['frame_id']}\n"
            f"{_hms_ms(float(fdef['timestamp_seconds']))}  fps={fdef.get('requested_fps', '?')}"
        )
        draw.multiline_text(
            (x + 4, y + cell_h - 32),
            label,
            fill=NEUTRAL_FG,
            font=font_cell,
            spacing=2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG"
    ext = "jpg"
    if cfg.image_format.lower() in {"png"}:
        fmt = "PNG"
        ext = "png"
    if not str(out_path).endswith(f".{ext}"):
        out_path = out_path.with_suffix(f".{ext}")
    if fmt == "JPEG":
        canvas.save(out_path, format="JPEG", quality=85, optimize=True)
    else:
        canvas.save(out_path, format="PNG", optimize=True)

    sha = sha256_file(out_path)
    return TemporalContactSheetModel(
        sheet_id=sheet_id,
        interval_id=interval_id,
        frame_ids=[str(f["frame_id"]) for f in cell_frames],
        timestamps=[round(float(f["timestamp_seconds"]), 6) for f in cell_frames],
        path=out_path,
        columns=columns,
        rows=rows,
        width=sheet_w,
        height=sheet_h,
        overlap_with_previous=[],
        sha256=sha,
    )


def build_contact_sheets(
    *,
    interval_id: str,
    frames: list[dict],
    out_dir: Path,
    cfg: RefinementConfig,
) -> list[TemporalContactSheetModel]:
    """Build all contact sheets for one interval.

    `frames`: ordered list of dicts with frame_id, timestamp_seconds, path,
    requested_fps. Sheets are emitted with ``frames_per_sheet`` frames each
    and ``overlap_frames`` overlap with the previous sheet.
    """
    if not frames:
        return []
    fps_cfg = max(1, int(cfg.frames_per_sheet))
    overlap = max(0, min(fps_cfg - 1, int(cfg.sheet_overlap_frames)))
    if overlap >= fps_cfg:
        overlap = fps_cfg - 1
    step = fps_cfg - overlap
    if step <= 0:
        step = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[TemporalContactSheetModel] = []
    sheet_no = 1
    i = 0
    while i < len(frames):
        chunk = frames[i : i + fps_cfg]
        sid = f"sheet_{sheet_no:04d}"
        prev_sheet = sheets[-1] if sheets else None
        overlap_ids: list[str] = []
        if prev_sheet is not None and overlap > 0:
            overlap_ids = prev_sheet.frame_ids[-overlap:]
        sheet = _build_sheet(
            sheet_id=f"{interval_id}__{sid}",
            interval_id=interval_id,
            cell_frames=chunk,
            out_path=out_dir / f"{sid}.jpg",
            cfg=cfg,
        )
        if overlap_ids:
            sheet = sheet.model_copy(
                update={"overlap_with_previous": overlap_ids}
            )
        sheets.append(sheet)
        sheet_no += 1
        if i + fps_cfg >= len(frames):
            break
        i += step

    return sheets


__all__ = ["build_contact_sheets", "ContactSheetInputs"]