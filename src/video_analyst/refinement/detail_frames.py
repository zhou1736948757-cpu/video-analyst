"""Detail frame selection (deterministic, no model calls).

First version rule: pick the first and last legal frame of each interval.
If they coincide, keep only one.

Epic 4 遗留修复：本模块只负责"选哪些帧"成为 detail frame；
真正的高分辨率抽取由 ``refinement.detail_extractor`` 单独完成，
不再复用 1600 缩放后的普通 refinement frame。
"""

from __future__ import annotations

from dataclasses import dataclass

from video_analyst.config import RefinementConfig
from video_analyst.refinement.deduplicate import DeduplicatedFrame
from video_analyst.refinement.models import DetailFrameReference


@dataclass
class DetailFrameSelection:
    """detail frame 的选择规格（未抽取）。"""

    frame_id: str
    timestamp_seconds: float
    selection_reason: str


def select_detail_frames(
    frames: list[DeduplicatedFrame],
    *,
    interval_id: str,
    interval_start: float,
    interval_end: float,
    cfg: RefinementConfig,
) -> list[DetailFrameReference]:
    """返回至多 ``detail_max_per_interval`` 个 detail frame 引用。

    兼容旧调用方：返回 ``DetailFrameReference``，但 ``path`` 指向普通
    refinement frame（低分辨率）。Epic 3 pipeline 现在会在选择后调用
    ``extract_detail_frame`` 把这些引用替换为真正的高分辨率版本。
    """
    selections = select_detail_frame_specs(
        frames,
        interval_id=interval_id,
        interval_start=interval_start,
        interval_end=interval_end,
        cfg=cfg,
    )
    out: list[DetailFrameReference] = []
    for sel in selections:
        # 找到对应的 DeduplicatedFrame 取 path（低分辨率回退）
        match = next((d for d in frames if d.frame_id == sel.frame_id), None)
        path = match.path if match is not None else None  # type: ignore[assignment]
        out.append(
            DetailFrameReference(
                frame_id=sel.frame_id,
                timestamp_seconds=sel.timestamp_seconds,
                path=path,  # type: ignore[arg-type]
                selection_reason=sel.selection_reason,
            )
        )
    return out


def select_detail_frame_specs(
    frames: list[DeduplicatedFrame],
    *,
    interval_id: str,
    interval_start: float,
    interval_end: float,
    cfg: RefinementConfig,
) -> list[DetailFrameSelection]:
    """返回 detail frame 的选择规格（frame_id / timestamp / reason）。

    真正的高分辨率抽取由调用方完成。
    """
    if not frames:
        return []
    max_n = max(1, int(cfg.detail_max_per_interval))

    first = frames[0]
    last = frames[-1]
    selected: list[DetailFrameSelection] = []
    selected.append(
        DetailFrameSelection(
            frame_id=first.frame_id,
            timestamp_seconds=float(first.timestamp_seconds),
            selection_reason="interval_start",
        )
    )
    if last.frame_id != first.frame_id and len(selected) < max_n:
        single = (interval_end - interval_start) <= 1e-3
        reason = "single_frame_interval" if single else "interval_end"
        selected.append(
            DetailFrameSelection(
                frame_id=last.frame_id,
                timestamp_seconds=float(last.timestamp_seconds),
                selection_reason=reason,
            )
        )
    return selected[:max_n]


__all__ = [
    "DetailFrameSelection",
    "select_detail_frame_specs",
    "select_detail_frames",
]