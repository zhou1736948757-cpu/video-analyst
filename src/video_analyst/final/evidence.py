"""Evidence index + image copy (Epic 5).

构建 EvidenceIndex 索引所有结构化证据,并按配置复制/硬链接证据图片.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import FinalAnalysisConfig
from video_analyst.final.models import (
    EvidenceEntry,
    EvidenceIndex,
    MergedTemporalEvent,
    ShotAnalysis,
)
from video_analyst.fine.models import FineBatchAnalysis


def build_evidence_index(
    *,
    layout: OutputLayout,
    fine_analyses: dict[str, Any],
    merged_events: list[MergedTemporalEvent],
    shots: list[ShotAnalysis],
    terminal_origin: bool = False,
) -> EvidenceIndex:
    """构建 EvidenceIndex.

    fine_analyses may be FineBatchAnalysis (agent path) without contact_sheets.
    Contact sheets / detail frames are also read from Epic 3 visual inputs.
    """
    entries: list[EvidenceEntry] = []
    seen_ids: set[str] = set()
    cache = CacheStore(layout)

    def add(entry: EvidenceEntry) -> None:
        if entry.evidence_id in seen_ids:
            return
        seen_ids.add(entry.evidence_id)
        entries.append(entry)

    # 1. contact sheets + detail frames from refinement visual inputs
    visual_index = layout.refinement_fine_batches_json
    if visual_index.is_file():
        try:
            data = cache.read_json(visual_index)
        except Exception:
            data = {}
        for b in data.get("batches") or []:
            fb_id = str(b.get("fine_batch_id") or b.get("batch_id") or "batch")
            for cs in b.get("contact_sheets") or []:
                if not isinstance(cs, dict):
                    continue
                sid = str(cs.get("sheet_id") or Path(str(cs.get("path") or "sheet")).stem)
                eid = f"EVID-CS-{fb_id}-{sid}"
                ts = list(cs.get("timestamps") or [])
                path = cs.get("path") or ""
                add(
                    EvidenceEntry(
                        evidence_id=eid,
                        evidence_type="contact_sheet",
                        start=float(ts[0]) if ts else 0.0,
                        end=float(ts[-1]) if ts else 0.0,
                        description=f"contact sheet {sid} for {fb_id}",
                        image_paths=[str(path)] if path else [],
                        source_json_paths=[
                            str(visual_index.relative_to(layout.root))
                            if str(visual_index).startswith(str(layout.root))
                            else str(visual_index)
                        ],
                        confidence=0.7,
                        terminal_origin=terminal_origin,
                    )
                )
            for df in b.get("detail_frames") or []:
                if not isinstance(df, dict):
                    continue
                fid = str(df.get("frame_id") or "detail")
                eid = f"EVID-DF-{fb_id}-{fid}"
                add(
                    EvidenceEntry(
                        evidence_id=eid,
                        evidence_type="detail_frame",
                        start=float(df.get("timestamp_seconds") or 0.0),
                        end=float(df.get("timestamp_seconds") or 0.0),
                        description=f"detail frame {fid} from {fb_id}",
                        image_paths=[str(df.get("path") or "")] if df.get("path") else [],
                        source_json_paths=[],
                        confidence=0.85,
                        terminal_origin=terminal_origin,
                    )
                )

    # Also accept objects that already expose contact_sheets / detail_frames
    for fb_id, a in fine_analyses.items():
        sheets = getattr(a, "contact_sheets", None) or []
        for cs in sheets:
            sid = getattr(cs, "sheet_id", None) or "sheet"
            eid = f"EVID-CS-{fb_id}-{sid}"
            ts = list(getattr(cs, "timestamps", None) or [])
            path = getattr(cs, "path", None)
            add(
                EvidenceEntry(
                    evidence_id=eid,
                    evidence_type="contact_sheet",
                    start=float(ts[0]) if ts else 0.0,
                    end=float(ts[-1]) if ts else 0.0,
                    description=f"contact sheet {sid} for {fb_id}",
                    image_paths=[str(path)] if path else [],
                    source_json_paths=[],
                    confidence=0.7,
                    terminal_origin=terminal_origin,
                )
            )
        details = getattr(a, "detail_frames", None) or []
        for df in details:
            fid = getattr(df, "frame_id", None) or "detail"
            eid = f"EVID-DF-{fb_id}-{fid}"
            ts = float(getattr(df, "timestamp_seconds", 0.0) or 0.0)
            path = getattr(df, "path", None)
            add(
                EvidenceEntry(
                    evidence_id=eid,
                    evidence_type="detail_frame",
                    start=ts,
                    end=ts,
                    description=f"detail frame {fid} from {fb_id}",
                    image_paths=[str(path)] if path else [],
                    source_json_paths=[],
                    confidence=0.85,
                    terminal_origin=terminal_origin,
                )
            )

    # 2. structured fine analysis events (agent JSON)
    for fb_id, a in fine_analyses.items():
        if not isinstance(a, FineBatchAnalysis):
            continue
        batch_rel = f"analysis/fine/batches/{fb_id}.json"
        for ev in a.temporal_events:
            eid = f"EVID-EV-{ev.event_id}"
            add(
                EvidenceEntry(
                    evidence_id=eid,
                    evidence_type="fine_event",
                    start=ev.boundary.earliest,
                    end=ev.boundary.latest,
                    description=f"{ev.category}: {ev.description}",
                    image_paths=[],
                    source_json_paths=[batch_rel],
                    confidence=ev.confidence,
                    terminal_origin=terminal_origin,
                )
            )
        for ta in a.text_animations:
            eid = f"EVID-EV-{ta.event_id}"
            add(
                EvidenceEntry(
                    evidence_id=eid,
                    evidence_type="fine_event",
                    start=ta.boundary.earliest,
                    end=ta.boundary.latest,
                    description=f"text_animation: {ta.text_content or ta.event_id}",
                    image_paths=[],
                    source_json_paths=[batch_rel],
                    confidence=ta.confidence,
                    terminal_origin=False,
                )
            )
        for ga in a.graphic_animations:
            eid = f"EVID-EV-{ga.event_id}"
            add(
                EvidenceEntry(
                    evidence_id=eid,
                    evidence_type="fine_event",
                    start=ga.boundary.earliest,
                    end=ga.boundary.latest,
                    description=f"graphic_animation: {ga.graphic_subject}",
                    image_paths=[],
                    source_json_paths=[batch_rel],
                    confidence=ga.confidence,
                    terminal_origin=False,
                )
            )
        for tr in a.transitions:
            eid = f"EVID-EV-{tr.event_id}"
            add(
                EvidenceEntry(
                    evidence_id=eid,
                    evidence_type="fine_event",
                    start=tr.boundary.earliest,
                    end=tr.boundary.latest,
                    description=f"transition: {tr.transition_type}",
                    image_paths=[],
                    source_json_paths=[batch_rel],
                    confidence=tr.confidence,
                    terminal_origin=False,
                )
            )

    # 3. merged events (ensure every merged event has an EVID)
    for m in merged_events:
        ev = m.active
        eid = f"EVID-EV-{ev.event_id}"
        add(
            EvidenceEntry(
                evidence_id=eid,
                evidence_type="terminal_event" if m.source_round == "terminal" else "fine_event",
                start=ev.boundary.earliest,
                end=ev.boundary.latest,
                description=f"{ev.category}: {ev.description} (source={m.source_round})",
                image_paths=[],
                source_json_paths=[],
                confidence=ev.confidence,
                terminal_origin=terminal_origin or m.source_round == "terminal",
            )
        )

    # 4. media metadata + subtitles (stable evidence types for coverage reports)
    if layout.media_info_json.is_file():
        add(
            EvidenceEntry(
                evidence_id="EVID-META-media",
                evidence_type="media_metadata",
                start=0.0,
                end=0.0,
                description="source media_info.json",
                image_paths=[],
                source_json_paths=["media_info.json"],
                confidence=1.0,
                terminal_origin=False,
            )
        )
    if layout.subtitles_json.is_file():
        add(
            EvidenceEntry(
                evidence_id="EVID-SUB-active",
                evidence_type="subtitle",
                start=0.0,
                end=0.0,
                description="active subtitles/subtitles.json",
                image_paths=[],
                source_json_paths=["subtitles/subtitles.json"],
                confidence=0.5,
                terminal_origin=False,
            )
        )

    # 5. shots
    for shot in shots:
        eid = f"EVID-SHOT-{shot.shot_id}"
        etype = {
            "coarse_only": "coarse_frame",
            "fine": "refinement_frame",
            "terminal_refined": "terminal_frame",
        }.get(shot.analysis_level, "coarse_frame")
        add(
            EvidenceEntry(
                evidence_id=eid,
                evidence_type=etype,  # type: ignore[arg-type]
                start=shot.start,
                end=shot.end,
                description=f"shot {shot.shot_id} ({shot.analysis_level})",
                image_paths=[],
                source_json_paths=[],
                confidence=shot.confidence,
                terminal_origin=terminal_origin,
            )
        )

    return EvidenceIndex(video_id=layout.root.name, entries=entries)


def save_evidence_index(layout: OutputLayout, index: EvidenceIndex) -> Path:
    cache = CacheStore(layout)
    cache.write_json(layout.final_evidence_index_json, index.model_dump(mode="json"))
    return layout.final_evidence_index_json


def copy_evidence_images(
    *,
    index: EvidenceIndex,
    images_dir: Path,
    config: FinalAnalysisConfig,
) -> list[Path]:
    """Copy/symlink evidence images when paths exist. Returns copied paths."""
    mode = (config.evidence_mode or "copy").lower()
    copied: list[Path] = []
    if mode == "none":
        return copied
    images_dir.mkdir(parents=True, exist_ok=True)
    for entry in index.entries:
        for i, ip in enumerate(entry.image_paths):
            src = Path(ip)
            if not src.is_file():
                continue
            dest = images_dir / f"{entry.evidence_id}_{i}{src.suffix or '.jpg'}"
            try:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                if mode == "symlink":
                    dest.symlink_to(src.resolve())
                elif mode == "hardlink":
                    dest.hardlink_to(src)
                else:
                    shutil.copy2(src, dest)
                copied.append(dest)
            except Exception:
                try:
                    shutil.copy2(src, dest)
                    copied.append(dest)
                except Exception:
                    pass
    return copied


__all__ = ["build_evidence_index", "copy_evidence_images", "save_evidence_index"]
