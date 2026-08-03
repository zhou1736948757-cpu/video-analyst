"""Tests for the FineVisualBatch builder."""

from __future__ import annotations

from pathlib import Path

from video_analyst.config import RefinementConfig
from video_analyst.refinement.batch_builder import (
    build_fine_batch,
    build_fine_batches_index,
    load_active_subtitles_for_interval,
    load_fine_batches_index,
    persist_fine_batch,
    persist_fine_batches_index,
)
from video_analyst.refinement.deduplicate import DeduplicatedFrame
from video_analyst.refinement.models import (
    DetailFrameReference,
    RefinementSource,
    TemporalContactSheet,
)

from tests._epic3_fixtures import seed_refinement_layout


def _df(t: float, *, path: str, interval_id: str = "interval_0001", sha: str = "abcd") -> DeduplicatedFrame:
    return DeduplicatedFrame(
        interval_id=interval_id,
        local_index=-1,
        frame_id=f"{interval_id}__f{int(t * 1000):08d}",
        path=Path(path),
        timestamp_seconds=t,
        relative_timestamp_seconds=t,
        requested_fps=10,
        width=320,
        height=240,
        size_bytes=1024,
        sha256=sha,
        source_video_sha256="vsha",
    )


def test_load_subtitles_uses_active_only(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path, active_source="llm")
    # ASR JSON present but should be ignored
    layout.subtitles_asr_json.write_text(
        '{"segments": [{"index": 99, "start": 0.0, "end": 5.0, "text": "asr-only", "language": "Japanese", "is_code_switched": false, "source_segment_ids": [], "source": "asr", "merge_reason": null, "calibration_status": null}]}',
        encoding="utf-8",
    )
    subs, meta = load_active_subtitles_for_interval(
        layout,
        interval_start=5.0,
        interval_end=10.0,
        context_before=1.0,
        context_after=1.0,
    )
    # Subtitle overlaps [4, 6] and [40, 45] inside our 60s video; first is in window
    assert any("ja" in s.text for s in subs)
    assert meta.active_source == "llm"
    assert meta.calibration_model == "llama-3.3-70b-versatile"
    # The "asr-only" segment (only in subtitles_asr) MUST NOT appear
    assert not any("asr-only" in s.text for s in subs)


def test_load_subtitles_context_window(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    # context_before=2.0 means we look at [-2, +0] around interval start
    # The 6s (≈6.0) ja segment is at duration*0.1=6.0; with before=2.0,
    # we look at [4, 6] which captures 6.0..6.0 boundary... use a narrower test
    subs, _ = load_active_subtitles_for_interval(
        layout,
        interval_start=5.0,
        interval_end=10.0,
        context_before=2.0,
        context_after=1.0,
    )
    # Should include the 'ja' segment at [6, 11] (or [duration*0.1, +5])
    assert any("ja" in s.text for s in subs)


def test_load_subtitles_no_review(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    layout.review_json.unlink()
    # Falls back to subtitles.json origin
    subs, meta = load_active_subtitles_for_interval(
        layout,
        interval_start=5.0,
        interval_end=10.0,
        context_before=1.0,
        context_after=1.0,
    )
    assert meta.active_source == "llm"


def test_load_subtitles_no_files(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    layout.subtitles_json.unlink()
    layout.review_json.unlink()
    subs, meta = load_active_subtitles_for_interval(
        layout,
        interval_start=0.0,
        interval_end=10.0,
        context_before=1.0,
        context_after=1.0,
    )
    assert subs == []
    assert meta.active_source == "asr"


def test_build_fine_batch_basic(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    interval = RefinementSource(
        interval_id="interval_0001",
        start=5.0,
        end=10.0,
        requested_fps=10,
        priority="high",
        questions=["what is happening?"],
    )
    frames = [
        _df(5.0, path="/tmp/interval_0001/f1.jpg"),
        _df(7.5, path="/tmp/interval_0001/f2.jpg"),
        _df(10.0, path="/tmp/interval_0001/f3.jpg"),
    ]
    detail = [
        DetailFrameReference(
            frame_id=frames[0].frame_id,
            timestamp_seconds=5.0,
            path=Path(frames[0].path),
            selection_reason="interval_start",
        ),
        DetailFrameReference(
            frame_id=frames[-1].frame_id,
            timestamp_seconds=10.0,
            path=Path(frames[-1].path),
            selection_reason="interval_end",
        ),
    ]
    contact_sheets: list[TemporalContactSheet] = []
    fine = build_fine_batch(
        interval=interval,
        frames=frames,
        contact_sheets=contact_sheets,
        detail_frames=detail,
        layout=layout,
        cfg=RefinementConfig(),
        user_request="x",
    )
    assert fine.fine_batch_id == "fine_interval_0001"
    assert fine.interval.start == 5.0
    assert fine.interval.end == 10.0
    assert fine.refinement_questions == ["what is happening?"]
    # Detail frames are flagged
    is_detail_flags = {f.frame_id: f.is_detail for f in fine.frames}
    assert is_detail_flags[frames[0].frame_id] is True
    assert is_detail_flags[frames[-1].frame_id] is True
    assert is_detail_flags[frames[1].frame_id] is False


def test_persist_and_load_fine_batches_index(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    interval = RefinementSource(
        interval_id="interval_0001",
        start=5.0,
        end=10.0,
        requested_fps=10,
        priority="high",
    )
    fine = build_fine_batch(
        interval=interval,
        frames=[],
        contact_sheets=[],
        detail_frames=[],
        layout=layout,
        cfg=RefinementConfig(),
        user_request="",
    )
    path = persist_fine_batch(layout, fine)
    assert path.is_file()

    # Build & persist index
    index = build_fine_batches_index([fine], video_id="v")
    assert index.total_batches == 1
    p2 = persist_fine_batches_index(layout, index)
    assert p2.is_file()

    # Load
    loaded = load_fine_batches_index(layout)
    assert loaded is not None
    assert loaded.total_batches == 1
    assert loaded.batches[0].fine_batch_id == fine.fine_batch_id


def test_fine_batch_includes_coarse_context(tmp_path: Path):
    layout = seed_refinement_layout(tmp_path)
    interval = RefinementSource(
        interval_id="interval_0001",
        start=5.0,
        end=10.0,
        requested_fps=10,
        priority="high",
        source_batch_ids=["batch_0001", "batch_0002"],
        evidence_frame_ids=["frame_a"],
    )
    fine = build_fine_batch(
        interval=interval,
        frames=[],
        contact_sheets=[],
        detail_frames=[],
        layout=layout,
        cfg=RefinementConfig(),
        user_request="",
        coarse_analysis_paths={"manifest": "analysis/coarse/manifest.json"},
    )
    assert fine.coarse_context["source_batch_ids"] == ["batch_0001", "batch_0002"]
    assert fine.coarse_context["evidence_frame_ids"] == ["frame_a"]
    assert fine.coarse_context["coarse_analysis_paths"]["manifest"].endswith("manifest.json")