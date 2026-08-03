from pathlib import Path

from video_analyst.models import (
    ASREngine,
    SubtitleDocument,
    SubtitleMetadata,
    SubtitleSegment,
)
from video_analyst.phase2.prepare import build_visual_batches, map_subtitles_to_window


def _doc() -> SubtitleDocument:
    return SubtitleDocument(
        source_video="/tmp/v.mp4",
        engine=ASREngine.GROQ_WHISPER_LARGE_V3_TURBO,
        duration=40.0,
        origin="llm",
        segments=[
            SubtitleSegment(index=1, start=0.0, end=5.0, text="ja", language="Japanese"),
            SubtitleSegment(index=2, start=10.0, end=15.0, text="zh", language="Chinese"),
            SubtitleSegment(index=3, start=25.0, end=30.0, text="en", language="English"),
        ],
    )


def test_map_subtitles_to_window():
    doc = _doc()
    mid = map_subtitles_to_window(doc, 8.0, 20.0)
    assert len(mid) == 1
    assert mid[0]["text"] == "zh"


def test_build_visual_batches_uses_active_meta():
    doc = _doc()
    meta = SubtitleMetadata(active_source="llm", reviewed=False, needs_review=True)
    frames = [{"path": "/tmp/f.jpg", "time": 1.0, "kind": "fixed_fps", "fps": 1.0}]
    batches = build_visual_batches(
        doc=doc,
        meta=meta,
        fixed_frames=frames,
        scene_frames=[],
        batch_seconds=20.0,
        duration=40.0,
    )
    assert len(batches) == 2
    assert batches[0].subtitle_metadata.active_source == "llm"
    assert batches[0].subtitle_metadata.needs_review is True
    assert batches[0].analysis_instruction["subtitle_is_not_absolute_ground_truth"] is True
