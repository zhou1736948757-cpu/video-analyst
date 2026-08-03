"""Tests for overlap dedupe and same-language merge rules."""

from video_analyst.asr.merge import (
    dedupe_overlap_segments,
    merge_same_language_segments,
    offset_raw_segments,
)
from video_analyst.config import MergeConfig


def test_offset_raw_segments_absolute_times():
    chunks = [
        {
            "chunk_id": "chunk_0001",
            "absolute_start": 10.0,
            "absolute_end": 15.0,
            "detected_language": "Japanese",
            "segments": [{"start": 0.0, "end": 2.0, "text": "こんにちは"}],
        }
    ]
    segs = offset_raw_segments(chunks)
    assert segs[0]["start"] == 10.0
    assert segs[0]["end"] == 12.0
    assert segs[0]["id"].startswith("raw_seg_")


def test_dedupe_overlap_same_text():
    segs = [
        {"id": "a", "start": 0.0, "end": 2.0, "text": "hello", "language": "en"},
        {"id": "b", "start": 0.2, "end": 2.1, "text": "hello", "language": "en"},
    ]
    out = dedupe_overlap_segments(segs)
    assert len(out) == 1


def test_merge_short_gap_same_language():
    segs = [
        {
            "id": "raw_seg_0001",
            "start": 0.0,
            "end": 2.9,
            "text": "みんなさんこんにちは。今日はこのアプリに",
            "language": "Japanese",
        },
        {
            "id": "raw_seg_0002",
            "start": 3.04,
            "end": 5.06,
            "text": "次でご紹介いただきます。",
            "language": "Japanese",
        },
        {
            "id": "raw_seg_0003",
            "start": 5.79,
            "end": 10.0,
            "text": "大家好",
            "language": "Chinese",
        },
    ]
    merged = merge_same_language_segments(segs, cfg=MergeConfig())
    assert len(merged) == 2
    assert merged[0].language == "Japanese"
    assert "次でご紹介" in merged[0].text
    assert merged[0].source_segment_ids == ["raw_seg_0001", "raw_seg_0002"]
    assert merged[1].language == "Chinese"


def test_no_merge_across_language():
    segs = [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "あ", "language": "Japanese"},
        {"id": "2", "start": 1.1, "end": 2.0, "text": "你好", "language": "Chinese"},
    ]
    merged = merge_same_language_segments(segs)
    assert len(merged) == 2


def test_no_merge_after_sentence_end_with_long_gap():
    segs = [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "終わり。", "language": "Japanese"},
        {"id": "2", "start": 2.5, "end": 3.5, "text": "次です。", "language": "Japanese"},
    ]
    merged = merge_same_language_segments(
        segs, cfg=MergeConfig(auto_merge_gap_seconds=0.8, conditional_merge_gap_seconds=1.8)
    )
    assert len(merged) == 2
