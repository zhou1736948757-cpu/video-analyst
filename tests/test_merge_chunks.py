from video_analyst.asr.groq_whisper import (
    merge_chunk_payloads,
    merge_same_language_segments,
)


def test_merge_chunk_payloads_offsets_and_languages():
    chunks = [
        (
            0.0,
            {
                "language": "Japanese",
                "text": "こんにちは",
                "duration": 3.0,
                "segments": [
                    {"id": 0, "start": 0.0, "end": 2.5, "text": "こんにちは"},
                ],
            },
        ),
        (
            5.0,
            {
                "language": "Chinese",
                "text": "大家好",
                "duration": 3.0,
                "segments": [
                    {"id": 0, "start": 0.1, "end": 2.0, "text": "大家好"},
                ],
            },
        ),
        (
            10.0,
            {
                "language": "English",
                "text": "Hello",
                "duration": 4.0,
                "segments": [
                    {"id": 0, "start": 0.0, "end": 3.5, "text": "Hello"},
                ],
            },
        ),
    ]
    merged = merge_chunk_payloads(chunks, total_duration=20.0)
    assert merged["duration"] == 20.0
    assert len(merged["segments"]) == 3
    assert merged["segments"][0]["start"] == 0.0
    assert merged["segments"][1]["start"] == 5.1
    assert merged["segments"][2]["text"] == "Hello"
    assert "Japanese" in merged["language"]
    assert "Chinese" in merged["language"]
    assert "English" in merged["language"]


def test_same_language_pause_fragments_join_into_one_sentence():
    """Mid-sentence breath must not become two Japanese subtitle lines."""
    chunks = [
        (
            0.0,
            {
                "language": "Japanese",
                "text": "みんなさんこんにちは。今日はこのアプリに",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 2.9,
                        "text": "みんなさんこんにちは。今日はこのアプリに",
                    }
                ],
            },
        ),
        (
            3.04,
            {
                "language": "Japanese",
                "text": "次でご紹介いただきます。",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 2.02,
                        "text": "次でご紹介いただきます。",
                    }
                ],
            },
        ),
        (
            5.79,
            {
                "language": "Chinese",
                "text": "大家好。",
                "segments": [{"id": 0, "start": 0.0, "end": 0.8, "text": "大家好。"}],
            },
        ),
        (
            6.90,
            {
                "language": "Chinese",
                "text": "今天的话我会为大家介绍这个软件。",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 3.16,
                        "text": "今天的话我会为大家介绍这个软件。",
                    }
                ],
            },
        ),
        (
            11.62,
            {
                "language": "English",
                "text": "Hello everyone.",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 4.8, "text": "Hello everyone."}
                ],
            },
        ),
    ]
    merged = merge_chunk_payloads(chunks, total_duration=20.5)
    segs = merged["segments"]
    # JP+JP joined, ZH+ZH joined, EN alone → 3 language blocks
    assert len(segs) == 3
    assert segs[0]["language"] == "Japanese"
    assert "次でご紹介" in segs[0]["text"]
    assert segs[0]["start"] == 0.0
    assert abs(segs[0]["end"] - 5.06) < 1e-6
    assert segs[1]["language"] == "Chinese"
    assert "今天的话" in segs[1]["text"]
    assert segs[2]["language"] == "English"


def test_merge_same_language_respects_language_change():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "あ", "language": "Japanese"},
        {"start": 1.1, "end": 2.0, "text": "你好", "language": "Chinese"},
    ]
    out = merge_same_language_segments(segs, max_gap=2.0)
    assert len(out) == 2
