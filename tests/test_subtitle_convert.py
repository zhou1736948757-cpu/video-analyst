from pathlib import Path

from video_analyst.models import ASREngine
from video_analyst.subtitle.convert import raw_asr_to_document


def test_raw_asr_to_document(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "audio.mp3"
    video.write_bytes(b"x")
    audio.write_bytes(b"y")

    raw = {
        "text": "Hello world",
        "language": "en",
        "duration": 2.5,
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": " Hello"},
            {"id": 1, "start": 1.0, "end": 2.5, "text": "world "},
            {"id": 2, "start": 2.5, "end": 2.5, "text": "  "},
        ],
    }
    doc = raw_asr_to_document(
        raw,
        source_video=video,
        audio_path=audio,
        engine=ASREngine.GROQ_WHISPER_LARGE_V3_TURBO,
    )
    assert doc.language == "en"
    assert len(doc.segments) == 2
    assert doc.segments[0].text == "Hello"
    assert doc.segments[1].index == 2
    assert doc.engine == ASREngine.GROQ_WHISPER_LARGE_V3_TURBO
