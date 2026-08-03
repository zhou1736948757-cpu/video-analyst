"""ASR pipeline tests — Groq cloud only, never calls the real API."""

from pathlib import Path
from unittest.mock import patch

import pytest

from video_analyst.asr.pipeline import transcribe_audio
from video_analyst.exceptions import ASRError, GroqASRError
from video_analyst.models import ASREngine


def test_transcribe_prefers_groq(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    raw = {
        "text": "hi",
        "language": "en",
        "duration": 1.0,
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}],
    }

    with (
        patch("video_analyst.asr.pipeline.groq_available", return_value=True),
        patch(
            "video_analyst.asr.pipeline.transcribe_chunked_groq",
            return_value=(raw, ASREngine.GROQ_WHISPER_LARGE_V3_TURBO, None),
        ) as groq_mock,
    ):
        result, engine, doc = transcribe_audio(audio)
        assert engine == ASREngine.GROQ_WHISPER_LARGE_V3_TURBO
        assert result["text"] == "hi"
        assert doc is None
        groq_mock.assert_called_once()


def test_transcribe_raises_when_groq_unavailable(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")

    # Groq 未配置且 local Whisper 不可用时，必须抛 ASRError 指向 GROQ_API_KEY
    with (
        patch("video_analyst.asr.pipeline.groq_available", return_value=False),
        patch("video_analyst.asr.pipeline.local_whisper_installed", return_value=False),
    ):
        with pytest.raises(ASRError, match="GROQ_API_KEY"):
            transcribe_audio(audio)


def test_transcribe_raises_when_groq_fails(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")

    with (
        patch("video_analyst.asr.pipeline.groq_available", return_value=True),
        patch(
            "video_analyst.asr.pipeline.transcribe_chunked_groq",
            side_effect=GroqASRError("api error"),
        ),
    ):
        with pytest.raises(ASRError, match="Groq ASR failed"):
            transcribe_audio(audio)
