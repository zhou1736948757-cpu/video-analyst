"""Mixed-language fixture verification — Groq cloud primary with local Whisper fallback."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from video_analyst.asr.pipeline import transcribe_audio
from video_analyst.config import AppConfig
from video_analyst.exceptions import ASRError, GroqASRError
from video_analyst.models import ASREngine

from tests.fixtures.synthetic_audio import create_mixed_language_fixture


class TestMixedLanguageFixture:
    """Synthetic mixed-language fixture validation."""

    def test_fixture_characteristics(self, tmp_path: Path) -> None:
        wav = tmp_path / "mixed_lang.wav"
        meta = create_mixed_language_fixture(wav)

        assert wav.is_file(), "WAV fixture was not written"
        assert wav.stat().st_size > 0, "WAV fixture is empty"

        labels = [s["label"] for s in meta["segments"]]
        languages = [s["language"] for s in meta["segments"] if s["language"] is not None]

        assert "ja" in set(languages), "No Japanese-language segment in fixture"
        assert "zh" in set(languages), "No Chinese-language segment in fixture"
        assert "en" in set(languages), "No English-language segment in fixture"

        breath_labels = [label for label in labels if label.startswith("breath")]
        assert len(breath_labels) >= 2, "Fewer than 2 breathing-like pauses"
        assert "no_speech" in labels, "No no-speech interval in fixture"
        assert any("bgm" in label for label in labels), "No BGM marker in fixture"
        assert meta["duration"] >= 10.0, f"Fixture too short: {meta['duration']:.1f}s"

    def test_llm_calibration_enabled_for_groq_path(self) -> None:
        cfg = AppConfig()
        assert cfg.subtitle_calibration.enabled is True
        assert cfg.subtitle_calibration.provider == "groq"
        assert cfg.asr.provider == "auto"


class TestGroqCloudPath:
    """Groq-cloud routing for the mixed-language fixture."""

    def _mock_raw(self, meta: dict) -> dict:
        segments = []
        combined = []
        for i, seg in enumerate(meta["segments"]):
            if seg["language"] is None:
                continue
            segments.append(
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
            )
            combined.append(seg["text"])
        return {
            "text": " ".join(combined),
            "language": None,
            "duration": meta["duration"],
            "segments": segments,
            "task": "transcribe",
            "_engine": ASREngine.GROQ_WHISPER_LARGE_V3_TURBO.value,
            "_model": "whisper-large-v3-turbo",
        }

    def test_prefers_groq(self, tmp_path: Path) -> None:
        wav = tmp_path / "mixed_lang.wav"
        meta = create_mixed_language_fixture(wav)
        mock_raw = self._mock_raw(meta)

        with (
            patch("video_analyst.asr.pipeline.groq_available", return_value=True),
            patch(
                "video_analyst.asr.pipeline.transcribe_chunked_groq",
                return_value=(
                    mock_raw,
                    ASREngine.GROQ_WHISPER_LARGE_V3_TURBO,
                    None,
                ),
            ) as groq_mock,
        ):
            result, engine, doc = transcribe_audio(wav, multilingual=True)
            groq_mock.assert_called_once()
            assert engine == ASREngine.GROQ_WHISPER_LARGE_V3_TURBO
            assert doc is None
            assert len(result.get("segments", [])) > 0

    def test_raises_when_groq_unavailable(self, tmp_path: Path) -> None:
        wav = tmp_path / "mixed_lang.wav"
        create_mixed_language_fixture(wav)

        # Groq 未配置且 local Whisper 不可用时，必须抛 ASRError 指向 GROQ_API_KEY
        with (
            patch("video_analyst.asr.pipeline.groq_available", return_value=False),
            patch("video_analyst.asr.pipeline.local_whisper_installed", return_value=False),
        ):
            with pytest.raises(ASRError, match="GROQ_API_KEY"):
                transcribe_audio(wav, multilingual=True)

    def test_raises_when_groq_fails(self, tmp_path: Path) -> None:
        wav = tmp_path / "mixed_lang.wav"
        create_mixed_language_fixture(wav)

        with (
            patch("video_analyst.asr.pipeline.groq_available", return_value=True),
            patch(
                "video_analyst.asr.pipeline.transcribe_chunked_groq",
                side_effect=GroqASRError("api error"),
            ),
        ):
            with pytest.raises(ASRError, match="Groq ASR failed"):
                transcribe_audio(wav, multilingual=True)


class TestOfflineLocalPath:
    """Offline/local Whisper routing — proves zero provider-call attempts."""

    def _mock_local_raw(self, meta: dict) -> dict:
        segments = []
        combined = []
        for i, seg in enumerate(meta["segments"]):
            if seg["language"] is None:
                continue
            segments.append(
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
            )
            combined.append(seg["text"])
        return {
            "text": " ".join(combined),
            "language": None,
            "duration": meta["duration"],
            "segments": segments,
            "task": "transcribe",
            "_engine": ASREngine.LOCAL_WHISPER_SMALL.value,
            "_model": "small",
        }

    def test_offline_mixed_language_verification(self, tmp_path: Path) -> None:
        """Core AC: force local Whisper, disable LLM calibration, prove zero Groq/LLM provider calls."""
        wav = tmp_path / "mixed_lang.wav"
        meta = create_mixed_language_fixture(wav)
        mock_raw = self._mock_local_raw(meta)

        # Disable LLM calibration for the offline path
        cfg = AppConfig()
        cfg.subtitle_calibration.enabled = False
        cfg.subtitle_calibration.provider = "local"

        groq_spy = patch("video_analyst.asr.pipeline.transcribe_chunked_groq")
        llm_spy = patch("video_analyst.llm.calibrate.calibrate_subtitles")

        with (
            patch("video_analyst.asr.pipeline.groq_available", return_value=True),
            patch(
                "video_analyst.asr.pipeline.local_whisper_installed",
                return_value=True,
            ),
            patch(
                "video_analyst.asr.pipeline.transcribe_with_local_whisper",
                return_value=(mock_raw, ASREngine.LOCAL_WHISPER_SMALL),
            ) as local_mock,
            groq_spy as groq_mock,
            llm_spy as llm_mock,
        ):
            result, engine, doc = transcribe_audio(
                wav,
                force_local=True,
                multilingual=True,
                config=cfg,
            )

            # Local Whisper was called
            local_mock.assert_called_once()

            # Zero Groq provider-call attempts
            groq_mock.assert_not_called()

            # Zero LLM calibration provider-call attempts
            llm_mock.assert_not_called()

            # Engine is local Whisper
            assert engine == ASREngine.LOCAL_WHISPER_SMALL

            # All fixture languages are preserved
            assert len(result.get("segments", [])) > 0

            # LLM calibration is disabled in config
            assert cfg.subtitle_calibration.enabled is False

    def test_falls_back_to_local_when_groq_unavailable(self, tmp_path: Path) -> None:
        """Groq unavailable + local installed = transparent local fallback."""
        wav = tmp_path / "mixed_lang.wav"
        meta = create_mixed_language_fixture(wav)
        mock_raw = self._mock_local_raw(meta)

        with (
            patch("video_analyst.asr.pipeline.groq_available", return_value=False),
            patch(
                "video_analyst.asr.pipeline.local_whisper_installed",
                return_value=True,
            ),
            patch(
                "video_analyst.asr.pipeline.transcribe_with_local_whisper",
                return_value=(mock_raw, ASREngine.LOCAL_WHISPER_SMALL),
            ) as local_mock,
            patch("video_analyst.asr.pipeline.transcribe_chunked_groq") as groq_mock,
        ):
            result, engine, doc = transcribe_audio(wav, multilingual=True)
            local_mock.assert_called_once()
            groq_mock.assert_not_called()
            assert engine == ASREngine.LOCAL_WHISPER_SMALL

    def test_force_local_routes_to_local_whisper(self, tmp_path: Path) -> None:
        """force_local=True picks local even when Groq is available."""
        wav = tmp_path / "mixed_lang.wav"
        meta = create_mixed_language_fixture(wav)
        mock_raw = self._mock_local_raw(meta)

        with (
            patch("video_analyst.asr.pipeline.groq_available", return_value=True),
            patch(
                "video_analyst.asr.pipeline.local_whisper_installed",
                return_value=True,
            ),
            patch(
                "video_analyst.asr.pipeline.transcribe_with_local_whisper",
                return_value=(mock_raw, ASREngine.LOCAL_WHISPER_SMALL),
            ) as local_mock,
            patch("video_analyst.asr.pipeline.transcribe_chunked_groq") as groq_mock,
        ):
            result, engine, doc = transcribe_audio(wav, force_local=True)
            local_mock.assert_called_once()
            groq_mock.assert_not_called()
            assert engine == ASREngine.LOCAL_WHISPER_SMALL

    def test_offline_mode_env_var_enforces_no_groq(self, tmp_path: Path) -> None:
        """VIDEO_ANALYST_FORCE_LOCAL_WHISPER=1 enforces offline routing."""
        wav = tmp_path / "mixed_lang.wav"
        meta = create_mixed_language_fixture(wav)
        mock_raw = self._mock_local_raw(meta)

        with (
            patch.dict(
                os.environ,
                {"VIDEO_ANALYST_FORCE_LOCAL_WHISPER": "1"},
                clear=False,
            ),
            patch("video_analyst.asr.pipeline.groq_available", return_value=True),
            patch(
                "video_analyst.asr.pipeline.local_whisper_installed",
                return_value=True,
            ),
            patch(
                "video_analyst.asr.pipeline.transcribe_with_local_whisper",
                return_value=(mock_raw, ASREngine.LOCAL_WHISPER_SMALL),
            ) as local_mock,
            patch("video_analyst.asr.pipeline.transcribe_chunked_groq") as groq_mock,
        ):
            transcribe_audio(wav)
            local_mock.assert_called_once()
            groq_mock.assert_not_called()
