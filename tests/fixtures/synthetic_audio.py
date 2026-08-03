"""Generate a synthetic mixed-language audio fixture for offline Whisper testing.

The fixture contains Japanese, Chinese, and English-like acoustic segments
separated by breathing-like pauses, a no-speech silence interval, and a
low-frequency tone zone simulating BGM presence.  All synthesis uses only
stdlib ``wave`` + ``struct`` + ``math`` — no external audio dependencies.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000  # 16 kHz mono, Whisper's native rate
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


def _write_sine_stereo(
    samples: list[float],
    freq_hz: float,
    duration_sec: float,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> None:
    """Append a sine tone to *samples*."""
    n = int(sr * duration_sec)
    for i in range(n):
        t = i / sr
        samples.append(amplitude * math.sin(2.0 * math.pi * freq_hz * t))


def _write_harmonic(
    samples: list[float],
    base_hz: float,
    duration_sec: float,
    amplitude: float = 0.25,
    sr: int = SAMPLE_RATE,
) -> None:
    """Append a tone with harmonics (more speech-like)."""
    n = int(sr * duration_sec)
    for i in range(n):
        t = i / sr
        val = amplitude * (
            0.5 * math.sin(2.0 * math.pi * base_hz * t)
            + 0.3 * math.sin(2.0 * math.pi * base_hz * 2.0 * t)
            + 0.15 * math.sin(2.0 * math.pi * base_hz * 3.0 * t)
        )
        samples.append(val)


def _write_silence(samples: list[float], duration_sec: float, sr: int = SAMPLE_RATE) -> None:
    n = int(sr * duration_sec)
    samples.extend([0.0] * n)


def _write_ambient_noise(
    samples: list[float],
    duration_sec: float,
    amplitude: float = 0.005,
    sr: int = SAMPLE_RATE,
) -> None:
    """Low-level noise to represent ambient / no-speech sections."""
    import random

    n = int(sr * duration_sec)
    for _ in range(n):
        samples.append(amplitude * (random.random() - 0.5))


def create_mixed_language_fixture(
    path: Path,
    *,
    sr: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """Build a synthetic WAV at *path* with mixed-language acoustic segments.

    Returns a dict describing each segment (start, end, label, language) which
    mirrors the mock response ``transcribe_with_local_whisper`` would return
    for this audio.
    """
    samples: list[float] = []
    sr = sr

    # Segment descriptions: (label, language, duration_sec, synth_fn_kwargs)
    segments_meta = [
        # --- Japanese-like segment ---
        ("greeting_jp", "ja", 2.0, {"base_hz": 180}),
        # --- Breathing pause ---
        ("breath_1", None, 0.4, {}),
        # --- Chinese-like segment ---
        ("discuss_zh", "zh", 2.5, {"base_hz": 240}),
        # --- Short breathing pause ---
        ("breath_2", None, 0.3, {}),
        # --- English term ---
        ("term_en1", "en", 1.2, {"base_hz": 140}),
        # --- Breathing pause ---
        ("breath_3", None, 0.35, {}),
        # --- Japanese continued ---
        ("continue_jp", "ja", 1.8, {"base_hz": 200}),
        # --- Chinese with English term code-switch ---
        ("code_switch_zh_en", "zh", 2.0, {"base_hz": 220}),
        # --- Breathing pause ---
        ("breath_4", None, 0.5, {}),
        # --- English segment ---
        ("term_en2", "en", 1.5, {"base_hz": 130}),
        # --- No-speech silence interval (ambient) ---
        ("no_speech", None, 2.0, {}),
        # --- BGM marker section (low tone + speech-like harmonics) ---
        ("bgm_marker_with_voice", "ja", 3.0, {"base_hz": 100}),
    ]

    segment_info: list[dict[str, Any]] = []
    current_time = 0.0

    for label, lang, dur, kwargs in segments_meta:
        start = current_time
        if label.startswith("breath") or label == "no_speech":
            _write_ambient_noise(samples, dur, amplitude=0.003 if "breath" in label else 0.001)
        elif label == "bgm_marker_with_voice":
            # Low BGM-like tone mixed with voice harmonics
            n = int(sr * dur)
            for i in range(n):
                t = i / sr
                bgm = 0.15 * math.sin(2.0 * math.pi * 80 * t)  # low BGM hum
                voice = 0.2 * (
                    0.5 * math.sin(2.0 * math.pi * kwargs["base_hz"] * t)
                    + 0.3 * math.sin(2.0 * math.pi * kwargs["base_hz"] * 2.0 * t)
                )
                samples.append(bgm + voice)
        else:
            _write_harmonic(samples, kwargs["base_hz"], dur)
        end = current_time + dur
        segment_info.append(
            {
                "label": label,
                "language": lang,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": dur,
                "text": _label_to_display_text(label, lang),
            }
        )
        current_time = end

    # Normalize amplitude to avoid clipping
    if samples:
        max_amp = max(abs(s) for s in samples)
        if max_amp > 0.95:
            scale = 0.9 / max_amp
            samples = [s * scale for s in samples]

    # Write WAV
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sr)
        raw = b"".join(struct.pack("<h", int(s * 32767)) for s in samples)
        wf.writeframes(raw)

    return {
        "path": str(path),
        "sample_rate": sr,
        "duration": round(current_time, 3),
        "segments": segment_info,
    }


def _label_to_display_text(label: str, language: str | None) -> str:
    """Return representative text for mock transcription."""
    mapping = {
        "greeting_jp": "こんにちは、AI技術について話しましょう",
        "discuss_zh": "今天我们讨论API设计和系统架构",
        "term_en1": "Let's review the request flow",
        "continue_jp": "このAPIは複数の言語をサポートしています",
        "code_switch_zh_en": "我们需要确保the latency is under 200ms",
        "term_en2": "The batch pipeline processes concurrent jobs",
        "bgm_marker_with_voice": "BGM背景音乐中の音声です",
    }
    return mapping.get(label, "")
