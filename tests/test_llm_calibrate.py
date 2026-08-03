"""LLM calibration tests — never call the real API."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from video_analyst.llm.calibrate import calibrate_subtitles, _parse_llm_json
from video_analyst.llm.review import apply_active_version, save_review_package
from video_analyst.llm.validate import (
    char_change_ratio,
    looks_like_translation,
    validate_and_build_llm_document,
)
from video_analyst.cache.store import OutputLayout
from video_analyst.models import (
    ASREngine,
    CalibrationStatus,
    SegmentDiff,
    SubtitleDocument,
    SubtitleReviewPackage,
    SubtitleSegment,
    SubtitleVersion,
)


def _asr_doc() -> SubtitleDocument:
    return SubtitleDocument(
        source_video="/tmp/v.mp4",
        audio_path="/tmp/a.mp3",
        language="Japanese+Chinese+English",
        engine=ASREngine.GROQ_WHISPER_LARGE_V3_TURBO,
        duration=20.0,
        origin="asr",
        segments=[
            SubtitleSegment(index=1, start=0.0, end=5.0, text="こんにちは", language="Japanese"),
            SubtitleSegment(index=2, start=5.0, end=10.0, text="张家好。", language="Chinese"),
            SubtitleSegment(
                index=3,
                start=11.0,
                end=16.0,
                text="Hello everyone.",
                language="English",
            ),
        ],
    )


def test_parse_llm_json_with_fence():
    raw = '```json\n{"segments": [{"index": 1, "text": "hi"}]}\n```'
    data = _parse_llm_json(raw)
    assert data["segments"][0]["text"] == "hi"


def test_calibrate_without_key_falls_back_to_asr():
    with patch.dict("os.environ", {}, clear=True):
        import os

        os.environ.pop("GROQ_API_KEY", None)
        pkg = calibrate_subtitles(_asr_doc())
    assert pkg.active == SubtitleVersion.ASR
    assert pkg.llm is None
    assert pkg.llm_meta.error or pkg.llm_error
    assert "GROQ_API_KEY" in (pkg.llm_meta.error or pkg.llm_error or "")


def test_calibrate_success_sets_active_llm():
    asr = _asr_doc()
    llm_body = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"segments":['
                        '{"index":1,"text":"こんにちは","note":null},'
                        '{"index":2,"text":"大家好。","note":"同音误识"},'
                        '{"index":3,"text":"Hello everyone.","note":null}'
                        "]}"
                    )
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = llm_body

    with (
        patch("video_analyst.llm.calibrate.llm_available", return_value=True),
        patch("video_analyst.llm.calibrate.os.environ.get", return_value="gsk_test"),
        patch("video_analyst.llm.calibrate.httpx.Client") as client_cls,
    ):
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client.post.return_value = mock_resp
        client_cls.return_value = client

        pkg = calibrate_subtitles(asr)

    assert pkg.active == SubtitleVersion.LLM
    assert pkg.llm is not None
    assert pkg.llm.segments[1].text == "大家好。"
    assert pkg.diffs[1].changed is True
    assert pkg.diffs[0].changed is False
    assert pkg.reviewed is False


def test_validate_rejects_structure_change():
    asr = _asr_doc()
    llm_items = [{"index": 1, "text": "x"}]  # missing entries
    doc, diffs, ok, err = validate_and_build_llm_document(asr, llm_items)
    assert doc is None
    assert ok is False
    assert err


def test_validate_rejects_translation():
    assert looks_like_translation("大家好今天介绍软件", "Hello everyone today", "zh")
    assert char_change_ratio("张家好", "大家好") > 0


def test_validate_rejects_timestamp_tamper():
    asr = _asr_doc()
    items = [
        {"index": 1, "text": "こんにちは", "start": 0.0, "end": 5.0},
        {"index": 2, "text": "大家好。", "start": 9.0, "end": 10.0},  # start changed
        {"index": 3, "text": "Hello everyone.", "start": 11.0, "end": 16.0},
    ]
    doc, diffs, ok, err = validate_and_build_llm_document(asr, items)
    assert doc is None
    assert ok is False


def test_save_and_switch_active(tmp_path: Path):
    asr = _asr_doc()
    llm = asr.model_copy(deep=True)
    llm.origin = "llm"
    llm.segments[1].text = "大家好。"

    pkg = SubtitleReviewPackage(
        source_video=asr.source_video,
        active=SubtitleVersion.LLM,
        reviewed=False,
        asr=asr,
        llm=llm,
        diffs=[
            SegmentDiff(
                index=2,
                start=5.0,
                end=10.0,
                language="Chinese",
                asr_text="张家好。",
                llm_text="大家好。",
                changed=True,
                status=CalibrationStatus.ACCEPTED,
                note="fix",
            )
        ],
        llm_model="test-model",
    )
    layout = OutputLayout.create(tmp_path / "out")
    save_review_package(pkg, layout)

    assert layout.subtitles_asr_srt.is_file()
    assert layout.subtitles_llm_srt.is_file()
    assert layout.review_json.is_file()
    active_text = layout.subtitles_srt.read_text(encoding="utf-8")
    assert "大家好" in active_text
    assert "张家好" not in active_text

    # create dummy phase2 mapping that should be cleared
    layout.phase2_mapping_json.write_text("{}", encoding="utf-8")
    layout.phase2_batches_json.write_text("{}", encoding="utf-8")

    apply_active_version(layout, SubtitleVersion.ASR, reviewed=True)
    active_text2 = layout.subtitles_srt.read_text(encoding="utf-8")
    assert "张家好" in active_text2
    assert not layout.phase2_mapping_json.is_file()
    assert not layout.phase2_batches_json.is_file()
