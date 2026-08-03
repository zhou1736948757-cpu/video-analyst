"""LLM post-calibration of ASR subtitles with automatic validation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from video_analyst.config import AppConfig, DEFAULT_CONFIG
from video_analyst.exceptions import LLMCalibrationError
from video_analyst.llm.validate import validate_and_build_llm_document
from video_analyst.logging.setup import get_logger
from video_analyst.models import (
    ASRMeta,
    CalibrationStatus,
    LLMMeta,
    SegmentDiff,
    SourceHashes,
    SubtitleDocument,
    SubtitleReviewPackage,
    SubtitleVersion,
    utc_now,
)
from video_analyst.util.hashing import sha256_json

log = get_logger("llm.calibrate")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
PROMPT_VERSION = "subtitle-calibration-v1"

SYSTEM_PROMPT = """你是字幕质检与校准助手。这不是润色、翻译或改写任务。

用户提供 Whisper 多语言字幕（可能含日语、中文、英语）。

规则：
1. 只有存在明确语音识别错误时才修改正文（如「张家好」→「大家好」）。
2. 必须保留原本语言；禁止翻译成其他语言。
3. 必须保留口语重复、停顿词和不完整表达；禁止删语气词、禁止书面化润色。
4. 禁止修改 index、start、end。
5. 禁止增加、删除、合并或拆分字幕条目。
6. 拿不准时原样返回。
7. 只输出 JSON 对象，不要 Markdown。

输出：
{
  "segments": [
    {"index": 1, "text": "校准后的文本", "note": "可选"}
  ]
}
segments 必须覆盖输入每一个 index，数量一致。
"""


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def calibrate_subtitles(
    asr_doc: SubtitleDocument,
    *,
    model: str | None = None,
    timeout: float = 120.0,
    skip_if_unavailable: bool = True,
    config: AppConfig | None = None,
    source_hashes: SourceHashes | None = None,
    asr_meta: ASRMeta | None = None,
) -> SubtitleReviewPackage:
    cfg = config or DEFAULT_CONFIG
    model_name = (
        model
        or os.environ.get("VIDEO_ANALYST_LLM_MODEL")
        or cfg.subtitle_calibration.model
        or DEFAULT_LLM_MODEL
    )

    base_meta = asr_meta or ASRMeta()
    hashes = source_hashes or SourceHashes(
        asr_subtitles=sha256_json(asr_doc.model_dump(mode="json"))
    )

    def _package(
        *,
        active: SubtitleVersion,
        llm: SubtitleDocument | None,
        diffs: list[SegmentDiff],
        llm_status: str,
        error: str | None,
        needs_review: bool = False,
    ) -> SubtitleReviewPackage:
        pkg = SubtitleReviewPackage(
            source_video=asr_doc.source_video,
            active=active,
            reviewed=False,
            needs_review=needs_review,
            asr=asr_doc.model_copy(deep=True),
            llm=llm,
            diffs=diffs,
            asr_meta=base_meta,
            llm_meta=LLMMeta(
                enabled=cfg.subtitle_calibration.enabled,
                provider=cfg.subtitle_calibration.provider,
                model=model_name,
                prompt_version=cfg.subtitle_calibration.prompt_version or PROMPT_VERSION,
                status=llm_status,
                error=error,
            ),
            source_hashes=hashes,
            llm_model=model_name,
            llm_error=error,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        pkg.recompute_summary()
        return pkg

    if not asr_doc.segments:
        return _package(
            active=SubtitleVersion.ASR,
            llm=None,
            diffs=[],
            llm_status="skipped",
            error="no segments to calibrate",
        )

    if not cfg.subtitle_calibration.enabled:
        return _package(
            active=SubtitleVersion.ASR,
            llm=None,
            diffs=[],
            llm_status="skipped",
            error="LLM calibration disabled",
        )

    if not llm_available():
        msg = "GROQ_API_KEY not configured; skipped LLM calibration"
        log.warning(msg)
        if not skip_if_unavailable:
            raise LLMCalibrationError(msg)
        return _package(
            active=SubtitleVersion.ASR,
            llm=None,
            diffs=[],
            llm_status="skipped",
            error=msg,
        )

    try:
        corrected, raw_response = _call_groq_calibrate(
            asr_doc, model=model_name, timeout=timeout
        )
    except LLMCalibrationError as exc:
        log.warning("LLM calibration failed, active=asr: %s", exc)
        return _package(
            active=SubtitleVersion.ASR,
            llm=None,
            diffs=[],
            llm_status="failed",
            error=str(exc),
        )

    llm_doc, diffs, overall_ok, val_err = validate_and_build_llm_document(
        asr_doc,
        corrected,
        cfg=cfg.subtitle_calibration.validation,
        model_name=model_name,
    )

    if llm_doc is None:
        return _package(
            active=SubtitleVersion.ASR,
            llm=None,
            diffs=diffs,
            llm_status="failed",
            error=val_err or "validation failed",
            needs_review=True,
        )

    # Attach raw preview without secrets
    llm_doc.metadata["llm_raw_preview"] = str(raw_response)[:500]

    needs = any(d.status == CalibrationStatus.NEEDS_REVIEW for d in diffs)
    active = SubtitleVersion.LLM if overall_ok else SubtitleVersion.ASR
    if not overall_ok:
        needs = True

    changed = sum(1 for d in diffs if d.changed)
    log.info(
        "LLM calibration validated: model=%s changed=%d/%d active=%s",
        model_name,
        changed,
        len(diffs),
        active.value,
    )
    return _package(
        active=active,
        llm=llm_doc,
        diffs=diffs,
        llm_status="completed",
        error=None,
        needs_review=needs,
    )


def _call_groq_calibrate(
    asr_doc: SubtitleDocument,
    *,
    model: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], str]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    payload_segments = [
        {
            "index": s.index,
            "start": s.start,
            "end": s.end,
            "language": s.language,
            "text": s.text,
        }
        for s in asr_doc.segments
    ]
    user_content = (
        "请校准下列字幕 JSON。只修正明确识别错误；禁止翻译/润色；禁止改时间轴。\n\n"
        + json.dumps({"segments": payload_segments}, ensure_ascii=False, indent=2)
    )
    body = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    log.info("Calling LLM calibration model=%s segments=%d", model, len(payload_segments))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(GROQ_CHAT_URL, headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise LLMCalibrationError(f"LLM HTTP error: {exc}") from exc

    if response.status_code >= 400:
        raise LLMCalibrationError(
            f"LLM API error {response.status_code}: {response.text[:800]}"
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMCalibrationError(f"Unexpected LLM response shape: {exc}") from exc

    parsed = _parse_llm_json(content)
    segments = parsed.get("segments")
    if not isinstance(segments, list) or not segments:
        raise LLMCalibrationError("LLM JSON missing segments list")

    normalized: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        normalized.append({"index": idx, "text": text, "note": item.get("note")})

    if not normalized:
        raise LLMCalibrationError("LLM returned no usable segments")

    have = {x["index"] for x in normalized}
    for seg in asr_doc.segments:
        if seg.index not in have:
            normalized.append(
                {"index": seg.index, "text": seg.text, "note": "missing in LLM output"}
            )

    return normalized, content if isinstance(content, str) else json.dumps(content)


def _parse_llm_json(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc2:
                raise LLMCalibrationError(f"LLM returned invalid JSON: {exc2}") from exc2
        else:
            raise LLMCalibrationError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMCalibrationError("LLM JSON root must be an object")
    return data
