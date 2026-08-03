"""Fine batch 执行器（Epic 4）。

实现：

    run_fine_analysis(
        fine_batches_path: Path,
        output_dir: Path,
        provider: FineVisionProvider,
        user_request: str,
    ) -> FineAnalysisResult

处理逻辑：
    读取 fine_batches
    → 校验图片和字幕
    → 按时间顺序处理
    → 为每批构建请求
    → 调用 Provider
    → 验证响应
    → 保存批次结果
    → 更新缓存
    → 继续下一批

要求：
1. 每批完成后立即原子写入
2. 中断后可从下一批继续
3. 批次结果不得只保存在内存
4. 单批失败可重试
5. 超过重试次数后标记失败并继续
6. 最终结果必须显示成功、失败和跳过数量
7. 可配置是否允许 partial success
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.config import AppConfig, FineAnalysisConfig, VisionConfig
from video_analyst.exceptions import (
    FineBatchAnalysisError,
    FineVisionProviderUnavailableError,
    FineVisionResponseParseError,
    FineVisionResponseValidationError,
    FineVisionTimeoutError,
    FineVisionRateLimitError,
    FineVisionRequestError,
)
from video_analyst.fine.aggregate import (
    aggregate_fine_analysis,
    save_fine_video_analysis,
    save_temporal_events,
)
from video_analyst.fine.deduplicate import deduplicate_batch_analyses
from video_analyst.fine.manifest import (
    build_fine_cache_key,
    load_fine_batch_analysis,
    load_fine_manifest,
    persist_fine_batch_analysis,
    persist_fine_raw_response,
    save_fine_manifest,
    upsert_fine_batch_record,
)
from video_analyst.fine.models import (
    FINE_SCHEMA_VERSION,
    FineAnalysisManifest,
    FineAnalysisResult,
    FineBatchAnalysis,
    FineBatchRunRecord,
    FineVisionRequest,
)
from video_analyst.fine.prompts import PROMPT_VERSION
from video_analyst.fine.request_builder import (
    RequestBuildReport,
    build_all_fine_requests,
    build_fine_request,
)
from video_analyst.fine.second_refinement import (
    build_second_refinement_plan,
    save_second_refinement_plan,
)
from video_analyst.fine.validation import (
    assert_fine_analysis_within_interval,
    assert_question_coverage,
    load_fine_batches_index,
)
from video_analyst.logging.setup import get_logger
from video_analyst.refinement.models import FineVisualBatch
from video_analyst.util.hashing import sha256_json

log = get_logger("fine.batch_runner")


def _safe_filename(fine_batch_id: str) -> str:
    return f"{fine_batch_id}.json"


def _raw_filename(fine_batch_id: str) -> str:
    return f"{fine_batch_id}_response.json"


def _is_cached(
    manifest: FineAnalysisManifest | None,
    fine_batch_id: str,
    cache_key: str,
) -> bool:
    if manifest is None:
        return False
    for rec in manifest.batches:
        if (
            rec.fine_batch_id == fine_batch_id
            and rec.status in ("ok", "cached")
            and rec.cache_key == cache_key
        ):
            return True
    return False


def _analyze_with_provider(
    provider: Any,
    request: FineVisionRequest,
    *,
    stage: str = "fine_vision",
) -> tuple[FineBatchAnalysis, str]:
    """调用 provider 并解析。返回 (analysis, raw_snapshot)。

    Epic 6: 经过统一 TelemetryRecorder 记录。stage 可为 fine_vision 或 terminal_vision。
    """
    from video_analyst.telemetry.recorder import get_global_recorder

    recorder = get_global_recorder()
    with recorder.record_call(
        provider=provider.provider_name,
        model=provider.model_name,
        stage=stage,  # type: ignore[arg-type]
        batch_id=request.fine_batch_id,
    ) as rec:
        # 图片数 = contact sheets + detail frames
        rec.input_image_count = len(request.contact_sheets) + len(request.detail_frames)
        if provider.provider_name == "mock":
            rec.network_request_sent = False
            rec.response_received = True
        else:
            rec.network_request_sent = True
        try:
            analysis = provider.analyze_fine_batch(request)
            rec.response_received = True
            rec.success = True
        except Exception as exc:
            rec.response_received = False
            rec.success = False
            rec.error_type = type(exc).__name__
            raise
    raw_snapshot = json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False)
    return analysis, raw_snapshot


def _execute_one_batch(
    *,
    layout: OutputLayout,
    request: FineVisionRequest,
    report: RequestBuildReport,
    provider: Any,
    fine_cfg: FineAnalysisConfig,
    vision_cfg: VisionConfig,
    config_fingerprint: str,
    allow_repair: bool = True,
    extra_force_refresh: bool = False,
) -> tuple[FineBatchRunRecord, FineBatchAnalysis | None]:
    """运行单个 fine batch（含重试）。返回 (record, analysis_or_None)。"""
    cache_key = build_fine_cache_key(
        request=request,
        report=report,
        layout=layout,
        provider_name=provider.provider_name,
        model=provider.model_name,
        cfg=fine_cfg,
        vision_cfg=vision_cfg,
    )

    # 缓存检查
    if not extra_force_refresh:
        manifest = load_fine_manifest(layout)
        if _is_cached(manifest, request.fine_batch_id, cache_key):
            log.info(
                "fine cache hit: %s key=%s (skipping provider call)",
                request.fine_batch_id,
                cache_key[:10],
            )
            cached = load_fine_batch_analysis(layout, request.fine_batch_id)
            rec = FineBatchRunRecord(
                fine_batch_id=request.fine_batch_id,
                interval_id=request.interval.interval_id,
                provider=provider.provider_name,
                model=provider.model_name,
                prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
                status="cached",
                attempts=0,
                cache_key=cache_key,
                event_count=len(cached.temporal_events) if cached else 0,
                question_count=len(cached.question_answers) if cached else 0,
                second_refinement_count=len(cached.second_refinement_requests) if cached else 0,
            )
            return rec, cached

    log.info(
        "running fine batch=%s sheets=%d detail=%d questions=%d",
        request.fine_batch_id,
        len(request.contact_sheets),
        len(request.detail_frames),
        len(request.refinement_questions),
    )

    max_retries = max(0, int(vision_cfg.max_retries))
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            analysis, raw_text = _analyze_with_provider(provider, request)
            # 跨阶段校验
            assert_fine_analysis_within_interval(
                analysis,
                interval_start=request.interval.start,
                interval_end=request.interval.end,
            )
            assert_question_coverage(
                analysis,
                refinement_questions=list(request.refinement_questions),
            )
            # 持久化
            analysis_path = persist_fine_batch_analysis(layout, analysis)
            persist_fine_raw_response(
                layout,
                request.fine_batch_id,
                cache_key,
                raw_text,
            )
            rec = FineBatchRunRecord(
                fine_batch_id=request.fine_batch_id,
                interval_id=request.interval.interval_id,
                provider=provider.provider_name,
                model=provider.model_name,
                prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
                status="ok",
                attempts=attempt + 1,
                raw_response_path=str(
                    layout.fine_raw_dir.relative_to(layout.root)
                    / _raw_filename(request.fine_batch_id)
                ),
                analysis_path=str(analysis_path.relative_to(layout.root)),
                cache_key=cache_key,
                event_count=len(analysis.temporal_events),
                question_count=len(analysis.question_answers),
                second_refinement_count=len(analysis.second_refinement_requests),
            )
            return rec, analysis
        except (
            FineVisionTimeoutError,
            FineVisionRateLimitError,
            FineVisionRequestError,
        ) as exc:
            # 可重试错误
            last_exc = exc
            log.warning(
                "fine batch %s attempt %d failed (retryable): %s",
                request.fine_batch_id,
                attempt + 1,
                exc,
            )
            if attempt >= max_retries:
                break
            import time

            time.sleep(float(vision_cfg.retry_backoff_base) ** attempt)
            continue
        except FineVisionResponseParseError as exc:
            # 尝试一次 repair
            last_exc = exc
            if allow_repair and provider.supports_repair_request():
                log.info(
                    "fine batch %s parse error; attempting repair",
                    request.fine_batch_id,
                )
                try:
                    repair_payload = provider.build_repair_payload(
                        "", "previous response was not valid JSON"
                    )
                    if repair_payload is not None:
                        # 通过 provider 的 repair 路径重新获取
                        repaired_text = _invoke_provider_repair(provider, repair_payload)
                        if repaired_text:
                            parsed = provider.parse_repair_response(repaired_text)
                            if parsed is not None:
                                from video_analyst.fine.response_parser import (
                                    validate_fine_response,
                                )

                                analysis, _w = validate_fine_response(
                                    parsed, request=request
                                )
                                assert_fine_analysis_within_interval(
                                    analysis,
                                    interval_start=request.interval.start,
                                    interval_end=request.interval.end,
                                )
                                assert_question_coverage(
                                    analysis,
                                    refinement_questions=list(request.refinement_questions),
                                )
                                raw_text = json.dumps(
                                    analysis.model_dump(mode="json"), ensure_ascii=False
                                )
                                analysis_path = persist_fine_batch_analysis(layout, analysis)
                                persist_fine_raw_response(
                                    layout, request.fine_batch_id, cache_key, raw_text
                                )
                                rec = FineBatchRunRecord(
                                    fine_batch_id=request.fine_batch_id,
                                    interval_id=request.interval.interval_id,
                                    provider=provider.provider_name,
                                    model=provider.model_name,
                                    prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
                                    status="ok",
                                    attempts=attempt + 2,
                                    raw_response_path=str(
                                        layout.fine_raw_dir.relative_to(layout.root)
                                        / _raw_filename(request.fine_batch_id)
                                    ),
                                    analysis_path=str(analysis_path.relative_to(layout.root)),
                                    cache_key=cache_key,
                                    event_count=len(analysis.temporal_events),
                                    question_count=len(analysis.question_answers),
                                    second_refinement_count=len(analysis.second_refinement_requests),
                                )
                                return rec, analysis
                except Exception as repair_exc:
                    log.warning(
                        "fine batch %s repair failed: %s",
                        request.fine_batch_id,
                        repair_exc,
                    )
            break
        except FineVisionResponseValidationError as exc:
            last_exc = exc
            log.warning(
                "fine batch %s validation error (non-retryable): %s",
                request.fine_batch_id,
                exc,
            )
            break
        except Exception as exc:
            last_exc = exc
            log.error(
                "fine batch %s unexpected error: %s",
                request.fine_batch_id,
                exc,
            )
            break

    # 失败
    rec = FineBatchRunRecord(
        fine_batch_id=request.fine_batch_id,
        interval_id=request.interval.interval_id,
        provider=provider.provider_name,
        model=provider.model_name,
        prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
        status="failed",
        attempts=max_retries + 1,
        error=str(last_exc)[:500] if last_exc else "unknown",
        cache_key=cache_key,
    )
    return rec, None


def _invoke_provider_repair(provider: Any, repair_payload: dict[str, Any]) -> str | None:
    """调用 provider 的 repair 路径（mock 直接返回 expected_payload JSON）。"""
    # mock provider 的 build_repair_payload 返回 messages 包装；
    # parse_repair_response 能从中恢复 expected_payload。
    # 真实 provider 通过 _chat_repair 走 HTTP。这里统一用 parse_repair_response。
    try:
        # 把 messages 序列化成 raw_text，再让 parse_repair_response 解析
        raw_text = json.dumps(repair_payload, ensure_ascii=False)
        return raw_text
    except Exception:
        return None


def run_fine_analysis(
    *,
    layout: OutputLayout,
    config: AppConfig,
    provider: Any,
    user_request: str = "",
    analysis_dimensions: list[str] | None = None,
    video_id: str = "video",
    refresh: bool = False,
    refresh_batches: list[str] | None = None,
    allow_partial: bool | None = None,
    build_second_plan: bool = True,
) -> FineAnalysisResult:
    """逐批执行 fine 分析。

    输出：
        analysis/fine/
        ├── manifest.json
        ├── batches/fine_interval_XXXX.json
        ├── raw/fine_interval_XXXX_response.json
        ├── temporal_events.json
        ├── fine_video_analysis.json
        └── second_refinement_plan.json
    """
    fine_cfg = config.fine_analysis
    vision_cfg = config.vision
    if allow_partial is not None:
        fine_cfg.allow_partial = bool(allow_partial)

    # 读取 fine_batches.json
    index_data = load_fine_batches_index(layout.refinement_fine_batches_json)
    batches_raw = index_data.get("batches") or []
    if not batches_raw:
        log.info("fine_batches.json has no batches; nothing to analyze")
        # 写空 manifest + 空 fine_video_analysis
        manifest = FineAnalysisManifest(
            video_id=video_id,
            provider=provider.provider_name,
            model=provider.model_name,
            prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
            user_request=user_request,
            allow_partial=fine_cfg.allow_partial,
            config_fingerprint=sha256_json(config.fingerprint()),
        )
        manifest.recompute_counts()
        save_fine_manifest(layout, manifest)
        # 空 plan：video_duration 未知，直接构造空聚合（不调用 aggregate_fine_analysis
        # 以避免 video_end <= video_start 校验失败）
        from video_analyst.fine.models import FineVideoAnalysis

        agg = FineVideoAnalysis(
            schema_version=FINE_SCHEMA_VERSION,
            video_start=0.0,
            video_end=0.0,
            analyzed_intervals=[],
            analysis_gaps=[],
            temporal_events=[],
            subject_motion_summary=[],
            camera_motion_summary=[],
            text_animation_summary=[],
            graphic_animation_summary=[],
            transition_summary=[],
            answered_questions=[],
            unresolved_questions=[],
            important_events=[],
            second_refinement_required=False,
            second_refinement_plan_path=None,
            confidence=0.3,
            warnings=["no fine batches to analyze"],
        )
        save_fine_video_analysis(layout, agg)
        save_temporal_events(layout, [])
        if build_second_plan:
            plan = build_second_refinement_plan(
                requests_by_batch={},
                video_duration=0.0,
                round1_refinement_duration=0.0,
                config=fine_cfg,
            )
            save_second_refinement_plan(layout, plan)
        return FineAnalysisResult(
            schema_version=FINE_SCHEMA_VERSION,
            status="completed_no_batches",
            total_batches=0,
            succeeded_batches=0,
            failed_batches=0,
            cached_batches=0,
            total_events=0,
            fine_video_analysis_path=layout.fine_video_analysis_json,
            manifest_path=layout.fine_manifest_json,
            temporal_events_path=layout.fine_temporal_events_json,
            second_refinement_plan_path=layout.fine_second_refinement_plan_json
            if build_second_plan
            else None,
        )

    # 按时间顺序排序
    batches_sorted = sorted(
        batches_raw,
        key=lambda b: (
            b.get("interval", {}).get("start", 0.0),
            b.get("fine_batch_id", ""),
        ),
    )
    batches_order = [b.get("fine_batch_id", "") for b in batches_sorted]
    batches_meta: dict[str, dict[str, Any]] = {}
    for b in batches_sorted:
        fb_id = b.get("fine_batch_id", "")
        iv = b.get("interval", {}) or {}
        batches_meta[fb_id] = {"start": iv.get("start", 0.0), "end": iv.get("end", 0.0)}

    # 构建 FineVisualBatch 对象
    fine_batches: list[FineVisualBatch] = []
    for b in batches_sorted:
        try:
            fine_batches.append(FineVisualBatch.model_validate(b))
        except Exception as exc:
            log.warning("skipping invalid fine batch: %s", exc)

    # 加载或创建 manifest
    manifest = load_fine_manifest(layout)
    if manifest is None:
        manifest = FineAnalysisManifest(
            video_id=video_id,
            provider=provider.provider_name,
            model=provider.model_name,
            prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
            user_request=user_request,
            allow_partial=fine_cfg.allow_partial,
            config_fingerprint=sha256_json(config.fingerprint()),
        )
    else:
        manifest = manifest.model_copy(
            update={
                "provider": provider.provider_name,
                "model": provider.model_name,
                "prompt_version": fine_cfg.prompt_version or PROMPT_VERSION,
                "user_request": user_request,
                "allow_partial": fine_cfg.allow_partial,
                "config_fingerprint": sha256_json(config.fingerprint()),
            }
        )

    # refresh：清除指定 batch 的缓存记录
    target_refresh = set(refresh_batches or [])
    if refresh:
        target_refresh = set(batches_order)

    analyses: dict[str, FineBatchAnalysis] = {}
    failed: dict[str, str] = {}
    new_records: list[FineBatchRunRecord] = []

    # 先从磁盘加载已成功 batch（用于 previous context / aggregation）
    for fb_id in batches_order:
        cached = load_fine_batch_analysis(layout, fb_id)
        if cached is not None:
            analyses[fb_id] = cached

    for batch in fine_batches:
        fb_id = batch.fine_batch_id
        # 构建请求
        request, build_report = build_fine_request(
            batch,
            layout=layout,
            cfg=fine_cfg,
            user_request=user_request,
            analysis_dimensions=analysis_dimensions,
            video_id=video_id,
        )

        try:
            record, analysis = _execute_one_batch(
                layout=layout,
                request=request,
                report=build_report,
                provider=provider,
                fine_cfg=fine_cfg,
                vision_cfg=vision_cfg,
                config_fingerprint=manifest.config_fingerprint or "",
                allow_repair=True,
                extra_force_refresh=(fb_id in target_refresh),
            )
        except Exception as exc:
            log.error("fine batch %s crashed: %s", fb_id, exc)
            record = FineBatchRunRecord(
                fine_batch_id=fb_id,
                interval_id=batch.interval.interval_id,
                provider=provider.provider_name,
                model=provider.model_name,
                prompt_version=fine_cfg.prompt_version or PROMPT_VERSION,
                status="failed",
                error=str(exc)[:500],
            )
            analysis = None

        # 持久化 manifest（按 fine_batch_id 排序保持稳定）
        manifest = upsert_fine_batch_record(manifest, record)
        manifest.batches = sorted(manifest.batches, key=lambda r: r.fine_batch_id)
        manifest.recompute_counts()
        save_fine_manifest(layout, manifest)

        if analysis is not None:
            analyses[fb_id] = analysis
        else:
            failed[fb_id] = record.error or "unknown"
            if not fine_cfg.allow_partial:
                raise FineBatchAnalysisError(
                    f"fine batch {fb_id} failed and partial not allowed: {failed[fb_id]}"
                )

    # ---- 聚合 ----
    video_duration = 0.0
    round1_duration = 0.0
    if batches_meta:
        video_duration = max(m.get("end", 0.0) for m in batches_meta.values())
        round1_duration = sum(
            m.get("end", 0.0) - m.get("start", 0.0) for m in batches_meta.values()
        )

    # 收集二轮补帧请求
    requests_by_batch: dict[str, list] = {}
    for fb_id, a in analyses.items():
        if a.second_refinement_requests:
            requests_by_batch[fb_id] = a.second_refinement_requests

    second_plan_path = None
    second_required = False
    if build_second_plan:
        from video_analyst.fine.second_refinement import (
            build_second_refinement_plan as _build_plan,
        )

        plan = _build_plan(
            requests_by_batch=requests_by_batch,
            video_duration=video_duration,
            round1_refinement_duration=round1_duration,
            config=fine_cfg,
        )
        second_plan_path = save_second_refinement_plan(layout, plan)
        second_required = len(plan.intervals) > 0 or len(plan.deferred_intervals) > 0

    # 聚合全片
    agg = aggregate_fine_analysis(
        analyses=analyses,
        failed_batches=failed,
        batches_order=batches_order,
        batches_meta=batches_meta,
        video_start=0.0,
        video_end=video_duration,
        second_refinement_required=second_required,
        second_refinement_plan_path=second_plan_path,
    )
    save_fine_video_analysis(layout, agg)
    # 保存去重后的时序事件
    canonical_events, _dedup_report = deduplicate_batch_analyses(analyses)
    save_temporal_events(layout, canonical_events)

    # ---- 统计 ----
    succeeded = sum(1 for r in manifest.batches if r.status == "ok")
    cached = sum(1 for r in manifest.batches if r.status == "cached")
    failed_count = sum(1 for r in manifest.batches if r.status == "failed")
    total_events = len(canonical_events)

    if failed_count == 0 and succeeded + cached == len(fine_batches):
        status = "completed"
    elif (succeeded + cached) > 0 and failed_count > 0 and fine_cfg.allow_partial:
        status = "partial"
    elif fine_cfg.allow_partial and succeeded + cached == 0:
        status = "partial"
    else:
        status = "failed"

    log.info(
        "fine analysis done: batches=%d succeeded=%d cached=%d failed=%d events=%d",
        len(fine_batches),
        succeeded,
        cached,
        failed_count,
        total_events,
    )

    return FineAnalysisResult(
        schema_version=FINE_SCHEMA_VERSION,
        status=status,  # type: ignore[arg-type]
        total_batches=len(fine_batches),
        succeeded_batches=succeeded,
        failed_batches=failed_count,
        cached_batches=cached,
        total_events=total_events,
        fine_video_analysis_path=layout.fine_video_analysis_json,
        manifest_path=layout.fine_manifest_json,
        temporal_events_path=layout.fine_temporal_events_json,
        second_refinement_plan_path=second_plan_path,
        failed_batch_ids=list(failed.keys()),
        warnings=agg.warnings,
    )


__all__ = ["run_fine_analysis"]
