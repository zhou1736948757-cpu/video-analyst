"""Resumable pipeline cache and hierarchical output layout."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_analyst.exceptions import CacheError
from video_analyst.models import PipelineState, SubtitleDocument, utc_now


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    audio_dir: Path
    audio_chunks_dir: Path
    asr_dir: Path
    asr_chunks_dir: Path
    asr_raw_dir: Path
    asr_groq_raw_dir: Path
    subtitles_dir: Path
    cache_dir: Path
    logs_dir: Path
    raw_dir: Path
    phase2_dir: Path
    phase2_fixed_frames_dir: Path
    phase2_scene_frames_dir: Path
    phase2_pinned_frames_dir: Path
    media_info_json: Path
    audio_mp3: Path
    asr_manifest_json: Path
    subtitles_asr_segments_json: Path
    subtitles_json: Path
    subtitles_srt: Path
    subtitles_asr_json: Path
    subtitles_asr_srt: Path
    subtitles_llm_json: Path
    subtitles_llm_srt: Path
    review_json: Path
    asr_raw_json: Path  # legacy flat cache
    pipeline_state_json: Path
    run_log: Path
    groq_response_json: Path
    llm_response_json: Path
    phase2_batches_json: Path
    phase2_mapping_json: Path
    analysis_dir: Path
    coarse_dir: Path
    coarse_batches_dir: Path
    coarse_raw_dir: Path
    coarse_manifest_json: Path
    coarse_structure_json: Path
    coarse_refinement_plan_json: Path
    # Epic 3 — refinement / fine-frame extraction outputs
    refinement_dir: Path
    refinement_frames_dir: Path
    refinement_contact_sheets_dir: Path
    refinement_detail_frames_dir: Path
    refinement_visual_inputs_dir: Path
    refinement_batches_dir: Path
    refinement_intervals_dir: Path
    refinement_manifest_json: Path
    refinement_fine_batches_json: Path
    # Epic 4 — fine temporal vision analysis outputs
    fine_dir: Path
    fine_batches_out_dir: Path
    fine_raw_dir: Path
    fine_manifest_json: Path
    fine_temporal_events_json: Path
    fine_video_analysis_json: Path
    fine_second_refinement_plan_json: Path
    # Epic 5 — terminal refinement + final report
    final_dir: Path
    final_report_dir: Path
    final_evidence_dir: Path
    final_evidence_images_dir: Path
    final_analysis_dir: Path
    final_merged_temporal_events_json: Path
    final_merged_question_answers_json: Path
    final_unresolved_items_json: Path
    final_analysis_coverage_json: Path
    final_video_analysis_json: Path
    final_report_md: Path
    final_report_json: Path
    final_shot_table_csv: Path
    final_event_table_csv: Path
    final_unresolved_table_csv: Path
    final_run_summary_json: Path
    final_evidence_index_json: Path
    # Epic 5 — terminal refinement outputs (reuses Epic 3 + Epic 4 patterns)
    terminal_refinement_dir: Path
    terminal_refinement_frames_dir: Path
    terminal_refinement_contact_sheets_dir: Path
    terminal_refinement_detail_frames_dir: Path
    terminal_refinement_intervals_dir: Path
    terminal_refinement_manifest_json: Path
    terminal_refinement_fine_batches_json: Path
    terminal_refinement_batches_dir: Path
    terminal_fine_dir: Path
    terminal_fine_batches_out_dir: Path
    terminal_fine_raw_dir: Path
    terminal_fine_manifest_json: Path
    terminal_fine_temporal_events_json: Path
    terminal_fine_analysis_json: Path

    @classmethod
    def create(cls, output_dir: Path) -> OutputLayout:
        root = output_dir.expanduser().resolve()
        audio_dir = root / "audio"
        audio_chunks_dir = audio_dir / "chunks"
        asr_dir = root / "asr"
        asr_chunks_dir = asr_dir / "chunks"
        asr_raw_dir = asr_dir / "raw"
        asr_groq_raw_dir = asr_raw_dir / "groq_responses"
        subtitles_dir = root / "subtitles"
        cache_dir = root / "cache"
        logs_dir = root / "logs"
        raw_dir = root / "raw"
        phase2_dir = root / "phase2"
        phase2_fixed = phase2_dir / "frames_1fps"
        phase2_scene = phase2_dir / "frames_scene"
        phase2_pinned = phase2_dir / "frames_pinned"
        analysis_dir = root / "analysis"
        coarse_dir = analysis_dir / "coarse"
        coarse_batches_dir = coarse_dir / "batches"
        coarse_raw_dir = coarse_dir / "raw"
        refinement_dir = analysis_dir / "refinement"
        refinement_frames_dir = refinement_dir / "frames"
        refinement_contact_sheets_dir = refinement_dir / "contact_sheets"
        refinement_detail_frames_dir = refinement_dir / "detail_frames"
        refinement_visual_inputs_dir = refinement_dir / "visual_inputs"
        refinement_batches_dir = refinement_visual_inputs_dir / "batches"
        refinement_intervals_dir = refinement_dir / "intervals"
        fine_dir = analysis_dir / "fine"
        fine_batches_out_dir = fine_dir / "batches"
        fine_raw_dir = fine_dir / "raw"
        # Epic 5 directories
        final_dir = root / "report"
        final_report_dir = final_dir
        final_evidence_dir = final_dir / "evidence"
        final_evidence_images_dir = final_evidence_dir / "images"
        final_analysis_dir = analysis_dir / "final_analysis"
        terminal_refinement_dir = analysis_dir / "terminal_refinement"
        terminal_refinement_frames_dir = terminal_refinement_dir / "frames"
        terminal_refinement_contact_sheets_dir = terminal_refinement_dir / "contact_sheets"
        terminal_refinement_detail_frames_dir = terminal_refinement_dir / "detail_frames"
        terminal_refinement_intervals_dir = terminal_refinement_dir / "intervals"
        terminal_refinement_batches_dir = terminal_refinement_dir / "visual_inputs" / "batches"
        terminal_fine_dir = analysis_dir / "terminal_fine"
        terminal_fine_batches_out_dir = terminal_fine_dir / "batches"
        terminal_fine_raw_dir = terminal_fine_dir / "raw"
        for d in (
            audio_dir,
            audio_chunks_dir,
            asr_dir,
            asr_chunks_dir,
            asr_raw_dir,
            asr_groq_raw_dir,
            subtitles_dir,
            cache_dir,
            logs_dir,
            raw_dir,
            phase2_dir,
            phase2_fixed,
            phase2_scene,
            phase2_pinned,
            analysis_dir,
            coarse_dir,
            coarse_batches_dir,
            coarse_raw_dir,
            refinement_dir,
            refinement_frames_dir,
            refinement_contact_sheets_dir,
            refinement_detail_frames_dir,
            refinement_visual_inputs_dir,
            refinement_batches_dir,
            refinement_intervals_dir,
            fine_dir,
            fine_batches_out_dir,
            fine_raw_dir,
            final_dir,
            final_evidence_dir,
            final_evidence_images_dir,
            final_analysis_dir,
            terminal_refinement_dir,
            terminal_refinement_frames_dir,
            terminal_refinement_contact_sheets_dir,
            terminal_refinement_detail_frames_dir,
            terminal_refinement_intervals_dir,
            terminal_refinement_batches_dir,
            terminal_fine_dir,
            terminal_fine_batches_out_dir,
            terminal_fine_raw_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            audio_dir=audio_dir,
            audio_chunks_dir=audio_chunks_dir,
            asr_dir=asr_dir,
            asr_chunks_dir=asr_chunks_dir,
            asr_raw_dir=asr_raw_dir,
            asr_groq_raw_dir=asr_groq_raw_dir,
            subtitles_dir=subtitles_dir,
            cache_dir=cache_dir,
            logs_dir=logs_dir,
            raw_dir=raw_dir,
            phase2_dir=phase2_dir,
            phase2_fixed_frames_dir=phase2_fixed,
            phase2_scene_frames_dir=phase2_scene,
            phase2_pinned_frames_dir=phase2_pinned,
            media_info_json=root / "media_info.json",
            audio_mp3=audio_dir / "audio.mp3",
            asr_manifest_json=asr_dir / "manifest.json",
            subtitles_asr_segments_json=subtitles_dir / "subtitles_asr_segments.json",
            subtitles_json=subtitles_dir / "subtitles.json",
            subtitles_srt=subtitles_dir / "subtitles.srt",
            subtitles_asr_json=subtitles_dir / "subtitles_asr.json",
            subtitles_asr_srt=subtitles_dir / "subtitles_asr.srt",
            subtitles_llm_json=subtitles_dir / "subtitles_llm.json",
            subtitles_llm_srt=subtitles_dir / "subtitles_llm.srt",
            review_json=subtitles_dir / "review.json",
            asr_raw_json=cache_dir / "asr_raw.json",
            pipeline_state_json=cache_dir / "pipeline_state.json",
            run_log=logs_dir / "run.log",
            groq_response_json=raw_dir / "groq_response.json",
            llm_response_json=raw_dir / "llm_calibration.json",
            phase2_batches_json=phase2_dir / "visual_batches.json",
            phase2_mapping_json=phase2_dir / "subtitle_mapping.json",
            analysis_dir=analysis_dir,
            coarse_dir=coarse_dir,
            coarse_batches_dir=coarse_batches_dir,
            coarse_raw_dir=coarse_raw_dir,
            coarse_manifest_json=coarse_dir / "manifest.json",
            coarse_structure_json=coarse_dir / "video_structure.json",
            coarse_refinement_plan_json=coarse_dir / "refinement_plan.json",
            refinement_dir=refinement_dir,
            refinement_frames_dir=refinement_frames_dir,
            refinement_contact_sheets_dir=refinement_contact_sheets_dir,
            refinement_detail_frames_dir=refinement_detail_frames_dir,
            refinement_visual_inputs_dir=refinement_visual_inputs_dir,
            refinement_batches_dir=refinement_batches_dir,
            refinement_intervals_dir=refinement_intervals_dir,
            refinement_manifest_json=refinement_dir / "manifest.json",
            refinement_fine_batches_json=refinement_visual_inputs_dir / "fine_batches.json",
            fine_dir=fine_dir,
            fine_batches_out_dir=fine_batches_out_dir,
            fine_raw_dir=fine_raw_dir,
            fine_manifest_json=fine_dir / "manifest.json",
            fine_temporal_events_json=fine_dir / "temporal_events.json",
            fine_video_analysis_json=fine_dir / "fine_video_analysis.json",
            fine_second_refinement_plan_json=fine_dir / "second_refinement_plan.json",
            # Epic 5
            final_dir=final_dir,
            final_report_dir=final_report_dir,
            final_evidence_dir=final_evidence_dir,
            final_evidence_images_dir=final_evidence_images_dir,
            final_analysis_dir=final_analysis_dir,
            final_merged_temporal_events_json=final_analysis_dir / "merged_temporal_events.json",
            final_merged_question_answers_json=final_analysis_dir / "merged_question_answers.json",
            final_unresolved_items_json=final_analysis_dir / "unresolved_items.json",
            final_analysis_coverage_json=final_analysis_dir / "analysis_coverage.json",
            final_video_analysis_json=final_analysis_dir / "final_video_analysis.json",
            final_report_md=final_dir / "report.md",
            final_report_json=final_dir / "report.json",
            final_shot_table_csv=final_dir / "shot_table.csv",
            final_event_table_csv=final_dir / "event_table.csv",
            final_unresolved_table_csv=final_dir / "unresolved_items.csv",
            final_run_summary_json=final_dir / "run_summary.json",
            final_evidence_index_json=final_evidence_dir / "index.json",
            terminal_refinement_dir=terminal_refinement_dir,
            terminal_refinement_frames_dir=terminal_refinement_frames_dir,
            terminal_refinement_contact_sheets_dir=terminal_refinement_contact_sheets_dir,
            terminal_refinement_detail_frames_dir=terminal_refinement_detail_frames_dir,
            terminal_refinement_intervals_dir=terminal_refinement_intervals_dir,
            terminal_refinement_manifest_json=terminal_refinement_dir / "manifest.json",
            terminal_refinement_fine_batches_json=terminal_refinement_dir / "visual_inputs" / "fine_batches.json",
            terminal_refinement_batches_dir=terminal_refinement_batches_dir,
            terminal_fine_dir=terminal_fine_dir,
            terminal_fine_batches_out_dir=terminal_fine_batches_out_dir,
            terminal_fine_raw_dir=terminal_fine_raw_dir,
            terminal_fine_manifest_json=terminal_fine_dir / "manifest.json",
            terminal_fine_temporal_events_json=terminal_fine_dir / "temporal_events.json",
            terminal_fine_analysis_json=terminal_fine_dir / "terminal_fine_analysis.json",
        )


class CacheStore:
    def __init__(self, layout: OutputLayout) -> None:
        self.layout = layout

    def load_state(self, video_path: Path) -> PipelineState:
        path = self.layout.pipeline_state_json
        if not path.is_file():
            return PipelineState(
                video_path=str(video_path.resolve()),
                output_dir=str(self.layout.root),
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # tolerate legacy stage names
            state = PipelineState.model_validate(data)
            if Path(state.video_path).resolve() != video_path.resolve():
                return PipelineState(
                    video_path=str(video_path.resolve()),
                    output_dir=str(self.layout.root),
                )
            return state
        except Exception as exc:
            raise CacheError(f"Failed to load pipeline state: {exc}") from exc

    def save_state(self, state: PipelineState) -> None:
        state.updated_at = utc_now()
        self._write_json(self.layout.pipeline_state_json, state.model_dump(mode="json"))

    def write_json(self, path: Path, data: Any) -> None:
        self._write_json(path, data)

    def _write_json(self, path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception as exc:
            raise CacheError(f"Failed to write {path}: {exc}") from exc

    def read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CacheError(f"Failed to read {path}: {exc}") from exc

    def save_subtitle_document(self, doc: SubtitleDocument) -> None:
        self._write_json(self.layout.subtitles_json, doc.model_dump(mode="json"))

    def clear_asr_tree(self) -> None:
        """Remove ASR hierarchy + subtitles + phase2 subtitle-derived data."""
        layout = self.layout
        for p in (
            layout.asr_manifest_json,
            layout.subtitles_asr_segments_json,
            layout.subtitles_json,
            layout.subtitles_srt,
            layout.subtitles_asr_json,
            layout.subtitles_asr_srt,
            layout.subtitles_llm_json,
            layout.subtitles_llm_srt,
            layout.review_json,
            layout.asr_raw_json,
            layout.groq_response_json,
            layout.llm_response_json,
            layout.phase2_batches_json,
            layout.phase2_mapping_json,
        ):
            if p.is_file():
                p.unlink()
        for d in (
            layout.asr_chunks_dir,
            layout.asr_groq_raw_dir,
            layout.audio_chunks_dir,
        ):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)

    def clear_phase2_subtitle_derived(self) -> None:
        """Invalidate mapping/batches after active subtitle switch (keep frames)."""
        for p in (self.layout.phase2_batches_json, self.layout.phase2_mapping_json):
            if p.is_file():
                p.unlink()

    def clear_epic2_tree(self) -> None:
        """Remove all Epic 2 vision/coarse analysis outputs."""
        layout = self.layout
        if layout.coarse_dir.is_dir():
            shutil.rmtree(layout.coarse_dir, ignore_errors=True)
        # Recreate the parent dir for next run
        layout.analysis_dir.mkdir(parents=True, exist_ok=True)

    def clear_epic3_tree(self) -> None:
        """Remove all Epic 3 refinement outputs (preserve Epic 2 coarse tree)."""
        layout = self.layout
        if layout.refinement_dir.is_dir():
            shutil.rmtree(layout.refinement_dir, ignore_errors=True)
        # Recreate the parent dir for next run
        layout.analysis_dir.mkdir(parents=True, exist_ok=True)

    def clear_epic3_interval(self, interval_id: str) -> None:
        """Remove one interval's Epic 3 artifacts (frames, contact_sheets,
        detail_frames, intervals/*.json, and any fine_batch entry). Other intervals stay."""
        layout = self.layout
        # Frames
        fr_dir = layout.refinement_frames_dir / interval_id
        if fr_dir.is_dir():
            shutil.rmtree(fr_dir, ignore_errors=True)
        cs_dir = layout.refinement_contact_sheets_dir / interval_id
        if cs_dir.is_dir():
            shutil.rmtree(cs_dir, ignore_errors=True)
        df_dir = layout.refinement_detail_frames_dir / interval_id
        if df_dir.is_dir():
            shutil.rmtree(df_dir, ignore_errors=True)
        # Interval manifest
        iv_path = layout.refinement_intervals_dir / f"{interval_id}.json"
        if iv_path.is_file():
            iv_path.unlink()
        # Fine batch file
        fb_path = layout.refinement_batches_dir / f"fine_{interval_id}.json"
        if fb_path.is_file():
            fb_path.unlink()

    def clear_epic4_tree(self) -> None:
        """Remove all Epic 4 fine analysis outputs (preserve Epic 3 outputs)."""
        layout = self.layout
        if layout.fine_dir.is_dir():
            shutil.rmtree(layout.fine_dir, ignore_errors=True)
        # Recreate for next run
        layout.fine_dir.mkdir(parents=True, exist_ok=True)
        layout.fine_batches_out_dir.mkdir(parents=True, exist_ok=True)
        layout.fine_raw_dir.mkdir(parents=True, exist_ok=True)

    def clear_epic4_batch(self, fine_batch_id: str) -> None:
        """Remove one fine batch's analysis + raw response. Other batches stay."""
        layout = self.layout
        b_path = layout.fine_batches_out_dir / f"{fine_batch_id}.json"
        if b_path.is_file():
            b_path.unlink()
        r_path = layout.fine_raw_dir / f"{fine_batch_id}_response.json"
        if r_path.is_file():
            r_path.unlink()

    def clear_epic5_tree(self) -> None:
        """Remove all Epic 5 outputs (preserve Epic 1–4 results)."""
        layout = self.layout
        for p in (
            layout.final_dir,
            layout.final_analysis_dir,
            layout.terminal_refinement_dir,
            layout.terminal_fine_dir,
        ):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        # Recreate directories
        for p in (
            layout.final_dir,
            layout.final_evidence_dir,
            layout.final_evidence_images_dir,
            layout.final_analysis_dir,
            layout.terminal_refinement_dir,
            layout.terminal_refinement_frames_dir,
            layout.terminal_refinement_contact_sheets_dir,
            layout.terminal_refinement_detail_frames_dir,
            layout.terminal_refinement_intervals_dir,
            layout.terminal_refinement_batches_dir,
            layout.terminal_fine_dir,
            layout.terminal_fine_batches_out_dir,
            layout.terminal_fine_raw_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
