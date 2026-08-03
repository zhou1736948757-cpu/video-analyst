"""CLI entrypoint for the in-session agent video analysis workflow.

Primary product commands:
  prepare → (agent coarse) → refine-prepare → (agent fine)
  → terminal-prepare? → (agent terminal?) → finalize

Scripts never perform visual inference. Vision providers / API keys are
not part of the product path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape

from video_analyst import __version__
from video_analyst.env.detect import collect_environment_report, get_os_family
from video_analyst.env.setup_ffmpeg import (
    ensure_tools_directory,
    install_ffmpeg,
    recommended_install_commands,
)
from video_analyst.exceptions import (
    EnvironmentBlockedError,
    FFmpegInstallError,
    FFmpegNotFoundError,
    VideoAnalystError,
)
from video_analyst.models import SubtitleVersion

app = typer.Typer(
    name="video-analyst",
    help=(
        "Agent-assisted local video analysis. "
        "Local scripts extract media; the current conversation Agent inspects frames."
    ),
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if not value:
        return
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        v = _pkg_version("video-analyst")
    except PackageNotFoundError:
        v = __version__
    typer.echo(v)
    raise typer.Exit()


@app.callback()
def _main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit (reads from package metadata).",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """Video Analyst — agent-assisted local workflow."""


def _print_doctor() -> int:
    report = collect_environment_report()

    if report.python_ok:
        console.print(f"Python: OK — {report.python_version}")
    else:
        console.print(f"Python: FAIL — {report.python_version} (need 3.11+)")

    if report.ffmpeg:
        console.print(f"FFmpeg: OK — {report.ffmpeg.ffmpeg_path}")
        console.print(f"ffprobe: OK — {report.ffmpeg.ffprobe_path}")
        console.print(f"FFmpeg version: {report.ffmpeg.ffmpeg_version}")
        console.print(f"ffprobe version: {report.ffmpeg.ffprobe_version}")
    else:
        console.print("FFmpeg: missing")
        console.print("ffprobe: missing")

    if report.groq_api_key_configured:
        console.print("Groq API Key: configured (ASR / subtitle calibration only)")
    else:
        console.print("Groq API Key: not configured (optional; local Whisper may work)")

    if report.local_whisper_installed:
        console.print("Local Whisper Small: installed")
    else:
        console.print("Local Whisper Small: not installed")

    console.print("Vision API Key: not required (in_session_agent mode)")
    console.print("Vision execution mode: in_session_agent")
    console.print(f"OS: {report.os_family}")
    console.print(f"Tools directory: {report.tools_bin}")

    if report.ready:
        console.print("Environment status: ready")
        return 0

    console.print("Environment status: blocked")
    if not report.python_ok:
        console.print("\nInstall Python 3.11 or higher, then re-run doctor.")
    if report.ffmpeg is None:
        plan = recommended_install_commands()
        console.print(f"\nRecommended action for {plan.os_family}:")
        if plan.commands:
            for cmd in plan.commands:
                console.print("  " + " ".join(cmd))
        for note in plan.notes:
            console.print(f"  {note}")
        console.print("\nOr run: video-analyst setup")
    return 1


@app.command("doctor")
def doctor_cmd(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON (no secrets).",
    ),
) -> None:
    """Detect Python, FFmpeg, ffprobe, ASR backends. Vision API not required."""
    if json_output:
        import json as _json

        report = collect_environment_report()
        payload = {
            "python_ok": report.python_ok,
            "python_version": report.python_version,
            "ffmpeg": report.ffmpeg is not None,
            "ffmpeg_path": str(report.ffmpeg.ffmpeg_path) if report.ffmpeg else None,
            "ffprobe_path": str(report.ffmpeg.ffprobe_path) if report.ffmpeg else None,
            "groq_api_key_configured": report.groq_api_key_configured,
            "local_whisper_installed": report.local_whisper_installed,
            "vision_api_key_required": False,
            "vision_execution_mode": "in_session_agent",
            "os_family": report.os_family,
            "ready": report.ready,
            "version": __version__,
        }
        console.print(_json.dumps(payload, ensure_ascii=False, indent=2))
        raise typer.Exit(0 if report.ready else 1)
    code = _print_doctor()
    raise typer.Exit(code)


@app.command("setup")
def setup_cmd(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip interactive confirmation (still prints planned commands).",
    ),
) -> None:
    """Detect OS and FFmpeg; install FFmpeg if missing after user confirmation."""
    console.print(f"video-analyst setup v{__version__}")
    console.print(f"OS family: {get_os_family()}")
    tools = ensure_tools_directory()
    console.print(f"Shared tools directory: {tools}")

    try:
        env = install_ffmpeg(confirm=True, assume_yes=yes)
    except FFmpegInstallError as exc:
        console.print(f"[red]Setup failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except FFmpegNotFoundError as exc:
        console.print(f"[red]FFmpeg still missing:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(f"[green]FFmpeg OK[/green] — {env.ffmpeg_path}")
    console.print(f"[green]ffprobe OK[/green] — {env.ffprobe_path}")
    console.print("")
    code = _print_doctor()
    raise typer.Exit(code)


@app.command("clean")
def clean_cmd(
    output: Path = typer.Argument(..., help="Output directory to clean."),
    analysis_only: bool = typer.Option(False, "--analysis-only", help="Remove analysis/ only."),
    cache_only: bool = typer.Option(False, "--cache-only", help="Remove cache/ only."),
    all_generated: bool = typer.Option(
        False, "--all-generated", help="Remove all generated outputs (keep source video)."
    ),
) -> None:
    """Clean generated outputs. Never deletes source video."""
    import shutil

    out = output.expanduser().resolve()
    if not out.is_dir():
        console.print(f"[red]Not a directory:[/red] {out}")
        raise typer.Exit(1)
    removed: list[str] = []
    if all_generated:
        for sub in (
            "analysis",
            "report",
            "phase2",
            "audio",
            "asr",
            "subtitles",
            "cache",
            "logs",
            "raw",
            "telemetry",
            "metadata",
        ):
            p = out / sub
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p))
    elif analysis_only:
        p = out / "analysis"
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
    elif cache_only:
        p = out / "cache"
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
    else:
        console.print("Specify --analysis-only / --cache-only / --all-generated")
        raise typer.Exit(2)
    for r in removed:
        console.print(f"[green]Removed:[/green] {r}")
    console.print("[green]Clean complete. Source video preserved.[/green]")


@app.command("prepare")
def prepare_cmd(
    video: Path = typer.Argument(..., help="Local video file path"),
    output: Path = typer.Option(
        Path("./video_analyst_output"),
        "--output",
        "-o",
        help="Output directory",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Optional single-language lock for ASR (e.g. zh, ja, en).",
    ),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore cache and re-run all stages"),
    refresh_asr: bool = typer.Option(
        False, "--refresh-asr", help="Keep media/audio but re-run ASR + subtitles"
    ),
    multilingual: bool = typer.Option(
        True, "--multilingual/--no-multilingual", help="Chunked multilingual ASR (default on)."
    ),
    llm_calibrate: bool = typer.Option(
        True,
        "--llm-calibrate/--no-llm-calibrate",
        help="After ASR, run LLM calibration (default on). Needs GROQ_API_KEY.",
    ),
    user_request: str = typer.Option(
        "",
        "--user-request",
        help="Optional analysis request stored into batch contracts.",
    ),
) -> None:
    """Stage 1: local prepare (media + ASR + coarse frames). No vision API."""
    from video_analyst.workflow.prepare import run_prepare

    try:
        result = run_prepare(
            video_path=video,
            output_dir=output,
            language=language,
            resume=not no_resume,
            refresh_asr=refresh_asr,
            multilingual=multilingual,
            llm_calibrate=llm_calibrate,
            user_request=user_request,
        )
    except EnvironmentBlockedError as exc:
        console.print(f"[red]Environment blocked:[/red] {escape(str(exc))}")
        console.print("Run: video-analyst doctor")
        raise typer.Exit(2) from exc
    except VideoAnalystError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    out = Path(result["output"])
    console.print(f"[green]STATUS: {result['status']}[/green]")
    console.print(result["message"])
    console.print(f"Manifest: {result['manifest_path']}")
    console.print(f"Coarse frames: {out / 'analysis' / 'coarse' / 'inputs' / 'frames'}")
    console.print(
        f"Contact sheets: {out / 'analysis' / 'coarse' / 'inputs' / 'contact_sheets'}"
    )
    console.print(
        "\nNext: current Agent inspects images and writes:\n"
        f"  {out / 'analysis' / 'coarse' / 'batches'}/<batch_id>.json\n"
        f"  {out / 'analysis' / 'coarse' / 'video_structure.json'}\n"
        f"  {out / 'analysis' / 'coarse' / 'refinement_plan.json'}\n"
        f"Then: video-analyst validate-coarse {out}"
    )


@app.command("analyze")
def analyze_cmd(
    video: Path = typer.Argument(..., exists=False, help="Local video file path"),
    output: Path = typer.Option(
        Path("./video_analyst_output"),
        "--output",
        "-o",
        help="Output directory for media info, audio, subtitles, cache, logs",
    ),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    no_resume: bool = typer.Option(False, "--no-resume"),
    refresh_asr: bool = typer.Option(False, "--refresh-asr"),
    multilingual: bool = typer.Option(True, "--multilingual/--no-multilingual"),
    llm_calibrate: bool = typer.Option(True, "--llm-calibrate/--no-llm-calibrate"),
    phase2_prep: bool = typer.Option(True, "--phase2-prep/--no-phase2-prep"),
    user_request: str = typer.Option("", "--user-request"),
) -> None:
    """ASR + optional Phase 2 prep only (legacy alias). Prefer `prepare` for full Stage 1."""
    from video_analyst.pipeline import run_phase1

    try:
        doc = run_phase1(
            video_path=video,
            output_dir=output,
            language=language,
            resume=not no_resume,
            refresh_asr=refresh_asr,
            multilingual=multilingual,
            llm_calibrate=llm_calibrate,
            run_phase2_prep=phase2_prep,
            user_request=user_request,
        )
    except EnvironmentBlockedError as exc:
        console.print(f"[red]Environment blocked:[/red] {escape(str(exc))}")
        console.print("Run: video-analyst doctor")
        raise typer.Exit(2) from exc
    except VideoAnalystError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    out = output.expanduser().resolve()
    sub = out / "subtitles"
    console.print("[green]Phase 1 complete[/green]")
    console.print(f"Engine: {doc.engine.value if doc.engine else 'unknown'}")
    console.print(f"Language: {doc.language or 'unknown'}")
    console.print(f"Segments: {len(doc.segments)}")
    console.print(f"Active JSON: {sub / 'subtitles.json'}")
    console.print(
        "Tip: use `video-analyst prepare` to also package coarse agent inputs."
    )


@app.command("refine-prepare")
def refine_prepare_cmd(
    video: Path = typer.Argument(..., help="Source video path"),
    output: Path = typer.Option(
        Path("./video_analyst_output"), "--output", "-o", help="Output directory"
    ),
    user_request: str = typer.Option("", "--user-request"),
    refresh_refinement: bool = typer.Option(False, "--refresh-refinement"),
    refresh_interval: list[str] = typer.Option([], "--refresh-interval"),
    frames_per_sheet: int = typer.Option(4, "--frames-per-sheet"),
    sheet_overlap: int = typer.Option(1, "--sheet-overlap"),
    sheet_columns: int = typer.Option(2, "--contact-sheet-columns"),
    max_frame_long_edge: int = typer.Option(1600, "--max-frame-long-edge"),
    contact_sheet_max_edge: int = typer.Option(4096, "--contact-sheet-max-edge"),
    image_format: str = typer.Option("jpg", "--image-format"),
    jpeg_quality: int = typer.Option(2, "--jpeg-quality"),
    no_contact_sheets: bool = typer.Option(False, "--no-contact-sheets/--with-contact-sheets"),
    allow_partial: bool = typer.Option(False, "--allow-partial/--no-allow-partial"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Stage 3: local fine-frame extraction from refinement_plan.json (no vision API)."""
    from video_analyst.cache.store import OutputLayout
    from video_analyst.config import AppConfig
    from video_analyst.exceptions import (
        Epic3PipelineError,
        RefinementPlanLoadError,
        RefinementPlanValidationError,
    )
    from video_analyst.logging.setup import setup_logging
    from video_analyst.pipeline.epic3 import run_epic3
    from video_analyst.workflow.state import detect_workflow_status, save_workflow_state

    cfg = AppConfig()
    cfg.refinement.frames_per_sheet = int(frames_per_sheet)
    cfg.refinement.sheet_overlap_frames = int(sheet_overlap)
    cfg.refinement.sheet_columns = int(sheet_columns)
    cfg.refinement.max_frame_long_edge = int(max_frame_long_edge)
    cfg.refinement.contact_sheet_max_edge = int(contact_sheet_max_edge)
    cfg.refinement.image_format = str(image_format)
    cfg.refinement.jpeg_quality = int(jpeg_quality)
    cfg.refinement.no_contact_sheets = bool(no_contact_sheets)
    cfg.refinement.allow_partial = bool(allow_partial)

    layout = OutputLayout.create(output)
    setup_logging(layout.run_log, level=10 if verbose else 20)

    if not layout.coarse_refinement_plan_json.is_file():
        console.print(
            f"[red]No refinement_plan.json at {layout.coarse_refinement_plan_json}[/red]\n"
            "Agent must write coarse analysis + refinement plan first."
        )
        raise typer.Exit(1)

    console.print(f"video-analyst refine-prepare v{__version__}")
    console.print(f"Output: {layout.root}")

    try:
        result = run_epic3(
            video_path=video,
            output_dir=output,
            config=cfg,
            user_request=user_request,
            refresh=refresh_refinement,
            refresh_intervals=list(refresh_interval) if refresh_interval else None,
            allow_partial=cfg.refinement.allow_partial,
        )
    except (RefinementPlanLoadError, RefinementPlanValidationError, Epic3PipelineError) as exc:
        console.print(f"[red]refine-prepare failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except VideoAnalystError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    state = detect_workflow_status(layout.root)
    state.set_status(
        "waiting_for_fine_agent_analysis",
        message="Refinement preparation complete",
    )
    save_workflow_state(layout.root, state)

    console.print(f"[green]STATUS: waiting_for_fine_agent_analysis[/green]")
    console.print(
        "Refinement preparation complete.\n"
        "The current Agent must now inspect fine contact sheets and detail frames."
    )
    console.print(f"Total intervals: {result.total_intervals}")
    console.print(f"Succeeded:       {result.succeeded_intervals}")
    console.print(f"Total frames:    {result.total_frames}")
    console.print(f"Total sheets:    {result.total_contact_sheets}")
    console.print(f"Fine batches:    {result.fine_batches_path}")
    console.print(f"Contact sheets:  {layout.refinement_contact_sheets_dir}")
    console.print(f"Detail frames:   {layout.refinement_detail_frames_dir}")
    console.print(
        f"\nNext: Agent writes analysis/fine/ then:\n"
        f"  video-analyst validate-fine {layout.root}"
    )


@app.command("terminal-prepare")
def terminal_prepare_cmd(
    video: Path = typer.Argument(..., help="Source video path"),
    output: Path = typer.Option(
        Path("./video_analyst_output"), "--output", "-o", help="Output directory"
    ),
    user_request: str = typer.Option("", "--user-request"),
) -> None:
    """Optional terminal refinement frame extraction (once only). No vision API."""
    from video_analyst.workflow.terminal_prepare import run_terminal_prepare

    try:
        result = run_terminal_prepare(
            video_path=video,
            output_dir=output,
            user_request=user_request,
        )
    except VideoAnalystError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(f"[green]STATUS: {result['status']}[/green]")
    console.print(result["message"])
    if result.get("manifest_path"):
        console.print(f"Manifest: {result['manifest_path']}")


@app.command("finalize")
def finalize_cmd(
    video: Path = typer.Argument(..., help="Source video path"),
    output: Path = typer.Option(
        Path("./video_analyst_output"), "--output", "-o", help="Output directory"
    ),
    user_request: str = typer.Option("", "--user-request"),
    allow_partial: bool = typer.Option(True, "--allow-partial/--no-allow-partial"),
    no_markdown: bool = typer.Option(False, "--no-markdown"),
    no_csv: bool = typer.Option(False, "--no-csv"),
    no_evidence_copy: bool = typer.Option(False, "--no-evidence-copy"),
) -> None:
    """Stage 5: deterministic aggregation + report (no vision inference)."""
    from video_analyst.workflow.finalize import run_finalize

    try:
        result = run_finalize(
            video_path=video,
            output_dir=output,
            user_request=user_request,
            allow_partial=allow_partial,
            no_markdown=no_markdown,
            no_csv=no_csv,
            no_evidence_copy=no_evidence_copy,
        )
    except VideoAnalystError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(f"[green]STATUS: completed (report={result.status})[/green]")
    console.print("vision_execution_mode=in_session_agent")
    console.print("external_api_used=false  api_key_used=false")
    if result.report_md_path:
        console.print(f"Markdown: {result.report_md_path}")
    if result.report_json_path:
        console.print(f"JSON:     {result.report_json_path}")
    if result.shot_table_path:
        console.print(f"Shots CSV: {result.shot_table_path}")


@app.command("report")
def report_cmd(
    video: Path = typer.Argument(..., help="Source video path"),
    output: Path = typer.Option(Path("./video_analyst_output"), "--output", "-o"),
    user_request: str = typer.Option("", "--user-request"),
    allow_partial: bool = typer.Option(True, "--allow-partial/--no-allow-partial"),
    no_markdown: bool = typer.Option(False, "--no-markdown"),
    no_csv: bool = typer.Option(False, "--no-csv"),
    no_evidence_copy: bool = typer.Option(False, "--no-evidence-copy"),
) -> None:
    """Alias for finalize (Stage 5)."""
    finalize_cmd(
        video=video,
        output=output,
        user_request=user_request,
        allow_partial=allow_partial,
        no_markdown=no_markdown,
        no_csv=no_csv,
        no_evidence_copy=no_evidence_copy,
    )


@app.command("status")
def status_cmd(
    output: Path = typer.Argument(..., help="Output directory"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show pipeline stage, present/missing files, and next agent action."""
    from video_analyst.workflow.state import (
        detect_workflow_status,
        format_status_report,
        save_workflow_state,
    )

    state = detect_workflow_status(output)
    save_workflow_state(output, state)
    if json_output:
        import json as _json

        console.print(
            _json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2)
        )
    else:
        console.print(format_status_report(state))


@app.command("validate-coarse")
def validate_coarse_cmd(
    output: Path = typer.Argument(..., help="Output directory"),
) -> None:
    """Validate agent-written coarse analysis JSON (schema + evidence)."""
    from video_analyst.workflow.validate_agent import (
        format_validation_result,
        validate_coarse,
    )

    result = validate_coarse(output)
    console.print(format_validation_result(result))
    raise typer.Exit(0 if result.ok else 1)


@app.command("validate-fine")
def validate_fine_cmd(
    output: Path = typer.Argument(..., help="Output directory"),
) -> None:
    """Validate agent-written fine analysis JSON (schema + evidence + questions)."""
    from video_analyst.workflow.validate_agent import (
        format_validation_result,
        validate_fine,
    )

    result = validate_fine(output)
    console.print(format_validation_result(result))
    raise typer.Exit(0 if result.ok else 1)


@app.command("validate-terminal")
def validate_terminal_cmd(
    output: Path = typer.Argument(..., help="Output directory"),
) -> None:
    """Validate agent-written terminal analysis JSON."""
    from video_analyst.workflow.validate_agent import (
        format_validation_result,
        validate_terminal,
    )

    result = validate_terminal(output)
    console.print(format_validation_result(result))
    raise typer.Exit(0 if result.ok else 1)


@app.command("run")
def run_cmd(
    video: Path = typer.Argument(..., help="Source video path"),
    output: Path = typer.Option(Path("./video_analyst_output"), "--output", "-o"),
    user_request: str = typer.Option("", "--user-request"),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    allow_partial: bool = typer.Option(True, "--allow-partial/--no-allow-partial"),
) -> None:
    """Advance local stages only; pause when the Agent must inspect images.

    Never calls mock or external vision APIs to fill analysis results.
    """
    from video_analyst.workflow.run import run_until_agent_needed

    try:
        result = run_until_agent_needed(
            video_path=video,
            output_dir=output,
            user_request=user_request,
            language=language,
            resume=resume,
            allow_partial=allow_partial,
        )
    except VideoAnalystError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(f"[bold]STATUS: {result['status']}[/bold]")
    console.print(result.get("message") or "")
    if result.get("actions"):
        console.print(f"Actions run: {', '.join(result['actions'])}")
    if result.get("report"):
        console.print("")
        console.print(result["report"])
    # Exit 0 even when waiting for agent — not a failure
    raise typer.Exit(0)


@app.command("review")
def review_cmd(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Analyze output directory containing subtitles/review.json",
    ),
    use: Optional[str] = typer.Option(
        None,
        "--use",
        help="Set active version: asr | llm. Omit to only display the review table.",
    ),
) -> None:
    """Show ASR vs LLM subtitle diffs; optionally choose which version is active."""
    from video_analyst.cache.store import OutputLayout
    from video_analyst.llm.review import apply_active_version, format_review_table, load_review

    layout = OutputLayout.create(output)
    if not layout.review_json.is_file():
        console.print(
            f"[red]No review.json at {layout.review_json}[/red]\n"
            "Run prepare/analyze first (with LLM calibration)."
        )
        raise typer.Exit(1)

    try:
        if use is not None:
            version = SubtitleVersion(use.strip().lower())
            package = apply_active_version(layout, version, reviewed=True)
            console.print(f"[green]Active version set to:[/green] {package.active.value}")
        else:
            package = load_review(layout)
    except ValueError as exc:
        console.print(f"[red]Invalid --use value:[/red] {escape(str(exc))}")
        console.print("Use: --use asr  or  --use llm")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(format_review_table(package))
    console.print("")
    console.print(f"Active files: {layout.subtitles_srt}")
    console.print(f"ASR files:    {layout.subtitles_asr_srt}")
    console.print(f"LLM files:    {layout.subtitles_llm_srt}")


# ---- Removed product modes (explicit rejection) ----


@app.command("provider-check", hidden=True)
def provider_check_cmd() -> None:
    """Removed: vision provider check is not part of the product."""
    console.print(
        "[red]provider-check removed.[/red]\n"
        "Video Analyst no longer uses external vision providers.\n"
        "Vision execution mode: in_session_agent (current conversation Agent inspects frames)."
    )
    raise typer.Exit(2)


@app.command("coarse-analyze", hidden=True)
def coarse_analyze_cmd() -> None:
    """Removed: vision providers are not product runtime modes."""
    console.print(
        "[red]coarse-analyze (vision provider path) removed.[/red]\n"
        "The corresponding runtime mode was intentionally removed from the product.\n"
        "Workflow:\n"
        "  1. video-analyst prepare VIDEO -o OUT\n"
        "  2. Current Agent inspects coarse images and writes analysis/coarse/\n"
        "  3. video-analyst validate-coarse OUT"
    )
    raise typer.Exit(2)


@app.command("fine-analyze", hidden=True)
def fine_analyze_cmd() -> None:
    """Removed: vision providers are not product runtime modes."""
    console.print(
        "[red]fine-analyze (vision provider path) removed.[/red]\n"
        "The corresponding runtime mode was intentionally removed from the product.\n"
        "Workflow:\n"
        "  1. video-analyst refine-prepare VIDEO -o OUT\n"
        "  2. Current Agent inspects fine frames and writes analysis/fine/\n"
        "  3. video-analyst validate-fine OUT"
    )
    raise typer.Exit(2)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    app()


if __name__ == "__main__":
    main()
