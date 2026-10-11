#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video-analyst CLI dispatcher (SKILL.md CLI summary — all commands).

Commands:
  prepare / refine-prepare / terminal-prepare / finalize / validate-coarse /
  validate-fine / validate-terminal / inspect-prepare / import-url / status /
  doctor / clean / run  (+ setup-key for the key protocol).

Key discipline: this entry never prints a key value; config writes go through
video_analyst.config.write_key (chmod 600, verified).
"""

import argparse
import json
import sys


def _add_video_output(p, video=True):
    if video:
        p.add_argument("video")
    p.add_argument("-o", "--output", required=True, help="OUTPUT directory")


def _err(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def _dump(obj, as_json=False):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="video-analyst",
                                     description="Agent-assisted local video analysis "
                                                 "(stdlib + ffmpeg/yt-dlp; no vision API)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Stage 1: precheck/ASR/coarse frames/scene/near-dup/contact sheets")
    _add_video_output(p)

    p = sub.add_parser("refine-prepare", help="Stage 3: extract detail frames per refinement plan")
    _add_video_output(p)

    p = sub.add_parser("terminal-prepare", help="Optional terminal refinement (once only)")
    _add_video_output(p)

    p = sub.add_parser("finalize", help="Stage 5: deterministic aggregation -> md/json/csv")
    p.add_argument("output")

    p = sub.add_parser("status", help="Read state.json -> 12-state position and progress")
    p.add_argument("output")

    for name, stage in (("validate-coarse", "coarse"), ("validate-fine", "fine"),
                        ("validate-terminal", "terminal")):
        p = sub.add_parser(name)
        p.add_argument("output")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("inspect-prepare", help="Deterministic absolute-time inspection inputs")
    p.add_argument("video")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--at", action="append", metavar="TIME",
                   help="pinned timestamps (seconds / MM:SS / HH:MM:SS); repeatable")
    p.add_argument("--start", help="inspection range start")
    p.add_argument("--end", help="inspection range end")
    p.add_argument("--fps", type=int, default=5, choices=[5, 10, 15])

    p = sub.add_parser("import-url", help="Explicit one-public-URL yt-dlp import (with source manifest)")
    p.add_argument("url")
    p.add_argument("-o", "--output", required=True, dest="out_dir")
    p.add_argument("--subs", default="all", help="subtitle langs e.g. zh.*,ja.*,en.* (default all)")
    p.add_argument("--dry-run", action="store_true", help="print command, download nothing")

    p = sub.add_parser("doctor", help="Environment self-check (ffmpeg/ffprobe/yt-dlp/PIL/ASR/key)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("clean", help="Remove ALL generated artifacts (state + user inputs preserved)")
    p.add_argument("output")
    p.add_argument("--all-generated", action="store_true", required=True,
                   help="confirm removal of generated artifacts")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("run", help="Local stages in sequence; pauses at the agent-vision node")
    _add_video_output(p)

    p = sub.add_parser("setup-key", help="Write a provider key into ~/.config/video-analyst/config.json (600)")
    p.add_argument("provider", nargs="?", default="groq")
    p.add_argument("key", nargs="?", help="key value (--stdin instead for safety)")
    p.add_argument("--stdin", action="store_true", help="read the key from stdin instead of argv")

    args = parser.parse_args(argv)
    cmd = args.command

    try:
        if cmd == "prepare":
            from .prepare import run_prepare
            res = run_prepare(args.video, args.output)
            _dump(res)
            print("==> PAUSE for current Agent: open analysis/coarse/inputs/ frames, "
                  "contact sheets, manifest and subtitles/subtitles.json, then write "
                  "CoarseBatchAnalysis / VideoStructure / RefinementPlan and run "
                  "`video-analyst validate-coarse OUTPUT`.", file=sys.stderr)
        elif cmd == "refine-prepare":
            from .refine import run_refine
            _dump(run_refine(args.video, args.output))
        elif cmd == "terminal-prepare":
            from .terminal import run_terminal
            _dump(run_terminal(args.video, args.output))
        elif cmd == "finalize":
            from .finalize import run_finalize
            _dump(run_finalize(args.output))
        elif cmd == "status":
            from .status import build_status, render_status
            print(render_status(build_status(args.output)))
        elif cmd in ("validate-coarse", "validate-fine", "validate-terminal"):
            from .validate import run_validate
            stage = cmd.split("-")[1]
            res = run_validate(args.output, stage)
            if not res["ok"]:
                print(f"VALIDATE-{stage.upper()} FAIL")
                for e in res["errors"]:
                    print("  - " + e)
                return 1
            print(f"VALIDATE-{stage.upper()} PASS  (state={res['state']})")
        elif cmd == "inspect-prepare":
            from .inspect import run_inspect
            _dump(run_inspect(args.video, args.output, at_times=args.at,
                              start=args.start, end=args.end, fps=args.fps))
        elif cmd == "import-url":
            from .import_url import run_import
            _dump(run_import(args.url, args.out_dir, subs=args.subs, dry_run=args.dry_run),
                  as_json=True)
        elif cmd == "doctor":
            from .doctor import collect, render_doctor
            d = collect()
            if args.json:
                _dump(d)
            else:
                print(render_doctor(d))
        elif cmd == "clean":
            from .clean import run_clean
            res = run_clean(args.output)
            if args.json:
                _dump(res)
            else:
                print("cleaned:" + (" none" if not res["removed"] else ""))
                for r in res["removed"]:
                    print("  - " + r)
                print(f"state.json preserved: {res['state_preserved']}")
        elif cmd == "run":
            from .prepare import run_prepare
            from . import state as state_mod
            res = run_prepare(args.video, args.output)
            _dump(res)
            print()
            print("==> PAUSE for current Agent: open analysis/coarse/inputs/ frames, "
                  "contact sheets, manifest and subtitles/subtitles.json, then write "
                  "CoarseBatchAnalysis / VideoStructure / RefinementPlan and run "
                  "`video-analyst validate-coarse OUTPUT`.")
        elif cmd == "setup-key":
            from .config import write_key
            if args.stdin:
                key = sys.stdin.readline().strip()
            elif args.key:
                key = args.key.strip()
            else:
                key = None
            if not key:
                print("no key provided — use: video-analyst setup-key groq <KEY> "
                      "or video-analyst setup-key groq --stdin", file=sys.stderr)
                return 2
            path = write_key(key, provider=args.provider)
            print(f"key written for {args.provider} -> {path} (chmod 600 verified)")
            print("doctor will now report groq_api_key: present (value never displayed).")
        else:
            parser.error(f"unknown command {cmd!r}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report honestly
        return _err(f"{exc}")


if __name__ == "__main__":
    sys.exit(main())