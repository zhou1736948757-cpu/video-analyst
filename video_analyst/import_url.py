#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""import-url — explicit one-public-URL yt-dlp import (SKILL.md).

* accepts exactly one public HTTP(S) URL
* invokes `yt-dlp --no-playlist --ignore-config`, no cookie arguments
* subtitle selection: `all` (default) or explicit `zh.*,ja.*,en.*` pattern
* writes source_manifest.json next to the downloaded media
* --dry-run prints the command that would run and touches nothing (offline use)

In this release the CLI is implemented and parameter-validated; the actual
download branch is only driven by an explicit user URL, never implicitly.
"""

import json
import os

from .common import ensure_dir, iso_now, run, which


def _ensure_yt_dlp():
    n, p = which("yt-dlp")
    if not n:
        raise RuntimeError("yt-dlp not found on PATH — install it (e.g. ~/.local/bin) first")
    return p


def _validate_url(url):
    import urllib.parse

    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise ValueError(f"import-url needs one public HTTP(S) URL, got {url!r}")
    return url


def build_command(url, out_dir, subs):
    """Returns argv for yt-dlp. Pure/parameterized; no side effects."""
    _validate_url(url)
    yt = _ensure_yt_dlp()
    cmd = [yt, "--no-playlist", "--ignore-config"]
    if subs and subs != "all":
        cmd += ["--write-subs", "--sub-langs", subs]
    else:
        cmd += ["--write-subs", "--sub-langs", "all"]
    cmd += ["-o", os.path.join(out_dir, "%(title)s.%(ext)s"), url]
    return cmd


def _make_source_manifest(url, out_dir, files):
    data = {
        "format": "video-analyst/source-manifest/1",
        "url": url,
        "downloaded_at": iso_now(),
        "yt_dlp_args": ["--no-playlist", "--ignore-config"],
        "cookie_used": False,
        "files": files,
    }
    path = os.path.join(out_dir, "source_manifest.json")
    ensure_dir(out_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def run_import(url, out_dir, subs="all", dry_run=False):
    """Execute import (or dry-run). Returns summary. No cookies, never implicit."""
    _validate_url(url)
    if not out_dir:
        raise ValueError("-o LOCAL_IMPORT_DIR is required")
    if subs and subs != "all":
        for pat in subs.split(","):
            if not pat.strip():
                raise ValueError("--subs must be a comma-separated pattern list like zh.*,ja.*,en.*")
    cmd = build_command(url, out_dir, subs)
    if dry_run:
        return {
            "dry_run": True,
            "url": url,
            "command": cmd,
            "note": "no files downloaded",
            "manifest": None,
        }
    ensure_dir(out_dir)
    run(cmd, timeout=3600)
    files = sorted(f for f in os.listdir(out_dir)
                   if not f.startswith(".") and f != "source_manifest.json")
    manifest = _make_source_manifest(url, out_dir, files)
    return {"dry_run": False, "url": url, "command": cmd, "files": files,
            "manifest": os.path.relpath(manifest, out_dir)}