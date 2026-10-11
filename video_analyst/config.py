#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Groq API key handling for the ASR path.

Security contract (AC-4 / task VAF-002):
  * key value is only ever read in memory; never printed, logged, or written to
    analysis JSON / reports / state.json / subtitles.
  * source of truth priority: env GROQ_API_KEY, then ~/.config/video-analyst/config.json.
  * config file is written with chmod 600 and its permission is verified.
"""

import os
import stat

from .common import CONFIG_DIR, CONFIG_PATH


def load_config():
    from .common import load_json

    return load_json(CONFIG_PATH, default={}) or {}


def read_key():
    """Returns the Groq key or None. Env takes priority over config."""
    env = os.environ.get("GROQ_API_KEY", "").strip()
    if env:
        return env
    cfg = load_config()
    try:
        k = (cfg or {}).get("providers", {}).get("groq", {}).get("api_key") or ""
    except AttributeError:
        k = ""
    return k.strip() or None


def key_present():
    return read_key() is not None


def config_writable():
    """True when ~/.config/video-analyst exists (or can be created, mode 700) and accepts files."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        os.chmod(CONFIG_DIR, 0o700)
        probe = os.path.join(CONFIG_DIR, ".writeprobe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def write_key(key, provider="groq"):
    """Write the key into config.json with 600 permissions. Returns config path.

    The key is never returned or printed; callers only see the path + permission check.
    """
    if not key or not isinstance(key, str) or not key.strip():
        raise ValueError("empty key")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.chmod(CONFIG_DIR, 0o700)
    cfg = load_config() or {}
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    if not isinstance(providers, dict):
        providers = {}
    provider_cfg = providers.get(provider)
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}
    provider_cfg["api_key"] = key.strip()
    providers[provider] = provider_cfg
    cfg["providers"] = providers
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json_dump(cfg, fh)
    os.chmod(CONFIG_PATH, 0o600)
    perms = stat.S_IMODE(os.stat(CONFIG_PATH).st_mode)
    if perms != 0o600:
        raise RuntimeError(f"config permission check failed: {CONFIG_PATH} mode={oct(perms)}")
    return CONFIG_PATH


def json_dump(data, fh):
    import json

    json.dump(data, fh, ensure_ascii=False, indent=2)
    fh.write("\n")


def format_key_status():
    """Presence-only display string; never includes the key value."""
    if key_present():
        return "present"
    return "MISSING"


def config_permission_ok():
    """True when config file (if exists) is chmod 600."""
    if not os.path.exists(CONFIG_PATH):
        return True
    perms = stat.S_IMODE(os.stat(CONFIG_PATH).st_mode)
    return (perms & 0o077) == 0 and (perms & 0o600) == 0o600 or perms == 0o600