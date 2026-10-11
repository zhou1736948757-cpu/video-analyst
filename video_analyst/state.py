#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12-state deterministic state machine for video-analyst outputs.

States (SKILL.md Status machine):
  not_started | preparing | waiting_for_coarse_agent_analysis | coarse_analysis_complete |
  preparing_refinement | waiting_for_fine_agent_analysis | fine_analysis_complete |
  waiting_for_terminal_agent_analysis | terminal_analysis_complete | ready_to_finalize |
  completed | failed

Transitions are event-driven and validated against a table. Validation of an
agent-written stage is what advances the machine (validate-coarse/fine/terminal).
"""

import os

from .common import dump_json, iso_now, load_json

STATES = [
    "not_started",
    "preparing",
    "waiting_for_coarse_agent_analysis",
    "coarse_analysis_complete",
    "preparing_refinement",
    "waiting_for_fine_agent_analysis",
    "fine_analysis_complete",
    "waiting_for_terminal_agent_analysis",
    "terminal_analysis_complete",
    "ready_to_finalize",
    "completed",
    "failed",
]

INDEX = {s: i for i, s in enumerate(STATES)}


class StateError(Exception):
    pass


class RejectionError(Exception):
    """Precondition violation (e.g. terminal already done, bad plan). Does NOT mark
    the workflow as failed — the prior valid state stays intact."""


def state_path(output_dir):
    return os.path.join(output_dir, "state.json")


def load(output_dir):
    st = load_json(state_path(output_dir))
    if not st or not isinstance(st, dict) or "state" not in st:
        ensure_state(output_dir)
        st = load_json(state_path(output_dir))
    if st["state"] not in INDEX:
        st["state"] = "failed"
        save(output_dir, st)
    return st


def ensure_state(output_dir, source=None):
    path = state_path(output_dir)
    if os.path.exists(path):
        return load(output_dir)
    st = {
        "state": "not_started",
        "source": source,
        "terminal_refinement_count": 0,
        "history": [],
        "updated_at": iso_now(),
    }
    save(output_dir, st)
    return st


def save(output_dir, st):
    st["updated_at"] = iso_now()
    dump_json(state_path(output_dir), st, indent=2)
    return st


def _record(st, event, note=None):
    entry = {"event": event, "state": st["state"], "at": iso_now()}
    if note:
        entry["note"] = note
    st.setdefault("history", []).append(entry)


def _fail(output_dir, st, event, note, exc=None):
    st["state"] = "failed"
    entry = {"event": event, "state": "failed", "at": iso_now(), "note": note}
    if exc:
        entry["error"] = str(exc)[:500]
    st.setdefault("history", []).append(entry)
    save(output_dir, st)
    return st


def current(output_dir):
    return load(output_dir)["state"]


# ------------------------------------------------------------------- events
# Each event: {"from": set-of-eligible-states-or-"*", "to": target, "note": ...}
# from="*" allows any state (prepare/clean/fail are re-runnable by design).

_EVENTS = {
    "prepare_start": {"from": "*", "to": "preparing"},
    "prepare_done": {"from": {"preparing"}, "to": "waiting_for_coarse_agent_analysis"},
    "coarse_validated": {
        "from": {
            "waiting_for_coarse_agent_analysis",
            "coarse_analysis_complete",
            "preparing_refinement",
            "waiting_for_fine_agent_analysis",
            "fine_analysis_complete",
            "waiting_for_terminal_agent_analysis",
            "terminal_analysis_complete",
            "ready_to_finalize",
        },
        "to": "coarse_analysis_complete",
        "idempotent": True,
    },
    "refine_start": {"from": {"coarse_analysis_complete"}, "to": "preparing_refinement"},
    "refine_done": {"from": {"preparing_refinement"}, "to": "waiting_for_fine_agent_analysis"},
    "fine_validated": {
        "from": {"waiting_for_fine_agent_analysis", "fine_analysis_complete", "ready_to_finalize"},
        "to": "fine_analysis_complete",
        "idempotent": True,
    },
    "terminal_start": {
        "from": {"fine_analysis_complete"},
        "to": "waiting_for_terminal_agent_analysis",
    },
    "terminal_validated": {
        "from": {
            "waiting_for_terminal_agent_analysis",
            "terminal_analysis_complete",
            "ready_to_finalize",
        },
        "to": "terminal_analysis_complete",
        "idempotent": True,
    },
    "terminal_ready": {
        "from": {"fine_analysis_complete", "terminal_analysis_complete", "ready_to_finalize"},
        "to": "ready_to_finalize",
        "idempotent": True,
    },
    "finalize_done": {
        "from": {"ready_to_finalize", "terminal_analysis_complete", "fine_analysis_complete",
                 "completed"},
        "to": "completed",
    },
    "fail": {"from": "*", "to": "failed"},
}


def transition(output_dir, event, source=None, note=None):
    """Validate a state transition, mutate + persist state.json. Returns state dict."""
    st = ensure_state(output_dir, source=source)
    cur = st["state"]
    spec = _EVENTS.get(event)
    if spec is None:
        raise StateError(f"unknown transition event: {event!r}")
    eligible = spec["from"]
    if eligible != "*" and cur not in eligible:
        if spec.get("idempotent") and INDEX.get(cur, 0) >= INDEX[spec["to"]]:
            _record(st, event, note or "idempotent re-validation")
            save(output_dir, st)
            return st
        raise StateError(
            f"invalid transition {event!r} from state {cur!r} (allowed: {sorted(eligible)})"
        )
    st["state"] = spec["to"]
    if source is not None:
        st["source"] = source
    _record(st, event, note)
    save(output_dir, st)
    return st