#!/usr/bin/env python3
"""
Tiny JSON-file-backed state store tracking auto-capture progress per session.

Public API:
  get_state(session_id)                  -- {} if never seen before
  set_state(session_id, **fields)        -- merge-updates fields for a session
  all_states()                           -- full dict, session_id -> state

Tracked fields per session:
  last_finalized_line  -- how many lines of this session's transcript have
                           already been processed into jot()s. Everything
                           after this line (up to current EOF) is "new".
  last_finalized_at     -- unix timestamp of the last successful finalize pass
  cwds_seen             -- list of distinct cwd values observed in this
                           session's transcript entries (a single session can
                           span multiple working directories over its life --
                           confirmed empirically, see the 2026-07-25 session
                           that touched 5 different cwds under one session_id)

Deliberately a flat JSON file, not sqlite -- expected volume is one entry per
Claude Code session on this machine, not a high-write-rate structure. Matches
the project's existing convention of small flat state files (.recall_session_cache).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent / ".auto_capture_state.json"


def _load() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        # corrupt/unreadable state file shouldn't crash a hook -- treat as empty
        # and let it get rewritten on the next successful save.
        return {}


def _save(data: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(STATE_PATH)  # atomic on POSIX -- avoids a torn write if interrupted


def get_state(session_id: str) -> dict:
    return _load().get(session_id, {})


def set_state(session_id: str, **fields) -> None:
    data = _load()
    existing = data.get(session_id, {})
    existing.update(fields)
    existing["last_finalized_at"] = time.time() if "last_finalized_line" in fields else existing.get("last_finalized_at")
    data[session_id] = existing
    _save(data)


def all_states() -> dict:
    return _load()
