#!/usr/bin/env python3
"""
SessionEnd hook -- best-effort finalize pass. Processes everything new in
this session's transcript since its last checkpoint into jot()s. SessionEnd
is confirmed NOT to fire reliably on abnormal termination (crash/force-quit),
so this is a bonus when it does fire, not something relied on as guaranteed
-- see session_start_backstop.py for the actual safety net.

Never blocks, never raises past main(): exits 0 on any failure, logs to
../backlink_errors/ (shared error dir with the other hooks in this project).
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HOOK_DIR  = Path(__file__).resolve().parent.parent
ERROR_DIR = HOOK_DIR / "backlink_errors"


def log_error(exc):
    ERROR_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (ERROR_DIR / f"session_end_capture_{stamp}.txt").write_text(
        f"time: {stamp}\nerror: {exc!r}\n\n{traceback.format_exc()}"
    )


def run():
    # See session_start_backstop.py for the full explanation -- claude -p extraction
    # subprocesses are real Claude Code invocations subject to this same global hook.
    if os.environ.get("MEMORY_PROJECT_NESTED_EXTRACTION"):
        return

    raw = sys.stdin.read().strip()
    if not raw:
        return
    data = json.loads(raw)

    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    if not session_id or not transcript_path:
        return

    sys.path.insert(0, str(HOOK_DIR))
    from auto_capture import capture_session
    capture_session(session_id, transcript_path)


def main():
    try:
        run()
    except Exception as exc:
        log_error(exc)


if __name__ == "__main__":
    main()
