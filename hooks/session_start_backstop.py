#!/usr/bin/env python3
"""
SessionStart hook -- the actual safety net (SessionEnd is best-effort and
confirmed not to fire reliably on abnormal termination). On a genuinely fresh
session start (source == "startup" -- not resume/clear/compact/fork of an
already-running session), scans ALL ~/.claude/projects/*/*.jsonl transcripts
for sessions that have new content since their last checkpoint and are
confirmed not still open by a live process, and catches them up via the same
capture_session() pipeline SessionEnd uses.

Liveness is checked via `lsof` (is any process holding this exact file open),
not a time-based mtime guess. An earlier version used "not modified in the
last N minutes" as a proxy for "safe to touch" -- that's wrong for the exact
scenario this hook exists to cover: an abrupt crash/force-quit followed by the
user immediately opening a new session leaves a transcript with a VERY recent
mtime but no live writer at all, and a time-based guard would skip it, missing
precisely the case it was built for. `lsof` answers the real question directly
instead of approximating it from a timestamp.

Scans every project folder, not just one derived from the current cwd --
confirmed empirically that a single session's transcript can span multiple
cwds over its life and lives wherever its FIRST entry resolved to, not
necessarily anywhere related to where it's read from later. Folder-level
scoping is not reliable for finding "this project's" sessions.

Capped per run (MAX_ORPHANS_PER_SWEEP) so a large backlog gets caught up
gradually across future session starts rather than all at once -- avoids a
single SessionStart hook invocation running long or spending a lot of quota
in one shot.

Never blocks (SessionStart hooks don't support blocking anyway), never raises
past main(): exits 0 on any failure, logs to ../backlink_errors/.
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HOOK_DIR  = Path(__file__).resolve().parent.parent
ERROR_DIR = HOOK_DIR / "backlink_errors"
PROJECTS_DIR = Path.home() / ".claude" / "projects"

MAX_ORPHANS_PER_SWEEP = 3         # cap work per invocation; backlog clears gradually, not all at once


def log_error(exc):
    ERROR_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (ERROR_DIR / f"session_start_backstop_{stamp}.txt").write_text(
        f"time: {stamp}\nerror: {exc!r}\n\n{traceback.format_exc()}"
    )


def _session_id_from_path(path: Path) -> str:
    return path.stem  # <session-id>.jsonl -> <session-id>


def _line_count(path: Path) -> int:
    n = 0
    with open(path, "r", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def _is_open_by_a_process(path: Path) -> bool:
    """True if any process currently holds this exact file open. lsof exits 1
    with no output when nothing has it open, 0 with a listing when something
    does -- a direct answer to "is this still being written to", unlike a
    time-based guess from mtime."""
    try:
        result = subprocess.run(["lsof", str(path)], capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # lsof missing or hung -- fail safe by treating as "maybe still open"
        # rather than risk racing a live writer.
        return True
    return result.returncode == 0


def find_orphans(current_session_id: str, get_state) -> list[Path]:
    if not PROJECTS_DIR.exists():
        return []
    candidates = []
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for transcript in project_dir.glob("*.jsonl"):
            sid = _session_id_from_path(transcript)
            if sid == current_session_id:
                continue  # never sweep the session that's just starting
            try:
                mtime = transcript.stat().st_mtime
            except OSError:
                continue
            if _is_open_by_a_process(transcript):
                continue  # a live process still has this open -- don't race it
            state = get_state(sid)
            last_line = state.get("last_finalized_line", 0)
            try:
                current_lines = _line_count(transcript)
            except OSError:
                continue
            if current_lines > last_line:
                candidates.append((mtime, transcript, sid))
    # oldest-touched first -- clear the longest-abandoned backlog first
    candidates.sort(key=lambda c: c[0])
    return [(t, sid) for _, t, sid in candidates[:MAX_ORPHANS_PER_SWEEP]]


def run():
    raw = sys.stdin.read().strip()
    if not raw:
        return
    data = json.loads(raw)

    current_session_id = data.get("session_id", "")

    sys.path.insert(0, str(HOOK_DIR))

    if data.get("source") == "compact":
        # Mini consolidation cycle: a session that compacts many times but never hits SessionEnd
        # (a long-running interactive session, or eventually the always-on Agent SDK backend) got
        # ZERO enrichment for its entire runtime under the old startup-only logic -- everything
        # about to become compacted/lossy just stayed that way until the session finally ended.
        # This runs the same proven capture_session() extraction on THIS session's own transcript
        # immediately after every compaction, so nothing has to wait until the end to be captured
        # with full fidelity from the raw source. (2026-07-31 -- see CLAUDE.md for the full
        # reasoning: PreCompact can't touch the compaction result itself, so the fix has to be
        # "enrich memory from the raw transcript right after," not "intervene during compaction.")
        transcript_path = data.get("transcript_path", "")
        if transcript_path and current_session_id:
            from auto_capture import capture_session
            capture_session(current_session_id, transcript_path)
        return

    if data.get("source") != "startup":
        return  # only sweep orphans on a genuinely fresh session, not resume/clear/fork

    from capture_state import get_state
    from auto_capture import capture_sessions_parallel

    orphans = find_orphans(current_session_id, get_state)
    if orphans:
        capture_sessions_parallel([(sid, str(transcript)) for transcript, sid in orphans])


def main():
    try:
        run()
    except Exception as exc:
        log_error(exc)


if __name__ == "__main__":
    main()
