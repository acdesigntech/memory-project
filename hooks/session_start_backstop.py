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


# Distinctive opening text from this project's own claude -p prompt templates
# (auto_capture.py's EXTRACTION_PROMPT_TEMPLATE, memory_store.py's
# FEEDBACK_DRAFT_PROMPT_TEMPLATE). Every one-shot `claude -p` call this project
# makes embeds its prompt as the first turn's content, so this is present on
# line 1 of every transcript such a call produces.
SYNTHETIC_CALL_SIGNATURES = (
    "You are extracting durable, cross-session-worthy memories",
    "separate feedback notes a coding assistant recorded",
)


def _is_synthetic_extraction_call(path: Path) -> bool:
    """True if `path` is a transcript from this project's own claude -p
    extraction machinery, not a real user session. Found 2026-08-01: of the
    9,507 transcripts on this machine, 9,502 turned out to be exactly this --
    a one-shot claude -p call (auto_capture.py's extraction, or chain-11's
    draft_rule_from_cluster()) leaves behind its own tiny transcript file, and
    since nothing legitimately captures these (they're not real conversations
    worth extracting facts FROM), find_orphans() would otherwise re-examine an
    ever-growing pile of them on every single sweep forever -- the checkpoint-
    cache added earlier the same night can never mark them "clean" for the
    same reason. This is the real fix: recognize and skip them outright,
    before any of the other checks even run, rather than trying to make
    re-examining them cheaper. Reads only the first line, not the whole file."""
    try:
        with open(path, "r", errors="ignore") as f:
            first_line = f.readline()
    except OSError:
        return False
    return any(sig in first_line for sig in SYNTHETIC_CALL_SIGNATURES)


def _has_new_content(path: Path, last_line: int) -> bool:
    """True if `path` has more than `last_line` lines. find_orphans() only ever
    needs this yes/no answer, never the exact count, so this stops reading the
    instant it's proven true instead of counting the whole file every time
    (replaces a full _line_count()). Found 2026-08-01: this machine's transcript
    history is dominated by files that are already fully captured (checkpoint at
    or near the file's true end) -- for that common case there's nothing to
    short-circuit on until near EOF anyway, so this alone isn't a full fix (a
    real-world timing run only moved ~13min -> ~9min, not the order-of-magnitude
    the lsof-call-count reduction alone would suggest). It's still strictly
    better than a full count and does meaningfully help the never-yet-captured
    case (last_line=0 against a large file), so kept as a genuine improvement,
    documented honestly rather than oversold."""
    n = 0
    with open(path, "r", errors="ignore") as f:
        for _ in f:
            n += 1
            if n > last_line:
                return True
    return False


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


def find_orphans(current_session_id: str, get_state, set_state=None) -> list[Path]:
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
            if _is_synthetic_extraction_call(transcript):
                continue  # this project's own claude -p byproduct, not a real session
            try:
                mtime = transcript.stat().st_mtime
            except OSError:
                continue

            state = get_state(sid)

            # Cache hit: this exact mtime was already confirmed clean (no new
            # content) on a previous sweep, and the file hasn't been touched
            # since -- skip without even opening it. Found 2026-08-01: after
            # the checkpoint-first reorder below, a real-world timing run still
            # cost ~9min, because the remaining bottleneck was opening and
            # reading ~9,500 already-fully-processed files on every single
            # sweep, forever, even though their content never changes again
            # once a session ends. This is what actually answers Andrew's
            # original mtime idea safely: not "recent mtime = might be live"
            # (an absolute guess, wrong for a long-idle-but-genuinely-live
            # session), but "mtime unchanged since I last checked = content is
            # provably identical to last time" (a comparison against OUR OWN
            # prior observation, not a guess about someone else's process).
            # First sweep after this ships still pays the full ~9min to build
            # the cache; every sweep after that should be dramatically faster
            # for the ~9,500 files that are permanently static.
            if state.get("checked_mtime") == mtime:
                continue

            # Cheap check next: most of this machine's transcript history is
            # already fully captured (checkpointed via SessionEnd/backstop/compact
            # over time) -- skip straight past those without ever shelling out to
            # lsof. Found 2026-07-31: the old lsof-before-checkpoint-check order
            # made every genuine session startup pay an lsof call per historical
            # transcript regardless of whether it had anything new at all --
            # confirmed at ~81ms/call x 9,507 files = ~13 min real-world cost.
            # Safe regardless of order: get_state() was never a liveness signal,
            # only a "how much has already been captured" one -- if there's
            # nothing new, it's irrelevant whether the session is alive or dead,
            # so it's always correct to skip the liveness check in that case. The
            # abrupt-crash scenario this hook exists for still works: a crash
            # interrupts active writing before a final checkpoint can run, so a
            # crashed session's transcript almost always has content past its
            # last checkpoint and still reaches the real lsof check below.
            last_line = state.get("last_finalized_line", 0)
            try:
                has_new_content = _has_new_content(transcript, last_line)
            except OSError:
                continue
            if not has_new_content:
                if set_state is not None:
                    set_state(sid, checked_mtime=mtime)
                continue  # nothing new -- no reason to even ask if it's live

            # Only for sessions with genuinely new, uncaptured content does
            # liveness actually matter. Deliberately NOT mtime-based -- a
            # genuinely live session can sit open with zero writes for a long
            # idle stretch (user stepped away mid-conversation) and must still be
            # correctly detected as live; an mtime cutoff would misclassify that
            # exact case as "safe," reintroducing the concurrent-writer race this
            # check exists to prevent. lsof answers the real question directly.
            if _is_open_by_a_process(transcript):
                continue  # a live process still has this open -- don't race it

            candidates.append((mtime, transcript, sid))
    # oldest-touched first -- clear the longest-abandoned backlog first
    candidates.sort(key=lambda c: c[0])
    return [(t, sid) for _, t, sid in candidates[:MAX_ORPHANS_PER_SWEEP]]


def run():
    # `claude -p` extraction subprocesses launched by this project (auto_capture.py,
    # memory_store.py's draft_rule_from_cluster) are full Claude Code invocations and
    # fire this same global SessionStart hook. Without this guard, an extraction call
    # re-triggers the orphan sweep below, which can launch MORE extraction calls,
    # which fire SessionStart again -- confirmed 2026-07-31 as a real runaway process
    # tree that had to be killed by hand. The env var is set on every such subprocess.
    if os.environ.get("MEMORY_PROJECT_NESTED_EXTRACTION"):
        return

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

    from capture_state import get_state, set_state
    from auto_capture import capture_sessions_parallel

    orphans = find_orphans(current_session_id, get_state, set_state)
    if orphans:
        capture_sessions_parallel([(sid, str(transcript)) for transcript, sid in orphans])


def main():
    try:
        run()
    except Exception as exc:
        log_error(exc)


if __name__ == "__main__":
    main()
