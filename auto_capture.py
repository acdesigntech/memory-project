#!/usr/bin/env python3
"""
Automatic session capture: turns a raw Claude Code transcript segment into
jot()-quality memories, without a human in the loop (per the user's explicit
"make it perform like jot()" direction, 2026-07-25).

Public API:
  capture_session(session_id, transcript_path) -> int
      Process everything new since this session's last recorded checkpoint
      (capture_state.get_state) through to current end-of-file, extract
      distinct jot()-worthy facts, jot() each one, advance the checkpoint.
      Returns the number of facts jotted. Safe to call repeatedly/idempotently
      -- a session with nothing new since last time is a fast no-op.

Design notes (see PROJECT_PLAN.md for full rationale):
  - Reads the RAW transcript file directly, not the calling session's live
    (possibly already-compacted) context -- the raw file never loses history
    (compaction only ever appends a `compact_boundary` marker entry, never
    deletes prior lines; confirmed empirically against a real 3-day, 94MB,
    multi-compaction session transcript on 2026-07-25).
  - Chunks a large new segment at `compact_boundary` markers where present
    (natural conversational boundaries the platform itself already draws),
    falling back to a character budget for stretches without one.
  - Extraction itself is a subprocess `claude -p` call per chunk, not the
    calling session reasoning about its own context -- keeps this fully
    decoupled from whatever hook process invoked it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_state import get_state, set_state
from memory_store import jot

# Character budget per extraction chunk -- rough guard against feeding an
# enormous stretch (no compact_boundary in it) to a single claude -p call.
# Not a hard science; generous enough for real conversational stretches,
# small enough that a single call stays fast/cheap.
CHUNK_CHAR_BUDGET = 60_000

EXTRACTION_PROMPT_TEMPLATE = """You are extracting durable, cross-session-worthy memories from a slice of a coding assistant conversation. Read the transcript below and identify distinct facts worth remembering long-term, using the SAME low-bar judgment a human note-taker would: a stated user preference, a correction the user gave, a decision/design resolved in conversation, or something surprising/non-obvious that came up. Skip routine tool mechanics, restating what a file already says, or anything only meaningful within this one exchange.

Working directory context for this slice: {cwd}

Output ONLY a JSON array (no other text), where each element is:
  {{"text": "<the fact, written as a standalone memory -- someone with zero other context should understand it>", "topic_hint": "<short topic slug, e.g. the project directory name>"}}

If nothing in this slice is worth remembering, output an empty array: []

Transcript slice:
---
{transcript_text}
---

JSON array:"""


def _iter_entries(transcript_path: Path, from_line: int):
    """Yield (line_number, parsed_dict) for every parseable line after from_line."""
    with open(transcript_path, "r", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            if i <= from_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def _entry_text(entry: dict) -> str | None:
    """Render one user/assistant entry as a readable conversational line, or
    None if it's not conversationally meaningful (tool mechanics, thinking,
    non-message entry types)."""
    t = entry.get("type")
    if t not in ("user", "assistant"):
        return None
    msg = entry.get("message", {})
    content = msg.get("content")
    role = "User" if t == "user" else "Claude"

    if isinstance(content, str):
        return f"{role}: {content}"

    if isinstance(content, list):
        parts = []
        for block in content:
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        if parts:
            return f"{role}: " + "\n".join(parts)
    return None


def _last_cwd(entries: list[tuple[int, dict]]) -> str | None:
    for _, e in reversed(entries):
        if "cwd" in e:
            return e["cwd"]
    return None


def _chunk_entries(entries: list[tuple[int, dict]]) -> list[list[tuple[int, dict]]]:
    """Split into chunks at compact_boundary markers first, then further by
    character budget within any chunk that's still too large."""
    chunks: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    for i, e in entries:
        if e.get("type") == "system" and e.get("subtype") == "compact_boundary":
            if current:
                chunks.append(current)
                current = []
            continue  # the marker itself isn't conversational content
        current.append((i, e))
    if current:
        chunks.append(current)

    # further split any chunk whose rendered text would exceed the budget
    final: list[list[tuple[int, dict]]] = []
    for chunk in chunks:
        piece: list[tuple[int, dict]] = []
        piece_len = 0
        for i, e in chunk:
            text = _entry_text(e)
            add_len = len(text) if text else 0
            if piece and piece_len + add_len > CHUNK_CHAR_BUDGET:
                final.append(piece)
                piece = []
                piece_len = 0
            piece.append((i, e))
            piece_len += add_len
        if piece:
            final.append(piece)
    return final


def _render_chunk(chunk: list[tuple[int, dict]]) -> str:
    lines = []
    for _, e in chunk:
        text = _entry_text(e)
        if text:
            lines.append(text)
    return "\n\n".join(lines)


# Kept under the SessionStart/SessionEnd hook timeout (300s, ~/.claude/settings.json)
# so a single slow claude -p call gets a chance to time out cleanly on its own
# rather than the whole hook process getting killed externally mid-call.
EXTRACTION_TIMEOUT = 270


def _start_extraction(transcript_text: str, cwd: str) -> subprocess.Popen | None:
    """Launch one claude -p extraction subprocess without blocking on it.
    Returns None if there's nothing worth extracting from (caller treats
    that the same as a no-facts result)."""
    if not transcript_text.strip():
        return None
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(cwd=cwd or "unknown", transcript_text=transcript_text)
    try:
        return subprocess.Popen(
            ["claude", "-p", prompt],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except FileNotFoundError:
        return None


def _collect_extraction(proc: subprocess.Popen | None) -> list[dict]:
    """Block until one already-running extraction subprocess finishes and
    parse its output. Safe to call on several procs in sequence after they
    were all started together -- each call only waits out whatever time is
    left on that proc, since it's been running concurrently in the
    background since _start_extraction returned."""
    if proc is None:
        return []
    try:
        stdout, _ = proc.communicate(timeout=EXTRACTION_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return []
    if proc.returncode != 0:
        return []
    raw = stdout.strip()
    # tolerate a model that wraps the array in a code fence despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(facts, list):
        return []
    return [f for f in facts if isinstance(f, dict) and f.get("text")]


def capture_sessions_parallel(sessions: list[tuple[str, str]]) -> int:
    """Process several sessions' new transcript content at once.

    The slow part -- one claude -p subprocess per chunk -- runs concurrently
    across every session passed in, since each subprocess is fully
    independent and touches neither ChromaDB nor capture_state.json.
    Recording results (jot() + set_state()) happens back in this single
    process, strictly sequentially, only after every subprocess has already
    been started -- so there's never more than one writer touching the
    ChromaDB store or the state file, without needing any locking.
    """
    session_jobs: dict[str, list[subprocess.Popen | None]] = {}
    session_meta: dict[str, dict] = {}

    for session_id, transcript_path in sessions:
        path = Path(transcript_path)
        if not path.exists():
            continue

        state = get_state(session_id)
        from_line = state.get("last_finalized_line", 0)
        entries = list(_iter_entries(path, from_line))
        if not entries:
            continue

        new_max_line = entries[-1][0]
        cwd = _last_cwd(entries) or state.get("cwd")
        cwds_seen = set(state.get("cwds_seen", []))
        for _, e in entries:
            if "cwd" in e:
                cwds_seen.add(e["cwd"])

        session_jobs[session_id] = [
            _start_extraction(_render_chunk(chunk), cwd) for chunk in _chunk_entries(entries)
        ]
        session_meta[session_id] = {"new_max_line": new_max_line, "cwd": cwd, "cwds_seen": cwds_seen}

    # Every subprocess above is already running in the background. Collecting
    # them here is sequential, but each collect only blocks on time the proc
    # hasn't already used up concurrently with the others -- total wall time
    # is bounded by the slowest single job, not the sum of all of them.
    total_jotted = 0
    for session_id, jobs in session_jobs.items():
        meta = session_meta[session_id]
        for proc in jobs:
            for fact in _collect_extraction(proc):
                jot(fact["text"], topic_hint=fact.get("topic_hint") or (Path(meta["cwd"]).name if meta["cwd"] else None))
                total_jotted += 1
        set_state(
            session_id,
            last_finalized_line=meta["new_max_line"],
            cwd=meta["cwd"],
            cwds_seen=sorted(meta["cwds_seen"]),
        )
    return total_jotted


def capture_session(session_id: str, transcript_path: str) -> int:
    return capture_sessions_parallel([(session_id, transcript_path)])


if __name__ == "__main__":
    # manual CLI test: python auto_capture.py <session_id> <transcript_path>
    if len(sys.argv) == 3:
        n = capture_session(sys.argv[1], sys.argv[2])
        print(f"jotted {n} facts")
