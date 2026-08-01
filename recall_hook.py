#!/usr/bin/env python3
"""
UserPromptSubmit hook — two things, on different schedules.

1. Real current date/time, injected on EVERY prompt, not just the first.
   Found 2026-08-01: a long-running session drifted many hours past its own
   starting context (which had opened in the evening) without ever re-
   checking the actual time, leading to confidently wrong "goodnight" /
   "still nighttime" framing well into the next afternoon. A jot()ted
   reminder to "check the time" is a soft, semantically-gated fix -- it only
   surfaces if some future prompt happens to score a recall() hit against
   it. This is the actual fix: ground truth, unconditionally, every time,
   cheap enough (a datetime call, no embeddings/ChromaDB) that there's no
   reason to gate it at all.

2. Relevant memories, surfaced before Claude sees the first message of a
   session. Skips subsequent messages in the same session -- this part IS
   gated, since repeating the same recalled context on every message would
   be noisy and recall() itself has a real cost (embeddings + ChromaDB).

Never blocks: exits 0 on any failure, logs to backlink_errors/.
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HOOK_DIR   = Path(__file__).resolve().parent
ERROR_DIR  = HOOK_DIR / "backlink_errors"
SESSION_CACHE = HOOK_DIR / ".recall_session_cache"
MIN_SCORE  = 0.35
N_RESULTS  = 4

# Topics never surfaced via this automatic path (still fully queryable via a
# direct recall() call when the user actually raises the topic themselves —
# this only blocks the unprompted hook path). Configure your own sensitive
# topics in local_config.py (gitignored, not this file) rather than here, so
# the published script never carries anyone's real topic names.
try:
    from local_config import EXCLUDE_TOPICS
except ImportError:
    EXCLUDE_TOPICS = set()


def log_error(exc):
    ERROR_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (ERROR_DIR / f"recall_hook_{stamp}.txt").write_text(
        f"time: {stamp}\nerror: {exc!r}\n\n{traceback.format_exc()}"
    )


def extract_prompt(data: dict) -> str:
    for key in ("prompt", "message", "content", "text"):
        if key in data and isinstance(data[key], str) and data[key].strip():
            return data[key].strip()
    return ""


def format_time_context() -> str:
    now = datetime.now().astimezone()
    return f"[Current date/time] {now.strftime('%A, %B %-d, %Y, %-I:%M %p %Z')}"


def format_context(hits: list[dict]) -> str:
    lines = ["[Memory context — relevant past sessions]", ""]
    for h in hits:
        excerpt = " ".join(h["document"].split())[:250]
        lines.append(f"• {h['title']}  [{h['topic']}]  (score={h['score']:.2f})")
        lines.append(f"  {excerpt}…")
        lines.append(f"  file: {h['file_path']}")
        lines.append("")
    lines.append("[End memory context]")
    return "\n".join(lines)


def already_ran_this_session(session_id: str) -> bool:
    if not session_id or not SESSION_CACHE.exists():
        return False
    return SESSION_CACHE.read_text().strip() == session_id


def mark_session(session_id: str) -> None:
    SESSION_CACHE.write_text(session_id)


def run():
    raw = sys.stdin.read().strip()
    if not raw:
        return

    data = json.loads(raw)
    session_id = data.get("session_id", "")

    context_parts = [format_time_context()]

    if not already_ran_this_session(session_id):
        prompt = extract_prompt(data)
        if prompt:
            cwd = data.get("cwd", "")
            topic_hint = Path(cwd).name if cwd else None

            sys.path.insert(0, str(HOOK_DIR))
            from memory_store import recall

            hits = recall(prompt, n_results=N_RESULTS, topic_hint=topic_hint, exclude_topic=EXCLUDE_TOPICS)
            hits = [h for h in hits if h["score"] >= MIN_SCORE]

            mark_session(session_id)

            if hits:
                context_parts.append(format_context(hits))
        # prompt empty -- deliberately don't mark_session, so a later prompt in
        # this same session that DOES have extractable content still gets a
        # shot at the memory-recall pass, matching the original retry behavior.

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n\n".join(context_parts),
        }
    }))


def main():
    try:
        run()
    except Exception as exc:
        log_error(exc)


if __name__ == "__main__":
    main()
