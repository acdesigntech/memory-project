#!/usr/bin/env python3
"""
UserPromptSubmit hook — surface relevant memories before Claude sees the first
message of a session. Skips subsequent messages in the same session.
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
    if already_ran_this_session(session_id):
        return

    prompt = extract_prompt(data)
    if not prompt:
        return

    cwd = data.get("cwd", "")
    topic_hint = Path(cwd).name if cwd else None

    sys.path.insert(0, str(HOOK_DIR))
    from memory_store import recall

    hits = recall(prompt, n_results=N_RESULTS, topic_hint=topic_hint, exclude_topic=EXCLUDE_TOPICS)
    hits = [h for h in hits if h["score"] >= MIN_SCORE]

    mark_session(session_id)

    if not hits:
        return

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": format_context(hits),
        }
    }))


def main():
    try:
        run()
    except Exception as exc:
        log_error(exc)


if __name__ == "__main__":
    main()
