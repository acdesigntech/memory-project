# Changelog

Notable changes to this project, newest first. No version tags/releases — this is a live, continuously-evolving personal system, so entries are dated instead.

## 2026-07-29 — Fix SessionStart/SessionEnd hook timeouts; parallelize orphan capture

### Fixed

- `SessionStart` and `SessionEnd` hook timeouts (`~/.claude/settings.json`, and `install.py` for fresh installs) were 60s and 120s respectively — far too short. A real `claude -p` subprocess call was measured at ~2 minutes wall-clock even for a trivial extraction prompt (mostly CLI/session startup overhead, not generation). Any extraction that ran past its hook's timeout got killed externally by Claude Code's own hook-timeout enforcement — a SIGTERM/SIGKILL, not a Python exception — which happened *before* `capture_state.py`'s checkpoint could advance and *before* anything reached `backlink_errors/`. Net effect: some crashed/killed sessions were silently never captured, with no error visible anywhere. Both hook timeouts raised to 300s.
- `auto_capture.py`'s internal per-`claude -p`-call timeout (`EXTRACTION_TIMEOUT`) lowered from 120s to 270s — it was previously *shorter* than the empirically measured baseline call time (backwards), and now stays under the new 300s hook ceiling with a buffer for surrounding overhead (embedding model load, transcript scanning).
- `install.py` still hardcoded the old 60s/120s values for fresh installs — updated to match, so new installs don't reintroduce this bug.

### Changed

- `auto_capture.py`: replaced the sequential per-orphan `capture_session()` loop with `capture_sessions_parallel()`. Every `claude -p` extraction subprocess across every orphaned session in a sweep now launches concurrently (`Popen`, non-blocking) instead of one after another — total wall time for a sweep is bounded by the slowest single extraction call, not the sum of all of them. Recording results (`jot()` + `set_state()`) still happens strictly sequentially, back in the single parent process, only after every subprocess has already been launched — deliberately, to avoid needing any locking around `capture_state.json` or ChromaDB's `PersistentClient`, since neither is ever written to concurrently under this design. `capture_session(session_id, transcript_path)` is now a one-line wrapper around `capture_sessions_parallel()` for the single-session case (used by `session_end_capture.py`).
- `hooks/session_start_backstop.py` now hands all orphans from one sweep to `capture_sessions_parallel()` in a single call, instead of looping `capture_session()` once per orphan.

### Docs

- `CLAUDE.md`: auto-capture section updated with the corrected timeout values and the new parallel-capture design.
- `README.md`: new "Capture timing" subsection under the memory chain table, setting expectations for startup latency after an abnormal session end.
- `CLAUDE.md.example`: added a heads-up that a session may pause for a few minutes on startup while the backstop hook catches up a previous crashed session — expected behavior, not a hang.
