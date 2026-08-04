# Changelog

Notable changes to this project, newest first. No version tags/releases — this is a live, continuously-evolving personal system, so entries are dated instead.

## 2026-08-04 — purge() vector-index scrub, correction-encoding tombstones, lazy prune() archiving

Closed three of the four items from the 2026-08-03 parking-lot reprioritization (external code review). Only "auto-discovery of `CORPUS_DIR`" remains.

### Fixed

- `purge(doc_id)` previously only called `col.delete(ids=[doc_id])` on the Chroma collection — a soft delete against its hnswlib-backed persistent index, so a "permanently" purged embedding could remain physically present in `.chromadb/*/data_level0.bin` until that slot got reused by a future insert. `purge()` now does a full collection rebuild instead (fetch every remaining entry, `delete_collection()`, recreate, re-add everything except the purged doc) — the only available guarantee, since neither `PersistentClient` nor `Collection` exposes a compact/vacuum call in this Chroma version. O(corpus size) instead of O(1), an accepted tradeoff since `purge()` is rare/deliberate, not routine.
- `regression_test.py`'s `test_prune_cold_revive_purge()` cached a `Collection` handle before calling `purge()` and reused it afterward — `purge()`'s rebuild replaces the module-level handle, so a cached one goes stale (raises Chroma's `NotFoundError`). Fixed by re-fetching after the call.
- `prune()`'s full-corpus sweep was never called automatically — a memory that decayed below `DELETION_FLOOR` was already invisible to `recall()` (`RETRIEVAL_FLOOR` catches it regardless of `archived` state) but sat unarchived, and therefore unreachable via `recall_cold()`, until someone ran `prune()` by hand.

### Added

- `reject_claim(text, reason='', topic_hint=None, source_doc_id='')` — records a tombstone (a claim deliberately marked wrong) in the same Chroma collection as ordinary memories (`capture_tier="tombstone"`), reusing the existing nearest-neighbor index rather than a second vector store.
- `purge(doc_id, tombstone=False, reason='')` — gained an opt-in `tombstone` param. `tombstone=True` calls `reject_claim()` with the purged text right after the rebuild, so a corrected-and-purged fact can't silently reappear. Deliberately not automatic: plain `purge()` (the accidentally-jotted-secret case) needs the embedding to actually vanish, the opposite of what a tombstone wants.
- `jot()` now checks `_nearest_tombstone()` before writing (reusing its own already-computed embedding) and refuses the write — returns `None` instead of a doc_id — if the new text is a near-duplicate (>=0.82 raw cosine similarity, `TOMBSTONE_MATCH_THRESHOLD`) of a rejected claim. This is what stops a corrected fact from resurrecting itself via a later `jot()` or an autonomous `auto_capture.py` re-extraction. `jot()`'s return type is now `str | None`.
- `_archive_entry(col, doc_id, meta)` — shared archiving helper factored out of `prune()`'s loop. `_score_hits()` (shared by `recall()`/`recall_associative()`) now calls it the moment a raw candidate's strength drops below `DELETION_FLOOR`, so a decayed memory gets archived as a side effect of the next real query that happens to surface it — no scheduler/cron needed (PowerMem's lazy-at-retrieval-time lead). Not exhaustive: an entry nobody ever queries near still only gets archived by an explicit `prune()` sweep.

### Changed

- `auto_capture.py`'s `jot()` call site now only counts a jot as successful when the return value isn't `None`, to match `jot()`'s new tombstone-refusal behavior.
- `_score_hits()` filters out `capture_tier == "tombstone"` entries unconditionally, same pattern as the existing `archived` filter — tombstones are metadata about what *not* to write, never themselves a retrievable memory.

### Docs

- `PROJECT_PLAN.md`: three new `## ... (DONE 2026-08-04)` sections (purge() vector-index scrub, correction encoding, lazy prune() archiving), each moved out of the parking lot.
- `regression_test.py`: new `test_reject_claim_tombstone()` (7 checks) and `test_lazy_archive_on_recall()`. Full suite: 84 passed, 0 failed.

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
