#!/usr/bin/env python3
"""
Full-stack regression test for memory-project's public API.

Unit tests belong next to a new feature. This is different: it exercises
the WHOLE system end to end — every public function across memory_store.py,
auto_capture.py, capture_state.py, and all four hooks — the way a change to
one function (e.g. jot()'s metadata shape) can silently break something far
away (e.g. find_feedback_patterns()'s clustering) that no single feature's
unit tests would ever catch. Run this "every once in a while," per Andrew's
framing, after anything that touches shared plumbing.

Origin: built 2026-08-01 to codify a full manual regression pass run the
night before (14 areas, one real finding — see PROJECT_PLAN.md). Same 14
areas, same assertions, every time this runs — that determinism is the
whole point; a skill/agent re-deriving "what to test" from scratch each
run would drift.

All synthetic data uses topic_hint values prefixed "regression-test" (or a
dedicated temp dir for the orphan-sweep test) and is purged in a `finally`
block, so a crash mid-run still leaves the real corpus untouched — verified
via a final corpus-wide sweep, not just the individual per-test purges.

Usage:
  .venv/bin/python regression_test.py            # full run, incl. live claude -p calls
  .venv/bin/python regression_test.py --quick     # skip the slow/costly claude -p-dependent checks

Exit code 0 if every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hooks"))

# Gitignored (see .gitignore) -- a run's report can echo real corpus content
# (doc titles, topic names) surfaced during e.g. the EXCLUDE_TOPICS check, so
# reports live on disk for Andrew's own reference but never get committed,
# same reasoning as activity.log/session_summaries/*.
REPORTS_DIR = ROOT / "regression_reports"

import memory_store as ms
import auto_capture as ac
import capture_state as cs

REGRESSION_TOPIC_PREFIX = "regression-test"

results: list[tuple[str, bool, str]] = []
_created_ids: list[str] = []  # doc_ids created outside the topic-prefix sweep (e.g. curated ingest paths)


def check(name: str, condition: bool, detail: str = "") -> bool:
    condition = bool(condition)
    results.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def skip(name: str, reason: str) -> None:
    results.append((name, None, reason))
    print(f"[SKIP] {name} — {reason}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ---------------------------------------------------------------- core recall loop

def test_jot_variants():
    section("jot() — metadata paths")
    id1 = ms.jot("Regression fact: default topic_hint should derive from cwd basename.")
    id2 = ms.jot("Regression fact: explicit topic_hint override.", topic_hint=REGRESSION_TOPIC_PREFIX)
    id3 = ms.jot("Regression fact: tagged with category=feedback.", topic_hint=REGRESSION_TOPIC_PREFIX, category="feedback")
    id4 = ms.jot("Regression fact: tagged with an explicit session_id.", topic_hint=REGRESSION_TOPIC_PREFIX, session_id="regression-test-session-abc")
    _created_ids.extend([id1, id2, id3, id4])

    col = ms._get_collection()
    res = col.get(ids=[id1, id2, id3, id4], include=["metadatas"])
    m = dict(zip(res["ids"], res["metadatas"]))

    check("jot() default topic_hint derives cwd basename", m[id1]["topic"] == ROOT.name, m[id1]["topic"])
    check("jot() explicit topic_hint applied", m[id2]["topic"] == REGRESSION_TOPIC_PREFIX)
    check("jot() category stored correctly", m[id3]["category"] == "feedback")
    check("jot() session_id stored correctly", m[id4]["session_id"] == "regression-test-session-abc")
    check("jot() fragment defaults correct", all(
        m[i]["capture_tier"] == "fragment" and m[i]["memory_type"] == "episodic" and m[i]["stability"] == ms.EPISODIC_BASE_STABILITY
        for i in (id1, id2, id3, id4)
    ))
    return id2  # reused by recall tests


def test_recall(fact_id: str):
    section("recall() — matching, exclusion, topic boost")
    query = "regression fact explicit topic_hint override"

    hits = ms.recall(query, n_results=3)
    check("recall() basic match surfaces the right doc top-ranked", hits and hits[0]["id"] == fact_id)

    hits_excl = ms.recall(query, n_results=10, exclude_topic=REGRESSION_TOPIC_PREFIX)
    check("recall() exclude_topic filters out the whole topic", not any(h["topic"] == REGRESSION_TOPIC_PREFIX for h in hits_excl))

    hits_boost = ms.recall("something something", n_results=5, topic_hint=REGRESSION_TOPIC_PREFIX)
    check("recall() topic_hint boost surfaces topic despite a vague query",
          any(h["topic"] == REGRESSION_TOPIC_PREFIX for h in hits_boost))

    col = ms._get_collection()
    before = col.get(ids=[fact_id], include=["metadatas"])["metadatas"][0]
    ms.recall(query, n_results=1)
    after = col.get(ids=[fact_id], include=["metadatas"])["metadatas"][0]
    expected_stability = before["stability"] * ms.STABILITY_GROWTH
    check("recall() reinforcement grows stability by STABILITY_GROWTH exactly",
          math.isclose(after["stability"], expected_stability, rel_tol=1e-6),
          f"{after['stability']:.1f} vs expected {expected_stability:.1f}")
    check("recall() reinforcement increments access_count", after["access_count"] == before["access_count"] + 1)


def test_recall_associative():
    section("recall_associative() + confirm_activation()")
    ids = [
        ms.jot("Regression cluster: prefers terse replies with no trailing summary.", topic_hint=REGRESSION_TOPIC_PREFIX, category="feedback"),
        ms.jot("Regression cluster: annoyed by an unnecessary recap sentence.", topic_hint=REGRESSION_TOPIC_PREFIX, category="feedback"),
        ms.jot("Regression cluster: explicitly said stop summarizing after finishing.", topic_hint=REGRESSION_TOPIC_PREFIX, category="feedback"),
    ]
    weak_id = ms.jot("Regression cluster loosely adjacent: something about writing style in general.", topic_hint=REGRESSION_TOPIC_PREFIX)
    _created_ids.extend(ids + [weak_id])

    res = ms.recall_associative("terse replies no summary recap stop summarizing", n_results=5, n_activated=5, topic_hint=REGRESSION_TOPIC_PREFIX)
    confident_ids = {h["id"] for h in res["confident"]}
    activated_ids = {h["id"] for h in res["activated"]}

    check("recall_associative() splits into confident/activated tiers", confident_ids or activated_ids)
    check("recall_associative() confident tier only contains score >= CONFIDENT_THRESHOLD",
          all(h["score"] >= ms.CONFIDENT_THRESHOLD for h in res["confident"]))
    check("recall_associative() activated tier only contains scores in the activation band",
          all(ms.ACTIVATION_FLOOR <= h["score"] < ms.CONFIDENT_THRESHOLD for h in res["activated"]))

    target = next((h for h in res["activated"]), None) or next((h for h in res["confident"]), None)
    if target is None:
        check("recall_associative()/confirm_activation() — usable candidate found", False, "no hit in either tier to test confirm_activation against")
        return

    col = ms._get_collection()
    before = col.get(ids=[target["id"]], include=["metadatas"])["metadatas"][0]
    ms.confirm_activation(target["id"])
    after = col.get(ids=[target["id"]], include=["metadatas"])["metadatas"][0]
    expected = min(before["stability"] * ms.ACTIVATION_STABILITY_BOOST, ms.EPISODIC_BASE_STABILITY * ms.STABILITY_CAP_FACTOR)
    check("confirm_activation() applies the larger ACTIVATION_STABILITY_BOOST exactly",
          math.isclose(after["stability"], expected, rel_tol=1e-6),
          f"{after['stability']:.1f} vs expected {expected:.1f}")


# ---------------------------------------------------------------- curated memory

def test_ingest():
    section("ingest() — curated file path")
    test_dir = ms.CORPUS_DIR / "regression-test-ingest"
    test_dir.mkdir(parents=True, exist_ok=True)
    doc_path = test_dir / "doc.md"
    doc_path.write_text("# Regression Ingest Doc\n\nMarker: zzzregressiontestmarkerzzz\n")
    doc_id = "regression-test-ingest/doc"

    try:
        ms.ingest(doc_path)
        col = ms._get_collection()
        m = col.get(ids=[doc_id], include=["metadatas"])["metadatas"]
        check("ingest() creates a curated entry at CURATED_INITIAL_STABILITY",
              bool(m) and m[0]["stability"] == ms.CURATED_INITIAL_STABILITY and m[0]["capture_tier"] == "curated")

        hits = ms.recall("zzzregressiontestmarkerzzz", n_results=1)
        check("ingest()'d doc is recallable", hits and hits[0]["id"] == doc_id)

        after_recall = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
        doc_path.write_text(doc_path.read_text() + "\nEdited for re-ingest test.\n")
        ms.ingest(doc_path)
        after_reingest = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
        check("re-ingest after edit preserves access_count/stability instead of resetting",
              after_reingest["access_count"] == after_recall["access_count"]
              and after_reingest["stability"] == after_recall["stability"])
        check("re-ingest after edit updates content", "Edited for re-ingest test" in col.get(ids=[doc_id], include=["documents"])["documents"][0])
    finally:
        ms.purge(doc_id)
        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------- memory lifecycle

def test_prune_cold_revive_purge():
    section("prune() / recall_cold() / revive_from_cold() / purge()")
    doc_id = ms.jot("Regression fact: artificially aged to force prune()/cold-storage cycle.", topic_hint=REGRESSION_TOPIC_PREFIX)
    query = "regression fact artificially aged force prune cold-storage cycle"

    col = ms._get_collection()
    m = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
    m["last_accessed"] = time.time() - 60 * 86400  # 60 days ago, well past DELETION_FLOOR at 7-day stability
    col.update(ids=[doc_id], metadatas=[m])

    n = ms.prune()
    m2 = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
    check("prune() archives a decayed doc", n >= 1 and m2.get("archived") is True)

    hits = ms.recall(query, n_results=10)
    check("archived doc invisible to recall()", not any(h["id"] == doc_id for h in hits))
    assoc = ms.recall_associative(query, n_results=10, n_activated=10)
    check("archived doc invisible to recall_associative() (both tiers)",
          not any(h["id"] == doc_id for h in assoc["confident"] + assoc["activated"]))

    cold_hits = ms.recall_cold(query, n_results=5)
    check("archived doc findable via recall_cold()", any(h["id"] == doc_id for h in (cold_hits or [])))

    ok = ms.revive_from_cold(doc_id)
    m3 = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
    check("revive_from_cold() un-archives and resets last_accessed",
          ok and m3.get("archived") is False and (time.time() - m3["last_accessed"]) < 5)
    hits_after = ms.recall(query, n_results=10)
    check("revived doc visible to recall() again", any(h["id"] == doc_id for h in hits_after))

    ok_purge = ms.purge(doc_id)
    still_present = col.get(ids=[doc_id], include=["metadatas"])["ids"]
    check("purge() removes the doc completely", ok_purge and not still_present)
    check("purge() on a nonexistent id returns False, no crash", ms.purge("fragment/does-not-exist-regression") is False)


def test_consolidation():
    section("episodic -> semantic consolidation trigger")
    doc_id = ms.jot("Regression fact: engineered to just barely trigger episodic-to-semantic consolidation.", topic_hint=REGRESSION_TOPIC_PREFIX)
    col = ms._get_collection()
    m = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
    episodic_cap = ms.EPISODIC_BASE_STABILITY * ms.STABILITY_CAP_FACTOR
    m["stability"] = episodic_cap - 3 * 86400.0  # just under the cap; one more recall pushes it over
    col.update(ids=[doc_id], metadatas=[m])

    ms.recall("regression fact engineered to just barely trigger episodic-to-semantic consolidation", n_results=1)
    m2 = col.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
    check("consolidation flips memory_type to semantic at the episodic cap", m2["memory_type"] == "semantic")
    check("consolidation increments consolidation_level", m2["consolidation_level"] == 1)
    check("consolidation resets stability to SEMANTIC_BASE_STABILITY", m2["stability"] == ms.SEMANTIC_BASE_STABILITY)
    ms.purge(doc_id)


# ---------------------------------------------------------------- capture pipeline

def test_chunking_logic():
    section("auto_capture chunking — compact_boundary and char-budget paths")
    entries_boundary = [
        (1, {"type": "user", "message": {"content": "a"}}),
        (2, {"type": "system", "subtype": "compact_boundary", "content": "x"}),
        (3, {"type": "user", "message": {"content": "b"}}),
    ]
    chunks = ac._chunk_entries(entries_boundary)
    check("compact_boundary splits into separate chunks", len(chunks) == 2 and chunks[0][-1][0] == 1 and chunks[1][-1][0] == 3)

    big = "x" * 25000
    entries_budget = [
        (1, {"type": "user", "message": {"content": big}}),
        (2, {"type": "assistant", "message": {"content": big}}),
        (3, {"type": "user", "message": {"content": big}}),
    ]
    chunks2 = ac._chunk_entries(entries_budget)
    total_len = lambda c: sum(len(e.get("message", {}).get("content", "")) for _, e in c)
    check("char-budget splits when no compact_boundary present",
          len(chunks2) == 2 and all(total_len(c) <= ac.CHUNK_CHAR_BUDGET for c in chunks2))


def test_capture_state():
    section("capture_state.py — get_state/set_state/all_states")
    sid = "regression-test-state-session"
    check("get_state() on unseen session returns {}", cs.get_state(sid) == {})
    cs.set_state(sid, last_finalized_line=10, cwd="/tmp/foo")
    s1 = cs.get_state(sid)
    check("set_state() writes fields correctly", s1.get("last_finalized_line") == 10 and s1.get("cwd") == "/tmp/foo")
    cs.set_state(sid, last_finalized_line=20)
    s2 = cs.get_state(sid)
    check("set_state() merge-updates without clobbering other fields", s2.get("last_finalized_line") == 20 and s2.get("cwd") == "/tmp/foo")
    check("all_states() includes the session", sid in cs.all_states())
    _cleanup_state_session(sid)


def test_capture_extraction_live():
    section("auto_capture end-to-end (live claude -p) — chunking, checkpointing, idempotency")
    tmpdir = Path(tempfile.mkdtemp(prefix="regression-capture-"))
    transcript = tmpdir / "fake.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"content":"Regression: always run tests with pytest -x, stop at first failure."},"cwd":"/tmp/regressioncapturetest"}\n'
        '{"type":"assistant","message":{"content":"Noted."},"cwd":"/tmp/regressioncapturetest"}\n'
        '{"type":"system","subtype":"compact_boundary","content":"x"}\n'
        '{"type":"user","message":{"content":"Regression: never deploy on Fridays, a past deploy caused a weekend outage."},"cwd":"/tmp/regressioncapturetest"}\n'
        '{"type":"assistant","message":{"content":"Noted."},"cwd":"/tmp/regressioncapturetest"}\n'
    )
    sid = "regression-test-capture-session"
    try:
        n1 = ac.capture_sessions_parallel([(sid, str(transcript))])
        state1 = cs.get_state(sid)
        check("capture_sessions_parallel() advances checkpoint to end of file", state1.get("last_finalized_line") == 5)
        check("capture_sessions_parallel() attributes cwd correctly", state1.get("cwd") == "/tmp/regressioncapturetest")

        t0 = time.time()
        n2 = ac.capture_sessions_parallel([(sid, str(transcript))])
        elapsed = time.time() - t0
        check("re-running on unchanged content is an idempotent no-op", n2 == 0 and elapsed < 1.0, f"{elapsed:.2f}s")

        _purge_jots_from_session_topic("regressioncapturetest")
    finally:
        _cleanup_state_session(sid)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------- hooks

def test_orphan_sweep_and_liveness():
    section("session_start_backstop.py — orphan sweep logic + lsof liveness check")
    import session_start_backstop as backstop

    # liveness check, against a file we genuinely hold open vs. released
    tmpdir = Path(tempfile.mkdtemp(prefix="regression-liveness-"))
    f = tmpdir / "held.txt"
    f.write_text("x")
    import subprocess
    proc = subprocess.Popen(["tail", "-f", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    check("_is_open_by_a_process() True while genuinely held open", backstop._is_open_by_a_process(f))
    proc.terminate()
    proc.wait()
    time.sleep(0.3)
    check("_is_open_by_a_process() False after release", not backstop._is_open_by_a_process(f))

    # find_orphans() logic, against an isolated fake PROJECTS_DIR (not the real
    # machine history -- keeps this fast and deterministic regardless of how many
    # real transcripts have ever existed; the real machine's per-call lsof cost is
    # a separately-documented, already-flagged finding, not re-measured every run)
    fake_projects = tmpdir / "projects"
    fake_project_dir = fake_projects / "-fake-project"
    fake_project_dir.mkdir(parents=True)

    current_sid = "current-session-excluded"
    orphan_sid = "genuine-orphan"
    processed_sid = "already-processed"

    (fake_project_dir / f"{current_sid}.jsonl").write_text('{"type":"user","message":{"content":"x"}}\n')
    (fake_project_dir / f"{orphan_sid}.jsonl").write_text('{"type":"user","message":{"content":"x"}}\n')
    (fake_project_dir / f"{processed_sid}.jsonl").write_text('{"type":"user","message":{"content":"x"}}\n')

    def fake_get_state(sid):
        if sid == processed_sid:
            return {"last_finalized_line": 999}  # already covers everything -> not an orphan
        return {}

    orig_projects_dir = backstop.PROJECTS_DIR
    backstop.PROJECTS_DIR = fake_projects
    try:
        orphans = backstop.find_orphans(current_sid, fake_get_state)
        orphan_sids = {sid for _, sid in orphans}
        check("find_orphans() excludes the current session", current_sid not in orphan_sids)
        check("find_orphans() excludes an already-fully-processed session", processed_sid not in orphan_sids)
        check("find_orphans() finds a genuine orphan", orphan_sid in orphan_sids)
    finally:
        backstop.PROJECTS_DIR = orig_projects_dir
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_session_end_capture_live():
    section("session_end_capture.py — direct invocation (live claude -p)")
    import subprocess
    tmpdir = Path(tempfile.mkdtemp(prefix="regression-sessionend-"))
    transcript = tmpdir / "fake.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"content":"Regression: the deploy script must always run with --dry-run first, a past deploy without it wiped a shared staging config."},"cwd":"/tmp/regressionsessionendtest"}\n'
        '{"type":"assistant","message":{"content":"Understood."},"cwd":"/tmp/regressionsessionendtest"}\n'
    )
    sid = "regression-test-sessionend-session"
    payload = json.dumps({"session_id": sid, "transcript_path": str(transcript)})
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "hooks" / "session_end_capture.py")],
            input=payload, capture_output=True, text=True, timeout=270,
        )
        check("session_end_capture.py exits 0", proc.returncode == 0, proc.stderr[-300:] if proc.returncode else "")
        state = cs.get_state(sid)
        check("session_end_capture.py advances checkpoint", state.get("last_finalized_line") == 2)
        _purge_jots_from_session_topic("regressionsessionendtest")
    finally:
        _cleanup_state_session(sid)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_recall_hook():
    section("recall_hook.py — UserPromptSubmit hook")
    import subprocess

    real_cache_path = ROOT / ".recall_session_cache"
    real_cache_backup = real_cache_path.read_text() if real_cache_path.exists() else None

    marker_id = ms.jot("Regression recall_hook marker: zzzrecallhookmarkerzzz unique phrase for hook test.", topic_hint=REGRESSION_TOPIC_PREFIX)
    _created_ids.append(marker_id)

    try:
        def run_hook(session_id: str, prompt: str) -> str:
            payload = json.dumps({"session_id": session_id, "prompt": prompt, "cwd": "/tmp/regressionhooktest"})
            proc = subprocess.run(
                [sys.executable, str(ROOT / "recall_hook.py")],
                input=payload, capture_output=True, text=True, timeout=30,
            )
            return proc.stdout.strip()

        out1 = run_hook("regression-test-hook-session-1", "zzzrecallhookmarkerzzz unique phrase for hook test")
        check("recall_hook.py produces hookSpecificOutput for a matching prompt", '"hookSpecificOutput"' in out1)

        out2 = run_hook("regression-test-hook-session-1", "zzzrecallhookmarkerzzz unique phrase for hook test")
        check("recall_hook.py session cache blocks a repeat call in the same session", out2 == "")

        out3 = run_hook("regression-test-hook-session-2", "zzzrecallhookmarkerzzz unique phrase for hook test")
        check("recall_hook.py allows a fresh session_id to produce output again", '"hookSpecificOutput"' in out3)

        try:
            from local_config import EXCLUDE_TOPICS
        except ImportError:
            EXCLUDE_TOPICS = set()
        if EXCLUDE_TOPICS:
            excluded_topic = next(iter(EXCLUDE_TOPICS))
            col = ms._get_collection()
            all_entries = col.get(include=["metadatas"])
            has_excluded_content = any(m.get("topic") == excluded_topic and not m.get("archived") for m in all_entries["metadatas"])
            if has_excluded_content:
                sample = next(m for m in all_entries["metadatas"] if m.get("topic") == excluded_topic and not m.get("archived"))
                out4 = run_hook("regression-test-hook-session-3", sample["title"])
                check(f"recall_hook.py EXCLUDE_TOPICS hides '{excluded_topic}' content", excluded_topic not in out4)
            else:
                skip("recall_hook.py EXCLUDE_TOPICS check", f"no non-archived '{excluded_topic}' content currently in the corpus to test against")
        else:
            skip("recall_hook.py EXCLUDE_TOPICS check", "local_config.py defines no EXCLUDE_TOPICS on this machine")
    finally:
        if real_cache_backup is not None:
            real_cache_path.write_text(real_cache_backup)
        elif real_cache_path.exists():
            real_cache_path.unlink()


def test_incremental_backlink():
    section("incremental_backlink.py — PostToolUse hook")
    import subprocess
    dir_a = ms.CORPUS_DIR / "regression-test-backlink-a"
    dir_b = ms.CORPUS_DIR / "regression-test-backlink-b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    shared_text = (
        "This disposable test document discusses the zzzbacklinktestmarkerzzz unique "
        "cross-topic linking scenario with sufficient shared vocabulary overlap "
        "zzzbacklinktestmarkerzzz cross-topic linking scenario shared vocabulary "
        "overlap zzzbacklinktestmarkerzzz cross-topic linking vocabulary overlap test."
    )
    doc_a = dir_a / "doc_a.md"
    doc_b = dir_b / "doc_b.md"
    doc_a.write_text(f"# Regression Backlink Test Doc A\n\n{shared_text}\n")
    doc_b.write_text(f"# Regression Backlink Test Doc B\n\n{shared_text}\n")
    doc_b_id = "regression-test-backlink-b/doc_b"

    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "incremental_backlink.py"), str(doc_b)],
            capture_output=True, text=True, timeout=60,
        )
        check("incremental_backlink.py exits cleanly", proc.returncode == 0, proc.stderr[-300:] if proc.returncode else "")

        col = ms._get_collection()
        found = col.get(ids=[doc_b_id], include=["metadatas"])["ids"]
        check("incremental_backlink.py ingests the target file", bool(found))

        check("target doc gets a Related section linking the partner", "doc_a.md" in doc_b.read_text())
        check("partner doc gets a reciprocal Related link", "doc_b.md" in doc_a.read_text())
    finally:
        ms.purge(doc_b_id)
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)


# ---------------------------------------------------------------- secondary tools

def test_secondary_tools():
    section("secondary tools — sanity")
    import subprocess
    proc = subprocess.run([sys.executable, str(ROOT / "backlinks.py")], capture_output=True, text=True, timeout=60)
    check("backlinks.py (read-only diagnostic) runs clean against the real corpus", proc.returncode == 0, proc.stderr[-300:] if proc.returncode else "")

    classify_used = False
    for py_file in ROOT.glob("*.py"):
        if py_file.name in ("classify.py", "regression_test.py"):
            continue
        text = py_file.read_text(errors="ignore")
        if "import classify" in text or "from classify" in text:
            classify_used = True
    check("classify.py confirmed not imported anywhere live (known prototype)", not classify_used)


# ---------------------------------------------------------------- cleanup helpers

def _cleanup_state_session(sid: str) -> None:
    state_path = cs.STATE_PATH
    try:
        data = json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if sid in data:
        del data[sid]
        state_path.write_text(json.dumps(data, indent=2))


def _purge_jots_from_session_topic(topic: str) -> None:
    col = ms._get_collection()
    all_entries = col.get(include=["metadatas"])
    for doc_id, m in zip(all_entries["ids"], all_entries["metadatas"]):
        if m.get("topic") == topic:
            ms.purge(doc_id)


def final_sweep() -> int:
    """Purge anything left over under the regression-test topic namespace, plus
    anything explicitly tracked in _created_ids -- a real safety net, not just
    trusting the per-test cleanup happened (a crash mid-test would skip it)."""
    col = ms._get_collection()
    all_entries = col.get(include=["metadatas"])
    leftover = [
        doc_id for doc_id, m in zip(all_entries["ids"], all_entries["metadatas"])
        if m.get("topic", "").startswith(REGRESSION_TOPIC_PREFIX) or doc_id in _created_ids
    ]
    for doc_id in leftover:
        ms.purge(doc_id)
    for sid in ("regression-test-state-session", "regression-test-capture-session", "regression-test-sessionend-session"):
        _cleanup_state_session(sid)
    return len(leftover)


def write_report(mode: str, elapsed: float, n_swept: int) -> Path:
    passed = sum(1 for _, ok, _ in results if ok is True)
    failed = [(n, d) for n, ok, d in results if ok is False]
    skipped = [(n, d) for n, ok, d in results if ok is None]

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    report_path = REPORTS_DIR / f"{stamp}_{mode}.md"

    lines = [
        f"# Regression report — {mode} run",
        "",
        f"- **When**: {stamp}",
        f"- **Result**: {passed} passed, {len(failed)} failed, {len(skipped)} skipped — {elapsed:.1f}s",
        f"- **Cleanup**: final sweep purged {n_swept} leftover doc(s)",
        "",
    ]

    current_group = None
    for name, ok, detail in results:
        status = {"True": "PASS", "False": "FAIL", "None": "SKIP"}[str(ok)]
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
        lines.append(f"- `{mark}` **{status}** — {name}" + (f" ({detail})" if detail else ""))

    if failed:
        lines += ["", "## Failures", ""]
        for name, detail in failed:
            lines.append(f"- {name}" + (f" — {detail}" if detail else ""))

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slow/costly checks that require a live claude -p call")
    args = parser.parse_args()

    t0 = time.time()
    print(f"memory-project regression test — {'quick' if args.quick else 'full'} run\n")

    try:
        fact_id = test_jot_variants()
        test_recall(fact_id)
        test_recall_associative()
        test_ingest()
        test_prune_cold_revive_purge()
        test_consolidation()
        test_chunking_logic()
        test_capture_state()
        test_orphan_sweep_and_liveness()
        test_incremental_backlink()
        test_secondary_tools()

        if args.quick:
            skip("auto_capture end-to-end (live claude -p)", "--quick")
            skip("session_end_capture.py direct invocation (live claude -p)", "--quick")
        else:
            test_capture_extraction_live()
            test_session_end_capture_live()

        test_recall_hook()  # local-only (recall() under the hood), runs in both modes
    finally:
        n_swept = final_sweep()

    elapsed = time.time() - t0
    passed = sum(1 for _, ok, _ in results if ok is True)
    failed = [(n, d) for n, ok, d in results if ok is False]
    skipped = sum(1 for _, ok, _ in results if ok is None)

    print(f"\n{'='*60}")
    print(f"{passed} passed, {len(failed)} failed, {skipped} skipped — {elapsed:.1f}s")
    print(f"final sweep purged {n_swept} leftover doc(s)")
    if failed:
        print("\nFAILURES:")
        for name, detail in failed:
            print(f"  - {name}" + (f" ({detail})" if detail else ""))
    print("=" * 60)

    report_path = write_report("quick" if args.quick else "full", elapsed, n_swept)
    print(f"report written to {report_path.relative_to(ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
