#!/usr/bin/env python3
"""
ChromaDB-backed memory store for Claude Code session summaries.

Public API:
  ingest(path)              — add or update a memory from a .md file (curated tier)
  jot(text, session_id)     — cheap unfiled capture for mid-session fragments (fragment tier)
  recall(text)              — retrieve relevant memories, log access, update stability
  recall_associative(text)  — two-tier "rings a bell" recall: confident hits (reinforced,
                               state as fact) + weak activated hits (not reinforced, hedge
                               and let Claude judge relevance) — see confirm_activation()
  confirm_activation(id)    — call after the user affirms an activated-tier hit; bigger
                               reinforcement boost than an ordinary recall (desirable-
                               difficulty effect). Do nothing if the user denies/corrects it.
  prune()                    — ARCHIVE (not delete) memories whose raw strength has dropped
                               below DELETION_FLOOR into cold storage: excluded from recall()/
                               recall_associative() entirely, but not erased — models human
                               "I haven't thought about that in 40 years!" memory rather than
                               true forgetting. See recall_cold()/revive_from_cold().
  recall_cold(text)          — search ONLY archived/cold memories, by raw similarity (not
                               decayed strength) against a much stricter bar than ordinary
                               recall — a cold memory needs a genuinely specific cue to
                               resurface, not just a loose association.
  revive_from_cold(id)       — call after the user confirms a recall_cold() hit is genuinely
                               the thing being recalled; un-archives it and refreshes it to
                               its memory-type's base stability, active again in ordinary recall.
  purge(id)                  — TRUE permanent deletion, bypassing cold storage entirely. For
                               something that should never have been recorded, not routine
                               cleanup — prune() is routine cleanup now, purge() is not.
"""

from __future__ import annotations

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import math
import re
import subprocess
import time
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_DIR   = Path(__file__).resolve().parent / "session_summaries"
DB_DIR       = Path(__file__).resolve().parent / ".chromadb"
ACTIVITY_LOG = Path(__file__).resolve().parent / "activity.log"
MODEL_NAME   = "all-MiniLM-L6-v2"
COLLECTION   = "memories"

# Set on every `claude -p` subprocess this project launches (auto_capture.py's
# extraction calls, and any future callers of the same pattern). `claude -p` is a
# full Claude Code invocation, subject to the SAME global SessionStart/SessionEnd
# hooks as any real session -- without this guard, each extraction call re-triggers
# session_start_backstop.py, including its orphan sweep, which can launch MORE
# extraction subprocesses, which trigger the hook again, recursively. Found
# 2026-07-31: an idle first-compaction catch-up spun up a growing tree of nested
# claude -p / hook processes (confirmed via `ps` parent-chain tracing) that had to
# be killed by hand. The hook scripts check for this var and no-op immediately.
NESTED_EXTRACTION_ENV_VAR = "MEMORY_PROJECT_NESTED_EXTRACTION"

EPISODIC_BASE_STABILITY = 7  * 86400.0   # 7 days in seconds
SEMANTIC_BASE_STABILITY = 90 * 86400.0   # 90 days in seconds
CURATED_INITIAL_STABILITY = 30 * 86400.0 # 30 days -- starting stability for a brand-new
                                          # ingest()'d (curated) memory. Deliberately between
                                          # EPISODIC_BASE_STABILITY and SEMANTIC_BASE_STABILITY:
                                          # a curated .md file was written up on purpose (unlike
                                          # a jot() fragment, whose whole design bar is "cheap,
                                          # low-effort, expected to decay unless reinforced" --
                                          # see CLAUDE.md), so it shouldn't start on the same
                                          # 7-day clock as a passing mention, but it also hasn't
                                          # earned the 90-day semantic tier through actual
                                          # reinforcement yet -- that's still consolidation's job.
                                          # memory_type still starts "episodic" either way; this
                                          # only changes the starting point on that clock, not
                                          # the growth/cap/consolidation math.
STABILITY_GROWTH        = 1.5            # multiplier on stability per recall
STABILITY_CAP_FACTOR    = 10.0           # stability capped at base * this
RETRIEVAL_FLOOR         = 0.1            # minimum strength to appear in recall()
DELETION_FLOOR          = 0.02           # minimum strength before prune() archives (not deletes) it
TOPIC_BOOST_FACTOR      = 1.5            # score multiplier for hits matching topic_hint
COLD_REVIVAL_THRESHOLD  = 0.6            # raw similarity (not strength-weighted) needed for a cold/
                                          # archived memory to surface via recall_cold() at all. An
                                          # archived memory's decayed strength is frozen and meaningless
                                          # as a ranking signal (it would never recover on its own), so
                                          # this bypasses decay entirely and requires a genuinely
                                          # specific, strong match instead of a loose association --
                                          # modeling that a long-dormant memory takes a real cue to
                                          # resurface, triggered by something CURRENT (whatever's live
                                          # in the conversation, i.e. the query text), not by another
                                          # stored memory cross-activating it.

_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _raw_strength(last_accessed: float, stability: float) -> float:
    return math.exp(-(time.time() - last_accessed) / stability)


def _grow_stability(current: float, memory_type: str) -> float:
    base = EPISODIC_BASE_STABILITY if memory_type == "episodic" else SEMANTIC_BASE_STABILITY
    return min(current * STABILITY_GROWTH, base * STABILITY_CAP_FACTOR)


def _maybe_consolidate(memory_type: str, consolidation_level: int, new_stability: float) -> dict:
    """
    Promote episodic -> semantic once reinforcement has grown a memory's
    stability up to the episodic cap — further episodic recalls can't grow
    it past that ceiling, so consolidation is exactly what continued
    reinforcement should unlock next. One-way: semantic memories never
    demote back to episodic (a semantic memory that stops being useful
    still decays via strength and can still be pruned, just on the slower
    90-day-based clock).
    """
    if memory_type != "episodic":
        return {}
    episodic_cap = EPISODIC_BASE_STABILITY * STABILITY_CAP_FACTOR
    if new_stability < episodic_cap:
        return {}
    return {
        "memory_type":         "semantic",
        "stability":           SEMANTIC_BASE_STABILITY,
        "consolidation_level": consolidation_level + 1,
    }


def _title_of(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else "Untitled"


def _topic_of(path: Path) -> str:
    rel = path.relative_to(CORPUS_DIR)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def _list_topics() -> set[str]:
    """
    Candidate topics for _match_topic(): curated session_summaries/ folders
    plus any topic already present in the store (so a project that only has
    jot() fragments — no curated folder yet — still becomes matchable once
    it has one fragment, not just after someone writes it up as a .md).
    """
    topics = set()
    if CORPUS_DIR.exists():
        topics.update(p.name for p in CORPUS_DIR.iterdir() if p.is_dir())

    col = _get_collection()
    if col.count() > 0:
        metas = col.get(include=["metadatas"])["metadatas"]
        topics.update(
            m["topic"] for m in metas if m.get("topic") not in (None, "unfiled", "root")
        )
    return topics


def _match_topic(hint: str) -> str | None:
    """
    Fuzzy-match a cwd basename (e.g. "smis-gitops") against known topic
    folders (e.g. "smis") so recall() can boost same-project memories even
    when the prompt text itself gives no semantic signal (e.g. "hi claude").
    """
    hint_norm = re.sub(r"[^a-z0-9]", "", hint.lower())
    if not hint_norm:
        return None

    best = None
    for topic in _list_topics():
        topic_norm = re.sub(r"[^a-z0-9]", "", topic.lower())
        if topic_norm and (topic_norm == hint_norm or topic_norm in hint_norm or hint_norm in topic_norm):
            if best is None or len(topic_norm) > len(best[1]):
                best = (topic, topic_norm)
    return best[0] if best else None


def _doc_id(path: Path) -> str:
    return path.relative_to(CORPUS_DIR).with_suffix("").as_posix()


def _log_activity(action: str, doc_id: str, topic: str, title: str) -> None:
    """Plain-text append-only log so activity is visible without querying ChromaDB."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}  {action:6s} {topic:15s} {title[:60]}  ({doc_id})\n"
    with open(ACTIVITY_LOG, "a") as f:
        f.write(line)


def ingest(path: "Path | str") -> None:
    """
    Add or update a memory from a session summary .md file.
    Re-ingesting an edited file preserves existing access metadata (and
    memory_type/consolidation_level) so a content update doesn't reset the
    memory's accumulated strength or silently demote a consolidated memory
    back to episodic.

    A brand-new entry starts at CURATED_INITIAL_STABILITY (30 days), not
    EPISODIC_BASE_STABILITY (7 days) -- a curated .md file was deliberately
    written up, unlike a jot() fragment, so it shouldn't decay on the same
    clock as a cheap unfiled mention. Still starts memory_type="episodic";
    consolidation to the 90-day semantic tier still has to be earned via
    reinforcement, same as any other episodic memory.
    """
    path = Path(path).resolve()
    if not path.exists() or path.suffix != ".md" or CORPUS_DIR not in path.parents:
        return

    text      = path.read_text(errors="ignore")
    doc_id    = _doc_id(path)
    embedding = _get_model().encode(text, normalize_embeddings=True).tolist()

    col = _get_collection()
    now = time.time()

    existing = col.get(ids=[doc_id], include=["metadatas"])
    if existing["ids"]:
        m                   = existing["metadatas"][0]
        created_at          = m["created_at"]
        last_accessed       = m["last_accessed"]
        access_count        = m["access_count"]
        stability           = m["stability"]
        memory_type         = m["memory_type"]
        consolidation_level = m["consolidation_level"]
    else:
        created_at          = now
        last_accessed       = now
        access_count        = 0
        stability           = CURATED_INITIAL_STABILITY
        memory_type         = "episodic"
        consolidation_level = 0

    col.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "file_path":           str(path),
            "title":               _title_of(text),
            "topic":               _topic_of(path),
            "memory_type":         memory_type,
            "consolidation_level": consolidation_level,
            "capture_tier":        "curated",
            "source_ids":          "",
            "created_at":          created_at,
            "last_accessed":       last_accessed,
            "access_count":        access_count,
            "stability":           stability,
        }],
    )
    _log_activity("ingest", doc_id, _topic_of(path), _title_of(text))


def jot(text: str, session_id: str = "", topic_hint: str | None = None, category: str | None = None) -> str:
    """
    Cheap, unfiled capture for mid-session fragments — no backing file, no
    classify.py filing, no incremental_backlink.py cross-topic pass (too
    expensive to run on every jot; reserved for the curated tier). Embeds
    straight into the store and is retrievable immediately via recall(),
    same as any other memory — decay, reinforcement, and consolidation all
    apply identically. Promote to a proper session summary later (through
    the file-based ingest() pipeline) if something here earns it.

    topic_hint defaults to cwd's basename (the project jot() is called from).
    It's resolved the same way recall()'s topic_hint is: matched against an
    existing curated/fragment topic if one fits, else stored as-is. Either
    way the fragment gets a real topic instead of "unfiled" so it can
    participate in recall()'s cwd-based topic boost — including for a
    project that has no curated session_summaries/ folder yet, since
    _list_topics() also looks at topics already in the store.

    category (added 2026-07-31, chain item 11): optional tag stored as real
    metadata, not inferred from text content. "feedback" is the first
    recognized value — lets find_feedback_patterns() query precisely for
    feedback-type memories instead of guessing from a "Feedback (date): ..."
    text-prefix convention, which is fragile for anything that needs to
    filter on it reliably. None (the default) means untagged, same as before
    this parameter existed.

    Returns the generated doc_id.
    """
    if topic_hint is None:
        topic_hint = Path(os.getcwd()).name
    topic = _match_topic(topic_hint) or topic_hint or "unfiled"

    doc_id    = f"fragment/{uuid.uuid4().hex}"
    title     = text.strip().split("\n", 1)[0][:80]
    embedding = _get_model().encode(text, normalize_embeddings=True).tolist()

    col = _get_collection()
    now = time.time()

    col.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "file_path":           "",
            "title":               title,
            "topic":               topic,
            "memory_type":         "episodic",
            "consolidation_level": 0,
            "capture_tier":        "fragment",
            "session_id":          session_id,
            "source_ids":          "",
            "category":            category or "",
            "created_at":          now,
            "last_accessed":       now,
            "access_count":        0,
            "stability":           EPISODIC_BASE_STABILITY,
        }],
    )
    _log_activity("jot", doc_id, topic, title)
    return doc_id


def _score_hits(
    query: str,
    fetch_floor: int,
    exclude_topics: set[str],
    topic_hint: str | None,
) -> list[dict]:
    """
    Shared embed + fetch + score logic used by both recall() and
    recall_associative(). Returns every hit above RETRIEVAL_FLOOR, sorted by
    score descending, each still carrying its raw "_meta" for the caller to
    decide what to do with (reinforce, or not). fetch_floor is the minimum
    number of candidates to pull from ChromaDB before scoring/filtering —
    callers that need a wider net (e.g. recall_associative(), which slices
    two separate tiers out of the same pool) should pass a larger value than
    a caller that only wants a handful of top hits.
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    matched_topic = _match_topic(topic_hint) if topic_hint else None
    embedding = _get_model().encode(query, normalize_embeddings=True).tolist()

    fetch_n = min(col.count(), max(fetch_floor, 20))
    raw = col.query(
        query_embeddings=[embedding],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc_id, doc, meta, distance in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        if meta["topic"] in exclude_topics:
            continue
        if meta.get("archived"):
            continue  # cold storage -- excluded from ordinary recall entirely, see recall_cold()

        # ChromaDB cosine distance = 1 - cosine_similarity for normalized vectors
        similarity = 1.0 - distance
        strength   = _raw_strength(meta["last_accessed"], meta["stability"])

        if strength < RETRIEVAL_FLOOR:
            continue

        score = similarity * strength
        if matched_topic and meta["topic"] == matched_topic:
            score *= TOPIC_BOOST_FACTOR

        hits.append({
            "id":          doc_id,
            "document":    doc,
            "title":       meta["title"],
            "topic":       meta["topic"],
            "memory_type": meta["memory_type"],
            "file_path":   meta["file_path"],
            "strength":    round(strength, 4),
            "similarity":  round(similarity, 4),
            "score":       round(score, 4),
            "_meta":       meta,
        })

    hits.sort(key=lambda h: -h["score"])
    return hits


def _reinforce(hits: list[dict], col: "chromadb.Collection", boost: float) -> None:
    """
    Shared reinforcement logic: grows stability by `boost` (capped the same
    way _grow_stability's own multiplier would be), updates last_accessed/
    access_count, runs consolidation check. Pops "_meta" off each hit dict
    in place (same contract recall() always had — hits returned to the
    caller don't carry the raw metadata).
    """
    now = time.time()
    for hit in hits:
        m = hit.pop("_meta")
        base = EPISODIC_BASE_STABILITY if m["memory_type"] == "episodic" else SEMANTIC_BASE_STABILITY
        new_stability = min(m["stability"] * boost, base * STABILITY_CAP_FACTOR)
        updated = {
            **m,
            "last_accessed": now,
            "access_count":  m["access_count"] + 1,
            "stability":     new_stability,
        }
        updated.update(_maybe_consolidate(m["memory_type"], m["consolidation_level"], new_stability))
        col.update(ids=[hit["id"]], metadatas=[updated])


def recall(
    query: str,
    n_results: int = 5,
    exclude_topic: str | set[str] | None = None,
    topic_hint: str | None = None,
) -> list[dict]:
    """
    Retrieve memories semantically relevant to query, ranked by similarity * strength.
    Updates last_accessed, access_count, and stability for every returned hit —
    this is the only code path that should increment those fields (besides
    recall_associative()'s confident tier and confirm_activation() — see below).

    topic_hint (e.g. the launch cwd's basename) is fuzzy-matched against known
    topic folders; hits from the matched topic get their score boosted by
    TOPIC_BOOST_FACTOR, so being in a project's directory surfaces its memories
    even when the prompt text alone carries no semantic signal.

    exclude_topic accepts a single topic or a set of topics to filter out
    entirely (e.g. recall_hook.py's EXCLUDE_TOPICS, for topics that shouldn't
    be surfaced via the automatic path — see that module for why).
    """
    col = _get_collection()
    exclude_topics = {exclude_topic} if isinstance(exclude_topic, str) else (exclude_topic or set())
    hits = _score_hits(query, n_results * 4, exclude_topics, topic_hint)[:n_results]
    _reinforce(hits, col, STABILITY_GROWTH)
    return hits


CONFIDENT_THRESHOLD      = 0.35   # matches recall_hook.py's own bar for a hit worth stating directly
ACTIVATION_FLOOR         = 0.15   # below this: too weak to even hedge about, treated as noise
ACTIVATION_STABILITY_BOOST = 2.5  # vs. STABILITY_GROWTH (1.5) for an ordinary recall hit -- a confirmed
                                   # "rings a bell" memory was retrieved under real uncertainty, and
                                   # effortful/uncertain retrieval that turns out correct reinforces a
                                   # memory more than an easy, confident one does (the "desirable
                                   # difficulty" / testing-effect finding from memory research) -- so
                                   # confirmation deliberately gets a bigger boost than passive re-recall.


def recall_associative(
    query: str,
    n_results: int = 5,
    n_activated: int = 3,
    exclude_topic: str | set[str] | None = None,
    topic_hint: str | None = None,
) -> dict[str, list[dict]]:
    """
    Two-tier "rings a bell" recall, layered on top of the same scoring recall() uses.

    - `confident` (score >= CONFIDENT_THRESHOLD): behaves exactly like recall() —
      reinforced immediately at the normal rate, safe to state as fact.
    - `activated` (ACTIVATION_FLOOR <= score < CONFIDENT_THRESHOLD): weak,
      uncertain associative matches — NOT auto-reinforced. These are candidates
      for Claude to apply judgment to: only surface the ones that actually seem
      relevant to the conversation, using hedging language ("this might be
      related...", "does this ring a bell...") rather than stating them as fact.

    If the user affirms an `activated` hit, call confirm_activation(doc_id) to
    apply the larger confirmation boost. If they deny or correct it, do nothing
    — no function call, no stability change; an unconfirmed weak activation
    should neither strengthen nor weaken on its own.
    """
    col = _get_collection()
    exclude_topics = {exclude_topic} if isinstance(exclude_topic, str) else (exclude_topic or set())
    fetch_floor = max(n_results, n_activated) * 4
    all_hits = _score_hits(query, fetch_floor, exclude_topics, topic_hint)

    confident = [h for h in all_hits if h["score"] >= CONFIDENT_THRESHOLD][:n_results]
    activated = [h for h in all_hits if ACTIVATION_FLOOR <= h["score"] < CONFIDENT_THRESHOLD][:n_activated]

    _reinforce(confident, col, STABILITY_GROWTH)
    # activated hits keep their "_meta" popped off for a consistent return shape,
    # but deliberately get NO reinforcement here — only confirm_activation() does.
    for hit in activated:
        hit.pop("_meta", None)

    return {"confident": confident, "activated": activated}


def confirm_activation(doc_id: str) -> bool:
    """
    Call after the user affirms a 'rings a bell' hit surfaced via
    recall_associative()'s `activated` tier. Applies ACTIVATION_STABILITY_BOOST
    (bigger than an ordinary recall's STABILITY_GROWTH) — see the note on
    ACTIVATION_STABILITY_BOOST above for why a confirmed uncertain guess should
    reinforce more than an easy, confident retrieval.

    Do NOT call this if the user denies or corrects the guess — the fragment
    quietly declines to reinforce; there is no separate "punish" path per the
    original design ("just do nothing with it if denied or corrected").

    Returns True if the memory was found and reinforced, False if doc_id
    doesn't exist (e.g. already pruned).
    """
    col = _get_collection()
    existing = col.get(ids=[doc_id], include=["metadatas"])
    if not existing["ids"]:
        return False

    m = existing["metadatas"][0]
    now = time.time()
    base = EPISODIC_BASE_STABILITY if m["memory_type"] == "episodic" else SEMANTIC_BASE_STABILITY
    new_stability = min(m["stability"] * ACTIVATION_STABILITY_BOOST, base * STABILITY_CAP_FACTOR)
    updated = {
        **m,
        "last_accessed": now,
        "access_count":  m["access_count"] + 1,
        "stability":     new_stability,
    }
    updated.update(_maybe_consolidate(m["memory_type"], m["consolidation_level"], new_stability))
    col.update(ids=[doc_id], metadatas=[updated])
    _log_activity("confirm", doc_id, m["topic"], m["title"])
    return True


def prune() -> int:
    """
    ARCHIVE (not delete) entries whose raw strength has decayed below DELETION_FLOOR —
    models human "cold storage" rather than true forgetting: excluded from recall()/
    recall_associative() entirely (you don't stumble on these in everyday thinking), but
    the embedding and content are kept, not erased. A sufficiently specific, strong cue
    can still resurface one via recall_cold() + revive_from_cold() — the "I haven't
    thought about that in 40 years!" phenomenon, rather than a permanent loss.
    Already-archived entries are left alone (idempotent). For true, deliberate permanent
    deletion, see purge() — that's a rare/manual action, not what routine cleanup does.
    Returns the count of entries newly archived this call.
    """
    col = _get_collection()
    if col.count() == 0:
        return 0

    all_entries = col.get(include=["metadatas"])
    to_archive = [
        (doc_id, meta)
        for doc_id, meta in zip(all_entries["ids"], all_entries["metadatas"])
        if not meta.get("archived") and _raw_strength(meta["last_accessed"], meta["stability"]) < DELETION_FLOOR
    ]

    if to_archive:
        now = time.time()
        col.update(
            ids=[doc_id for doc_id, _ in to_archive],
            metadatas=[{**meta, "archived": True, "archived_at": now} for _, meta in to_archive],
        )
        for doc_id, meta in to_archive:
            _log_activity("archive", doc_id, meta["topic"], meta["title"])

    return len(to_archive)


def purge(doc_id: str) -> bool:
    """
    TRUE permanent deletion, bypassing cold storage entirely — removes a memory from the
    store with no way to recall_cold() it back. Use only for something that should never
    have been recorded in the first place (e.g. accidentally jotted sensitive content),
    not as routine cleanup — prune() is routine cleanup now, this is not.
    Returns True if found and deleted, False if doc_id doesn't exist.
    """
    col = _get_collection()
    existing = col.get(ids=[doc_id], include=["metadatas"])
    if not existing["ids"]:
        return False
    meta = existing["metadatas"][0]
    col.delete(ids=[doc_id])
    _log_activity("purge", doc_id, meta["topic"], meta["title"])
    return True


def recall_cold(query: str, n_results: int = 3) -> list[dict]:
    """
    Search ONLY archived/cold-storage memories, ranked by raw similarity rather than the
    usual similarity * strength score — an archived memory's decayed strength is frozen
    and not a meaningful ranking signal (it would never recover on its own). Only returns
    hits >= COLD_REVIVAL_THRESHOLD (0.6), deliberately much stricter than the ordinary
    confident-recall bar (0.35): resurfacing something long-dormant needs a genuinely
    specific match, not a loose association. Triggered by whatever's CURRENT in the
    conversation (the query text) — this is not one memory cross-activating another.

    Does not reinforce or revive anything by itself — a hit here is just a candidate.
    Call revive_from_cold(doc_id) if the user confirms it's genuinely the thing they were
    trying to recall.
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    embedding = _get_model().encode(query, normalize_embeddings=True).tolist()
    fetch_n = min(col.count(), max(n_results * 4, 20))
    raw = col.query(
        query_embeddings=[embedding],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc_id, doc, meta, distance in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        if not meta.get("archived"):
            continue
        similarity = 1.0 - distance
        if similarity < COLD_REVIVAL_THRESHOLD:
            continue
        hits.append({
            "id":          doc_id,
            "document":    doc,
            "title":       meta["title"],
            "topic":       meta["topic"],
            "similarity":  round(similarity, 4),
            "archived_at": meta.get("archived_at"),
        })

    hits.sort(key=lambda h: -h["similarity"])
    return hits[:n_results]


def revive_from_cold(doc_id: str) -> bool:
    """
    Call after the user confirms a recall_cold() hit is genuinely the thing they were
    trying to recall. Un-archives it and refreshes stability to its memory_type's base
    (EPISODIC_BASE_STABILITY or SEMANTIC_BASE_STABILITY, whichever tier it was in when
    archived — revival doesn't reset it back to episodic if it had consolidated before
    going cold) — mirroring how a suddenly-remembered old memory becomes newly active in
    mind again, not just a one-off retrieval. Ordinary recall()/recall_associative() will
    find it again afterward.
    Returns True if found and revived, False if doc_id doesn't exist.
    """
    col = _get_collection()
    existing = col.get(ids=[doc_id], include=["metadatas"])
    if not existing["ids"]:
        return False

    m = existing["metadatas"][0]
    now = time.time()
    base = EPISODIC_BASE_STABILITY if m["memory_type"] == "episodic" else SEMANTIC_BASE_STABILITY
    updated = {
        **m,
        "archived":      False,
        "stability":     base,
        "last_accessed": now,
        "access_count":  m["access_count"] + 1,
    }
    col.update(ids=[doc_id], metadatas=[updated])
    _log_activity("revive", doc_id, m["topic"], m["title"])
    return True


# ---------------------------------------------------------------- feedback pattern consolidation
# Chain item 11 (2026-07-31): notice when several separate category="feedback" jot() fragments
# describe the same recurring correction/preference, and promote that into an actual candidate
# CLAUDE.md rule -- instead of that promotion only ever happening when a human manually notices
# the repetition across sessions. See PROJECT_PLAN.md for the full motivation/design writeup.

CONSOLIDATION_MIN_CLUSTER = 3
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.55
PENDING_RULES_PATH = Path(__file__).resolve().parent / "pending_rules.md"

FEEDBACK_DRAFT_PROMPT_TEMPLATE = """You are looking at {n} separate feedback notes a coding assistant recorded about the same recurring pattern in how it should behave, each written after a real correction or confirmation from the user. Synthesize them into ONE general standing rule, written in the same style already used in this project's CLAUDE.md: a clear rule statement, then a line starting with "**Why:**" explaining the reasoning, then a line starting with "**How to apply:**" explaining when it kicks in.

Feedback notes:
---
{notes}
---

Output ONLY the rule in that three-part format, no other commentary."""


def find_feedback_patterns(
    min_cluster_size: int = CONSOLIDATION_MIN_CLUSTER,
    similarity_threshold: float = CONSOLIDATION_SIMILARITY_THRESHOLD,
) -> list[list[dict]]:
    """
    Pulls every non-archived fragment tagged category="feedback" and greedily clusters
    them by embedding similarity, reusing the embeddings already computed at jot-time --
    no new encoding work. Embeddings are stored L2-normalized, so cosine similarity
    between two of them is just their dot product; no numpy needed at this scale.

    Greedy single-link clustering, not a real clustering library -- consistent with this
    project's existing preference for direct methods over pulling in a dependency for
    something this small (same reasoning as classify.py's TF-IDF centroids or
    apply_backlinks.py's threshold-based cross-topic linking).

    Purely read-only. Returns a list of clusters, each a list of
    {"id", "document", "meta"} dicts, one entry per cluster that reaches min_cluster_size.
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    all_entries = col.get(include=["metadatas", "documents", "embeddings"])
    feedback = [
        {"id": doc_id, "document": doc, "meta": meta, "embedding": emb}
        for doc_id, doc, meta, emb in zip(
            all_entries["ids"], all_entries["documents"], all_entries["metadatas"], all_entries["embeddings"]
        )
        if meta.get("category") == "feedback" and not meta.get("archived")
    ]
    if len(feedback) < min_cluster_size:
        return []

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    n = len(feedback)
    clustered = [False] * n
    clusters = []
    for i in range(n):
        if clustered[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if clustered[j]:
                continue
            if cosine(feedback[i]["embedding"], feedback[j]["embedding"]) >= similarity_threshold:
                group.append(j)
        if len(group) >= min_cluster_size:
            for idx in group:
                clustered[idx] = True
            clusters.append([feedback[idx] for idx in group])

    return clusters


def draft_rule_from_cluster(cluster: list[dict]) -> str:
    """
    One claude -p subprocess call -- the same mechanism auto_capture.py already uses for
    autonomous extraction -- synthesizing a cluster of feedback fragments into a single
    candidate CLAUDE.md rule. Never writes anywhere itself; the caller routes the result
    into the review queue. Returns "" on any failure (timeout, missing CLI, bad output) --
    same silent-failure contract as the rest of this file's subprocess-based extraction.
    """
    notes = "\n\n".join(f"- {f['document']}" for f in cluster)
    prompt = FEEDBACK_DRAFT_PROMPT_TEMPLATE.format(n=len(cluster), notes=notes)
    try:
        result = subprocess.run(
            ["claude", "-p", prompt], capture_output=True, text=True, timeout=270,
            env={**os.environ, NESTED_EXTRACTION_ENV_VAR: "1"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def review_feedback_patterns(
    min_cluster_size: int = CONSOLIDATION_MIN_CLUSTER,
    similarity_threshold: float = CONSOLIDATION_SIMILARITY_THRESHOLD,
) -> int:
    """
    Run on demand -- not scheduled anywhere automatically, consistent with how prune()
    also isn't automatically scheduled in this project today (v2 note in PROJECT_PLAN.md:
    a SessionStart-surfaced version, "N pending rule suggestions," is the natural next
    step, not required for v1).

    Finds recurring feedback patterns via find_feedback_patterns(), drafts a candidate
    rule for each via draft_rule_from_cluster(), and appends them to pending_rules.md for
    human review. CLAUDE.md is never written to directly by this process -- it only
    changes on explicit approval, since it's a standing instruction file, not something
    that should get silently rewritten by a background process.

    Returns the number of new candidate rules written.
    """
    clusters = find_feedback_patterns(min_cluster_size, similarity_threshold)
    if not clusters:
        return 0

    if not PENDING_RULES_PATH.exists():
        PENDING_RULES_PATH.write_text(
            "# Pending Rules — Feedback Pattern Consolidation\n\n"
            "Candidate CLAUDE.md rules drafted from recurring feedback patterns. Review "
            "each, then manually add the ones worth keeping to CLAUDE.md and delete the "
            "entry here. Nothing here is applied automatically.\n"
        )

    written = 0
    new_entries = []
    for cluster in clusters:
        draft = draft_rule_from_cluster(cluster)
        if not draft:
            continue
        stamp = time.strftime("%Y-%m-%d")
        ids = ", ".join(f["id"] for f in cluster)
        new_entries.append(f"\n## Candidate ({stamp}, from {len(cluster)} fragments: {ids})\n\n{draft}\n")
        written += 1

    if new_entries:
        with open(PENDING_RULES_PATH, "a") as f:
            f.write("".join(new_entries))
    return written
