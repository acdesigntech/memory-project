# memory-project

A local-first, associative long-term memory system for [Claude Code](https://claude.ai/code). ChromaDB-backed, embeddings via `sentence-transformers`, wired in globally via hooks so it works no matter which directory a session launches from.

The goal is EA-style memory, not a flat search index: context on past work should follow you across projects the way an executive assistant carries context across an executive's whole portfolio, not just one meeting at a time. It deliberately models a chain of phases mirroring human associative/episodic memory — encoding, filing, association, decay, reinforcement, retrieval, capture, consolidation, cued recall, and non-destructive forgetting — rather than just "embed everything, cosine-similarity search it."

## See it in action

**Day 1, working in `project-a/`:**
> "let's always use pytest fixtures instead of setUp/tearDown for this codebase — tried unittest style on a past project and sharing fixtures across test files was a pain."

Claude calls `jot()` — no file, no ceremony, just a one-line fragment stored and forgotten about.

**Day 19, working in `project-b/` — a different repo, a fresh session that's never seen `project-a/`:**
> "should I use `TestCase.setUp()` here or something else?"

> Claude: "This might be related — a few weeks back you mentioned preferring pytest fixtures over setUp/tearDown, since sharing them across files got painful on a past project. Does that still apply here?"

> "yeah, still applies, good call"

That's `recall_associative()` surfacing a weak, unreinforced "rings a bell" match (below the confident-recall bar, so it wasn't stated as fact — it was hedged) — and the user's confirmation calling `confirm_activation()`, which reinforces it *harder* than an easy direct hit would have, precisely because it was retrieved under real uncertainty and turned out to be right.

## Why not just use \<existing memory framework\>?

Most agent-memory tooling is built to bolt onto any agent stack generically. This is the opposite bet: it's deliberately, deeply specific to Claude Code — global hooks (`PostToolUse`, `UserPromptSubmit`, `SessionStart`, `SessionEnd`), raw transcript parsing, `claude -p` subprocess calls for autonomous extraction. If you want a generic memory layer for an arbitrary LLM app, look elsewhere. If you want to see what deep, native-feeling persistent memory could look like specifically inside Claude Code, this is that.

| | [Mem0](https://mem0.ai) | [Letta](https://letta.com) (MemGPT) | [Zep](https://www.getzep.com) | memory-project |
|---|---|---|---|---|
| Model | embedding store | OS-like stateful agent runtime | temporal knowledge graph | embedding store + Ebbinghaus-curve decay |
| Forgetting | persists until explicitly deleted | manually edited memory blocks | facts marked invalid, not removed | decays automatically; archives (never deletes) below a floor |
| Confidence-tiered recall | — | — | — | yes — weak "rings a bell" matches surfaced separately, reinforced only on human confirmation |
| Target | any agent stack (hosted API) | any agent stack (own runtime) | any agent stack (enterprise/temporal reasoning) | Claude Code specifically — global hooks, transcript-based auto-capture |
| Funding | $24.5M raised | $10M seed | — | none — solo, local-first, free |

The gap this fills isn't "better vector search" — Mem0 and Zep both do that well. It's that none of them model *forgetting* as a first-class, automatic, human-confirmed process, and none of them are trying to be the native memory layer for one specific tool rather than a generic API you integrate.

## What breaks without this (vs. Claude Code's built-in memory)

Claude Code already ships a built-in per-project memory feature (`~/.claude/projects/<encoded-cwd>/memory/*.md` plus an index file). It's a reasonable lightweight scratchpad for a single project — but it has real, concrete limits this system exists specifically to route around:

- **Directory-scoped, not identity-scoped.** The built-in memory's storage path is templated from the literal launch directory. Depending on exactly where a session launches from, work on what's conceptually the same project can end up filed into two disconnected memory buckets, each invisible to the other. This system resolves paths relative to its own install location, not `cwd` — it's scoped to *you*, not to whichever folder you happened to launch from.
- **A hard ceiling, not a soft one.** The built-in index file has a stated truncation limit — past a fixed number of lines, older entries just stop being visible, with nothing deciding what was worth keeping. This system's decay curve does that job continuously instead: things that stop being relevant quietly fall out of ordinary retrieval on their own, long before anything resembling a wall.
- **Flat, not ranked.** The built-in memory is a set of files read more or less wholesale — no embeddings, no similarity ranking, no sense of "how relevant is this to what's happening right now." Every fact is filed by hand, and finding the right one again later is on you.
- **All-or-nothing, not tiered.** There's no equivalent of confidence-tiered "rings a bell" recall — a fact is either in the index and gets read, or it isn't. No weak-signal middle ground, and no way for the system to earn confidence in a memory through repeated confirmation.
- **Manual capture, no crash safety net.** Built-in memory only gets written when the model deliberately decides to write a fact mid-session. If a session ends abnormally, whatever was said in it is just gone. This system's `SessionStart`/`SessionEnd` hooks close that gap — even a crashed session's raw transcript gets swept and mined for facts the next time a session starts.

None of this is a knock on the built-in feature for what it's meant to be. It's what starts to matter once memory is used the way a person actually would — across projects, over months, without hand-managing every fact yourself.

## The memory chain

| # | Phase | Mechanism |
|---|---|---|
| 1 | Encoding | `ingest(path)` — curated tier, file-backed |
| 2 | Filing / classification | `classify.py` — TF-IDF centroid topic classification |
| 3 | Association / backlinking | `incremental_backlink.py` (encoding-time) + `apply_backlinks.py` (periodic sweep) |
| 4 | Decay | `strength = exp(-t / stability)`, computed at query time, never stored |
| 5 | Reinforcement | every `recall()` hit grows `stability` (×1.5, capped at 10× base) |
| 6 | Retrieval, global | `recall_hook.py`, a `UserPromptSubmit` hook, works from any launch directory |
| 7 | Capture, global | `auto_capture.py` + `SessionStart`/`SessionEnd` hooks — memories get created even if a session crashes |
| 8 | Consolidation | `episodic` (7-day base stability) promotes to `semantic` (90-day base) once reinforcement maxes out the episodic ceiling |
| 9 | Cued / associative recall | `recall_associative()` — surfaces weak "rings a bell" matches for human-in-the-loop confirmation |
| 10 | Cold storage | `prune()` archives instead of deleting — non-destructive, human forgetting-shaped |

### Capture timing

Each finalize pass is a real `claude -p` subprocess call per transcript chunk — measured at ~2 minutes wall-clock even for a trivial extraction (mostly CLI/session startup overhead, not generation), so both the `SessionStart` and `SessionEnd` hooks are configured with a 300s timeout, not a short one. If a previous session ended abnormally, expect the *next* session's startup to pause for up to a few minutes while the `SessionStart` backstop catches it up — that's expected, not a hang. When a sweep has to catch up more than one orphaned session at once (capped at 3 per invocation), `capture_sessions_parallel()` runs their extraction subprocesses concurrently rather than one after another, so total wait time is bounded by the slowest single session, not the sum of all of them.

### Decay, made concrete

`strength = exp(-t / stability)` is easy to skim past. In actual days, for a fresh `jot()` (episodic, 7-day base stability, never recalled again):

- **Day ~16**: strength decays under `RETRIEVAL_FLOOR` (0.1) — invisible to ordinary `recall()`, though `recall_associative()`'s weaker `activated` tier can still surface it a bit longer.
- **Day ~27**: strength decays under `DELETION_FLOOR` (0.02) — the next `prune()` archives it. Not deleted; findable again later only via `recall_cold()` with a genuinely specific query.

Every time it *is* recalled, though, `stability` gets multiplied by 1.5× and the clock resets. Do that six times (`7 × 1.5⁶ ≈ 70`, the episodic cap) and it consolidates to `semantic`, resetting to a 90-day base — at which point the same math pushes the retrieval floor out to ~7 months of silence and the deletion floor to ~11.7 months. A fact you only ever mention once fades in about a month; a fact that keeps coming up organically becomes something the system will hold onto for the better part of a year without being touched.

## Public API (`memory_store.py`)

- **`jot(text, session_id='', topic_hint=None)`** — cheap, unfiled, fragment-tier capture. Meant to be called constantly and cheaply; the decay/reinforcement pipeline sorts out what turns out to matter, for free, over time.
- **`recall(query, n_results=5, exclude_topic=None, topic_hint=None)`** — semantic retrieval, ranked by `similarity * strength`. Reading is what reinforces a memory (and can trigger consolidation).
- **`recall_associative(query, n_results=5, n_activated=3, ...)`** + **`confirm_activation(doc_id)`** — splits results into a `confident` tier (score ≥ 0.35, auto-reinforced) and a weaker `activated` tier (0.15–0.35, *not* auto-reinforced — surfaced for human judgment). Confirming an activated hit applies a bigger stability boost (2.5×) than an ordinary recall (1.5×) — a confirmed uncertain retrieval should reinforce more than an easy one (the "desirable difficulty" effect from spaced-repetition research). Denial does nothing — no penalty, asymmetric by design.
- **`ingest(path)`** — the full curated tier: file-backed, classified, backlinked. Reserve for something worth filing and reading later as a standalone document.
- **`prune()`** — archives (not deletes) anything below the deletion floor. **`recall_cold(query, n_results=3)`** searches only archived entries, with a much stricter similarity bar (0.6 vs. the ordinary 0.35) — resurfacing something long-dormant should take a genuinely specific cue, not a loose association. **`revive_from_cold(doc_id)`** un-archives a confirmed hit. **`purge(doc_id)`** is true, permanent deletion — a rare, deliberate escape hatch, not routine cleanup.

## Setup

```bash
git clone <this repo> memory-project
cd memory-project
python3 install.py
```

That's it — `install.py` is a drop-in installer, not a manual checklist:

- Creates `.venv` and installs dependencies (`chromadb`, `sentence-transformers`, `numpy`, `scikit-learn`).
- Wires the four hooks (`PostToolUse`/`Write`, `UserPromptSubmit`, `SessionStart`, `SessionEnd`) into your **user-level** `~/.claude/settings.json` — not this project's own `.claude/settings.json`, which is intentionally left empty, since the whole point is that this works regardless of which directory a session launches from. Existing hooks for other events/tools are left untouched; re-running the installer updates its own entries in place rather than duplicating them.
- Renders `CLAUDE.md.example` with this repo's actual absolute path substituted in for every `/path/to/memory-project` placeholder, and writes it into `~/.claude/CLAUDE.md` inside a marked block — appended after any existing content you already have there, and safely replaced in place (not duplicated) on re-runs.

Start a new Claude Code session afterward (or restart a running one) to pick up the new hooks. Corpus discovery (`CORPUS_DIR`) resolves relative to each script's own file location, so everything works regardless of your current working directory once the hooks are wired up.

## A macOS/Apple Silicon gotcha

`numpy` linked against Apple's Accelerate BLAS framework has a **thread-safety bug**: `sklearn.metrics.pairwise.cosine_similarity` can produce different results across repeated runs on identical input (symptom: `RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul`). Fix: set `os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")` **before any numpy/sklearn import**. Already baked into every script here — copy the guard into any new script you add that imports numpy/sklearn/sentence-transformers transitively.

## Changelog

See `CHANGELOG.md` for a dated history of notable changes.

## License

AGPLv3 — see `LICENSE`. Chosen deliberately: this is source-available, but if you take it and run it as a hosted/network service, that service has to stay open too. See `NOTICE` (if present) for any additional attribution requirements.
