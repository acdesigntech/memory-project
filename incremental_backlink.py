#!/usr/bin/env python3
"""
Encoding-time backlink: when a single session summary is written, link it
into the corpus immediately instead of waiting for the next full
apply_backlinks.py sweep. Also embeds the file into the ChromaDB memory
store (memory_store.ingest()) unconditionally — every write to a corpus
.md file gets embedded, whether or not it has any cross-topic matches.

Scope is intentionally narrow — mirrors how learning something new forms
associations with existing memory without rewriting everything you
already knew: only the new file and the specific files it matches get
touched, not the whole corpus.

Usage: incremental_backlink.py <path-to-md-file>

Designed to be called unconditionally from a hook after every Write, so
it never raises past main() — on any failure it writes a report to
backlink_errors/ and exits 0, so a backlink bug can never block the write
it's reacting to. Check backlink_errors/ periodically to catch and fix
issues that got silently swallowed.
"""

import os

# Must be set before numpy/sklearn import (inside run(), below). Works around
# a macOS arm64 Accelerate BLAS thread-safety bug (numpy 2.0.x) that otherwise
# makes cosine_similarity nondeterministic across runs on identical input.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from backlink_lib import (
    CORPUS_DIR,
    THRESHOLD,
    append_related_link,
    build_related_section,
    load_corpus,
    relative_link,
    replace_related_section,
    title_of,
)

ERROR_DIR = Path(__file__).resolve().parent / "backlink_errors"


def log_error(target, exc):
    ERROR_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = ERROR_DIR / f"{stamp}.txt"
    report.write_text(
        f"target: {target}\n"
        f"time: {stamp}\n"
        f"error: {exc!r}\n\n"
        f"{traceback.format_exc()}"
    )
    print(f"incremental_backlink: error logged to {report}", file=sys.stderr)


def run(target_arg):
    target_path = Path(target_arg).resolve()
    if CORPUS_DIR not in target_path.parents or target_path.suffix != ".md":
        return  # not a session summary, nothing to do
    if not target_path.exists():
        return  # e.g. deleted after the hook fired

    docs = load_corpus()
    docs_by_path = {path: (rel, topic, text) for path, rel, topic, text in docs}
    if target_path not in docs_by_path:
        return

    from memory_store import ingest as ingest_memory
    ingest_memory(target_path)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [text for _, _, _, text in docs]
    vec = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=1)
    matrix = vec.fit_transform(texts)

    target_idx = [path for path, _, _, _ in docs].index(target_path)
    sims = cosine_similarity(matrix[target_idx], matrix)[0]

    target_rel, target_topic, target_text = docs_by_path[target_path]
    docs_by_rel = {rel: text for _, rel, _, text in docs}

    matches = []
    for i, (path, rel, topic, _) in enumerate(docs):
        if path == target_path or topic == target_topic or sims[i] < THRESHOLD:
            continue
        matches.append((rel, sims[i]))
    matches.sort(key=lambda m: -m[1])

    if not matches:
        return

    # 1. The new file's own Related section — full replace is correct here,
    #    it has no prior links to preserve.
    section = build_related_section(target_rel, matches, docs_by_rel)
    new_target_text = replace_related_section(target_text, section)
    if new_target_text != target_text:
        target_path.write_text(new_target_text)

    # 2. Reciprocal link into each matched partner — append-only, leaves
    #    every other existing link in that file untouched.
    target_title = title_of(target_text)
    for partner_rel, _score in matches:
        partner_path = CORPUS_DIR / partner_rel
        partner_text = partner_path.read_text(errors="ignore")
        link = relative_link(partner_rel, target_rel)
        updated = append_related_link(partner_text, link, target_title)
        if updated is not None:
            partner_path.write_text(updated)


def main():
    if len(sys.argv) != 2:
        return
    target = sys.argv[1]
    try:
        run(target)
    except Exception as exc:  # noqa: BLE001 - must never propagate, see module docstring
        log_error(target, exc)


if __name__ == "__main__":
    main()
