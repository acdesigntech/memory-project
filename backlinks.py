#!/usr/bin/env python3
"""
Surface candidate backlinks between session summaries: pairwise TF-IDF
cosine similarity across the whole corpus (not just doc-to-topic-centroid,
which is what classify.py does). For each file, print its top nearest
neighbors so a human can pick which ones are worth wiring in as an
explicit link, prioritizing cross-topic matches since same-topic files
are already discoverable by folder.

Output only — this script does not edit any files.
"""

import os

# Must be set before numpy/sklearn import below. Works around a macOS arm64
# Accelerate BLAS thread-safety bug (numpy 2.0.x) that otherwise makes
# cosine_similarity nondeterministic across runs on identical input.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = Path(__file__).resolve().parent / "session_summaries"
TOP_K = 3
MIN_SCORE = 0.15


def load_corpus():
    docs = []
    for path in sorted(CORPUS_DIR.rglob("*.md")):
        rel = path.relative_to(CORPUS_DIR)
        topic = rel.parts[0] if len(rel.parts) > 1 else "root"
        docs.append((rel, topic, path.read_text(errors="ignore")))
    return docs


def main():
    docs = load_corpus()
    texts = [text for _, _, text in docs]

    vec = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=1)
    matrix = vec.fit_transform(texts)
    sims = cosine_similarity(matrix)
    np.fill_diagonal(sims, -1)  # exclude self-match

    cross_topic_high_conf = []

    for i, (rel, topic, _) in enumerate(docs):
        ranked = sorted(range(len(docs)), key=lambda j: sims[i][j], reverse=True)[:TOP_K]
        candidates = [(docs[j][0], docs[j][1], sims[i][j]) for j in ranked if sims[i][j] >= MIN_SCORE]
        if not candidates:
            continue

        print(f"{rel}  [{topic}]")
        for cand_rel, cand_topic, score in candidates:
            flag = " <-- CROSS-TOPIC" if cand_topic != topic else ""
            print(f"    {score:.3f}  {cand_rel}  [{cand_topic}]{flag}")
        print()

    # Collect high-confidence cross-topic pairs (score >= 0.3), deduped (i<j)
    n = len(docs)
    seen = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if docs[i][1] == docs[j][1]:
                continue
            if sims[i][j] < 0.3:
                continue
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            cross_topic_high_conf.append((docs[i][0], docs[i][1], docs[j][0], docs[j][1], sims[i][j]))

    cross_topic_high_conf.sort(key=lambda r: -r[4])
    print("=" * 70)
    print(f"High-confidence cross-topic pairs (score >= 0.3): {len(cross_topic_high_conf)}")
    for a_rel, a_topic, b_rel, b_topic, score in cross_topic_high_conf:
        print(f"  {score:.3f}  {a_topic}/{a_rel.name}  <->  {b_topic}/{b_rel.name}")


if __name__ == "__main__":
    main()
