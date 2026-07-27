#!/usr/bin/env python3
"""
Prototype: auto-classify a piece of session content into an existing
session_summaries/<topic>/ folder, using the folder's own existing files
as its "centroid" — no more asking "what topic is this?" unless nothing
matches well.

NOTE ON EMBEDDINGS: this box has no network access (pypi is not on the
egress allowlist), so this prototype uses TF-IDF + cosine similarity —
a lexical/keyword-overlap measure, not true semantic embeddings. It is
enough to validate the *mechanism* (centroids, thresholding, leave-one-out
accuracy). On the Mac, swap TfidfVectorizer for a real embedding model
(sentence-transformers, or an API embedding call) and everything below
this line — centroid math, ranking, threshold logic — stays identical.
"""

import os

# Must be set before numpy/sklearn import below. Works around a macOS arm64
# Accelerate BLAS thread-safety bug (numpy 2.0.x) that otherwise makes
# cosine_similarity nondeterministic across runs on identical input.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = Path(__file__).resolve().parent / "session_summaries"


def load_corpus():
    """topic -> list of (path, text). Files directly under session_summaries/
    (no subfolder) are labeled 'root' — that's where the outlier lives."""
    docs = defaultdict(list)
    for path in CORPUS_DIR.rglob("*.md"):
        rel = path.relative_to(CORPUS_DIR)
        topic = rel.parts[0] if len(rel.parts) > 1 else "root"
        text = path.read_text(errors="ignore")
        docs[topic].append((path, text))
    return docs


def fit_vectorizer(all_texts):
    vec = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=1)
    vec.fit(all_texts)
    return vec


def leave_one_out_eval(docs, vec):
    """For each doc, build centroids from every OTHER doc, then check
    whether the doc's true topic is the top match. This is the honest
    test — a doc trivially matches a centroid that includes itself."""
    all_paths = [(topic, path, text) for topic, items in docs.items() for path, text in items]

    results = []
    skipped_singleton = 0

    for held_topic, held_path, held_text in all_paths:
        # Build centroids excluding the held-out doc
        centroids = {}
        for topic, items in docs.items():
            remaining = [text for path, text in items if path != held_path]
            if not remaining:
                continue  # singleton topic, nothing left to build a centroid from
            topic_vecs = vec.transform(remaining)
            centroids[topic] = np.asarray(topic_vecs.mean(axis=0)).ravel()

        if held_topic not in centroids:
            skipped_singleton += 1
            continue

        held_vec = np.asarray(vec.transform([held_text]).todense()).ravel()

        sims = {t: cosine_similarity([held_vec], [c])[0][0] for t, c in centroids.items()}
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        top_topic, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        results.append({
            "path": str(held_path.relative_to(CORPUS_DIR)),
            "true_topic": held_topic,
            "predicted": top_topic,
            "correct": top_topic == held_topic,
            "top_score": top_score,
            "margin": top_score - second_score,
        })

    return results, skipped_singleton


def main():
    docs = load_corpus()
    all_texts = [text for items in docs.values() for _, text in items]
    vec = fit_vectorizer(all_texts)

    print(f"Corpus: {sum(len(v) for v in docs.values())} files across {len(docs)} topic folders")
    print(f"Folder sizes: { {k: len(v) for k, v in sorted(docs.items(), key=lambda kv: -len(kv[1]))} }\n")

    results, skipped = leave_one_out_eval(docs, vec)

    correct = sum(r["correct"] for r in results)
    print(f"Leave-one-out classification accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")
    print(f"Skipped (singleton topics, no centroid possible without the held-out doc): {skipped}\n")

    print("--- Misclassifications ---")
    misses = [r for r in results if not r["correct"]]
    if not misses:
        print("(none)")
    for r in misses:
        print(f"  {r['path']}: true={r['true_topic']} predicted={r['predicted']} "
              f"(score={r['top_score']:.3f}, margin={r['margin']:.3f})")

    print("\n--- Margin distribution (correct predictions only) ---")
    correct_margins = sorted([r["margin"] for r in results if r["correct"]])
    if correct_margins:
        n = len(correct_margins)
        print(f"  min={correct_margins[0]:.3f}  p10={correct_margins[n//10]:.3f}  "
              f"median={correct_margins[n//2]:.3f}  max={correct_margins[-1]:.3f}")

    # The outlier: does the recipe file refuse to match any real topic strongly?
    root_result = next((r for r in results if r["true_topic"] == "root"), None)
    if root_result:
        print(f"\n--- Outlier check (protein-cookie-recipe.md) ---")
        print(f"  best match: {root_result['predicted']} at score={root_result['top_score']:.3f}")
        print(f"  (a real new-topic threshold should sit above this score)")


if __name__ == "__main__":
    main()
