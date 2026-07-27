#!/usr/bin/env python3
"""
Wire cross-topic backlinks into session_summaries/ files — the periodic
consolidation sweep. This recomputes every file's cross-topic matches
against the whole corpus and replaces its "## Related" section, so it's
the right tool for when topic boundaries have shifted enough that old
links deserve reconsidering. For linking a single new file in as it's
written, see incremental_backlink.py instead — that one only touches the
new file and its specific matches, not the whole corpus.

Idempotent: re-running replaces any existing "## Related" section rather
than duplicating it.
"""

import os

# Must be set before numpy/sklearn import below. Works around a macOS arm64
# Accelerate BLAS thread-safety bug (numpy 2.0.x) that otherwise makes
# cosine_similarity nondeterministic across runs on identical input.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backlink_lib import THRESHOLD, build_related_section, load_corpus, replace_related_section


def main():
    docs = load_corpus()
    texts = [text for _, _, _, text in docs]
    docs_by_rel = {rel: text for _, rel, _, text in docs}

    vec = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=1)
    matrix = vec.fit_transform(texts)
    sims = cosine_similarity(matrix)
    np.fill_diagonal(sims, -1)

    related = {}
    for i, (_, i_rel, i_topic, _) in enumerate(docs):
        for j, (_, j_rel, j_topic, _) in enumerate(docs):
            if i == j or i_topic == j_topic or sims[i][j] < THRESHOLD:
                continue
            related.setdefault(i_rel, []).append((j_rel, sims[i][j]))

    changed = 0
    for path, rel, _, text in docs:
        if rel not in related:
            continue
        matches = sorted(related[rel], key=lambda m: -m[1])
        section = build_related_section(rel, matches, docs_by_rel)
        new_text = replace_related_section(text, section)
        if new_text != text:
            path.write_text(new_text)
            changed += 1

    print(f"Updated {changed} files with a ## Related section (threshold={THRESHOLD}).")


if __name__ == "__main__":
    main()
