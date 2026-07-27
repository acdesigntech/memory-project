#!/usr/bin/env python3
"""
One-shot corpus ingestion: walks session_summaries/ and upserts every .md file
into the ChromaDB memory store. Safe to re-run — upsert preserves existing
access metadata (last_accessed, access_count, stability) on files already present.
"""

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
from pathlib import Path

from memory_store import CORPUS_DIR, _get_model, ingest

def main():
    paths = sorted(CORPUS_DIR.rglob("*.md"))
    if not paths:
        print(f"No .md files found under {CORPUS_DIR}")
        return

    print(f"Found {len(paths)} files. Loading model...")
    _get_model()  # load once before the loop, not on first ingest() call
    print("Model ready. Ingesting...\n")

    start = time.time()
    ok = 0
    errors = []

    for i, path in enumerate(paths, 1):
        rel = path.relative_to(CORPUS_DIR)
        try:
            ingest(path)
            ok += 1
            print(f"  [{i:>3}/{len(paths)}] {rel}")
        except Exception as exc:
            errors.append((rel, exc))
            print(f"  [{i:>3}/{len(paths)}] ERROR {rel}: {exc}")

    elapsed = time.time() - start
    print(f"\nDone. {ok}/{len(paths)} ingested in {elapsed:.1f}s.")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for rel, exc in errors:
            print(f"  {rel}: {exc}")

if __name__ == "__main__":
    main()
