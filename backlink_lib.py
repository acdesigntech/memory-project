#!/usr/bin/env python3
"""
Shared helpers for the two backlink writers (apply_backlinks.py's full
corpus sweep and incremental_backlink.py's single-file hook path) so they
can't drift apart on threshold, link format, or section format.
"""

import re
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "session_summaries"
THRESHOLD = 0.3


def load_corpus():
    docs = []
    for path in sorted(CORPUS_DIR.rglob("*.md")):
        rel = path.relative_to(CORPUS_DIR)
        topic = rel.parts[0] if len(rel.parts) > 1 else "root"
        docs.append((path, rel, topic, path.read_text(errors="ignore")))
    return docs


def title_of(text):
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "Related session"


def relative_link(from_rel, to_rel):
    from_dir = from_rel.parent
    if from_dir == Path("."):
        return (Path("..") / to_rel).as_posix()
    return (Path(*([".."] * len(from_dir.parts))) / to_rel).as_posix()


def build_related_section(from_rel, matches, docs_by_rel):
    lines = ["## Related", ""]
    for to_rel, _score in matches:
        lines.append(f"- [{title_of(docs_by_rel[to_rel])}]({relative_link(from_rel, to_rel)})")
    return "\n".join(lines) + "\n"


def replace_related_section(text, section):
    """Full replace — used when a file's complete match set is known (batch sweep, or a brand-new file)."""
    pattern = re.compile(r"\n## Related\n.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(text):
        return pattern.sub("\n" + section.rstrip("\n"), text)
    return text.rstrip("\n") + "\n\n" + section


def append_related_link(text, link, title):
    """Add one link to an existing Related section without touching its other entries.
    Returns the new text, or None if the link is already present (no-op)."""
    if f"]({link})" in text:
        return None
    bullet = f"- [{title}]({link})"
    pattern = re.compile(r"\n## Related\n(.*?)(?=\n## |\Z)", re.DOTALL)
    match = pattern.search(text)
    if match:
        body = match.group(1).rstrip("\n")
        new_body = f"{body}\n{bullet}\n" if body else f"{bullet}\n"
        return text[: match.start(1)] + new_body + text[match.end(1):]
    return text.rstrip("\n") + "\n\n## Related\n\n" + bullet + "\n"
