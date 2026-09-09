#!/usr/bin/env python3
"""
Query activity.log — the append-only, human-eyeball-only record every
jot()/ingest()/purge()/etc. call writes a line to (see _log_activity() in
memory_store.py). Read-only: never touches ChromaDB or activity.log itself.

Parsing approach: timestamp is fixed-width ("YYYY-MM-DD HH:MM:SS"), but the
action/topic fields are only left-padded to a *minimum* width (Python's
{x:6s}/{x:15s} don't truncate), so a longer value (e.g. action "archive",
7 chars, vs. the 6-char field it's nominally padded to) still parses fine —
splitting on whitespace runs (str.split(None, 2)) treats a value plus its
padding as a single token regardless of how much padding there is, as long
as the value itself never contains a space, true for both action and topic.
doc_id is recovered from the trailing "(...)" instead of further splitting,
since the title in between may itself contain spaces.

Usage:
  .venv/bin/python query_activity.py --action jot
  .venv/bin/python query_activity.py --topic memory-project --since 2026-09-01
  .venv/bin/python query_activity.py --action purge --action archive
  .venv/bin/python query_activity.py --doc-id fragment/2b0b88d0
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, NamedTuple

ACTIVITY_LOG = Path(__file__).resolve().parent / "activity.log"

_TRAILING_DOC_ID = re.compile(r"\((?P<doc_id>[^()]*)\)\s*$")


class Entry(NamedTuple):
    timestamp: datetime
    action: str
    topic: str
    title: str
    doc_id: str


def _parse_line(line: str) -> "Entry | None":
    line = line.rstrip("\n")
    if len(line) < 19:
        return None
    ts_str, rest = line[:19], line[19:]
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

    parts = rest.strip().split(None, 2)
    if len(parts) < 3:
        return None
    action, topic, tail = parts

    m = _TRAILING_DOC_ID.search(tail)
    if m:
        doc_id = m.group("doc_id")
        title = tail[: m.start()].rstrip()
    else:
        doc_id = ""
        title = tail.strip()

    return Entry(ts, action, topic, title, doc_id)


def iter_entries(log_path: Path = ACTIVITY_LOG) -> Iterator[Entry]:
    if not log_path.exists():
        return
    with open(log_path, errors="replace") as f:
        for line in f:
            entry = _parse_line(line)
            if entry is not None:
                yield entry


def query(
    actions: "set[str] | None" = None,
    topic: "str | None" = None,
    since: "datetime | None" = None,
    until: "datetime | None" = None,
    doc_id_prefix: "str | None" = None,
    log_path: Path = ACTIVITY_LOG,
) -> list[Entry]:
    """Filter activity.log entries. All filters are AND'd together; each
    is skipped (no-op) when left as None, matching this project's existing
    convention (see recall()'s exclude_topic) of an absent filter meaning
    "don't restrict on this axis" rather than "match nothing"."""
    results = []
    for entry in iter_entries(log_path):
        if actions is not None and entry.action not in actions:
            continue
        if topic is not None and entry.topic != topic:
            continue
        if since is not None and entry.timestamp < since:
            continue
        if until is not None and entry.timestamp > until:
            continue
        if doc_id_prefix is not None and not entry.doc_id.startswith(doc_id_prefix):
            continue
        results.append(entry)
    return results


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"unrecognized date/time: {s!r} (expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", action="append", dest="actions", help="filter by action (repeatable); one of jot/ingest/blocked/reject/confirm/archive/purge/revive")
    parser.add_argument("--topic", help="filter by exact topic")
    parser.add_argument("--since", type=_parse_date, help="only entries at/after this date (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--until", type=_parse_date, help="only entries at/before this date (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--doc-id", dest="doc_id_prefix", help="filter by doc_id prefix")
    parser.add_argument("--count", action="store_true", help="print only the matching count, not each line")
    args = parser.parse_args()

    actions = set(args.actions) if args.actions else None
    hits = query(
        actions=actions,
        topic=args.topic,
        since=args.since,
        until=args.until,
        doc_id_prefix=args.doc_id_prefix,
    )

    if args.count:
        print(len(hits))
        return

    for e in hits:
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts}  {e.action:8s} {e.topic:15s} {e.title[:60]}  ({e.doc_id})")
    print(f"\n{len(hits)} matching entr{'y' if len(hits) == 1 else 'ies'}")


if __name__ == "__main__":
    main()
