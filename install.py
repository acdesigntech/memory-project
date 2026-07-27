#!/usr/bin/env python3
"""Drop-in installer for memory-project.

Sets up the venv, wires the four global hooks into ~/.claude/settings.json,
and writes the operational instructions into ~/.claude/CLAUDE.md — all with
this repo's actual absolute path substituted in automatically.

Safe to re-run: hook entries and the CLAUDE.md block are both replaced in
place (matched by this repo's path / a marker comment), not duplicated.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
VENV_DIR = REPO_DIR / ".venv"
SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"
MARKER_START = "<!-- memory-project:start -->"
MARKER_END = "<!-- memory-project:end -->"


def backup(path: Path):
    """Timestamped backup of an existing file before it's touched. Never overwritten itself,
    even by two runs within the same second."""
    if not path.exists():
        return None
    stamp = int(time.time())
    backup_path = path.with_suffix(path.suffix + f".bak.{stamp}")
    n = 1
    while backup_path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak.{stamp}-{n}")
        n += 1
    shutil.copy2(path, backup_path)
    print(f"Backed up existing {path} -> {backup_path}")
    return backup_path


def setup_venv():
    if not VENV_DIR.exists():
        print("Creating venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    print("Installing dependencies (chromadb, sentence-transformers, numpy, scikit-learn)...")
    subprocess.run(
        [str(VENV_DIR / "bin" / "pip"), "install", "-q",
         "chromadb", "sentence-transformers", "numpy", "scikit-learn"],
        check=True,
    )


def python_bin():
    return str(VENV_DIR / "bin" / "python")


def build_hooks():
    py = python_bin()
    return {
        "PostToolUse": [{
            "matcher": "Write",
            "hooks": [{
                "type": "command",
                "command": (
                    "jq -r '.tool_input.file_path' | { read -r f; "
                    f"{py} {REPO_DIR}/incremental_backlink.py \"$f\"; }} 2>/dev/null || true"
                ),
                "timeout": 30,
            }],
        }],
        "UserPromptSubmit": [{
            "hooks": [{"type": "command", "command": f"{py} {REPO_DIR}/recall_hook.py", "timeout": 20}],
        }],
        "SessionEnd": [{
            "hooks": [{"type": "command", "command": f"{py} {REPO_DIR}/hooks/session_end_capture.py", "timeout": 120}],
        }],
        "SessionStart": [{
            "hooks": [{"type": "command", "command": f"{py} {REPO_DIR}/hooks/session_start_backstop.py", "timeout": 60}],
        }],
    }


def install_hooks():
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    backup(SETTINGS_PATH)
    settings = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    settings.setdefault("hooks", {})

    repo_marker = str(REPO_DIR)
    for event, new_entries in build_hooks().items():
        existing = settings["hooks"].setdefault(event, [])
        # Drop any previous entries this installer wrote for this repo (idempotent re-run),
        # leaving any unrelated hooks for that same event alone.
        existing[:] = [e for e in existing if repo_marker not in json.dumps(e)]
        existing.extend(new_entries)

    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Hooks installed in {SETTINGS_PATH}")


def install_claude_md():
    """Never overwrites the file. First install appends a clearly-marked block after
    whatever's already there; re-installs replace only that same block in place."""
    template = (REPO_DIR / "CLAUDE.md.example").read_text()
    rendered = template.replace("/path/to/memory-project", str(REPO_DIR))
    block = f"{MARKER_START}\n{rendered}\n{MARKER_END}\n"

    if not CLAUDE_MD_PATH.exists():
        CLAUDE_MD_PATH.write_text(block)
        print(f"Created {CLAUDE_MD_PATH} (didn't exist before)")
        return

    backup(CLAUDE_MD_PATH)
    existing = CLAUDE_MD_PATH.read_text()
    pattern = re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END) + r"\n?"
    has_matched_pair = re.search(pattern, existing, flags=re.S) is not None

    if has_matched_pair:
        existing = re.sub(pattern, block, existing, flags=re.S)
        print(f"Updated existing memory-project block in {CLAUDE_MD_PATH} (rest of file untouched)")
    elif MARKER_START in existing or MARKER_END in existing:
        # A marker is present but unpaired (e.g. hand-edited) — don't guess, don't clobber.
        print(f"WARNING: found an unpaired {MARKER_START!r}/{MARKER_END!r} marker in "
              f"{CLAUDE_MD_PATH} — leaving the file untouched. Remove the stray marker "
              f"by hand and re-run to install cleanly.")
        return
    else:
        existing = existing.rstrip("\n") + "\n\n" + block
        print(f"Appended memory-project block to existing {CLAUDE_MD_PATH} (nothing else changed)")

    CLAUDE_MD_PATH.write_text(existing)


def main():
    setup_venv()
    install_hooks()
    install_claude_md()
    print("\nDone. Start a new Claude Code session (or restart a running one) to pick up the hooks.")


if __name__ == "__main__":
    main()
