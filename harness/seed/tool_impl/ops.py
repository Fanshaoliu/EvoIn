"""Evidence tools of the seed harness: scan, grep, read_lines, bash, notepad.

Every tool receives the per-task ``global_storage`` from the runtime and returns
a dict whose ``result`` field is the text shown to the model.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

from evoin.runtime.state import (
    META_KEY,
    NOTEPAD_KEY,
    get_corpus_lines,
    get_workspace,
    storage_get,
    storage_set,
)

MAX_TOOL_CHARS = 80_000
MAX_READ_LINES = 500
MAX_GREP_PREVIEWS = 80


def _cap_text(text: str, max_chars: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n...[truncated {len(text) - max_chars} chars]...\n{text[-half:]}"


def _numbered(lines: list[str], start: int, end: int) -> str:
    return "\n".join(f"L{i}: {lines[i]}" for i in range(start, end))


def scan(global_storage: Any = None, chunk_size: int = 100, max_chunks: int = 20) -> dict[str, Any]:
    """Return a line-range map of the whole document."""
    lines = get_corpus_lines(global_storage)
    total = len(lines)
    if total == 0:
        meta = "(document is empty)"
        storage_set(global_storage, META_KEY, meta)
        return {"result": meta}

    chunk_size = max(1, int(chunk_size or 100))
    max_chunks = max(1, int(max_chunks or 20))
    if -(-total // chunk_size) > max_chunks:
        chunk_size = max(1, -(-total // max_chunks))

    entries = [
        "## Overview",
        f"- Total lines: {total}",
        "- Line numbers are 0-based. Use read_lines(start, end) for raw evidence.",
        "",
        "## Key Index",
    ]
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        non_empty = [line.strip() for line in lines[start:end] if line.strip()]
        preview = " / ".join(non_empty[:3]) if non_empty else "(blank lines)"
        entries.append(f"- L{start}-L{end - 1}: {_cap_text(preview, 500)}")

    meta = "\n".join(entries)
    storage_set(global_storage, META_KEY, meta)
    return {"result": meta}


def grep(
    pattern: str,
    use_regex: bool = False,
    context_lines: int = 1,
    global_storage: Any = None,
) -> dict[str, Any]:
    """Return every matching line number plus short previews."""
    lines = get_corpus_lines(global_storage)
    context = max(0, int(context_lines or 0))
    if use_regex:
        compiled = re.compile(pattern)
        matches = [i for i, line in enumerate(lines) if compiled.search(line)]
    else:
        needle = pattern.lower()
        matches = [i for i, line in enumerate(lines) if needle in line.lower()]

    if not matches:
        return {"result": f"No matches for {pattern!r}.", "matches": []}

    previews = []
    for idx in matches[:MAX_GREP_PREVIEWS]:
        start, end = max(0, idx - context), min(len(lines), idx + context + 1)
        previews.append(f"--- match L{idx}, preview L{start}-L{end - 1} ---\n{_numbered(lines, start, end)}")

    body = [
        f"Pattern: {pattern!r}",
        f"Total matches: {len(matches)}",
        f"All matching lines: {matches}",
        "",
        "## Previews",
        *previews,
    ]
    omitted = len(matches) - MAX_GREP_PREVIEWS
    if omitted > 0:
        body.append(f"... {omitted} additional matches omitted from previews; use read_lines for exact ranges.")
    return {"result": _cap_text("\n".join(body)), "matches": matches}


def read_lines(start: int, end: int, global_storage: Any = None) -> dict[str, Any]:
    """Return raw document lines in the half-open range [start, end)."""
    lines = get_corpus_lines(global_storage)
    total = len(lines)
    start = max(0, int(start))
    end = total if int(end) == -1 else min(total, int(end))
    end = max(start, end)
    truncated = end - start > MAX_READ_LINES
    if truncated:
        end = start + MAX_READ_LINES
    text = _numbered(lines, start, end)
    if truncated:
        text += f"\n...[truncated at {MAX_READ_LINES} lines; request a smaller or later range]..."
    return {"result": _cap_text(text), "start": start, "end": end, "total_lines": total}


def bash(command: str, timeout: int = 120, global_storage: Any = None) -> dict[str, Any]:
    """Run a shell command inside the per-task workspace."""
    workspace = get_workspace(global_storage)
    workspace.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=str(workspace),
        shell=True,
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout or 120)),
        check=False,
    )
    parts = (
        f"$ {command}",
        f"[exit_code] {completed.returncode}",
        completed.stdout.strip(),
        completed.stderr.strip(),
    )
    return {"result": _cap_text("\n".join(p for p in parts if p)), "exit_code": completed.returncode}


def notepad(action: str, content: str = "", global_storage: Any = None) -> dict[str, Any]:
    """Scratchpad that persists across tool rounds of one episode."""
    action = (action or "read").lower()
    if action == "write":
        storage_set(global_storage, NOTEPAD_KEY, content or "")
        return {"result": "Notepad updated."}
    if action == "read":
        return {"result": storage_get(global_storage, NOTEPAD_KEY, "") or "(notepad is empty)"}
    return {"result": f"Unknown notepad action: {action!r}"}
