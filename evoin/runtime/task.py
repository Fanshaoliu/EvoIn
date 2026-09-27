"""Task normalization and per-task document workspaces.

Accepted task records (JSON objects, one per line in a JSONL file):

* ``{"task_id", "question", "document", "answer", "rubrics"?}`` -- the simplest
  form; ``document`` is the full hidden text.
* ``{"task_id", "question", "context_messages": [{"role", "content"}], ...}``.
* SFT-style ``{"messages": [...]}`` whose user message holds the document
  followed by a ``Question:`` marker, and whose assistant message is the answer.

The document is rendered into a line list; tools address it with 0-based line
numbers and the workspace directory holds a copy as ``document.md``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from evoin.runtime.state import CORPUS_KEY, META_KEY, NOTEPAD_KEY, TASK_ID_KEY, WORKSPACE_KEY

MAX_LINE_LEN = 4000
_QUESTION_MARKERS = ("\n\u95ee\u9898:", "\n\u95ee\u9898\uff1a", "\nQuestion:", "\nQUESTION:")


def task_id_of(task: dict[str, Any]) -> str:
    return str(
        (task.get("metadata") or {}).get("task_id")
        or task.get("task_id")
        or task.get("id")
        or ""
    )


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``context_messages``, ``question`` and ``metadata.task_id``."""
    normalized = dict(task)
    if not normalized.get("context_messages") and isinstance(normalized.get("document"), str):
        normalized["context_messages"] = [{"role": "user", "content": normalized["document"]}]

    messages = list(normalized.get("messages") or [])
    if "answer" not in normalized:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                normalized["answer"] = message.get("content", "")
                break
    if not normalized.get("context_messages") and messages:
        user = next((m for m in messages if m.get("role") == "user"), messages[0])
        content = str(user.get("content", "") or "")
        question = normalized.get("question")
        document = content
        for marker in _QUESTION_MARKERS:
            if marker in content:
                document, question_part = content.rsplit(marker, 1)
                question = str(question or question_part).strip()
                break
        normalized["context_messages"] = [{"role": "user", "content": document.strip()}]
        if question:
            normalized["question"] = str(question).strip()

    metadata = dict(normalized.get("metadata") or {})
    metadata.setdefault("task_id", task_id_of(normalized) or uuid.uuid4().hex)
    normalized["metadata"] = metadata
    normalized.setdefault("question", "")
    return normalized


def _split_long_lines(lines: list[str]) -> list[str]:
    """Break very long lines so line-addressed tools can still reach them."""
    out: list[str] = []
    for line in lines:
        if len(line) <= MAX_LINE_LEN:
            out.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith(("{", "[")):
            try:
                out.extend(json.dumps(json.loads(stripped), ensure_ascii=False, indent=2).splitlines())
                continue
            except (json.JSONDecodeError, TypeError, RecursionError):
                pass
        for offset in range(0, len(line), MAX_LINE_LEN):
            chunk = line[offset:offset + MAX_LINE_LEN]
            out.append(chunk if offset == 0 else f"↩ {chunk}")
    return out


def build_corpus(context_messages: list[dict[str, Any]]) -> list[str]:
    """Render context messages into a role-delimited, searchable line list."""
    parts: list[str] = []
    for message in context_messages:
        role = message.get("role", "unknown")
        parts.append(f"--- [{role} message start] ---")
        parts.append(str(message.get("content", "") or ""))
        parts.append(f"--- [{role} message end] ---")
    return _split_long_lines("\n".join(parts).splitlines())


def write_workspace(directory: Path, corpus: list[str]) -> Path:
    workspace = Path(directory) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "document.md").write_text("\n".join(corpus), encoding="utf-8")
    return workspace


def new_task_storage(task: dict[str, Any], work_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize ``task`` and create the tool state for one episode."""
    normalized = normalize_task(task)
    task_id = str(normalized["metadata"]["task_id"])
    corpus = build_corpus(normalized.get("context_messages") or [])
    workspace = write_workspace(Path(work_dir), corpus)
    storage: dict[str, Any] = {
        TASK_ID_KEY: task_id,
        CORPUS_KEY: corpus,
        WORKSPACE_KEY: str(workspace),
        META_KEY: "",
        NOTEPAD_KEY: "",
    }
    return storage, normalized
