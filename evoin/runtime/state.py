"""Per-task tool state shared by all tool calls of one episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CORPUS_KEY = "evoin.corpus_lines"
META_KEY = "evoin.meta_md"
NOTEPAD_KEY = "evoin.notepad"
WORKSPACE_KEY = "evoin.workspace"
TASK_ID_KEY = "evoin.task_id"


def storage_get(global_storage: Any, key: str, default: Any = None) -> Any:
    if global_storage is None:
        return default
    getter = getattr(global_storage, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def storage_set(global_storage: Any, key: str, value: Any) -> None:
    if global_storage is None:
        raise RuntimeError("tool state is missing global_storage")
    setter = getattr(global_storage, "__setitem__", None)
    if not callable(setter):
        raise RuntimeError("unsupported global_storage object")
    setter(key, value)


def get_corpus_lines(global_storage: Any) -> list[str]:
    lines = storage_get(global_storage, CORPUS_KEY)
    if not isinstance(lines, list):
        raise RuntimeError("no document is loaded into global_storage")
    return [str(line) for line in lines]


def get_workspace(global_storage: Any) -> Path:
    raw = storage_get(global_storage, WORKSPACE_KEY)
    return Path(str(raw)) if raw else Path.cwd()
