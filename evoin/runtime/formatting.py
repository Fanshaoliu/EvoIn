"""Render raw tool outputs into the observation text the model sees."""

from __future__ import annotations

import json
from typing import Any

_DISPLAY_ONLY_KEYS = frozenset({"returnDisplay"})


def _strip_display_only(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_display_only(item) for item in value]
    if isinstance(value, dict):
        return {k: _strip_display_only(v) for k, v in value.items() if k not in _DISPLAY_ONLY_KEYS}
    return value


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _body_key(value: dict[str, Any], *, is_error: bool) -> str | None:
    preferred = (
        ("error", "traceback", "stderr", "message", "result", "content", "stdout")
        if is_error
        else ("result", "content", "stdout", "stderr", "message")
    )
    for key in preferred:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return key
    longest, length = None, -1
    for key, item in value.items():
        if isinstance(item, str) and "\n" in item and len(item) > length:
            longest, length = key, len(item)
    return longest


def _meta_line(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (str, int, float)):
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False)
    if "\n" in text:
        indented = "\n".join(f"  {line}" for line in text.splitlines())
        return f"- `{key}`:\n{indented}"
    return f"- `{key}`: {text}"


def format_tool_output(tool_output: Any, *, is_error: bool = False) -> str:
    """Flatten a tool output into stable Markdown text.

    A dict with a single ``result`` or ``content`` field is passed through as
    its bare value; other dicts become a ``## Tool Result`` document whose
    non-body keys are listed under ``### Metadata``.
    """
    value = _strip_display_only(tool_output)
    if isinstance(value, dict) and set(value) in ({"result"}, {"content"}):
        return _serialize(next(iter(value.values())))
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return "\n\n".join(["## Tool Result", f"### Body (`result`)\n\n{_serialize(value)}"])
    body_key = _body_key(value, is_error=is_error)
    meta: list[str] = []
    body = ""
    for key, item in value.items():
        if key == body_key:
            body = f"### Body (`{key}`)\n\n{_serialize(item)}"
            continue
        line = _meta_line(key, item)
        if line is not None:
            meta.append(line)
    sections = ["## Tool Result"]
    if meta:
        sections.append("### Metadata\n" + "\n".join(meta))
    if body:
        sections.append(body)
    return "\n\n".join(sections)


def wrap_tool_result(tool_name: str, body: str, *, is_error: bool) -> str:
    tag = "error" if is_error else "result"
    return (
        f"\n<tool_result>\n<tool_name>{tool_name}</tool_name>\n"
        f"<{tag}>{body}</{tag}>\n</tool_result>\n"
    )
