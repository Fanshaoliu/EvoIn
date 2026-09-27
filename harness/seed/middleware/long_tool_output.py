"""Head/tail truncation of long tool outputs.

Middleware interface used by the runtime: an object with an
``after_tool(tool_name, tool_input, output) -> output`` method, called on the
raw tool output before it is rendered into an observation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


class LongToolOutputMiddleware:
    """Keep the head and tail of an over-long tool output and mark the gap."""

    def __init__(
        self,
        *,
        max_output_chars: int = 12000,
        head_lines: int = 80,
        tail_lines: int = 40,
        head_chars: int = 6000,
        tail_chars: int = 6000,
        bypass_tool_names: Sequence[str] | None = None,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be >= 1")
        if min(head_lines, tail_lines, head_chars, tail_chars) < 0:
            raise ValueError("head/tail sizes must be >= 0")
        if head_chars + tail_chars > max_output_chars:
            raise ValueError("head_chars + tail_chars must be <= max_output_chars")
        self.max_output_chars = max_output_chars
        self.head_lines = head_lines
        self.tail_lines = tail_lines
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.bypass_tool_names = frozenset(bypass_tool_names or ())

    def after_tool(self, tool_name: str, tool_input: dict[str, Any], output: Any) -> Any:
        del tool_input
        if tool_name in self.bypass_tool_names:
            return output
        serialized = self._serialize(output)
        if len(serialized) <= self.max_output_chars:
            return output
        key, text = self._content_field(output)
        source = text if text is not None else serialized
        hint = (
            f"\n\n⚠️ [LongToolOutputMiddleware] The full output ({len(source):,} chars, "
            f"~{source.count(chr(10)) + 1} lines) has been truncated."
        )
        truncated = self._truncate(source) + hint
        if isinstance(output, dict):
            rebuilt = dict(output)
            rebuilt[key or ("content" if "content" in rebuilt else "result")] = truncated
            return rebuilt
        return truncated

    @staticmethod
    def _content_field(output: Any) -> tuple[str | None, str | None]:
        if isinstance(output, dict):
            for key in ("content", "result"):
                if isinstance(output.get(key), str):
                    return key, output[key]
        return None, None

    @staticmethod
    def _serialize(output: Any) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(output)

    def _truncate(self, text: str) -> str:
        """Keep the longer of the line- and char-based views that still fits."""
        candidates = [c for c in (self._by_lines(text), self._by_chars(text)) if c is not None]
        if not candidates:
            return text
        shorter = [c for c in candidates if len(c) < len(text)]
        if shorter:
            candidates = shorter
        fitting = [c for c in candidates if len(c) < self.max_output_chars]
        if fitting:
            return max(fitting, key=len)
        return candidates[0] if len(candidates) == 1 else min(candidates, key=len)

    def _by_lines(self, text: str) -> str | None:
        if self.head_lines == 0 and self.tail_lines == 0:
            return None
        lines = text.splitlines(keepends=True)
        if len(lines) <= self.head_lines + self.tail_lines:
            return text
        head = lines[: self.head_lines]
        tail = lines[-self.tail_lines:] if self.tail_lines > 0 else []
        omitted = len(lines) - self.head_lines - self.tail_lines
        return "".join(head) + f"\n... [{omitted} lines omitted] ...\n" + "".join(tail)

    def _by_chars(self, text: str) -> str | None:
        if self.head_chars == 0 and self.tail_chars == 0:
            return None
        if len(text) <= self.head_chars + self.tail_chars:
            return text
        head = text[: self.head_chars]
        tail = text[len(text) - self.tail_chars:] if self.tail_chars > 0 else ""
        omitted = len(text) - self.head_chars - self.tail_chars
        return head + f"\n... [{omitted} chars omitted] ...\n" + tail
