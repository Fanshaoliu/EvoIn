"""The seed harness that rewritten trajectories must be reproducible under."""

from __future__ import annotations

import json
import sys
import threading
from functools import lru_cache
from typing import Any, Callable

from evoin.runtime.agent import LoadedHarness, load_harness_runtime
from evoin.runtime.config import AGENT_YAML, import_symbol
from evoin.tailor import config

_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _harness() -> LoadedHarness:
    with _LOAD_LOCK:
        return load_harness_runtime(config.SEED_HARNESS_DIR)


def functions() -> dict[str, Callable[..., Any]]:
    return _harness().functions


def seed_system_prompt() -> str:
    return _harness().config.system_prompt.strip()


def seed_max_iterations() -> int:
    return int(_harness().config.max_iterations)


def tool_order() -> list[str]:
    return [tool.name for tool in _harness().config.tools]


def seed_tool_schemas() -> dict[str, dict[str, Any]]:
    return {tool.name: tool.input_schema for tool in _harness().config.tools}


@lru_cache(maxsize=1)
def seed_tools_block() -> str:
    chunks = [
        "\n".join((
            f"## Tool: {tool.name}",
            "",
            tool.description,
            "",
            "Arguments (JSON Schema):",
            "```json",
            json.dumps(tool.input_schema, ensure_ascii=False, indent=2),
            "```",
        ))
        for tool in _harness().config.tools
    ]
    return "\n\n".join(chunks).strip()


def cap_text(text: str) -> str:
    """The seed ``bash`` tool's own output cap, so replayed bash matches it."""
    bash = functions().get("bash")
    module = sys.modules.get(getattr(bash, "__module__", "") or "")
    capper = getattr(module, "_cap_text", None)
    return capper(text) if callable(capper) else text


@lru_cache(maxsize=1)
def replay_truncator() -> Callable[[str], str] | None:
    """Head/tail truncation from the seed's long-output middleware, if it has one."""
    for spec in _harness().config.middlewares:
        if not spec.import_spec.endswith(":LongToolOutputMiddleware"):
            continue
        params = {**spec.params, "max_output_chars": config.RUNTIME_OBSERVATION_LIMIT}
        head = int(params.get("head_chars", 0))
        tail = int(params.get("tail_chars", 0))
        if head + tail > config.RUNTIME_OBSERVATION_LIMIT:
            params["head_chars"] = params["tail_chars"] = config.RUNTIME_OBSERVATION_LIMIT // 2
        middleware = import_symbol(spec.import_spec)(**params)
        return middleware._truncate  # noqa: SLF001 - reuse the seed's truncation exactly
    return None


@lru_cache(maxsize=1)
def seed_fingerprints() -> dict[str, str]:
    harness_dir = config.SEED_HARNESS_DIR
    tool_files = {
        path.name: config.sha256_file(path)
        for path in sorted((harness_dir / "tools").glob("*.tool.yaml"))
    }
    code_files = {
        str(path.relative_to(harness_dir)): config.sha256_file(path)
        for folder in ("tool_impl", "middleware")
        for path in sorted((harness_dir / folder).glob("*.py"))
    }
    return {
        "seed_system_prompt_sha256": config.sha256_text(seed_system_prompt()),
        "seed_config_sha256": config.sha256_file(harness_dir / AGENT_YAML),
        "seed_tool_files_sha256": config.canonical_hash(tool_files),
        "seed_code_sha256": config.canonical_hash(code_files),
        "seed_tool_schemas_sha256": config.canonical_hash(seed_tool_schemas()),
        "seed_tool_order_sha256": config.canonical_hash(tool_order()),
    }


def seed_manifest() -> dict[str, Any]:
    return {
        "harness": config.SEED_HARNESS_DIR.name,
        "max_iterations": seed_max_iterations(),
        "tool_order": tool_order(),
        "fingerprints": seed_fingerprints(),
    }
