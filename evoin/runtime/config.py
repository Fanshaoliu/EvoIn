"""Load a harness directory into an executable agent configuration."""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

AGENT_YAML = "base_harness.yaml"
_SCHEMA_ONLY_KEYS = ("$schema",)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    binding: str


@dataclass
class MiddlewareSpec:
    import_spec: str
    params: dict[str, Any]


@dataclass
class AgentConfig:
    harness_dir: Path
    name: str
    system_prompt: str
    system_prompt_type: str
    max_iterations: int
    max_context_tokens: int
    tools: list[ToolSpec]
    stop_tools: list[str]
    middlewares: list[MiddlewareSpec] = field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _load_tool(harness_dir: Path, entry: dict[str, Any]) -> ToolSpec:
    name = str(entry.get("name") or "")
    if not name:
        raise ValueError("tool entry without a name")
    binding = str(entry.get("binding") or "")
    if not binding:
        raise ValueError(f"tool {name!r} has no binding")
    yaml_path = entry.get("yaml_path")
    spec: dict[str, Any] = {}
    if yaml_path:
        spec = _read_yaml((harness_dir / str(yaml_path)).resolve())
    description = str(spec.get("template_override") or spec.get("description") or "").strip()
    schema = dict(spec.get("input_schema") or {"type": "object", "properties": {}})
    for key in _SCHEMA_ONLY_KEYS:
        schema.pop(key, None)
    return ToolSpec(name=name, description=description, input_schema=schema, binding=binding)


def load_agent_config(harness_dir: str | Path) -> AgentConfig:
    harness_dir = Path(harness_dir).resolve()
    raw = _read_yaml(harness_dir / AGENT_YAML)
    prompt_ref = str(raw.get("system_prompt") or "")
    prompt_path = (harness_dir / prompt_ref).resolve() if prompt_ref.startswith(".") else None
    if prompt_path is not None:
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = prompt_ref
    tools = [_load_tool(harness_dir, entry) for entry in raw.get("tools") or [] if isinstance(entry, dict)]
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate tool names in {harness_dir / AGENT_YAML}: {names}")
    stop_tools = [str(tool) for tool in raw.get("stop_tools") or []]
    for stop in stop_tools:
        if stop not in names:
            raise ValueError(f"stop tool {stop!r} is not a configured tool")
    middlewares = [
        MiddlewareSpec(import_spec=str(item.get("import") or ""), params=dict(item.get("params") or {}))
        for item in raw.get("middlewares") or []
        if isinstance(item, dict)
    ]
    return AgentConfig(
        harness_dir=harness_dir,
        name=str(raw.get("name") or harness_dir.name),
        system_prompt=system_prompt,
        system_prompt_type=str(raw.get("system_prompt_type") or "string"),
        max_iterations=int(raw.get("max_iterations") or 30),
        max_context_tokens=int(raw.get("max_context_tokens") or 0),
        tools=tools,
        stop_tools=stop_tools,
        middlewares=middlewares,
    )


def activate_harness_path(harness_dir: str | Path) -> None:
    """Make ``tool_impl`` and ``middleware`` of this harness importable.

    Harness-local packages share names across harnesses, so any previously
    imported copies are evicted before the new harness is pushed to the front
    of ``sys.path``.
    """
    path = str(Path(harness_dir).resolve())
    for module_name in list(sys.modules):
        root = module_name.split(".", 1)[0]
        if root in {"tool_impl", "middleware"}:
            del sys.modules[module_name]
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def import_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"binding must be 'module:symbol', got {spec!r}")
    module_name, symbol = spec.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in symbol.split("."):
        obj = getattr(obj, part)
    return obj


def bind_tools(config: AgentConfig) -> dict[str, Callable[..., Any]]:
    functions: dict[str, Callable[..., Any]] = {}
    for tool in config.tools:
        function = import_symbol(tool.binding)
        if not callable(function):
            raise TypeError(f"tool {tool.name!r} binding is not callable: {tool.binding}")
        functions[tool.name] = function
    return functions


def build_middlewares(config: AgentConfig) -> list[Any]:
    return [import_symbol(spec.import_spec)(**spec.params) for spec in config.middlewares]


def accepts_global_storage(function: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return "global_storage" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def render_system_prompt(config: AgentConfig, context: dict[str, Any]) -> str:
    if config.system_prompt_type != "jinja":
        return config.system_prompt
    try:
        import jinja2
    except ImportError:
        rendered = config.system_prompt
        for key, value in context.items():
            rendered = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", str(value), rendered)
        return rendered
    return jinja2.Template(config.system_prompt).render(**context)


def tool_schemas(config: AgentConfig) -> list[dict[str, Any]]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in config.tools
    ]
