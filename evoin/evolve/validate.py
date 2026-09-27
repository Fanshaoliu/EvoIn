"""Static validation gate for candidate harnesses."""
from __future__ import annotations

import json
import py_compile
import traceback
from pathlib import Path
from typing import Any

from evoin.runtime.config import activate_harness_path, import_symbol, load_agent_config

REQUIRED_STOP_TOOL = "complete_task"


def _compile_local_py(harness_dir: Path) -> list[str]:
    compiled: list[str] = []
    for sub in ("tool_impl", "middleware"):
        base = harness_dir / sub
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            py_compile.compile(str(path), doraise=True)
            compiled.append(str(path.relative_to(harness_dir)))
    return compiled


def validate_harness_static(harness_dir: str | Path, *, write_json: bool = True) -> dict:
    """Validate a materialized harness without any model call.

    Checks that every harness-local module compiles, every configured tool binds
    to an importable callable, every middleware can be instantiated with its
    configured parameters, and the full agent configuration loads.
    """
    harness_dir = Path(harness_dir)
    report: dict[str, Any] = {"ok": False, "harness": harness_dir.name, "checks": {}, "errors": []}
    try:
        compiled = _compile_local_py(harness_dir)
        report["checks"]["py_compile"] = {"ok": True, "files": compiled}

        config = load_agent_config(harness_dir)
        report["checks"]["agent_config"] = {"ok": True}
        if REQUIRED_STOP_TOOL not in config.stop_tools:
            report["errors"].append(f"{REQUIRED_STOP_TOOL} must remain a stop tool")

        activate_harness_path(harness_dir)
        bindings = []
        for tool in config.tools:
            obj = import_symbol(tool.binding)
            if not callable(obj):
                report["errors"].append(f"tool {tool.name} binding is not callable: {tool.binding}")
            if tool.input_schema.get("type") != "object":
                report["errors"].append(f"tool {tool.name} input_schema must be an object schema")
            bindings.append({"tool": tool.name, "binding": tool.binding})
        report["checks"]["bindings"] = {"ok": not report["errors"], "items": bindings}

        middleware_items = []
        for spec in config.middlewares:
            import_symbol(spec.import_spec)(**spec.params)
            middleware_items.append({"import": spec.import_spec, "params": spec.params})
        report["checks"]["middlewares"] = {"ok": True, "items": middleware_items}
        report["ok"] = not report["errors"]
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()

    if write_json:
        (harness_dir / "validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report
