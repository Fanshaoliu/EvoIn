"""Tool-calling agent loop that executes a harness with a pluggable target model.

Trajectory steps use the rollout row format shared by evolution and Tailor::

    {"role": "user", "text": question}
    {"role": "assistant", "reasoning": str?, "text": str?, "tool_calls": [{"id", "name", "input"}]?}
    {"role": "tool", "observations": [{"tool_use_id", "content", "is_error"}]}
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evoin.models import AgentModel
from evoin.runtime.builtin import STOP_TOOL_FLAG
from evoin.runtime.config import (
    AgentConfig,
    accepts_global_storage,
    activate_harness_path,
    bind_tools,
    build_middlewares,
    load_agent_config,
    render_system_prompt,
    tool_schemas,
)
from evoin.runtime.formatting import format_tool_output, wrap_tool_result
from evoin.runtime.state import CORPUS_KEY, WORKSPACE_KEY
from evoin.runtime.task import new_task_storage

MAX_ITERATIONS_STATUS = "max_iterations_reached"
MAX_ITERATIONS_TEXT = "[Error: Maximum iteration limit reached.]"


@dataclass
class LoadedHarness:
    """A harness whose configuration and tool bindings are resolved once per process."""

    config: AgentConfig
    functions: dict[str, Callable[..., Any]]


def load_harness_runtime(harness_dir: str | Path) -> LoadedHarness:
    config = load_agent_config(harness_dir)
    activate_harness_path(config.harness_dir)
    return LoadedHarness(config=config, functions=bind_tools(config))


def _normalise_calls(raw_calls: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in raw_calls or []:
        raw = raw if isinstance(raw, dict) else {}
        name = str(raw.get("name") or "")
        args = raw.get("input")
        calls.append({
            "id": str(raw.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "name": name or "MALFORMED_TOOL_CALL",
            "input": args if isinstance(args, dict) else {},
            "malformed": not name or not isinstance(args, dict),
        })
    return calls


def _execute(
    name: str,
    args: dict[str, Any],
    *,
    functions: dict[str, Any],
    middlewares: list[Any],
    storage: dict[str, Any],
) -> tuple[str, bool, bool]:
    """Run one tool call; returns (observation, is_error, stop_requested)."""
    function = functions.get(name)
    if function is None:
        body = format_tool_output({"error": f"Tool '{name}' not found"}, is_error=True)
        return wrap_tool_result(name, body, is_error=True), True, False
    try:
        raw = function(**args, global_storage=storage) if accepts_global_storage(function) else function(**args)
        stop = isinstance(raw, dict) and bool(raw.get(STOP_TOOL_FLAG))
        for middleware in middlewares:
            hook = getattr(middleware, "after_tool", None)
            if callable(hook):
                raw = hook(name, args, raw)
        body = format_tool_output(raw, is_error=False)
        return wrap_tool_result(name, body, is_error=False), False, stop
    except Exception as exc:  # noqa: BLE001 - tool failures are shown to the model
        body = format_tool_output({"error": f"{type(exc).__name__}: {exc}"}, is_error=True)
        return wrap_tool_result(name, body, is_error=True), True, False


def run_agent(
    *,
    harness: LoadedHarness | str | Path,
    task: dict[str, Any],
    work_dir: str | Path,
    model: AgentModel,
) -> dict[str, Any]:
    """Run one episode of ``model`` under ``harness`` and return the rollout target."""
    if not isinstance(harness, LoadedHarness):
        harness = load_harness_runtime(harness)
    config, functions = harness.config, harness.functions
    middlewares = build_middlewares(config)
    storage, normalized = new_task_storage(task, Path(work_dir))
    task_id = str(normalized["metadata"]["task_id"])
    context = {
        "task_id": task_id,
        "document_path": "./document.md",
        "num_lines": len(storage[CORPUS_KEY]),
        "method": "base_harness",
    }
    question = str(normalized.get("question") or "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": render_system_prompt(config, context)},
        {"role": "user", "content": question},
    ]
    steps: list[dict[str, Any]] = [{"role": "user", "text": question}]
    tools = tool_schemas(config)
    status, final_answer, finished = "ok", "", False

    for _ in range(config.max_iterations):
        try:
            turn = model(messages, tools) or {}
        except Exception as exc:  # noqa: BLE001 - a failed model call ends the episode
            status = f"error:model:{type(exc).__name__}:{exc}"
            break
        reasoning = str(turn.get("reasoning") or "").strip()
        text = str(turn.get("text") or "").strip()
        calls = _normalise_calls(turn.get("tool_calls"))

        step: dict[str, Any] = {"role": "assistant"}
        if reasoning:
            step["reasoning"] = reasoning
        if text:
            step["text"] = text
        if calls:
            step["tool_calls"] = [
                {"id": c["id"], "name": c["name"], "input": c["input"], **({"malformed": True} if c["malformed"] else {})}
                for c in calls
            ]
        steps.append(step)
        messages.append({
            "role": "assistant",
            "content": text,
            **({"tool_calls": [{"id": c["id"], "name": c["name"], "input": c["input"]} for c in calls]}
               if calls else {}),
        })
        if not calls:
            final_answer, finished = text, True
            break

        observations: list[dict[str, Any]] = []
        stop = False
        for call in calls:
            if call["malformed"]:
                body = format_tool_output({"error": "Malformed tool call: arguments must be a JSON object"},
                                          is_error=True)
                content, is_error, requested = wrap_tool_result(call["name"], body, is_error=True), True, False
            else:
                content, is_error, requested = _execute(
                    call["name"], call["input"], functions=functions, middlewares=middlewares, storage=storage
                )
            observations.append({"tool_use_id": call["id"], "content": content, "is_error": is_error})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            if requested and call["name"] in config.stop_tools:
                final_answer, stop = str(call["input"].get("result", "")), True
        steps.append({"role": "tool", "observations": observations})
        if stop:
            finished = True
            break
    else:
        status = MAX_ITERATIONS_STATUS
        steps.append({"role": "assistant", "text": MAX_ITERATIONS_TEXT})

    if not finished and status == "ok":
        status = MAX_ITERATIONS_STATUS
    tool_calls = [call for step in steps for call in step.get("tool_calls", [])]
    return {
        "task_id": task_id,
        "status": status,
        "model_output_full": final_answer,
        "n_steps": len(steps),
        "tool_call_names": [c["name"] for c in tool_calls if not c.get("malformed")],
        "n_malformed_tool_calls": sum(1 for c in tool_calls if c.get("malformed")),
        "trajectory": {"steps": steps},
        "judge_task": normalized,
    }
