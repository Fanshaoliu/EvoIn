"""Strict conversion of rollout rows into ATIF trajectories.

A rollout row stores ``trajectory.steps`` as user / assistant / tool steps (see
``evoin.runtime.agent``). ATIF keeps each agent step together with the
observations of its own tool calls, bound by ``tool_call_id``.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PLACEHOLDER = (
    "(seed system prompt — provided separately in <seed_system_prompt>; "
    "this step is context only and is replaced downstream)"
)
ASSISTANT_TEXT_TERMINAL = "assistant_text"
COMPLETE_TASK_TERMINAL = "native_complete_task"
VALID_TERMINAL_MODES = frozenset({
    ASSISTANT_TEXT_TERMINAL,
    COMPLETE_TASK_TERMINAL,
})


class TrajectoryBindingError(ValueError):
    """Raised when source tool calls and observations cannot be bound exactly."""


def _source_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    steps = (record.get("trajectory") or {}).get("steps") or []
    if not isinstance(steps, list):
        raise TrajectoryBindingError("trajectory.steps is not a list")
    return steps


def _is_observation_step(step: dict[str, Any]) -> bool:
    return step.get("role") in {"user", "tool"} and "observations" in step


def terminal_mode_from_rollout(record: dict[str, Any]) -> str | None:
    """Classify a valid source terminal without changing its behavior."""
    final_assistant = next(
        (
            step
            for step in reversed(_source_steps(record))
            if isinstance(step, dict) and step.get("role") == "assistant"
        ),
        None,
    )
    if final_assistant is None:
        return None
    calls = final_assistant.get("tool_calls") or []
    if not isinstance(calls, list):
        raise TrajectoryBindingError("final assistant tool_calls is not a list")
    if any(
        isinstance(call, dict) and call.get("name") == "complete_task"
        for call in calls
    ):
        return COMPLETE_TASK_TERMINAL
    if calls:
        return None
    text = final_assistant.get("text")
    if isinstance(text, str) and text.strip():
        return ASSISTANT_TEXT_TERMINAL
    return None


def terminal_mode_from_atif(atif: dict[str, Any]) -> str | None:
    """Classify a valid ATIF terminal without requiring one fixed protocol."""
    final_agent = next(
        (
            step
            for step in reversed(atif.get("steps") or [])
            if isinstance(step, dict) and step.get("source") == "agent"
        ),
        None,
    )
    if final_agent is None:
        return None
    calls = final_agent.get("tool_calls") or []
    if not isinstance(calls, list):
        return None
    if any(
        isinstance(call, dict) and call.get("function_name") == "complete_task"
        for call in calls
    ):
        return COMPLETE_TASK_TERMINAL
    if calls:
        return None
    message = final_agent.get("message")
    if isinstance(message, str) and message.strip():
        return ASSISTANT_TEXT_TERMINAL
    return None


def final_answer_from_atif(atif: dict[str, Any]) -> str | None:
    """Return the answer from either supported terminal representation."""
    final_agent = next(
        (
            step
            for step in reversed(atif.get("steps") or [])
            if isinstance(step, dict) and step.get("source") == "agent"
        ),
        None,
    )
    if final_agent is None:
        return None
    for call in final_agent.get("tool_calls") or []:
        if (
            isinstance(call, dict)
            and call.get("function_name") == "complete_task"
        ):
            value = (call.get("arguments") or {}).get("result")
            return value if isinstance(value, str) and value.strip() else None
    if final_agent.get("tool_calls"):
        return None
    message = final_agent.get("message")
    return message if isinstance(message, str) and message.strip() else None


def convert_rollout_to_atif(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one rollout and reject ambiguous tool-result attachment."""
    steps_in = _source_steps(record)
    out: list[dict[str, Any]] = [{"source": "system", "message": SYSTEM_PLACEHOLDER}]
    agent_step_id = 0
    call_counter = 0
    index = 0
    while index < len(steps_in):
        source = steps_in[index]
        role = source.get("role")
        if role == "user" and "text" in source:
            out.append({"source": "user", "message": str(source.get("text") or "")})
            index += 1
            continue
        if role == "assistant":
            agent_step_id += 1
            atif_step: dict[str, Any] = {
                "source": "agent",
                "step_id": agent_step_id,
                "reasoning_content": str(source.get("reasoning") or ""),
                "message": str(source.get("text") or ""),
            }
            calls = source.get("tool_calls") or []
            if not isinstance(calls, list):
                raise TrajectoryBindingError(
                    f"assistant step {agent_step_id}: tool_calls is not a list"
                )
            atif_calls: list[dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, dict) or not call.get("name"):
                    raise TrajectoryBindingError(
                        f"assistant step {agent_step_id}: malformed tool call"
                    )
                call_counter += 1
                atif_calls.append({
                    "tool_call_id": f"c{call_counter}",
                    "function_name": str(call["name"]),
                    "arguments": call.get("input") or {},
                    "source_tool_call_id": call.get("id"),
                })
            if atif_calls:
                atif_step["tool_calls"] = atif_calls
                if index + 1 >= len(steps_in) or not _is_observation_step(steps_in[index + 1]):
                    raise TrajectoryBindingError(
                        f"assistant step {agent_step_id}: {len(atif_calls)} tool call(s) "
                        "have no immediate observation step"
                    )
                observations = steps_in[index + 1].get("observations") or []
                if not isinstance(observations, list) or len(observations) != len(atif_calls):
                    actual = len(observations) if isinstance(observations, list) else "non-list"
                    raise TrajectoryBindingError(
                        f"assistant step {agent_step_id}: {len(atif_calls)} tool call(s) "
                        f"but {actual} observation(s)"
                    )
                atif_step["observation"] = {
                    "results": [
                        {
                            "source_call_id": atif_calls[offset]["tool_call_id"],
                            "content": str(observation.get("content") or ""),
                            "is_error": bool(observation.get("is_error", False)),
                            "source_observation_index": offset,
                            "replay_status": "source_saved",
                        }
                        for offset, observation in enumerate(observations)
                    ]
                }
                out.append(atif_step)
                index += 2
                continue
            if index + 1 < len(steps_in) and _is_observation_step(steps_in[index + 1]):
                raise TrajectoryBindingError(
                    f"assistant step {agent_step_id}: observation follows a step with no tool calls"
                )
            out.append(atif_step)
            index += 1
            continue
        if _is_observation_step(source):
            raise TrajectoryBindingError(f"orphan observation at source step {index}")
        if role in {"system", None}:
            index += 1
            continue
        raise TrajectoryBindingError(f"unsupported source role {role!r} at step {index}")
    return {"steps": out}


def count_agent_steps(atif: dict[str, Any]) -> int:
    return sum(1 for step in atif.get("steps", []) if step.get("source") == "agent")


def ends_with_complete_task(record: dict[str, Any]) -> bool:
    """True only if the final assistant message calls complete_task."""
    return terminal_mode_from_rollout(record) == COMPLETE_TASK_TERMINAL
