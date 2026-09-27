"""Replay retained tool calls under the seed harness and regenerate observations.

Every observation that reaches training data is produced here by the seed
tools against the task document, never copied from the source rollout, so a
rewritten trajectory only cites evidence the seed harness really returns.
"""

from __future__ import annotations

import copy
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from evoin.runtime.builtin import complete_task
from evoin.runtime.config import accepts_global_storage
from evoin.runtime.formatting import format_tool_output, wrap_tool_result
from evoin.runtime.state import CORPUS_KEY, WORKSPACE_KEY
from evoin.runtime.task import new_task_storage
from evoin.tailor import config, seed
from evoin.tailor.trajectory import final_answer_from_atif

_TRUNCATION_MARKERS = (
    "[truncated",
    "[longtooloutputmiddleware]",
    " chars omitted]",
    " lines omitted]",
)
# A leading scalar, optionally after a Chinese "answer:" prefix.
_SCALAR_ANSWER_PATTERN = re.compile(
    r"^\s*(?:\u7b54\u6848\s*[:\uff1a]\s*)?(?P<value>[-+]?\d+(?:\.\d+)?)\b"
)
_BASH_REFERENCE_PATTERN = re.compile(
    r"\b(?:bash|python(?:3)?|terminal|command\s+line)\b",
    re.IGNORECASE,
)
_REPLAYABLE_TOOLS = frozenset({"bash", "grep", "scan", "notepad", "read_lines"})


def task_sha256(task: dict[str, Any]) -> str:
    return config.canonical_hash(task)


def _validate_arguments(function_name: str, arguments: Any) -> list[str]:
    schema = seed.seed_tool_schemas().get(function_name)
    if schema is None:
        return [f"tool {function_name!r} is not in the seed toolset"]
    if not isinstance(arguments, dict):
        return [f"arguments for {function_name!r} are not an object"]
    errors: list[str] = []
    properties = schema.get("properties") or {}
    for required in schema.get("required") or []:
        if required not in arguments:
            errors.append(f"missing required argument {required!r}")
    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                errors.append(f"unknown argument {key!r}")
    type_checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
    }
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        if not expected:
            continue
        choices = expected if isinstance(expected, list) else [expected]
        if not any(type_checks.get(choice, lambda _value: True)(value) for choice in choices):
            errors.append(f"argument {key!r} has type {type(value).__name__}, expected {expected!r}")
    return errors


def _render(function_name: str, raw_output: Any, *, is_error: bool) -> tuple[str, bool]:
    """Render a raw tool output exactly as the runtime does, with a size guard."""
    body = format_tool_output(raw_output, is_error=is_error)
    truncated = False
    if len(body) > config.RUNTIME_OBSERVATION_LIMIT:
        omitted = len(body) - config.RUNTIME_OBSERVATION_LIMIT
        truncate = seed.replay_truncator()
        if truncate is not None:
            head = truncate(body)
        else:
            half = config.RUNTIME_OBSERVATION_LIMIT // 2
            head = f"{body[:half]}\n... [{omitted} chars omitted] ...\n{body[-half:]}"
        body = (
            head
            + "\n\n⚠️ [LongToolOutputMiddleware] Full output was truncated during seed replay "
            f"({omitted:,} chars omitted)."
        )
        truncated = True
    return wrap_tool_result(function_name, body, is_error=is_error), truncated


def _denied_capability(command: str) -> str | None:
    lowered = command.lower()
    for token in config.BASH_REPLAY_DENY:
        if token in lowered:
            return token
    return None


def _bash_ulimit_prefix() -> str:
    """Apply rlimits through the shell: ``preexec_fn`` can deadlock in threaded callers."""
    return (
        "ulimit "
        f"-v {max(1, config.BASH_REPLAY_MEM_BYTES // 1024)} "
        f"-t {config.BASH_REPLAY_CPU_SECONDS} "
        f"-f {max(1, config.BASH_REPLAY_FSIZE_BYTES // 512)} "
        f"-u {config.BASH_REPLAY_NPROC} 2>/dev/null; "
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001 - best-effort reap
        pass


def _replay_bash(command: str, storage: dict[str, Any], *, timeout: Any = None) -> tuple[dict[str, Any], bool]:
    """Run a stored bash call like the seed ``bash`` tool, under resource limits.

    The payload carries exactly the keys the seed tool returns: every non-body
    key is rendered into the observation's metadata block.
    """
    workspace = Path(storage[WORKSPACE_KEY])
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        requested = int(timeout) if timeout is not None else config.BASH_REPLAY_TIMEOUT
    except (TypeError, ValueError):
        requested = config.BASH_REPLAY_TIMEOUT
    limit = max(1, min(requested, config.BASH_REPLAY_TIMEOUT_MAX))
    timed_out = False
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            _bash_ulimit_prefix() + command,
            cwd=str(workspace),
            shell=True,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
        out.seek(0)
        err.seek(0)
        stdout = out.read(config.BASH_REPLAY_OUTPUT_BYTES).decode("utf-8", "replace")
        stderr = err.read(config.BASH_REPLAY_OUTPUT_BYTES).decode("utf-8", "replace")
    if timed_out:
        stderr = f"{stderr}\n[seed replay: timed out after {limit}s; process group killed]".strip()
    output = "\n".join(
        part for part in (f"$ {command}", f"[exit_code] {proc.returncode}", stdout.strip(), stderr.strip())
        if part
    )
    return {"result": seed.cap_text(output), "exit_code": proc.returncode}, timed_out


def _final_scalar_anchor(atif: dict) -> str | None:
    match = _SCALAR_ANSWER_PATTERN.match(final_answer_from_atif(atif) or "")
    return match.group("value") if match else None


def _step_text(step: dict) -> str:
    fragments: list[str] = []
    for field in ("reasoning_content", "message"):
        value = step.get(field)
        if isinstance(value, str):
            fragments.append(value)
    for call in step.get("tool_calls") or []:
        key = {"notepad": "content", "complete_task": "result"}.get(call.get("function_name"))
        value = (call.get("arguments") or {}).get(key) if key else None
        if isinstance(value, str):
            fragments.append(value)
    return "\n".join(fragments)


def drop_nonload_bearing_bash_steps(atif: dict) -> tuple[dict, list[dict]]:
    """Remove an unsafe bash call that provably does not affect the answer.

    A single-call bash step whose command reaches a denied capability is removed
    only when the final scalar answer already appears in earlier reasoning after
    a raw read and no later retained text cites bash output.  Every other bash
    call is replayed under the sandbox.
    """
    trajectory = copy.deepcopy(atif)
    scalar_anchor = _final_scalar_anchor(trajectory)
    if not scalar_anchor:
        return trajectory, []
    anchor_pattern = re.compile(rf"(?<!\d){re.escape(scalar_anchor)}(?!\d)")
    steps = trajectory.get("steps") or []
    dropped: list[dict] = []
    retained: list[dict] = []
    prior_text: list[str] = []
    prior_raw_read = False
    for index, step in enumerate(steps):
        if step.get("source") != "agent":
            retained.append(step)
            continue
        calls = [call for call in (step.get("tool_calls") or []) if isinstance(call, dict)]
        is_unsafe_bash = (
            len(calls) == 1
            and calls[0].get("function_name") == "bash"
            and _denied_capability(str((calls[0].get("arguments") or {}).get("command") or "")) is not None
        )
        later_text = "\n".join(
            _step_text(candidate) for candidate in steps[index + 1:] if candidate.get("source") == "agent"
        )
        can_drop = (
            is_unsafe_bash
            and prior_raw_read
            and bool(anchor_pattern.search("\n".join([*prior_text, _step_text(step)])))
            and not _BASH_REFERENCE_PATTERN.search(later_text)
        )
        if can_drop:
            dropped.append({
                "step_id": step.get("step_id"),
                "tool_call_id": calls[0].get("tool_call_id"),
                "function_name": "bash",
                "reason": (
                    "unsafe scalar verification was non-load-bearing: its final scalar "
                    "appears in prior reasoning and later retained text does not cite bash output"
                ),
            })
            continue
        retained.append(step)
        prior_text.append(_step_text(step))
        if any(call.get("function_name") == "read_lines" for call in calls):
            prior_raw_read = True
    trajectory["steps"] = retained
    return trajectory, dropped


def _execute_call(*, function_name: str, arguments: Any, storage: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    args = dict(arguments) if isinstance(arguments, dict) else arguments
    errors = _validate_arguments(function_name, args)
    if errors:
        return f"[seed replay unavailable: {'; '.join(errors)}]", True, {
            "status": "invalid_seed_schema", "errors": errors, "runtime_truncated": False,
        }
    assert isinstance(args, dict)
    if function_name == "complete_task":
        content, truncated = _render(function_name, complete_task(**args), is_error=False)
        return content, False, {"status": "not_required_stop_tool", "runtime_truncated": truncated}
    if function_name == "bash":
        command = str(args.get("command") or "")
        denied = _denied_capability(command)
        if denied is not None:
            return (
                "[seed replay blocked: bash command reaches a denied capability]",
                True,
                {"status": "blocked_bash", "denied_capability": denied, "runtime_truncated": False},
            )
        try:
            raw, timed_out = _replay_bash(command, storage, timeout=args.get("timeout"))
        except Exception as exc:  # noqa: BLE001 - a replay failure must become data
            error = f"{type(exc).__name__}: {exc}"
            content, truncated = _render(function_name, {"error": error}, is_error=True)
            return content, True, {"status": "execution_error", "error": error, "runtime_truncated": truncated}
        content, truncated = _render(function_name, raw, is_error=False)
        return content, False, {
            "status": "replayed",
            "bash_exit_code": raw.get("exit_code"),
            "bash_timed_out": timed_out,
            "runtime_truncated": truncated,
            "raw_result_sha256": config.canonical_hash(raw),
        }
    function = seed.functions().get(function_name)
    if function_name not in _REPLAYABLE_TOOLS or function is None:
        return (
            f"[seed replay unavailable: tool {function_name!r} has no replay adapter]",
            True,
            {"status": "unsupported_tool", "runtime_truncated": False},
        )
    try:
        raw = function(**args, global_storage=storage) if accepts_global_storage(function) else function(**args)
    except Exception as exc:  # noqa: BLE001 - a replay failure must become auditable data
        error = f"{type(exc).__name__}: {exc}"
        content, truncated = _render(function_name, {"error": error}, is_error=True)
        return content, True, {"status": "execution_error", "error": error, "runtime_truncated": truncated}
    content, truncated = _render(function_name, raw, is_error=False)
    return content, False, {
        "status": "replayed",
        "runtime_truncated": truncated,
        "raw_result_sha256": config.canonical_hash(raw),
    }


def replay_trajectory(atif: dict[str, Any], task: dict[str, Any], *, phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay every retained call in order and replace every observation."""
    trajectory = copy.deepcopy(atif)
    with tempfile.TemporaryDirectory(prefix="evoin-replay-") as work_dir:
        storage, normalized = new_task_storage(task, Path(work_dir))
        report: dict[str, Any] = {
            "phase": phase,
            "task_sha256": task_sha256(task),
            "num_lines": len(storage.get(CORPUS_KEY) or []),
            "tool_order": seed.tool_order(),
            "calls": [],
            "blocked_call_ids": [],
            "execution_error_call_ids": [],
            "runtime_truncated_call_ids": [],
        }
        for step in trajectory.get("steps", []):
            calls = step.get("tool_calls") or []
            if step.get("source") != "agent" or not calls:
                continue
            results: list[dict[str, Any]] = []
            for call in calls:
                call_id = str(call.get("tool_call_id") or "")
                function_name = str(call.get("function_name") or "")
                content, is_error, replay_meta = _execute_call(
                    function_name=function_name, arguments=call.get("arguments"), storage=storage
                )
                status = replay_meta["status"]
                if status == "blocked_bash":
                    report["blocked_call_ids"].append(call_id)
                if status in {"execution_error", "unsupported_tool", "invalid_seed_schema"}:
                    report["execution_error_call_ids"].append(call_id)
                if replay_meta.get("runtime_truncated"):
                    report["runtime_truncated_call_ids"].append(call_id)
                report["calls"].append({
                    "step_id": step.get("step_id"),
                    "tool_call_id": call_id,
                    "function_name": function_name,
                    **replay_meta,
                })
                results.append({
                    "source_call_id": call_id,
                    "content": content,
                    "is_error": is_error,
                    "replay_status": status,
                    "seed_replay": {"phase": phase, **replay_meta},
                })
            step["observation"] = {"results": results}
    report["ok"] = not report["blocked_call_ids"] and not report["execution_error_call_ids"]
    report["normalized_task_id"] = str((normalized.get("metadata") or {}).get("task_id") or "")
    return trajectory, report


def observation_is_truncated(content: str) -> bool:
    lowered = (content or "").lower()
    return any(marker in lowered for marker in _TRUNCATION_MARKERS)
