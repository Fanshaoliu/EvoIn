"""Merge sparse rewrite edits, replay the result under the seed, and validate it."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from typing import Any

from evoin.tailor import config
from evoin.tailor.replay import replay_trajectory
from evoin.tailor.seed import seed_max_iterations, seed_tool_schemas, tool_order
from evoin.tailor.trajectory import final_answer_from_atif, terminal_mode_from_atif

REQUIRED_DERIVED_DIFF_FIELDS = {
    "ungrounded_terms",
    "tool_map",
    "unsupported_tools",
    "iteration_budget",
    "output_contract",
    "evidence_provenance_gaps",
    "trajectory_quality_issues",
}


def extract_json(text: str) -> dict:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else ""
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    try:
        parsed = json.loads(value.strip())
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model output is not a JSON object")
    return parsed


def _json_type_ok(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda current: isinstance(current, dict),
        "array": lambda current: isinstance(current, list),
        "string": lambda current: isinstance(current, str),
        "integer": lambda current: isinstance(current, int) and not isinstance(current, bool),
        "number": lambda current: isinstance(current, (int, float)) and not isinstance(current, bool),
        "boolean": lambda current: isinstance(current, bool),
        "null": lambda current: current is None,
    }
    return checks.get(expected, lambda _current: True)(value)


def _argument_errors(function_name: str, arguments: Any) -> list[str]:
    if not isinstance(arguments, dict):
        return [f"tool {function_name!r} arguments must be an object"]
    schema = seed_tool_schemas().get(function_name, {})
    properties = schema.get("properties") or {}
    errors: list[str] = []
    for required in schema.get("required") or []:
        if required not in arguments:
            errors.append(f"tool {function_name!r} missing required argument {required!r}")
    if schema.get("additionalProperties") is False:
        errors.extend(
            f"tool {function_name!r} has unknown argument {key!r}"
            for key in arguments if key not in properties
        )
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        if expected:
            choices = expected if isinstance(expected, list) else [expected]
            if not any(_json_type_ok(value, choice) for choice in choices):
                errors.append(
                    f"tool {function_name!r} argument {key!r} must be {expected!r}, "
                    f"got {type(value).__name__}"
                )
    return errors


def _calls_by_id(step: dict) -> dict[str, dict]:
    return {
        str(call.get("tool_call_id")): call
        for call in (step.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("tool_call_id") is not None
    }


def _one_target_call(step: dict, edit: dict) -> tuple[dict | None, str | None]:
    calls = [call for call in (step.get("tool_calls") or []) if isinstance(call, dict)]
    target_id = edit.get("tool_call_id")
    if target_id is None:
        if len(calls) == 1:
            return calls[0], None
        return None, "edit must identify exactly one existing tool call"
    matches = [
        call for call in calls
        if str(call.get("tool_call_id")) == str(target_id)
    ]
    if len(matches) != 1:
        return None, "edit must identify exactly one existing tool call"
    return matches[0], None


def _mark_replay_required(step: dict, call_ids: list[str]) -> None:
    step["observation"] = {
        "results": [],
        "needs_reground": list(dict.fromkeys(call_ids)),
        "replay_required": True,
    }


def apply_edits(atif: dict, edits: list[dict]) -> tuple[dict, list[dict], list[str]]:
    """Apply allowed sparse edits in order and identify required replays."""
    edits_by_step: dict[Any, list[dict]] = defaultdict(list)
    for edit in edits:
        if isinstance(edit, dict) and "step_id" in edit:
            edits_by_step[edit["step_id"]].append(edit)

    out_steps: list[dict] = []
    applied: list[dict] = []
    forced_replay_ids: list[str] = []
    seen_agent_step_ids: set[Any] = set()
    for source_step in atif.get("steps", []):
        if source_step.get("source") != "agent":
            out_steps.append(copy.deepcopy(source_step))
            continue
        step = copy.deepcopy(source_step)
        step_id = step.get("step_id")
        seen_agent_step_ids.add(step_id)
        step_edits = edits_by_step.get(step_id, [])
        if not step_edits:
            out_steps.append(step)
            continue

        original_call_ids = set(_calls_by_id(step))
        remapped_ids: list[str] = []
        drop_step = False
        for edit in step_edits:
            action = edit.get("action")
            log: dict[str, Any] = {
                "step_id": step_id,
                "action": action,
                "reason": edit.get("reason", ""),
                "changed": [],
            }
            if action == "drop":
                drop_step = True
                applied.append(log)
                break
            if action == "unsupported_capability":
                log["error"] = (
                    "unsupported_capability is only valid with feasible_under_seed=false"
                )
                applied.append(log)
                continue
            for key in ("reasoning_content", "message"):
                if key in edit and step.get(key) != edit[key]:
                    step[key] = edit[key]
                    log["changed"].append(key)
            if "tool_calls" in edit:
                candidate = edit["tool_calls"]
                if not isinstance(candidate, list):
                    log["error"] = "tool_calls edit must be a list"
                else:
                    candidate_ids = {
                        str(call.get("tool_call_id"))
                        for call in candidate
                        if isinstance(call, dict) and call.get("tool_call_id") is not None
                    }
                    if candidate_ids != original_call_ids or len(candidate_ids) != len(candidate):
                        log["error"] = (
                            "tool_calls edits cannot add, remove, or duplicate calls; "
                            "use an existing tool_call_id for every call"
                        )
                    else:
                        before = _calls_by_id(step)
                        after = {str(call["tool_call_id"]): call for call in candidate}
                        changed_ids = [
                            call_id for call_id in candidate_ids
                            if before[call_id].get("function_name") != after[call_id].get("function_name")
                            or before[call_id].get("arguments") != after[call_id].get("arguments")
                        ]
                        if changed_ids:
                            step["tool_calls"] = copy.deepcopy(candidate)
                            remapped_ids.extend(changed_ids)
                            log["changed"].append("tool_calls")
            if action == "rephrase" and "arguments" in edit:
                overlay = edit["arguments"]
                if not isinstance(overlay, dict):
                    log["error"] = "rephrase arguments must be an object"
                else:
                    target, target_error = _one_target_call(step, edit)
                    if target_error:
                        log["error"] = target_error
                    else:
                        current = target.get("arguments")
                        if not isinstance(current, dict):
                            log["error"] = "target tool call arguments must be an object"
                        else:
                            updated = {**current, **copy.deepcopy(overlay)}
                            if updated != current:
                                target["arguments"] = updated
                                remapped_ids.append(str(target.get("tool_call_id")))
                                log["changed"].append(
                                    f"tool_calls[{target.get('tool_call_id')}].arguments"
                                )
            if action == "tool_remap" and "tool_calls" not in edit:
                target, target_error = _one_target_call(step, edit)
                if target_error:
                    log["error"] = f"tool_remap {target_error}"
                else:
                    changed = False
                    for key in ("function_name", "arguments"):
                        if key in edit and target.get(key) != edit[key]:
                            target[key] = copy.deepcopy(edit[key])
                            changed = True
                    if changed:
                        remapped_ids.append(str(target.get("tool_call_id")))
                        log["changed"].append("tool_calls")
            if action == "reground_flag":
                requested = edit.get("reground")
                requested_ids = requested if isinstance(requested, list) else []
                if not requested_ids:
                    log["error"] = "reground_flag requires a non-empty reground list"
                invalid = [
                    str(call_id) for call_id in requested_ids
                    if str(call_id) not in original_call_ids
                ]
                if invalid:
                    log["error"] = f"reground refers to unknown call IDs: {invalid}"
                elif requested_ids:
                    remapped_ids.extend(str(call_id) for call_id in requested_ids)
                    log["changed"].append("reground")
            if action not in {
                "rephrase", "tool_remap", "reground_flag", "drop", "unsupported_capability",
            }:
                log["error"] = f"unsupported edit action {action!r}"
            log["changed"] = list(dict.fromkeys(log["changed"]))
            if action in {"rephrase", "tool_remap"} and not log["changed"] and not log.get("error"):
                log["error"] = (
                    f"{action} must change reasoning_content, message, tool_calls, "
                    "or a targeted arguments object"
                )
            applied.append(log)
        if drop_step:
            continue
        if remapped_ids:
            _mark_replay_required(step, remapped_ids)
            forced_replay_ids.extend(remapped_ids)
        out_steps.append(step)
    for step_id, unresolved_edits in edits_by_step.items():
        if step_id in seen_agent_step_ids:
            continue
        for edit in unresolved_edits:
            applied.append({
                "step_id": step_id,
                "action": edit.get("action"),
                "reason": edit.get("reason", ""),
                "changed": [],
                "error": "edit targets no agent step",
            })
    return {"steps": out_steps}, applied, list(dict.fromkeys(forced_replay_ids))


def validate_trajectory(
    atif: dict,
    *,
    expected_terminal_mode: str | None = None,
) -> dict:
    """Check structural and seed-schema legality after a full replay."""
    errors: list[str] = []
    seed_tools = set(tool_order())
    agent_steps = [step for step in atif.get("steps", []) if step.get("source") == "agent"]
    if len(agent_steps) > seed_max_iterations():
        errors.append(
            f"agent step count {len(agent_steps)} > seed_max_iterations {seed_max_iterations()}"
        )
    if not atif.get("steps") or atif["steps"][-1].get("source") != "agent":
        errors.append("trajectory does not end on an agent step")
    terminal_mode = terminal_mode_from_atif(atif)
    if terminal_mode not in config.allowed_terminal_modes():
        errors.append(
            f"terminal mode {terminal_mode!r} is not allowed by "
            "the complete_task terminal policy"
        )
    if (
        expected_terminal_mode is not None
        and terminal_mode != expected_terminal_mode
    ):
        errors.append(
            "terminal mode changed "
            f"from {expected_terminal_mode!r} to {terminal_mode!r}"
        )
    globally_seen: set[str] = set()
    for step in agent_steps:
        calls = step.get("tool_calls") or []
        call_ids = [str(call.get("tool_call_id")) for call in calls]
        if any(call.get("tool_call_id") is None for call in calls):
            errors.append(f"step {step.get('step_id')}: tool_call_id is missing")
        if len(set(call_ids)) != len(call_ids):
            errors.append(f"step {step.get('step_id')}: duplicate tool_call_id")
        repeated = set(call_ids) & globally_seen
        if repeated:
            errors.append(
                f"step {step.get('step_id')}: globally duplicate IDs {sorted(repeated)}"
            )
        globally_seen.update(call_ids)
        for call in calls:
            function_name = call.get("function_name")
            if function_name not in seed_tools:
                errors.append(f"step {step.get('step_id')}: seed does not expose {function_name!r}")
                continue
            errors.extend(
                f"step {step.get('step_id')}: {problem}"
                for problem in _argument_errors(str(function_name), call.get("arguments"))
            )
        results = (step.get("observation") or {}).get("results") or []
        if calls and len(results) != len(calls):
            errors.append(
                f"step {step.get('step_id')}: {len(calls)} calls but {len(results)} observation results"
            )
        result_ids = [str(result.get("source_call_id")) for result in results]
        if set(result_ids) != set(call_ids) or len(result_ids) != len(call_ids):
            errors.append(f"step {step.get('step_id')}: observation/call IDs do not bind one-to-one")
    return {"ok": not errors, "errors": errors}


def _final_answer(atif: dict) -> str | None:
    return final_answer_from_atif(atif)


def _base_artifact(custom_id: str, meta: dict) -> dict:
    return {
        "custom_id": custom_id,
        "schema_version": config.SCHEMA_VERSION,
        "task_id": meta.get("task_id"),
        "sample_idx": meta.get("sample_idx"),
        "document_profile": meta.get("document_profile"),
        "question": meta.get("question"),
        "gold": meta.get("gold"),
        "fingerprints": meta.get("fingerprints"),
        "done_key": meta.get("done_key"),
        "seed_manifest": meta.get("seed_manifest"),
        "task_sha256": meta.get("task_sha256"),
        "n_agent_steps_source": meta.get("n_agent_steps_source"),
        "n_agent_steps_hydrated": meta.get("n_agent_steps_hydrated"),
        "hydration": meta.get("hydration"),
        "preflight_sanitization": meta.get("preflight_sanitization") or [],
        "source_atif_trajectory": meta.get("source_atif_trajectory"),
        "hydrated_trajectory": meta.get("atif_trajectory"),
        "terminal_mode_source": meta.get("terminal_mode_source"),
        "terminal_mode_hydrated": meta.get("terminal_mode_hydrated"),
        "gates": {
            "ROLLOUT_SOLVED": {
                "passed": isinstance(meta.get("orig_score"), (int, float)) and meta["orig_score"] >= 1.0,
                "detail": (
                    "rollout candidate was fully solved with a valid "
                    f"{meta.get('terminal_mode_source')} terminal"
                ),
            },
            "TERMINAL_MODE_PRESERVED": {
                "passed": (
                    meta.get("terminal_mode_source") in config.allowed_terminal_modes()
                    and meta.get("terminal_mode_hydrated")
                    == meta.get("terminal_mode_source")
                ),
                "detail": (
                    f"source={meta.get('terminal_mode_source')!r}, "
                    f"hydrated={meta.get('terminal_mode_hydrated')!r}"
                ),
            }
        },
    }


def _quality_fields(parsed: dict) -> tuple[list[str], bool]:
    derived = parsed.get("derived_diff")
    missing = REQUIRED_DERIVED_DIFF_FIELDS - set(derived or {})
    errors = [f"derived_diff missing required fields: {sorted(missing)}"] if missing else []
    ready = parsed.get("sft_ready_after_edits")
    if not isinstance(ready, bool):
        errors.append("sft_ready_after_edits must be boolean")
        ready = False
    return errors, ready


def process_one(custom_id: str, meta: dict, result: dict, *, task: dict | None) -> dict:
    """Merge one rewrite response and replay the edited trajectory against ``task``."""
    artifact = _base_artifact(custom_id, meta)
    if meta.get("preflight_error"):
        artifact.update({
            "status": "rerollout_required",
            "rerollout_reason": meta["preflight_error"],
            "gates": {
                **artifact["gates"],
                "SEED_STRUCTURAL_OK": {"passed": False, "detail": meta["preflight_error"]},
                "REROLLOUT_REQUIRED": {
                    "passed": True,
                    "detail": "source cannot be safely converted or hydrated",
                },
            },
        })
        return artifact
    if not result.get("ok"):
        artifact.update({"status": "model_error", "error": result.get("error")})
        return artifact
    try:
        parsed = extract_json(str(result.get("text") or ""))
    except Exception as error:  # noqa: BLE001
        artifact.update({"status": "unparseable_json", "error": f"{type(error).__name__}: {error}"})
        return artifact
    artifact["rewrite_output"] = parsed
    artifact["overall_risk"] = parsed.get("overall_risk")
    artifact["feasible_under_seed"] = parsed.get("feasible_under_seed")
    artifact["sft_ready_after_edits"] = parsed.get("sft_ready_after_edits")
    artifact["trajectory_quality_issues"] = (
        (parsed.get("derived_diff") or {}).get("trajectory_quality_issues") or []
    )
    edits = parsed.get("edits") or []
    artifact["n_edits"] = len(edits) if isinstance(edits, list) else None
    artifact["actions"] = sorted({
        edit.get("action") for edit in edits if isinstance(edit, dict) and edit.get("action")
    }) if isinstance(edits, list) else []
    field_errors, _ready = _quality_fields(parsed)
    if field_errors:
        artifact.update({"status": "invalid_output", "error": "; ".join(field_errors)})
        return artifact
    if parsed.get("feasible_under_seed") is not True:
        artifact.update({
            "status": "infeasible",
            "infeasibility_reason": parsed.get("infeasibility_reason"),
            "gates": {
                **artifact["gates"],
                "SEED_STRUCTURAL_OK": {
                    "passed": True,
                    "detail": "strict binding and seed hydration succeeded",
                },
                "DELABELED_OK": {
                    "passed": False,
                    "detail": parsed.get("infeasibility_reason") or "model marked infeasible",
                },
            },
        })
        return artifact
    if not isinstance(edits, list) or any(not isinstance(edit, dict) for edit in edits):
        artifact.update({"status": "invalid_output", "error": "edits must be a list of objects"})
        return artifact
    if any(edit.get("action") == "unsupported_capability" for edit in edits):
        artifact.update({
            "status": "invalid_output",
            "error": "unsupported_capability cannot appear when feasible_under_seed is true",
        })
        return artifact
    rewritten, applied, forced_replay_ids = apply_edits(meta["atif_trajectory"], edits)
    merge_errors = [
        f"step {item.get('step_id')}: {item['error']}"
        for item in applied if item.get("error")
    ]
    artifact["applied"] = applied
    artifact["forced_replay_ids"] = forced_replay_ids
    if merge_errors:
        artifact.update({
            "status": "invalid_after_merge",
            "error": "; ".join(merge_errors),
            "rewritten_pre_replay": rewritten,
        })
        return artifact
    try:
        if task is None:
            raise ValueError("no task record is available for replay")
        replayed, replay = replay_trajectory(rewritten, task, phase="reground")
    except Exception as error:  # noqa: BLE001
        artifact.update({
            "status": "rerollout_required",
            "rerollout_reason": f"seed replay failed: {type(error).__name__}: {error}",
            "rewritten_pre_replay": rewritten,
        })
        return artifact
    terminal_mode_after = terminal_mode_from_atif(replayed)
    validation = validate_trajectory(
        replayed,
        expected_terminal_mode=meta.get("terminal_mode_source"),
    )
    artifact.update({
        "rewritten_pre_replay": rewritten,
        "rewritten_trajectory": replayed,
        "reground": replay,
        "validation": validation,
        "final_answer_before": _final_answer(meta["atif_trajectory"]),
        "final_answer_after": _final_answer(replayed),
        "terminal_mode_after": terminal_mode_after,
    })
    artifact["gates"] = {
        **artifact["gates"],
        "SEED_STRUCTURAL_OK": {
            "passed": True,
            "detail": "strict binding and initial seed hydration succeeded",
        },
        "DELABELED_OK": {
            "passed": True,
            "detail": "model marked the sparse edit set feasible under seed",
        },
        "REGROUND_OK": {
            "passed": bool(replay.get("ok")),
            "detail": (
                "all retained calls replayed under seed"
                if replay.get("ok") else "some retained calls could not replay"
            ),
        },
        "TERMINAL_MODE_PRESERVED": {
            "passed": terminal_mode_after == meta.get("terminal_mode_source"),
            "detail": (
                f"source={meta.get('terminal_mode_source')!r}, "
                f"rewritten={terminal_mode_after!r}"
            ),
        },
    }
    if not validation["ok"]:
        artifact["status"] = "invalid_after_replay"
    elif not replay.get("ok"):
        artifact["status"] = "rerollout_required"
        artifact["rerollout_reason"] = "retained calls could not be safely replayed"
        artifact["gates"]["REROLLOUT_REQUIRED"] = {
            "passed": True,
            "detail": artifact["rerollout_reason"],
        }
    else:
        artifact["status"] = "delabeled"
    return artifact
