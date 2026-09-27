"""Deterministic repair followed by at most one model repair call."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any, Callable

from evoin.tailor import config, edits as apply_edits, lint as evidence_linter
from evoin.tailor.prompts import repair_system_prompt
from evoin.tailor.replay import replay_trajectory
from evoin.tailor.rewrite_request import seed_context_block
from evoin.tailor.seed import seed_fingerprints
from evoin.tailor.trajectory import ASSISTANT_TEXT_TERMINAL, final_answer_from_atif, terminal_mode_from_atif

# call_fn(system_prompt, user_message, *, stage) -> response dict (see ``stages.StageCaller``)
StageCall = Callable[..., dict]

REPAIRABLE_BLOCKER_TYPES = frozenset({
    "residual_template",
    "unsupported_source_absence_claim",
    "reasoning_observation_mismatch",
    "reasoning_line_reference_mismatch",
    "contradictory_document_key_mapping",
    "empty_navigation_call",
    "repeated_failed_call",
    "truncated_full_read_claim",
    "decimal_precision_violation",
    "literal_newline_encoding",
})
_ALLOWED_REPAIR_ACTIONS = frozenset({
    "rephrase", "tool_remap", "reground_flag", "drop",
})
_DECIMAL_LITERAL_PATTERN = re.compile(r"(?<![\d.])[-+]?\d+\.(?P<fraction>\d+)(?!\d)")
_RAW_LINE_PATTERN = re.compile(r"(?m)^L(?P<line>\d+):\s?(?P<text>[^\n]*)$")
_LINE_THEN_QUOTE_PATTERN = re.compile(
    r"\bL(?P<first>\d+)(?:\s*(?:-|–|—|to)\s*L?(?P<last>\d+))?"
    r"\s*(?:[\uff1a:]\s*|(?:states?|says|mentions?|shows?|contains?|indicates?|reads?)\s*)"
    r"[\"“](?P<quote>[^\"”\n]{4,240})[\"”]",
    re.IGNORECASE,
)
_QUOTE_THEN_LINE_PATTERN = re.compile(
    r"[\"“](?P<quote>[^\"”\n]{4,240})[\"”]\s*"
    r"(?:at|in|on|from|\u4f4d\u4e8e|\u5728)\s*L(?P<first>\d+)"
    r"(?:\s*(?:-|–|—|to)\s*L?(?P<last>\d+))?",
    re.IGNORECASE,
)
_CITATION_PATTERN = re.compile(r"\bL\d+(?:\s*(?:-|–|—|to)\s*L?\d+)?", re.IGNORECASE)


def _normalise_line_text(text: str) -> str:
    return "".join(char.casefold() for char in text if not char.isspace())


def _final_answer(trajectory: dict) -> str | None:
    return final_answer_from_atif(trajectory)


def format_decimal_literals(text: str, digits: int) -> str:
    if digits < 0:
        raise ValueError("digits must be non-negative")

    def replace(match: re.Match[str]) -> str:
        if len(match.group("fraction")) <= digits:
            return match.group(0)
        literal = match.group(0)
        try:
            value = Decimal(literal)
            quantum = Decimal(1).scaleb(-digits)
            with localcontext() as context:
                context.prec = max(50, len(literal) + digits + 10)
                rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            return literal
        return format(rounded, f".{digits}f")

    return _DECIMAL_LITERAL_PATTERN.sub(replace, text)


def _allowed_answer_variants(original: str, question: str) -> set[str]:
    variants = {original, original.replace("\\n", "\n")}
    digits = evidence_linter._task_decimal_limit(question)  # noqa: SLF001
    if digits is not None:
        variants.update(format_decimal_literals(value, digits) for value in list(variants))
    return variants


def final_answer_guard(original: str | None, candidate: str | None, question: str) -> dict:
    allowed = (
        original is not None
        and candidate is not None
        and candidate in _allowed_answer_variants(original, question)
    )
    return {
        "passed": bool(allowed),
        "original": original,
        "candidate": candidate,
        "allowed_transformations": ["literal_newline", "decimal_round_half_up"],
    }


def _prior_observed_lines(step: dict, observed_lines: dict[int, set[str]]) -> None:
    calls = {
        str(call.get("tool_call_id")): call
        for call in (step.get("tool_calls") or [])
        if isinstance(call, dict)
    }
    for result in ((step.get("observation") or {}).get("results") or []):
        call = calls.get(str(result.get("source_call_id"))) or {}
        if (
            call.get("function_name") not in {"scan", "grep", "read_lines"}
            or result.get("is_error")
            or result.get("replay_status") != "replayed"
        ):
            continue
        for line in _RAW_LINE_PATTERN.finditer(str(result.get("content") or "")):
            observed_lines.setdefault(int(line.group("line")), set()).add(line.group("text"))


def _correct_text_line_references(
    text: str,
    observed_lines: dict[int, set[str]],
) -> tuple[str, list[dict]]:
    replacements: list[tuple[int, int, str]] = []
    details: list[dict] = []
    occupied: list[tuple[int, int]] = []
    for pattern in (_LINE_THEN_QUOTE_PATTERN, _QUOTE_THEN_LINE_PATTERN):
        for match in pattern.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            first = int(match.group("first"))
            last = int(match.group("last") or first)
            if last < first:
                first, last = last, first
            quote = _normalise_line_text(match.group("quote"))
            matching_lines = sorted({
                line for line, values in observed_lines.items()
                if any(quote in _normalise_line_text(value) for value in values)
            })
            if len(matching_lines) != 1 or first <= matching_lines[0] <= last:
                continue
            segment = match.group(0)
            corrected = _CITATION_PATTERN.sub(f"L{matching_lines[0]}", segment, count=1)
            if corrected == segment:
                continue
            replacements.append((match.start(), match.end(), corrected))
            occupied.append((match.start(), match.end()))
            details.append({
                "from": f"L{first}" if first == last else f"L{first}-L{last}",
                "to": f"L{matching_lines[0]}",
                "quote": match.group("quote"),
            })
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, details


def line_reference_corrections(trajectory: dict) -> list[dict]:
    observed_lines: dict[int, set[str]] = {}
    corrections: list[dict] = []
    for step in trajectory.get("steps") or []:
        if step.get("source") != "agent":
            continue
        for field in ("reasoning_content", "message"):
            value = step.get(field)
            if not isinstance(value, str):
                continue
            corrected, details = _correct_text_line_references(value, observed_lines)
            if corrected != value:
                corrections.append({
                    "step_id": step.get("step_id"),
                    "field": field,
                    "before": value,
                    "after": corrected,
                    "corrections": details,
                })
        _prior_observed_lines(step, observed_lines)
    return corrections


def deterministic_edits(artifact: dict, lint_result: dict) -> list[dict]:
    blocker_types = {
        str(item.get("type"))
        for item in (lint_result.get("blockers") or [])
        if isinstance(item, dict)
    }
    trajectory = artifact.get("rewritten_trajectory") or {}
    edits: list[dict] = []
    if "reasoning_line_reference_mismatch" in blocker_types:
        for correction in line_reference_corrections(trajectory):
            edits.append({
                "step_id": correction["step_id"],
                "action": "rephrase",
                str(correction["field"]): correction["after"],
                "reason": "deterministic unique-evidence line-reference correction",
            })
    digits = evidence_linter._task_decimal_limit(  # noqa: SLF001
        str(artifact.get("question") or "")
    )
    final_agent = next(
        (
            step
            for step in reversed(trajectory.get("steps") or [])
            if step.get("source") == "agent"
        ),
        None,
    )
    assistant_text_terminal = (
        terminal_mode_from_atif(trajectory) == ASSISTANT_TEXT_TERMINAL
    )
    for step in trajectory.get("steps") or []:
        if step.get("source") != "agent":
            continue
        step_id = step.get("step_id")
        for field in ("reasoning_content", "message"):
            value = step.get(field)
            if not isinstance(value, str):
                continue
            after = value
            reasons: list[str] = []
            if "literal_newline_encoding" in blocker_types and "\\n" in after:
                after = after.replace("\\n", "\n")
                reasons.append("replace a literal newline escape with a real line break")
            if (
                field == "message"
                and step is final_agent
                and assistant_text_terminal
                and "decimal_precision_violation" in blocker_types
                and digits is not None
            ):
                rounded = format_decimal_literals(after, digits)
                if rounded != after:
                    after = rounded
                    reasons.append(f"ROUND_HALF_UP to {digits} decimal places")
            if after != value:
                edits.append({
                    "step_id": step_id,
                    "action": "rephrase",
                    field: after,
                    "reason": " and ".join(reasons),
                })
        for call in step.get("tool_calls") or []:
            function_name = call.get("function_name")
            arguments = call.get("arguments") or {}
            call_id = call.get("tool_call_id")
            field = (
                "content" if function_name == "notepad"
                else "result" if function_name == "complete_task"
                else None
            )
            if field is None or not isinstance(arguments.get(field), str):
                continue
            before = arguments[field]
            after = before
            reasons: list[str] = []
            if "literal_newline_encoding" in blocker_types and "\\n" in after:
                after = after.replace("\\n", "\n")
                reasons.append("literal newline")
            if (
                function_name == "complete_task"
                and "decimal_precision_violation" in blocker_types
                and digits is not None
            ):
                rounded = format_decimal_literals(after, digits)
                if rounded != after:
                    after = rounded
                    reasons.append(f"ROUND_HALF_UP to {digits} decimal places")
            if after != before:
                edits.append({
                    "step_id": step_id,
                    "action": "rephrase",
                    "tool_call_id": call_id,
                    "arguments": {field: after},
                    "reason": " and ".join(reasons),
                })
    return edits


def _has_existing_read_call(trajectory: dict) -> bool:
    return any(
        call.get("function_name") == "read_lines"
        for step in trajectory.get("steps") or []
        if step.get("source") == "agent"
        for call in (step.get("tool_calls") or [])
    )


def classify_blockers(blockers: list[dict], trajectory: dict) -> dict:
    repairable: list[dict] = []
    rerollout: list[dict] = []
    for blocker in blockers:
        type_ = str(blocker.get("type") or "")
        if type_ == "truncated_full_read_claim" and not _has_existing_read_call(trajectory):
            rerollout.append(blocker)
        elif type_ in REPAIRABLE_BLOCKER_TYPES:
            repairable.append(blocker)
        else:
            rerollout.append(blocker)
    return {
        "eligible": bool(repairable) and not rerollout,
        "repairable": repairable,
        "rerollout": rerollout,
    }


def repair_done_key(artifact: dict, lint_result: dict) -> str:
    fingerprints = config.pipeline_fingerprint(seed_fingerprints())
    return config.canonical_hash({
        "stage": "repair",
        "custom_id": artifact.get("custom_id"),
        "trajectory_sha256": config.canonical_hash(artifact.get("rewritten_trajectory") or {}),
        "blocker_sha256": config.canonical_hash(lint_result.get("blockers") or []),
        "repair_prompt_sha256": config.sha256_file(config.REPAIR_PROMPT_MD),
        "pipeline_semantic_sha256": fingerprints["pipeline_sha256"],
    })


def repair_user_message(
    artifact: dict,
    initial_lint: dict,
    post_deterministic_lint: dict,
) -> str:
    task = json.dumps({"question": artifact.get("question")}, ensure_ascii=False)
    lint_findings = json.dumps({
        "initial": {
            "blockers": initial_lint.get("blockers") or [],
            "warnings": initial_lint.get("warnings") or [],
        },
        "post_deterministic": {
            "blockers": post_deterministic_lint.get("blockers") or [],
            "warnings": post_deterministic_lint.get("warnings") or [],
        },
    }, ensure_ascii=False, indent=2)
    derived_diff = json.dumps(
        (artifact.get("rewrite_output") or {}).get("derived_diff") or {},
        ensure_ascii=False,
        indent=2,
    )
    trajectory = json.dumps(
        artifact.get("rewritten_trajectory") or {},
        ensure_ascii=False,
        indent=2,
    )
    return (
        seed_context_block()
        + f"<task>\n{task}\n</task>\n\n"
        + f"<stage2_derived_diff>\n{derived_diff}\n</stage2_derived_diff>\n\n"
        + f"<structured_lint>\n{lint_findings}\n</structured_lint>\n\n"
        + f"<replayed_rewritten_trajectory>\n{trajectory}\n</replayed_rewritten_trajectory>\n"
    )


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in {"observation", "gold"} or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_repair_output(parsed: dict, trajectory: dict) -> list[str]:
    errors: list[str] = []
    if set(parsed) != {"edits"}:
        errors.append("repair output must contain only the edits field")
    edits = parsed.get("edits")
    if not isinstance(edits, list) or any(not isinstance(edit, dict) for edit in edits):
        return [*errors, "edits must be a list of objects"]
    agent_step_ids = {
        step.get("step_id")
        for step in trajectory.get("steps") or []
        if step.get("source") == "agent"
    }
    for index, edit in enumerate(edits):
        if _contains_forbidden_key(edit):
            errors.append(f"edit {index} contains forbidden observation/gold data")
        if edit.get("action") not in _ALLOWED_REPAIR_ACTIONS:
            errors.append(f"edit {index} has unsupported action {edit.get('action')!r}")
        if edit.get("step_id") not in agent_step_ids:
            errors.append(f"edit {index} targets no existing agent step")
    return errors


def _has_existing_read_remap(edits: list[dict], trajectory: dict) -> bool:
    calls_by_step: dict[Any, list[dict]] = {
        step.get("step_id"): [
            call for call in (step.get("tool_calls") or []) if isinstance(call, dict)
        ]
        for step in trajectory.get("steps") or []
        if step.get("source") == "agent"
    }
    for edit in edits:
        if edit.get("action") not in {"tool_remap", "rephrase"}:
            continue
        arguments = edit.get("arguments")
        if not isinstance(arguments, dict) or not ({"start", "end"} & set(arguments)):
            continue
        calls = calls_by_step.get(edit.get("step_id"), [])
        target_id = edit.get("tool_call_id")
        targets = (
            [call for call in calls if str(call.get("tool_call_id")) == str(target_id)]
            if target_id is not None
            else calls if len(calls) == 1
            else []
        )
        if len(targets) == 1 and targets[0].get("function_name") == "read_lines":
            return True
    return False


def _apply_and_replay(
    artifact: dict,
    edits: list[dict],
    *,
    task: dict,
    phase: str,
    answer_before_repair: str | None,
) -> tuple[dict | None, list[dict], dict, str | None]:
    trajectory = artifact.get("rewritten_trajectory") or {}
    merged, applied, forced_replay_ids = apply_edits.apply_edits(trajectory, edits)
    merge_errors = [
        f"step {item.get('step_id')}: {item.get('error')}"
        for item in applied if item.get("error")
    ]
    if merge_errors:
        return None, applied, {}, "; ".join(merge_errors)
    guard = final_answer_guard(
        answer_before_repair,
        _final_answer(merged),
        str(artifact.get("question") or ""),
    )
    artifact["final_answer_guard"] = guard
    if not guard["passed"]:
        return None, applied, {}, "terminal answer failed the answer guard"
    try:
        replayed, replay = replay_trajectory(merged, task, phase=phase)
    except Exception as error:  # noqa: BLE001
        return None, applied, {}, f"{type(error).__name__}: {error}"
    replay["forced_replay_ids"] = forced_replay_ids
    validation = apply_edits.validate_trajectory(
        replayed,
        expected_terminal_mode=artifact.get("terminal_mode_source"),
    )
    if not validation.get("ok"):
        return None, applied, replay, "; ".join(validation.get("errors") or [])
    if not replay.get("ok"):
        return None, applied, replay, "full seed replay did not complete successfully"
    return replayed, applied, replay, None


def _record_final_lint(artifact: dict, *, rerollout: bool = False) -> None:
    evidence_linter.apply_lint(artifact)
    if rerollout and artifact.get("status") == "rejected_by_linter":
        artifact["status"] = "rerollout_required"
        artifact["rerollout_reason"] = (
            "remaining blockers require new evidence or an unavailable replay path"
        )
        artifact.setdefault("gates", {})["REROLLOUT_REQUIRED"] = {
            "passed": True,
            "detail": artifact["rerollout_reason"],
        }


def run_repair_loop(
    artifact: dict,
    *,
    task: dict,
    call_fn: StageCall,
    enabled: bool = True,
) -> tuple[dict, dict | None]:
    """Run deterministic repair and at most one model repair, replaying after each."""
    artifact["lint_history"] = {}
    artifact.pop("evidence_lint", None)
    repair: dict[str, Any] = {
        "called": False,
        "eligible": False,
        "done_key": None,
        "input_blockers": [],
        "applied": [],
        "reground": {},
        "output": None,
        "error": None,
    }
    artifact["repair"] = repair
    if artifact.get("status") != "delabeled":
        _record_final_lint(artifact)
        return artifact, None
    answer_before_repair = _final_answer(artifact.get("rewritten_trajectory") or {})
    artifact["answer_before_repair"] = answer_before_repair
    initial = evidence_linter.lint(artifact)
    artifact["lint_history"]["initial"] = initial
    if initial.get("passed"):
        _record_final_lint(artifact)
        return artifact, None
    if not enabled:
        repair["error"] = "repair disabled by caller"
        _record_final_lint(artifact)
        return artifact, None
    safe_edits = deterministic_edits(artifact, initial)
    artifact["deterministic_repair"] = {"edits": safe_edits, "applied": [], "reground": {}}
    if safe_edits:
        replayed, applied, replay, error = _apply_and_replay(
            artifact,
            safe_edits,
            task=task,
            phase="deterministic_repair",
            answer_before_repair=answer_before_repair,
        )
        artifact["deterministic_repair"].update({"applied": applied, "reground": replay})
        if error or replayed is None:
            artifact["deterministic_repair"]["error"] = error
            _record_final_lint(artifact)
            artifact["status"] = (
                "invalid_after_repair_merge"
                if not (artifact.get("final_answer_guard") or {}).get("passed", True)
                or any(item.get("error") for item in applied)
                else "invalid_after_repair_replay"
            )
            return artifact, None
        artifact["rewritten_trajectory"] = replayed
        artifact["reground"] = replay
        artifact["final_answer_after"] = _final_answer(replayed)
        artifact["terminal_mode_after"] = terminal_mode_from_atif(replayed)
    artifact["stage2_sft_ready_after_edits"] = artifact.get("sft_ready_after_edits")
    artifact["sft_ready_after_edits"] = True
    post_deterministic = evidence_linter.lint(artifact)
    artifact["lint_history"]["post_deterministic"] = post_deterministic
    if post_deterministic.get("passed"):
        _record_final_lint(artifact)
        return artifact, None
    routing = classify_blockers(
        post_deterministic.get("blockers") or [],
        artifact.get("rewritten_trajectory") or {},
    )
    repair["eligible"] = routing["eligible"]
    repair["input_blockers"] = post_deterministic.get("blockers") or []
    repair["routing"] = routing
    if not routing["eligible"]:
        _record_final_lint(artifact, rerollout=bool(routing["rerollout"]))
        return artifact, None
    done_key = repair_done_key(artifact, post_deterministic)
    repair["done_key"] = done_key
    try:
        response = call_fn(
            repair_system_prompt(),
            repair_user_message(artifact, initial, post_deterministic),
            stage="repair",
        )
    except Exception as error:  # noqa: BLE001
        repair.update({
            "called": True,
            "prompt_sha256": config.sha256_file(config.REPAIR_PROMPT_MD),
            "error": f"{type(error).__name__}: {error}",
        })
        _record_final_lint(artifact)
        artifact["status"] = "repair_model_error"
        return artifact, None
    repair.update({
        "called": True,
        "prompt_sha256": config.sha256_file(config.REPAIR_PROMPT_MD),
        "schema_attempts": response.get("schema_attempts"),
    })
    if not response.get("ok"):
        repair["error"] = response.get("error")
        _record_final_lint(artifact)
        artifact["status"] = "repair_model_error"
        return artifact, response
    try:
        parsed = apply_edits.extract_json(str(response.get("text") or ""))
    except Exception as error:  # noqa: BLE001
        repair["error"] = f"{type(error).__name__}: {error}"
        _record_final_lint(artifact)
        artifact["status"] = "invalid_repair_output"
        return artifact, response
    repair["output"] = parsed
    errors = validate_repair_output(parsed, artifact.get("rewritten_trajectory") or {})
    if errors:
        repair["error"] = "; ".join(errors)
        _record_final_lint(artifact)
        artifact["status"] = "invalid_repair_output"
        return artifact, response
    if not parsed["edits"]:
        repair["error"] = (
            "repair model returned no safe edits; remaining blockers require new evidence "
            "or cannot be closed conservatively"
        )
        _record_final_lint(artifact)
        artifact["status"] = "rerollout_required"
        artifact["rerollout_reason"] = repair["error"]
        artifact.setdefault("gates", {})["REROLLOUT_REQUIRED"] = {
            "passed": True,
            "detail": repair["error"],
        }
        return artifact, response
    if (
        any(blocker.get("type") == "truncated_full_read_claim" for blocker in repair["input_blockers"])
        and not _has_existing_read_remap(parsed["edits"], artifact.get("rewritten_trajectory") or {})
    ):
        repair["error"] = "truncated evidence repair did not remap an existing read_lines call"
        _record_final_lint(artifact)
        artifact["status"] = "rerollout_required"
        artifact["rerollout_reason"] = repair["error"]
        artifact.setdefault("gates", {})["REROLLOUT_REQUIRED"] = {
            "passed": True,
            "detail": repair["error"],
        }
        return artifact, response
    replayed, applied, replay, error = _apply_and_replay(
        artifact,
        parsed["edits"],
        task=task,
        phase="repair",
        answer_before_repair=answer_before_repair,
    )
    repair["applied"] = applied
    repair["reground"] = replay
    if error or replayed is None:
        repair["error"] = error
        invalid_status = (
            "invalid_after_repair_merge"
            if any(item.get("error") for item in applied)
            or not (artifact.get("final_answer_guard") or {}).get("passed", True)
            else "invalid_after_repair_replay"
        )
        _record_final_lint(artifact)
        artifact["status"] = invalid_status
        return artifact, response
    artifact["rewritten_trajectory"] = replayed
    artifact["reground"] = replay
    artifact["final_answer_after"] = _final_answer(replayed)
    artifact["terminal_mode_after"] = terminal_mode_from_atif(replayed)
    post_repair = evidence_linter.lint(artifact)
    artifact["lint_history"]["post_repair"] = post_repair
    final_routing = classify_blockers(
        post_repair.get("blockers") or [],
        artifact.get("rewritten_trajectory") or {},
    )
    _record_final_lint(artifact, rerollout=bool(final_routing["rerollout"]))
    return artifact, response
