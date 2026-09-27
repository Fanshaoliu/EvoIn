"""Dispatch the rewrite, repair, and checking calls to their models.

Every call sends the stage's system prompt and the tagged user message
unchanged.  The response is checked against the stage's JSON contract, and a
response that violates it is regenerated up to ``config.SCHEMA_ATTEMPTS``
calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evoin.models import TailorModels
from evoin.tailor import config

STAGES = ("rewrite", "repair", "quality")
_DERIVED_DIFF_FIELDS = (
    "ungrounded_terms",
    "tool_map",
    "unsupported_tools",
    "iteration_budget",
    "output_contract",
    "evidence_provenance_gaps",
    "trajectory_quality_issues",
)
_TRAJECTORY_BLOCK = re.compile(
    r"<rewritten_trajectory>\s*(?P<value>\{.*\})\s*</rewritten_trajectory>",
    re.DOTALL,
)


def _agent_step_ids(user_message: str) -> list[int]:
    match = _TRAJECTORY_BLOCK.search(user_message)
    if not match:
        return []
    try:
        trajectory = json.loads(match.group("value"))
    except json.JSONDecodeError:
        return []
    return sorted({
        int(step["step_id"])
        for step in trajectory.get("steps") or []
        if isinstance(step, dict)
        and step.get("source") == "agent"
        and isinstance(step.get("step_id"), int)
        and not isinstance(step.get("step_id"), bool)
    })


def _strip_markdown_fence(text: str) -> str:
    value = (text or "").strip()
    if not value.startswith("```"):
        return value
    value = value[3:].lstrip()
    if value.lower().startswith("json"):
        value = value[4:].lstrip()
    if value.rstrip().endswith("```"):
        value = value.rstrip()[:-3].rstrip()
    return value


def normalise_stage_text(text: str, *, stage: str) -> tuple[str, list[str]]:
    stripped = _strip_markdown_fence(text)
    changes = ["strip_markdown_fence"] if stripped != (text or "").strip() else []
    if stage != "rewrite":
        return stripped, changes
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, changes
    if (
        isinstance(value, dict)
        and not isinstance(value.get("derived_diff"), dict)
        and all(field in value for field in _DERIVED_DIFF_FIELDS)
    ):
        value = dict(value)
        value["derived_diff"] = {field: value.pop(field) for field in _DERIVED_DIFF_FIELDS}
        changes.append("nest_derived_diff")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")), changes
    return stripped, changes


def stage_schema_error(text: str, *, stage: str, user_message: str) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return f"invalid JSON: {error}"
    if not isinstance(value, dict):
        return "top-level value is not an object"
    if stage == "rewrite":
        derived = value.get("derived_diff")
        if not isinstance(derived, dict):
            return "rewrite derived_diff is not an object"
        missing = sorted(set(_DERIVED_DIFF_FIELDS) - set(derived))
        if missing:
            return f"rewrite derived_diff is missing {missing}"
        if not isinstance(value.get("feasible_under_seed"), bool):
            return "rewrite feasible_under_seed is not boolean"
        if not isinstance(value.get("sft_ready_after_edits"), bool):
            return "rewrite sft_ready_after_edits is not boolean"
        if not isinstance(value.get("edits"), list):
            return "rewrite edits is not an array"
        return None
    if stage == "repair":
        return None if isinstance(value.get("edits"), list) else "repair edits is not an array"
    expected_ids = _agent_step_ids(user_message)
    if not isinstance(value.get("accept"), bool):
        return "quality accept is not boolean"
    if value.get("quality_risk") not in {"none", "light", "medium", "heavy"}:
        return "quality_risk is invalid"
    for field in ("blockers", "warnings", "reviewed_agent_step_ids", "consistency_findings"):
        if not isinstance(value.get(field), list):
            return f"quality {field} is not an array"
    if any(not isinstance(step_id, int) or isinstance(step_id, bool) for step_id in value["reviewed_agent_step_ids"]):
        return "quality reviewed_agent_step_ids contains a non-integer"
    if sorted(value["reviewed_agent_step_ids"]) != expected_ids:
        return "quality reviewed_agent_step_ids does not cover every agent step"
    if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
        return "quality rationale is empty"
    if value["accept"] and any(value.get(field) for field in ("blockers", "warnings", "consistency_findings")):
        return "quality accept=true has non-empty findings"
    return None


class StageCaller:
    """``caller(system, user, *, stage) -> {"ok", "text" | "error", ...}``."""

    def __init__(self, models: TailorModels) -> None:
        self.models = {"rewrite": models.rewrite, "repair": models.repair, "quality": models.check}

    def __call__(self, system_prompt: str, user_message: str, *, stage: str) -> dict[str, Any]:
        model = self.models[stage]
        schema_errors: list[str] = []
        response: dict[str, Any] = {"ok": False, "error": "no attempt"}
        for attempt in range(1, config.SCHEMA_ATTEMPTS + 1):
            try:
                raw = model(system_prompt, user_message)
            except Exception as exc:  # noqa: BLE001 - a failed call is recorded on the artifact
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                break
            text, changes = normalise_stage_text(str(raw or ""), stage=stage)
            error = stage_schema_error(text, stage=stage, user_message=user_message)
            response = {"ok": True, "text": text, "normalizations": changes}
            if error is None:
                break
            schema_errors.append(error)
        response.update({"schema_attempts": attempt, "discarded_schema_errors": schema_errors})
        return response
