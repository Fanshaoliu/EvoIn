"""Independent quality judge for trajectories that passed the deterministic checks."""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

from evoin.tailor import config
from evoin.tailor.prompts import quality_system_prompt
from evoin.tailor.rewrite_request import seed_context_block
from evoin.tailor.seed import seed_fingerprints

# call_fn(system_prompt, user_message, *, stage) -> response dict (see ``stages.StageCaller``)
StageCall = Callable[..., dict]


def _user_message(artifact: dict) -> str:
    task = json.dumps({"question": artifact.get("question")}, ensure_ascii=False)
    gate = json.dumps({
        "gates": artifact.get("gates"),
        "evidence_lint": artifact.get("evidence_lint"),
        "reground": artifact.get("reground"),
    }, ensure_ascii=False, indent=2)
    rewrite_audit = json.dumps({
        "sft_ready_after_edits": artifact.get("sft_ready_after_edits"),
        "derived_diff": (artifact.get("rewrite_output") or {}).get("derived_diff"),
        "applied": artifact.get("applied"),
        "deterministic_repair": artifact.get("deterministic_repair"),
        "repair": artifact.get("repair"),
    }, ensure_ascii=False, indent=2)
    trajectory = json.dumps(artifact.get("rewritten_trajectory"), ensure_ascii=False, indent=2)
    return (
        seed_context_block()
        + f"<task>\n{task}\n</task>\n\n"
        + f"<deterministic_gate>\n{gate}\n</deterministic_gate>\n\n"
        + f"<rewrite_audit>\n{rewrite_audit}\n</rewrite_audit>\n\n"
        + f"<rewritten_trajectory>\n{trajectory}\n</rewritten_trajectory>\n"
    )


def _agent_step_ids(artifact: dict) -> list[int]:
    ids: list[int] = []
    for step in (artifact.get("rewritten_trajectory") or {}).get("steps", []):
        if step.get("source") != "agent":
            continue
        step_id = step.get("step_id")
        if isinstance(step_id, int) and not isinstance(step_id, bool):
            ids.append(step_id)
    return sorted(ids)


def _parse(text: str) -> dict:
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
        raise ValueError("quality judge response is not a JSON object")
    return parsed


def _validate(decision: dict, expected_step_ids: list[int]) -> list[str]:
    errors: list[str] = []
    if not isinstance(decision.get("accept"), bool):
        errors.append("accept must be boolean")
    if decision.get("quality_risk") not in {"none", "light", "medium", "heavy"}:
        errors.append("quality_risk must be none|light|medium|heavy")
    for field in ("blockers", "warnings"):
        if not isinstance(decision.get(field), list):
            errors.append(f"{field} must be a list")
    reviewed = decision.get("reviewed_agent_step_ids")
    if (
        not isinstance(reviewed, list)
        or any(not isinstance(step_id, int) or isinstance(step_id, bool) for step_id in reviewed)
    ):
        errors.append("reviewed_agent_step_ids must be an integer list")
    elif len(reviewed) != len(set(reviewed)) or sorted(reviewed) != expected_step_ids:
        errors.append(
            "reviewed_agent_step_ids must list every agent step_id exactly once "
            f"(expected {expected_step_ids!r})"
        )
    findings = decision.get("consistency_findings")
    if not isinstance(findings, list):
        errors.append("consistency_findings must be a list")
    if decision.get("accept") is True and decision.get("blockers"):
        errors.append("accept=true requires an empty blockers list")
    if decision.get("accept") is True and decision.get("warnings"):
        errors.append("accept=true requires an empty warnings list")
    if decision.get("accept") is True and findings:
        errors.append("accept=true requires an empty consistency_findings list")
    return errors


def _require_clean_accept(decision: dict) -> dict:
    if decision.get("accept") is not True:
        return decision
    warnings = decision.get("warnings") or []
    findings = decision.get("consistency_findings") or []
    if not warnings and not findings:
        return decision
    normalized = copy.deepcopy(decision)
    normalized["accept"] = False
    blockers = list(normalized.get("blockers") or [])
    for warning in warnings:
        if isinstance(warning, dict):
            blockers.append({
                "type": "quality_warning_requires_rejection",
                "step_id": warning.get("step_id"),
                "detail": str(warning.get("detail") or warning.get("type") or "quality warning"),
            })
        else:
            blockers.append({"type": "quality_warning_requires_rejection", "detail": str(warning)})
    for finding in findings:
        if isinstance(finding, dict):
            blockers.append({
                "type": "consistency_finding_requires_rejection",
                "step_id": finding.get("step_id"),
                "detail": str(finding.get("detail") or finding.get("type") or "consistency finding"),
            })
        else:
            blockers.append({
                "type": "consistency_finding_requires_rejection",
                "detail": str(finding),
            })
    normalized["blockers"] = blockers
    if normalized.get("quality_risk") == "none":
        normalized["quality_risk"] = "light"
    return normalized


def quality_done_key(artifact: dict) -> str:
    fingerprints = config.pipeline_fingerprint(seed_fingerprints())
    return config.canonical_hash({
        "stage": "quality_judge",
        "custom_id": artifact.get("custom_id"),
        "artifact_done_key": artifact.get("done_key"),
        "trajectory": artifact.get("rewritten_trajectory"),
        "pipeline": fingerprints["pipeline_sha256"],
    })


def judge_artifact(
    artifact: dict,
    *,
    call_fn: StageCall,
) -> tuple[dict, dict | None]:
    """Call the independent judge only for deterministic-gate survivors."""
    if artifact.get("status") != "lint_passed":
        artifact.setdefault("quality_judge", {
            "called": False,
            "reason": f"pre-filter status {artifact.get('status')!r}",
        })
        return artifact, None
    done_key = quality_done_key(artifact)
    try:
        response = call_fn(
            quality_system_prompt(),
            _user_message(artifact),
            stage="quality",
        )
    except Exception as error:  # noqa: BLE001
        artifact["quality_judge"] = {
            "called": True,
            "done_key": done_key,
            "prompt_sha256": config.sha256_file(config.QUALITY_PROMPT_MD),
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
        artifact["status"] = "quality_model_error"
        return artifact, None
    judge: dict[str, Any] = {
        "called": True,
        "done_key": done_key,
        "prompt_sha256": config.sha256_file(config.QUALITY_PROMPT_MD),
        "schema_attempts": response.get("schema_attempts"),
    }
    if not response.get("ok"):
        judge.update({"ok": False, "error": response.get("error")})
        artifact["quality_judge"] = judge
        artifact["status"] = "quality_model_error"
        return artifact, response
    try:
        decision = _require_clean_accept(_parse(str(response.get("text") or "")))
        errors = _validate(decision, _agent_step_ids(artifact))
    except Exception as error:  # noqa: BLE001
        judge.update({"ok": False, "error": f"{type(error).__name__}: {error}"})
        artifact["quality_judge"] = judge
        artifact["status"] = "invalid_quality_judge_output"
        return artifact, response
    if errors:
        judge.update({"ok": False, "error": "; ".join(errors), "raw_decision": decision})
        artifact["quality_judge"] = judge
        artifact["status"] = "invalid_quality_judge_output"
        return artifact, response
    judge.update({"ok": True, "decision": decision})
    artifact["quality_judge"] = judge
    artifact.setdefault("gates", {})["SFT_QUALITY_OK"] = {
        "passed": decision["accept"],
        "detail": decision.get("rationale", ""),
        "blockers": decision.get("blockers") or [],
    }
    if decision["accept"]:
        artifact["gates"]["SFT_READY"] = {
            "passed": True,
            "detail": "deterministic evidence gate and independent quality judge passed",
        }
        artifact["status"] = "sft_ready"
    else:
        artifact["status"] = "rejected_by_quality_judge"
    return artifact, response
