"""Build the rewrite-stage request for one candidate trajectory."""

from __future__ import annotations

import json
from typing import Any

from evoin.tailor import config
from evoin.tailor.replay import (
    drop_nonload_bearing_bash_steps,
    replay_trajectory,
    task_sha256,
)
from evoin.tailor.seed import (
    seed_fingerprints,
    seed_manifest,
    seed_max_iterations,
    seed_system_prompt,
    seed_tools_block,
)
from evoin.tailor.trajectory import (
    TrajectoryBindingError,
    convert_rollout_to_atif,
    count_agent_steps,
    terminal_mode_from_atif,
    terminal_mode_from_rollout,
)


def seed_context_block() -> str:
    budget = json.dumps({"seed_max_iterations": seed_max_iterations()}, ensure_ascii=False)
    return (
        f"<seed_system_prompt>\n{seed_system_prompt()}\n</seed_system_prompt>\n\n"
        f"<seed_tools>\n{seed_tools_block()}\n</seed_tools>\n\n"
        f"<runtime_budget>\n{budget}\n</runtime_budget>\n\n"
    )


def user_message(
    question: str,
    gold: str,
    atif: dict,
    profile: dict[str, Any],
    preflight_sanitization: list[dict] | None = None,
) -> str:
    task = json.dumps({"question": question, "gold": gold}, ensure_ascii=False)
    return (
        seed_context_block()
        + f"<document_profile>\n{json.dumps(profile, ensure_ascii=False)}\n</document_profile>\n\n"
        + f"<task>\n{task}\n</task>\n\n"
        + "<preflight_sanitization>\n"
        + json.dumps(preflight_sanitization or [], ensure_ascii=False)
        + "\n</preflight_sanitization>\n\n"
        + f"<best_trajectory>\n{json.dumps(atif, ensure_ascii=False, indent=2)}\n</best_trajectory>\n"
    )


def _identity(candidate: dict) -> dict[str, Any]:
    return {
        "custom_id": str(candidate.get("custom_id") or ""),
        "task_id": candidate.get("task_id"),
        "sample_idx": candidate.get("sample_idx"),
        "orig_score": candidate.get("score"),
        "orig_model_output": (candidate.get("rollout_record") or {}).get("model_output"),
        "schema_version": config.SCHEMA_VERSION,
    }


def prepare_candidate(candidate: dict, task: dict[str, Any]) -> tuple[dict, str | None]:
    """Convert, sanitize and seed-replay one candidate, then build its request.

    Returns ``(meta, user_message)``; on failure ``meta["preflight_error"]`` is
    set and the message is ``None``.
    """
    record = candidate.get("rollout_record") or {}
    fingerprints = config.pipeline_fingerprint(seed_fingerprints())
    meta: dict[str, Any] = {**_identity(candidate), "fingerprints": fingerprints}
    profile: dict[str, Any] = {"num_lines": None}
    try:
        if not meta["custom_id"]:
            raise ValueError("candidate has no custom_id")
        expected_mode = str(candidate.get("terminal_mode") or "")
        if expected_mode not in config.allowed_terminal_modes():
            raise ValueError(
                f"terminal mode {expected_mode!r} is not allowed; the trajectory must close with complete_task"
            )
        if terminal_mode_from_rollout(record) != expected_mode:
            raise ValueError("candidate terminal mode does not match its rollout record")
        source_atif = convert_rollout_to_atif(record)
        if terminal_mode_from_atif(source_atif) != expected_mode:
            raise ValueError("ATIF conversion changed the terminal mode")
        for_hydration, preflight_sanitization = drop_nonload_bearing_bash_steps(source_atif)
        hydrated, hydration = replay_trajectory(for_hydration, task, phase="hydrate")
        hydration["preflight_sanitization"] = preflight_sanitization
        profile = {"num_lines": hydration.get("num_lines")}
        if not hydration.get("ok"):
            raise RuntimeError(
                "seed hydration failed; the trajectory needs a new rollout "
                f"(blocked={hydration.get('blocked_call_ids') or []}, "
                f"execution_errors={hydration.get('execution_error_call_ids') or []})"
            )
        hydrated_mode = terminal_mode_from_atif(hydrated)
        if hydrated_mode != expected_mode:
            raise RuntimeError(f"seed hydration changed the terminal mode ({expected_mode!r} != {hydrated_mode!r})")
        question = str(record.get("question") or task.get("question") or "")
        gold = str(record.get("answer") or task.get("answer") or "")
        message = user_message(question, gold, hydrated, profile, preflight_sanitization)
        user_sha = config.sha256_text(message)
        meta.update({
            "document_profile": profile,
            "question": question,
            "gold": gold,
            "terminal_mode_source": expected_mode,
            "terminal_mode_hydrated": hydrated_mode,
            "user_sha256": user_sha,
            "task_sha256": task_sha256(task),
            "n_agent_steps_source": count_agent_steps(source_atif),
            "n_agent_steps_hydrated": count_agent_steps(hydrated),
            "source_atif_trajectory": source_atif,
            "atif_trajectory": hydrated,
            "hydration": hydration,
            "preflight_sanitization": preflight_sanitization,
            "seed_manifest": seed_manifest(),
            "done_key": config.canonical_hash({
                "stage": "rewrite",
                "custom_id": meta["custom_id"],
                "user_sha256": user_sha,
                "pipeline": fingerprints["pipeline_sha256"],
            }),
        })
        return meta, message
    except (TrajectoryBindingError, FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        error_text = f"{type(error).__name__}: {error}"
        meta.update({
            "document_profile": profile,
            "task_sha256": task_sha256(task),
            "terminal_mode_source": candidate.get("terminal_mode"),
            "preflight_error": error_text,
            "done_key": config.canonical_hash({
                "stage": "rewrite",
                "custom_id": meta["custom_id"],
                "preflight_error": error_text,
                "pipeline": fingerprints["pipeline_sha256"],
            }),
        })
        return meta, None
