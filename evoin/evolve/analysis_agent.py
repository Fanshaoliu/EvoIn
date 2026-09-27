"""Analysis agent: model-based failure diagnosis and clustering of a rollout.

The analysis agent examines the successful and failed trajectories of the
current harness on the evolution split and summarizes failure patterns and
actionable feedback for the next proposal round:

  1. ``analysis.compact_rollout`` builds one compact record per task
     (question, answers, judge rationale, tool flow).
  2. Diagnosis: failed trajectories are diagnosed in batches, with a few
     passing trajectories as reference, using the shared failure taxonomy.
  3. Clustering: one call groups all diagnoses into a small set of failure
     patterns plus systemic recommendations for the proposal agent.
  4. The result is assembled into the overview consumed by
     ``analysis.render_evolution_query``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evoin.evolve import analysis as A
from evoin.evolve.proposal import FAILURE_TAXONOMY, extract_json_array, extract_json_object
from evoin.models import TextModel


DEBUG_BATCH_SYS = (
    "You are a failure-diagnosis expert for a long-context agent. The agent "
    "uses scan/grep/read_lines/bash/notepad to actively retrieve evidence from a HIDDEN long "
    "document and answer a question; a strict judge then scores the answer 0/1.\n\n"
    "I will give you a BATCH of FAILED trajectories from the SAME task class. A few PASSING "
    "trajectories from that class may also be included for reference (what a successful path looks "
    "like); they are absent for zero-pass classes. Each trajectory has: the question, the gold "
    "answer, the model's final answer, the judge rationale, and a compact tool_flow (the tool "
    "calls with key args).\n\n"
    "For EACH failed trajectory, do an EVIDENCE-DRIVEN root-cause analysis:\n"
    "- ROOT CAUSE: the fundamental reason this attempt failed. Cross-reference the judge rationale "
    "against the gold answer to distinguish 'the agent believed it succeeded but the judge "
    "disagrees' vs 'retrieval / reasoning genuinely went wrong'. The passing exemplars, if any, "
    "are a reference for the intended successful path.\n"
    "- FAILURE POINT / CRITICAL MISTAKE: where in the tool_flow it first went wrong (e.g. never "
    "scanned, grep too narrow, answered without read_lines, stopped early, wrong synthesis).\n"
    "- RESULT ROBUSTNESS: judge from THIS trajectory alone whether the failure is systematic (the "
    "agent would reliably fail this way) or fragile / incidental (a borderline judge call, a "
    "near-miss answer, or luck-dependent matching). Reason from evidence strength and the judge "
    "rationale, NOT from repeated sampling; reflect it in confidence and do not over-invest in "
    "fragile cases.\n"
    "- GENERAL MECHANISM: a structural mechanism (NOT task-specific knowledge, NOT tied to a "
    "particular document / question / answer) that would prevent THIS CLASS of failure.\n\n"
    "Assign a SINGLE primary taxonomy key that best captures the dominant failure. Each key must "
    "stay atomic - never merge two distinct issues into one label. If the failure is genuinely "
    "compound, you MAY list one or more additional atomic keys in secondary_failure_types "
    "(otherwise []). Allowed taxonomy keys:\n"
    + "\n".join(f"  - {t}" for t in FAILURE_TAXONOMY)
    + "\n\nReturn ONLY a strict JSON array - one object per failed trajectory, SAME order as given. "
    "No markdown fences, no prose, no comments:\n"
    '[{"task_id": str, "primary_failure_type": "<taxonomy key>", '
    '"secondary_failure_types": ["<taxonomy key>"], "root_cause": str, '
    '"critical_evidence_gap": str, "general_lesson": str, "confidence": "high|medium|low"}]\n'
    "root_cause and general_lesson must each be concise (<= 50 words). general_lesson MUST be a "
    "general discipline; it is FORBIDDEN to include any specific document / question / answer "
    "detail in general_lesson."
)


def _fmt_failed_block(details: list[dict]) -> str:
    blocks = []
    for i, d in enumerate(details):
        tail = f"\n  status: {d['status'][:48]}" if d.get("status") and d["status"] != "ok" else ""
        multi = ""
        if d.get("stability") == "unstable":
            multi = f"\n  note: task is UNSTABLE (pass_rate={d.get('pass_rate')}); some rollouts passed."
        blocks.append(
            f"[fail {i + 1}] task_id={d['task_id']}\n"
            f"  question: {d['question']}\n"
            f"  gold_answer: {d['gold_answer']}\n"
            f"  model_answer: {d['model_answer']}\n"
            f"  judge_rationale: {d['judge_rationale']}\n"
            f"  tool_flow: {d['tool_flow']}" + tail + multi
        )
    return "\n\n".join(blocks)


def _fmt_pass_exemplars(details: list[dict]) -> str:
    if not details:
        return "(no passing trajectory available in this class)"
    blocks = []
    for i, d in enumerate(details):
        blocks.append(
            f"[pass {i + 1}] task_id={d['task_id']}\n"
            f"  tool_flow: {d['tool_flow']}\n"
            f"  judge_rationale: {d['judge_rationale']}"
        )
    return "\n\n".join(blocks)


def _debug_user_message(*, class_name: str, failed: list[dict], passing: list[dict]) -> str:
    user = (
        f"Task class: {class_name}. This batch has {len(failed)} FAILED trajectories; "
        f"{len(passing)} PASSING exemplars are included for reference (successful path).\n\n"
        f"## Failed trajectories (analyze EVERY one, keep order)\n{_fmt_failed_block(failed)}\n\n"
        f"## Passing exemplars (reference only, do NOT emit objects for these)\n"
        f"{_fmt_pass_exemplars(passing)}\n\n"
        "Produce the strict JSON array per the system prompt."
    )
    return user


CLUSTER_SYS = (
    "You are a failure-pattern synthesis expert feeding a long-context agent harness optimizer. "
    "You are given per-trajectory failure diagnoses for ALL failed tasks in ONE task class (each: "
    "task_id, failure_type, root_cause, general_lesson), plus a summary of what passing "
    "trajectories did.\n\n"
    "Cluster the failures into a SMALL number (<= 6) of coherent failure patterns grouped by "
    "SHARED ROOT CAUSE. Merge semantically similar root causes even if their taxonomy labels "
    "differ; split a single label if it actually hides distinct root causes. The clustering exists "
    "to drive a harness optimizer whose ONLY objective is the pass rate.\n\n"
    "For each cluster provide: a concise failure_type label (prefer a taxonomy key), a one-line "
    "summary, the member task_ids, 2-4 representative task_ids (the clearest / most systematic), a "
    "general_mechanism (a STRUCTURAL fix direction, NOT task-specific), and a priority (high|"
    "medium|low) reflecting leverage = cluster size x how fixable it is via harness changes.\n\n"
    "Also synthesize systemic_recommendations: 1-3 highest-leverage, GENERAL harness change "
    "directions (these are handed to the mutation step; do NOT write task-specific hacks). The "
    "passing-trajectory summary is context for what already works.\n\n"
    "Return ONLY a strict JSON object. No markdown fences, no prose:\n"
    '{"failure_clusters": [{"failure_type": str, "summary": str, "count": int, '
    '"member_task_ids": [str], "representative_task_ids": [str], "general_mechanism": str, '
    '"priority": "high|medium|low"}], "systemic_recommendations": [str]}'
)


def _cluster_user_message(
    *, class_name: str, round_idx: int, diagnoses: list[dict], total_tasks: int,
    pass_rate: float, passing: list[dict],
) -> str:
    diag_lines = "\n".join(
        f"- task_id={d.get('task_id')} | type={d.get('primary_failure_type') or d.get('failure_type')} | "
        f"root_cause={str(d.get('root_cause', ''))[:200]} | "
        f"lesson={str(d.get('general_lesson', ''))[:160]}"
        for d in diagnoses
    ) or "(none)"
    common_tools = Counter(t for d in passing for t in d.get("tool_names", [])).most_common(8)
    pass_flows = "\n".join(f"  - {d['task_id']}: {d['tool_flow'][:180]}" for d in passing[:4]) or "  (none)"
    user = (
        f"Task class: {class_name}, round {round_idx}. "
        f"{len(diagnoses)} failed / {total_tasks} total tasks (pass_rate={pass_rate:.3f}).\n\n"
        f"## Per-trajectory failure diagnoses (from stage-1 debug)\n{diag_lines}\n\n"
        f"## Passing strategy summary\n"
        f"- passing tasks: {len(passing)}\n"
        f"- common tools in passing traces: {common_tools}\n"
        f"- exemplar passing tool_flows:\n{pass_flows}\n\n"
        "Cluster the failures and synthesize per the system prompt. Output the strict JSON object."
    )
    return user


def _call_json(analyst: TextModel, system: str, user: str, *, parse, valid, attempts: int, what: str):
    """Regenerate a response that violates its JSON contract, up to ``attempts`` calls."""
    head = ""
    for attempt in range(1, attempts + 1):
        raw = analyst(system, user)
        value = parse(raw)
        if valid(value):
            return value
        head = (raw or "")[:300]
        print(f"  [analysis] {what} attempt {attempt}/{attempts}: output violates the JSON contract", flush=True)
    raise RuntimeError(f"{what} failed after {attempts} attempts; last output head={head!r}")


def _example_from_detail(d: dict) -> dict:
    return {
        "task_id": d["task_id"], "score": d["score"], "stability": d["stability"],
        "question": d["question"], "judge_rationale": d["judge_rationale"],
        "tool_flow": d["tool_flow"], "root_cause": d["root_cause"],
    }


def analyze_rollout(
    analyst: TextModel,
    *,
    train_summary: dict,
    train_rows: list[dict],
    output_dir: str | Path,
    class_name: str,
    round_idx: int,
    correct_threshold: float = 1.0,
    batch_size: int = 10,
    max_pass_exemplars: int = 4,
    attempts: int = 4,
) -> tuple[dict, list[dict]]:
    """Diagnose and cluster the failures of one rollout; returns ``(overview, details)``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size if batch_size > 0 else 10

    details = A.compact_rollout(train_summary, train_rows, correct_threshold)
    passed = [d for d in details if d["failure_type"] == "passed"]
    failed = [d for d in details if d["failure_type"] not in ("passed", "infrastructure_error")]
    by_id = {d["task_id"]: d for d in details}

    diagnoses: list[dict] = []
    cluster_obj: dict = {"failure_clusters": [], "systemic_recommendations": []}
    if failed:
        pass_exemplars = passed[:max_pass_exemplars]
        n_batches = (len(failed) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = failed[bi * batch_size:(bi + 1) * batch_size]
            arr = _call_json(
                analyst, DEBUG_BATCH_SYS,
                _debug_user_message(class_name=class_name, failed=batch, passing=pass_exemplars),
                parse=extract_json_array, valid=bool, attempts=attempts, what="diagnosis",
            )
            diagnoses.extend(o for o in arr if isinstance(o, dict) and o.get("task_id"))

        diag_by_id = {str(o.get("task_id")): o for o in diagnoses}
        for d in failed:
            o = diag_by_id.get(d["task_id"])
            if not o:
                continue
            ft = str(o.get("primary_failure_type") or o.get("failure_type") or "").strip()
            if ft:
                d["failure_type"] = ft
            sec = o.get("secondary_failure_types")
            d["secondary_failure_types"] = (
                [str(x).strip() for x in sec if str(x).strip()] if isinstance(sec, list) else [])
            d["root_cause"] = str(o.get("root_cause") or d["root_cause"])
            d["critical_evidence_gap"] = str(o.get("critical_evidence_gap") or d["critical_evidence_gap"])
            d["general_lesson"] = str(o.get("general_lesson") or d["general_lesson"])
            d["confidence"] = str(o.get("confidence") or "")

        cluster_obj = _call_json(
            analyst, CLUSTER_SYS,
            _cluster_user_message(
                class_name=class_name, round_idx=round_idx,
                diagnoses=[diag_by_id[d["task_id"]] for d in failed if d["task_id"] in diag_by_id],
                total_tasks=len(details), pass_rate=(len(passed) / len(details) if details else 0.0),
                passing=passed,
            ),
            parse=extract_json_object, valid=lambda obj: bool(obj.get("failure_clusters")),
            attempts=attempts, what="clustering",
        )

    overview = _assemble_overview(
        details=details, by_id=by_id, cluster_obj=cluster_obj,
        class_name=class_name, round_idx=round_idx, train_summary=train_summary)
    with open(output_dir / "detail.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (output_dir / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(output_dir / "diagnoses.jsonl", "w", encoding="utf-8") as f:
        for o in diagnoses:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    (output_dir / "clusters.json").write_text(json.dumps(cluster_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return overview, details


def _assemble_overview(
    *, details: list[dict], by_id: dict[str, dict], cluster_obj: dict,
    class_name: str, round_idx: int, train_summary: dict,
) -> dict:
    total = len(details)
    pass_count = sum(1 for d in details if d["failure_type"] == "passed")
    fail_count = total - pass_count
    stability_counts = Counter(d["stability"] for d in details)
    failure_type_counts = Counter(d["failure_type"] for d in details)

    failure_clusters = []
    for c in cluster_obj.get("failure_clusters") or []:
        if not isinstance(c, dict):
            continue
        members = [str(t) for t in (c.get("member_task_ids") or []) if str(t) in by_id]
        reps = [str(t) for t in (c.get("representative_task_ids") or []) if str(t) in by_id]
        if not reps:
            reps = members[:4]
        count = int(c.get("count") or len(members) or 0)
        failure_clusters.append({
            "failure_type": str(c.get("failure_type") or "unspecified"),
            "description": str(c.get("summary") or A.FAILURE_TYPE_DESCRIPTIONS.get(
                str(c.get("failure_type") or ""), "")),
            "count": count,
            "rate": count / total if total else 0.0,
            "priority": str(c.get("priority") or ""),
            "general_lesson": str(c.get("general_mechanism") or ""),
            "representative_examples": [_example_from_detail(by_id[t]) for t in reps[:5]],
        })

    passed = [d for d in details if d["failure_type"] == "passed"]
    overview = {
        "schema_version": "evoin.analysis.v1",
        "class": class_name,
        "round": round_idx,
        "n_tasks": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_count / total if total else 0.0,
        "train_summary_pass_rate": train_summary.get("pass_rate"),
        "stability_counts": dict(stability_counts),
        "failure_type_counts": dict(failure_type_counts),
        "failure_clusters": failure_clusters,
        "pass_vs_fail": str(cluster_obj.get("pass_vs_fail") or ""),
        "systemic_recommendations": [str(x) for x in (cluster_obj.get("systemic_recommendations") or [])],
        "passing_strategy_summary": {
            "count": len(passed),
            "representative_examples": [_example_from_detail(d) for d in passed[:4]],
            "common_tools": Counter(t for d in passed for t in d.get("tool_names", [])).most_common(8),
        },
        "unstable_tasks": [
            {"task_id": d["task_id"], "pass_rate": d["pass_rate"], "failure_type": d["failure_type"]}
            for d in details if d["stability"] == "unstable"
        ][:20],
    }
    return overview
