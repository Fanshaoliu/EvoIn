"""Deterministic parts of the analysis step.

``compact_rollout`` turns a rollout on the evolution split into one compact
record per task (question, answers, judge rationale, tool flow) that the
analysis agent (``analysis_agent``) diagnoses; ``render_evolution_query``
renders the evolution query given to the proposal agent, following the
template in the paper appendix.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evoin.evolve.harness import HarnessView


FAILURE_TYPE_DESCRIPTIONS = {
    "passed": "The task passed; use only as contrast for successful behavior.",
    "infrastructure_error": "The rollout failed due to model/tool/runtime error, not agent behavior.",
    "truncated": "The agent hit max_iterations or stopped before producing a complete answer.",
    "no_scan_first": "The agent skipped an initial scan and lacked a document map.",
    "evidence_miss": "The agent missed required evidence or searched too narrowly.",
    "no_verify_lines": "The agent answered without opening/verifying raw evidence lines.",
    "no_citation": "The final answer lacks traceable raw-line support.",
    "incomplete": "The answer covers only part of a multi-part / exhaustive request.",
    "repeat_degenerate": "The agent repeated low-value calls or spun in place.",
    "format_violation": "The final response format is invalid or leaks tool/protocol text.",
    "wrong_reasoning": "Evidence was present but synthesis, comparison, or reasoning was wrong.",
}


def _cap(text: Any, n: int) -> str:
    s = str(text or "").replace("\r", " ").strip()
    s = "\n".join(line.strip() for line in s.splitlines() if line.strip())
    return s[:n]


def _task_id(row: dict) -> str:
    return str(row.get("task_id") or ((row.get("metadata") or {}).get("task_id")) or "unknown")


def _tool_flow(traj: dict, *, max_steps: int = 32) -> str:
    pieces: list[str] = []
    for step in (traj.get("steps") or []):
        for tc in step.get("tool_calls") or []:
            name = str(tc.get("name") or "?")
            args = tc.get("input") or {}
            arg = ""
            if "pattern" in args:
                arg = f"({ _cap(args.get('pattern'), 48)!r})"
            elif "command" in args:
                arg = f"({ _cap(args.get('command'), 48)!r})"
            elif "start" in args or "end" in args:
                arg = f"({args.get('start', '?')}..{args.get('end', '?')})"
            elif "result" in args:
                arg = f"(result={_cap(args.get('result'), 48)!r})"
            elif "reason" in args:
                arg = f"({_cap(args.get('reason'), 64)})"
            pieces.append(f"{name}{arg}")
            if len(pieces) >= max_steps:
                pieces.append("...")
                return " -> ".join(pieces)
    return " -> ".join(pieces) if pieces else "(no tool calls)"


def _tool_names(row: dict) -> list[str]:
    names = row.get("tool_call_names")
    if isinstance(names, list):
        return [str(x) for x in names]
    out: list[str] = []
    for step in ((row.get("trajectory") or {}).get("steps") or []):
        for tc in step.get("tool_calls") or []:
            out.append(str(tc.get("name") or "?"))
    return out


def _has_repetition(names: list[str]) -> bool:
    if len(names) < 8:
        return False
    counts = Counter(names)
    return bool(counts and max(counts.values()) >= max(6, len(names) // 2))


def _classify_failure(row: dict, *, threshold: float = 1.0) -> str:
    score = float(row.get("score", 0) or 0)
    status = str(row.get("status", "") or "")
    if score >= threshold:
        return "passed"
    if status.startswith(("fatal", "error")):
        return "infrastructure_error"
    names = _tool_names(row)
    rationale = str(row.get("rationale", "") or "").lower()
    output = str(row.get("model_output", "") or "").lower()
    combined = f"{rationale}\n{output}"
    if "max_iteration" in status.lower() or "truncated" in combined or "cut off" in combined:
        return "truncated"
    if "scan" not in names and ("grep" in names or len(names) <= 2):
        return "no_scan_first"
    if any(x in combined for x in ["json", "tool", "args", "complete_task", "format"]):
        return "format_violation"
    if "read_lines" not in names and any(x in combined for x in ["unsupported", "not supported", "evidence"]):
        return "no_verify_lines"
    if any(x in combined for x in ["line", "citation", "cite", "supporting evidence"]):
        return "no_citation"
    if any(x in combined for x in ["partial", "incomplete", "missing", "omits", "does not include",
                                   "not all", "fails to address", "only mentions"]):
        return "incomplete"
    if any(x in combined for x in ["not found", "cannot find", "missed", "no evidence", "wrong section",
                                   "irrelevant", "does not match"]):
        return "evidence_miss"
    if _has_repetition(names):
        return "repeat_degenerate"
    return "wrong_reasoning"


def _root_cause_for(ftype: str, row: dict) -> str:
    rationale = _cap(row.get("rationale", ""), 220)
    if ftype == "passed":
        return "Passed; use as positive contrast."
    if ftype == "infrastructure_error":
        return f"Rollout/runtime error: {_cap(row.get('status'), 160)}"
    if ftype == "truncated":
        return "The agent exhausted its budget or stopped before completing the requested answer."
    if ftype == "no_scan_first":
        return "The agent lacked a document map before searching or answering."
    if ftype == "evidence_miss":
        return rationale or "The search strategy likely missed the evidence-bearing region or alias."
    if ftype == "no_verify_lines":
        return "The agent relied on search summaries / guesses without enough raw-line verification."
    if ftype == "no_citation":
        return "The answer is not grounded in traceable raw lines."
    if ftype == "incomplete":
        return rationale or "The answer did not cover all requested sub-items or dimensions."
    if ftype == "repeat_degenerate":
        return "The tool flow repeats low-value actions instead of expanding evidence coverage."
    if ftype == "format_violation":
        return rationale or "The final output violates the requested answer/protocol format."
    return rationale or "The agent saw enough evidence but synthesized or compared it incorrectly."


def _general_lesson(ftype: str) -> str:
    lessons = {
        "passed": "Preserve the retrieval and verification behaviors seen in passing traces.",
        "infrastructure_error": "Ignore infrastructure failures for harness design unless they expose budget fragility.",
        "truncated": "Budget and closure rules must ensure complete answers before finalization.",
        "no_scan_first": "Always build a document map before targeted retrieval.",
        "evidence_miss": "Use broadened keyword ladders and range sweeps when exact search is sparse.",
        "no_verify_lines": "Open raw line windows before trusting any retrieved fact.",
        "no_citation": "Force every claim to be backed by line-anchored evidence.",
        "incomplete": "Decompose multi-part questions and audit every sub-item before final answer.",
        "repeat_degenerate": "Detect repeated low-yield calls and switch retrieval strategy.",
        "format_violation": "Add a hard final-format gate before complete_task.",
        "wrong_reasoning": "Require a draft-then-audit pass that checks qualifiers, comparisons, and semantic scope.",
    }
    return lessons.get(ftype, "Design a general mechanism that prevents this failure class.")


def compact_rollout(train_summary: dict, train_rows: list[dict], correct_threshold: float = 1.0) -> list[dict]:
    """One record per task; a failed sample is preferred as the representative."""
    per_task_meta = train_summary.get("per_task") or {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        grouped[_task_id(row)].append(row)

    details: list[dict] = []
    for tid, rows in sorted(grouped.items()):
        meta = per_task_meta.get(tid) or {}
        rep = next((r for r in rows if float(r.get("score", 0) or 0) < correct_threshold), rows[0])
        ftype = _classify_failure(rep, threshold=correct_threshold)
        scores = [float(r.get("score", 0) or 0) for r in rows]
        pass_rate = float(meta.get("pass_rate", mean(scores) if scores else 0.0) or 0.0)
        stability = str(meta.get("stability") or (
            "stable_pass" if pass_rate >= correct_threshold
            else "stable_fail" if pass_rate <= 0 else "unstable"))
        details.append({
            "task_id": tid,
            "score": float(rep.get("score", 0) or 0),
            "pass_rate": pass_rate,
            "stability": stability,
            "failure_type": ftype,
            "secondary_failure_types": [],
            "question": _cap(rep.get("question"), 360),
            "gold_answer": _cap(rep.get("answer"), 260),
            "model_answer": _cap(rep.get("model_output"), 420),
            "judge_rationale": _cap(rep.get("rationale"), 420),
            "tool_flow": _tool_flow(rep.get("trajectory") or {}),
            "tool_names": _tool_names(rep),
            "n_steps": int(rep.get("n_steps") or 0),
            "root_cause": _root_cause_for(ftype, rep),
            "critical_evidence_gap": _cap(rep.get("rationale"), 260),
            "general_lesson": _general_lesson(ftype),
            "status": str(rep.get("status", "") or ""),
        })
    return details


def _cap_block(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + f"\n...[truncated {len(text) - limit} chars]...\n" + text[-half:]


def _render_file_section(title: str, files: dict[str, str], *, limit: int) -> str:
    if not files:
        return f"### {title}\nNONE\n"
    parts = [f"### {title}"]
    for path, content in sorted(files.items()):
        parts.append(f"#### {path}\n```text\n{_cap_block(content, limit)}\n```")
    return "\n\n".join(parts) + "\n"


def render_harness_files(view: HarnessView) -> str:
    return "\n".join([
        "## Current Full-Harness Files",
        "These are the editable harness files for this round. Use them when proposing file_changes.",
        "### base_harness.yaml",
        "```yaml",
        _cap_block(view.base_harness_yaml, 10000),
        "```",
        _render_file_section("tools/*.tool.yaml", view.tool_yaml_files, limit=4000),
        _render_file_section("tool_impl/*.py", view.tool_impl_files, limit=9000),
        _render_file_section("middleware/*.py", view.middleware_files, limit=9000),
    ])


def render_evolution_query(
    *,
    overview: dict,
    view: HarnessView,
    available_tools: list[str],
    mandatory_tools: list[str],
    previous_attribution_text: str = "",
    best_ever_summary: str = "",
    n_candidates: int = 3,
) -> str:
    clusters = overview.get("failure_clusters") or []
    cluster_lines: list[str] = []
    for c in clusters:
        examples = "; ".join(
            f"{e['task_id']}({e['score']}): {e['root_cause'][:120]}"
            for e in (c.get("representative_examples") or [])[:3]
        )
        prio = f" [priority={c['priority']}]" if c.get("priority") else ""
        cluster_lines.append(
            f"- {c['failure_type']} x{c['count']} ({c['rate']:.1%}){prio}: "
            f"{c.get('general_lesson', '')}\n  representatives: {examples or 'NONE'}"
        )
    pass_examples = overview.get("passing_strategy_summary", {}).get("representative_examples") or []
    pass_lines = [
        f"- {e['task_id']}: {e['tool_flow'][:180]} | judge={e['judge_rationale'][:160]}"
        for e in pass_examples[:4]
    ]
    recs = overview.get("systemic_recommendations") or []
    recs_block = ("\nHigh-leverage recommendations from the failure analysis:\n"
                  + "\n".join(f"- {str(r)}" for r in recs)) if recs else ""
    unstable = overview.get("unstable_tasks") or []
    unstable_lines = [
        f"- {x['task_id']}: pass_rate={x['pass_rate']}, failure_type={x['failure_type']}"
        for x in unstable[:10]
    ]
    return (
        f"# EvoIn Evolution Query\n\n"
        f"## 1. Current Iteration Overview\n"
        f"- class: {overview.get('class')}\n"
        f"- round: {overview.get('round')}\n"
        f"- train tasks: {overview.get('n_tasks')}\n"
        f"- train pass_rate: {overview.get('pass_rate'):.3f}\n"
        f"- pass/fail: {overview.get('pass_count')}/{overview.get('fail_count')}\n"
        f"- stability_counts: {overview.get('stability_counts')}\n\n"
        f"## 2. Current Harness\n"
        f"- current tools: {view.enabled_tools}\n"
        f"- available tools: {available_tools}\n"
        f"- mandatory tools: {mandatory_tools}\n"
        f"- max_iterations: {view.max_iterations}\n\n"
        f"## 3. Failure Pattern Clusters\n"
        f"{chr(10).join(cluster_lines) if cluster_lines else 'NONE'}\n\n"
        f"## 4. Passing Strategy Summary\n"
        f"{chr(10).join(pass_lines) if pass_lines else 'NONE'}\n\n"
        f"## 5. Result Robustness\n"
        f"Judge each failure's robustness from its own trajectory (evidence strength + judge "
        f"rationale). Do not make aggressive changes driven by fragile / incidental cases.\n"
        f"{chr(10).join(unstable_lines) if unstable_lines else 'No multi-sample unstable tasks flagged.'}\n\n"
        f"## 6. Previous Iteration Change Attribution\n"
        f"{previous_attribution_text or 'NONE'}\n\n"
        f"## 7. Best-Ever Harness Summary\n"
        f"{best_ever_summary or 'NONE'}\n\n"
        f"## 8. Execution Instructions\n"
        f"Analyze failures -> group into pattern classes -> design general mechanisms -> propose "
        f"{n_candidates} orthogonal candidate harnesses. Optimize the pass rate. Do not overfit to a "
        f"specific task, document, or answer. Prefer changes that address large stable-fail clusters."
        f"{recs_block}\n\n"
        f"## 9. Current System Prompt\n"
        f"{view.system_prompt.rstrip()}\n\n"
        f"{render_harness_files(view)}"
    )
