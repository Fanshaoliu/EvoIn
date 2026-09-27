"""Batch rollout of one harness over a list of tasks, scored by the task evaluator.

``run_rollout`` writes ``OUT/rollout.jsonl`` (one row per task sample),
``OUT/summary.json`` and finally ``OUT/_SUCCESS``.
"""
from __future__ import annotations

import copy
import json
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evoin.models import AgentModel, Evaluator
from evoin.runtime.agent import load_harness_runtime, run_agent
from evoin.runtime.task import task_id_of

MAX_ANSWER_CHARS = 6000
MAX_GOLD_CHARS = 2000


def _cap(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


def _cap_trajectory(trajectory: dict, max_obs_chars: int) -> dict:
    capped = copy.deepcopy(trajectory)
    for step in capped.get("steps", []):
        for observation in step.get("observations", []):
            observation["content"] = _cap(str(observation.get("content", "")), max_obs_chars)
    return capped


def load_tasks(path: Path, max_tasks: int) -> list[dict]:
    tasks: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    for index, task in enumerate(tasks):
        if not task_id_of(task):
            task["task_id"] = f"task_{index}"
    return tasks


def score_target(target: dict, *, evaluator: Evaluator) -> dict:
    task = target["judge_task"]
    status = str(target.get("status") or "ok")
    judge_error = None
    if status.startswith("error"):
        judge = {"score": 0.0, "rationale": f"not judged: {status}", "requirement_status": []}
    else:
        try:
            judge = evaluator(task, str(target.get("model_output_full") or ""))
        except Exception as exc:  # noqa: BLE001
            judge_error = exc
            judge = {"score": 0.0, "rationale": f"judge error: {type(exc).__name__}: {exc}",
                     "requirement_status": []}
    if judge_error is not None:
        status = f"error:judge:{type(judge_error).__name__}:{judge_error}"
    return {
        "score": float(judge.get("score", 0) or 0),
        "rationale": str(judge.get("rationale", "")),
        "requirement_status": list(judge.get("requirement_status") or []),
        "status": status,
    }


def _run_one(
    task: dict,
    sample_idx: int,
    *,
    harness: Any,
    model: AgentModel,
    evaluator: Evaluator,
    work_root: Path,
    max_obs_chars: int,
    keep_workspaces: bool,
) -> dict:
    task_id = task_id_of(task)
    work_dir = work_root / f"{task_id}__s{sample_idx}"
    try:
        target = run_agent(harness=harness, task=task, work_dir=work_dir, model=model)
        scored = score_target(target, evaluator=evaluator)
    except Exception as exc:  # noqa: BLE001 - one failed task must not abort the batch
        traceback.print_exc()
        target = {"task_id": task_id, "model_output_full": "", "n_steps": 0, "tool_call_names": [],
                  "n_malformed_tool_calls": 0, "trajectory": {"steps": []}, "judge_task": task}
        scored = {"score": 0.0, "rationale": "", "requirement_status": [],
                  "status": f"error:{type(exc).__name__}:{exc}"}
    finally:
        if not keep_workspaces:
            shutil.rmtree(work_dir, ignore_errors=True)
    judge_task = target.get("judge_task") or task
    return {
        "task_id": task_id,
        "sample_idx": sample_idx,
        "question": _cap(str(judge_task.get("question") or ""), MAX_GOLD_CHARS),
        "answer": _cap(str(judge_task.get("answer") or ""), MAX_GOLD_CHARS),
        **scored,
        "model_output": _cap(str(target.get("model_output_full") or ""), MAX_ANSWER_CHARS),
        "n_steps": int(target.get("n_steps") or 0),
        "tool_call_names": list(target.get("tool_call_names") or []),
        "n_malformed_tool_calls": int(target.get("n_malformed_tool_calls") or 0),
        "trajectory": _cap_trajectory(target.get("trajectory") or {"steps": []}, max_obs_chars),
    }


def summarize(rows: list[dict], *, harness_dir: Path, n_tasks: int, samples: int) -> dict:
    """Aggregate scores with fixed denominators: a failed target or judge call scores zero."""
    per_task: dict[str, list[float]] = {}
    for row in rows:
        per_task.setdefault(str(row["task_id"]), []).append(float(row.get("score", 0) or 0))
    stats: dict[str, dict] = {}
    for task_id, scores in per_task.items():
        passes = sum(1 for s in scores if s >= 1.0)
        n = len(scores)
        stability = "stable_pass" if passes == n else "stable_fail" if passes == 0 else "unstable"
        stats[task_id] = {
            "n": n,
            "pass_count": passes,
            "pass_rate": passes / n,
            "mean_score": sum(scores) / n,
            "stability": stability,
            "scores": scores,
        }
    score_sum = sum(float(r.get("score", 0) or 0) for r in rows)
    passed = sum(1 for r in rows if float(r.get("score", 0) or 0) >= 1.0)
    return {
        "harness": Path(harness_dir).name,
        "n_tasks": n_tasks,
        "samples_per_task": samples,
        "n": len(rows),
        "passed": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "avg_score": score_sum / len(rows) if rows else 0.0,
        "scores": {task_id: v["mean_score"] for task_id, v in stats.items()},
        "per_task": stats,
        "n_errors": sum(1 for r in rows if str(r.get("status") or "").startswith("error")),
        "n_malformed_tool_calls": sum(int(r.get("n_malformed_tool_calls") or 0) for r in rows),
        "n_unstable": sum(1 for v in stats.values() if v["stability"] == "unstable"),
    }


def run_rollout(
    *,
    harness_dir: Path,
    tasks: list[dict],
    output_dir: Path,
    model: AgentModel,
    evaluator: Evaluator,
    concurrency: int = 8,
    samples_per_task: int = 1,
    max_obs_chars: int = 1500,
    keep_workspaces: bool = False,
) -> dict:
    """Roll out ``model`` under ``harness_dir`` on ``tasks``; ``max_obs_chars=0`` keeps full observations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_SUCCESS").unlink(missing_ok=True)
    harness = load_harness_runtime(harness_dir)
    samples = max(1, samples_per_task)
    jobs = [(index, task, k) for index, task in enumerate(tasks) for k in range(samples)]
    rows: list[tuple[int, int, dict]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _run_one, task, k, harness=harness, model=model, evaluator=evaluator,
                work_root=output_dir / "workspaces", max_obs_chars=max_obs_chars,
                keep_workspaces=keep_workspaces,
            ): (index, k)
            for index, task, k in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index, k = futures[future]
            row = future.result()
            rows.append((index, k, row))
            print(f"  [rollout] {done}/{len(jobs)} task={row['task_id']} sample={k} "
                  f"score={row['score']} status={row['status'][:60]}", flush=True)
    ordered = [row for _, _, row in sorted(rows, key=lambda item: (item[0], item[1]))]
    tmp = output_dir / "rollout.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output_dir / "rollout.jsonl")
    summary = summarize(ordered, harness_dir=harness_dir, n_tasks=len(tasks), samples=samples)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not keep_workspaces:
        shutil.rmtree(output_dir / "workspaces", ignore_errors=True)
    (output_dir / "_SUCCESS").touch()
    return summary


def read_rollout(output_dir: Path) -> tuple[dict, list[dict]]:
    """Load ``(summary, rows)`` written by :func:`run_rollout`."""
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    with (output_dir / "rollout.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return summary, rows
