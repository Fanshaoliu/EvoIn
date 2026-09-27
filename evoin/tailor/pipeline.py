"""Tailor: rewrite solved rollouts into seed-reproducible, evidence-closed trajectories.

Per candidate:

1. convert the rollout row to ATIF, drop provably non-load-bearing unsafe
   calls, and replay every call under the seed harness (``rewrite_request``);
2. ask the rewrite model for a sparse edit list;
3. merge the edits, replay again, and validate structure (``edits``);
4. run the deterministic checks, then at most one repair round (``repair``);
5. ask an independent checking model to admit the survivor (``quality``).

``run`` writes ``artifacts/<id>.json`` (one audit record per candidate, reused
on restart), ``accepted.jsonl`` and ``report.json`` under ``out_dir``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from evoin.models import TailorModels
from evoin.runtime.task import task_id_of
from evoin.tailor import config

RETRY_FROM_SCRATCH = frozenset({"model_error", "pipeline_error"})
RETRY_REPAIR = frozenset({"repair_model_error"})
RETRY_QUALITY = frozenset({"quality_model_error"})
ACCEPTED_STATUS = "sft_ready"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if isinstance(value, dict):
                yield value


def load_tasks(paths: list[Path]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for path in paths:
        for task in _read_jsonl(path):
            task_id = task_id_of(task)
            if not task_id:
                continue
            if task_id in tasks:
                duplicates.add(task_id)
            tasks[task_id] = task
    for task_id in duplicates:
        tasks.pop(task_id, None)
    if duplicates:
        print(f"[tailor] ignoring {len(duplicates)} task id(s) that are not unique", file=sys.stderr)
    return tasks


def select_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    min_score: float = 1.0,
    max_per_task: int = 0,
) -> tuple[list[dict[str, Any]], Counter]:
    """Keep rows with the full task score whose trajectory closes with a valid final answer."""
    from evoin.tailor.trajectory import TrajectoryBindingError, terminal_mode_from_rollout

    stats: Counter = Counter()
    seen: set[str] = set()
    per_task: Counter = Counter()
    candidates: list[dict[str, Any]] = []
    allowed = config.allowed_terminal_modes()
    for row in rows:
        stats["rows"] += 1
        score = row.get("score")
        if not isinstance(score, (int, float)) or score < min_score:
            stats["skip_score"] += 1
            continue
        if str(row.get("status") or "ok") != "ok":
            stats["skip_status"] += 1
            continue
        try:
            mode = terminal_mode_from_rollout(row)
        except TrajectoryBindingError:
            mode = None
        if mode not in allowed:
            stats["skip_terminal"] += 1
            continue
        task_id = str(row.get("task_id") or "")
        custom_id = f"{task_id}__s{row.get('sample_idx', 0)}"
        if not task_id or custom_id in seen:
            stats["skip_duplicate"] += 1
            continue
        if max_per_task and per_task[task_id] >= max_per_task:
            stats["skip_per_task_cap"] += 1
            continue
        seen.add(custom_id)
        per_task[task_id] += 1
        candidates.append({
            "custom_id": custom_id,
            "task_id": task_id,
            "sample_idx": row.get("sample_idx", 0),
            "score": float(score),
            "terminal_mode": mode,
            "rollout_record": row,
        })
    stats["candidates"] = len(candidates)
    return candidates, stats


def artifact_name(custom_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", custom_id)[:120]
    return f"{safe}.{hashlib.sha1(custom_id.encode('utf-8')).hexdigest()[:8]}.json"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_existing(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def process_candidate(
    candidate: dict[str, Any],
    task: dict[str, Any] | None,
    *,
    call_fn: Callable[..., dict],
    existing: dict[str, Any] | None = None,
    pipeline_sha256: str,
    run_repair: bool = True,
    run_quality: bool = True,
) -> dict[str, Any]:
    """Run the stages for one candidate, resuming a matching artifact when safe."""
    from evoin.tailor import edits, quality, repair, rewrite_request
    from evoin.tailor.prompts import rewrite_system_prompt
    from evoin.tailor.replay import task_sha256

    if task is None:
        meta = {
            "custom_id": candidate["custom_id"],
            "task_id": candidate.get("task_id"),
            "sample_idx": candidate.get("sample_idx"),
            "orig_score": candidate.get("score"),
            "terminal_mode_source": candidate.get("terminal_mode"),
            "preflight_error": f"KeyError: task {candidate.get('task_id')!r} is not in the task list",
        }
        return edits.process_one(meta["custom_id"], meta, {"ok": False}, task=None)

    resumable = (
        existing is not None
        and (existing.get("fingerprints") or {}).get("pipeline_sha256") == pipeline_sha256
        and existing.get("task_sha256") == task_sha256(task)
    )
    status = (existing or {}).get("status")
    if resumable and status in RETRY_REPAIR:
        artifact = existing
        artifact["status"] = "delabeled"
        artifact, _ = repair.run_repair_loop(artifact, task=task, call_fn=call_fn, enabled=run_repair)
    elif resumable and (status in RETRY_QUALITY or (run_quality and status == "lint_passed")):
        artifact = existing
        artifact["status"] = "lint_passed"
    else:
        meta, user_message = rewrite_request.prepare_candidate(candidate, task)
        if meta.get("preflight_error"):
            artifact = edits.process_one(meta["custom_id"], meta, {"ok": False}, task=task)
        else:
            assert user_message is not None
            response = call_fn(
                rewrite_system_prompt(),
                user_message,
                stage="rewrite",
            )
            artifact = edits.process_one(meta["custom_id"], meta, response, task=task)
            artifact["rewrite_call"] = {
                key: response.get(key)
                for key in ("schema_attempts", "discarded_schema_errors", "normalizations")
            }
        artifact, _ = repair.run_repair_loop(artifact, task=task, call_fn=call_fn, enabled=run_repair)

    if artifact.get("status") == "lint_passed":
        if run_quality:
            artifact, _ = quality.judge_artifact(artifact, call_fn=call_fn)
        else:
            artifact["quality_judge"] = {"called": False, "reason": "checking disabled by caller"}
    return artifact


def accepted_record(artifact: dict[str, Any]) -> dict[str, Any]:
    """Training-visible view of an admitted trajectory.

    The ATIF system step is a placeholder and is dropped; the seed system
    prompt is rendered per task when the trajectory becomes training data.
    """
    from evoin.tailor.trajectory import final_answer_from_atif

    trajectory = artifact.get("rewritten_trajectory") or {}
    return {
        "custom_id": artifact.get("custom_id"),
        "task_id": artifact.get("task_id"),
        "sample_idx": artifact.get("sample_idx"),
        "question": artifact.get("question"),
        "final_answer": final_answer_from_atif(trajectory),
        "terminal_mode": artifact.get("terminal_mode_after") or artifact.get("terminal_mode_source"),
        "seed_fingerprints": (artifact.get("seed_manifest") or {}).get("fingerprints"),
        "trajectory": {"steps": [step for step in trajectory.get("steps") or [] if step.get("source") != "system"]},
    }


def build_report(artifacts: list[dict[str, Any]], selection: Counter, fingerprints: dict[str, str]) -> dict[str, Any]:
    statuses = Counter(str(a.get("status")) for a in artifacts)
    blockers = Counter(
        str(item.get("type"))
        for a in artifacts
        for item in ((a.get("evidence_lint") or {}).get("blockers") or [])
    )
    return {
        "schema_version": config.SCHEMA_VERSION,
        "selection": dict(selection),
        "n_processed": len(artifacts),
        "n_accepted": statuses.get(ACCEPTED_STATUS, 0),
        "status_counts": dict(statuses.most_common()),
        "lint_blocker_counts": dict(blockers.most_common()),
        "repair_calls": sum(1 for a in artifacts if (a.get("repair") or {}).get("called")),
        "quality_calls": sum(1 for a in artifacts if (a.get("quality_judge") or {}).get("called")),
        "fingerprints": fingerprints,
    }


def run(
    *,
    rollouts: list[Path],
    tasks: list[Path],
    out_dir: Path,
    models: TailorModels,
    seed_harness: Path | None = None,
    concurrency: int = 4,
    limit: int = 0,
    min_score: float = 1.0,
    max_per_task: int = 0,
    run_repair: bool = True,
    run_quality: bool = True,
) -> dict[str, Any]:
    """Rewrite, replay, check, repair, and admit the solved trajectories in ``rollouts``.

    ``rollouts`` are ``rollout.jsonl`` files collected with the evolved harness,
    ``tasks`` the task JSONL files they were run on.  ``seed_harness`` defaults
    to ``harness/seed``.
    """
    from evoin.tailor.seed import seed_fingerprints
    from evoin.tailor.stages import StageCaller

    if seed_harness is not None:
        config.SEED_HARNESS_DIR = Path(seed_harness).resolve()
    fingerprints = config.pipeline_fingerprint(seed_fingerprints())
    pipeline_sha256 = fingerprints["pipeline_sha256"]
    task_index = load_tasks(tasks)
    rows = (row for path in rollouts for row in _read_jsonl(path))
    candidates, selection = select_candidates(rows, min_score=min_score, max_per_task=max_per_task)
    if limit > 0:
        candidates = candidates[:limit]
    selection["processed"] = len(candidates)
    call_fn = StageCaller(models)
    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tailor] {len(candidates)} candidate(s); selection={dict(selection)}", flush=True)

    retryable = RETRY_FROM_SCRATCH | RETRY_REPAIR | RETRY_QUALITY | ({"lint_passed"} if run_quality else set())

    def work(candidate: dict[str, Any]) -> dict[str, Any]:
        path = artifact_dir / artifact_name(candidate["custom_id"])
        existing = _read_existing(path)
        if (
            existing is not None
            and (existing.get("fingerprints") or {}).get("pipeline_sha256") == pipeline_sha256
            and existing.get("status") not in retryable
        ):
            return existing
        try:
            artifact = process_candidate(
                candidate,
                task_index.get(candidate["task_id"]),
                call_fn=call_fn,
                existing=existing,
                pipeline_sha256=pipeline_sha256,
                run_repair=run_repair,
                run_quality=run_quality,
            )
        except Exception as error:  # noqa: BLE001 - one bad candidate must not stop the run
            artifact = {
                "custom_id": candidate["custom_id"],
                "task_id": candidate.get("task_id"),
                "sample_idx": candidate.get("sample_idx"),
                "status": "pipeline_error",
                "error": f"{type(error).__name__}: {error}",
            }
        _atomic_write_json(path, artifact)
        return artifact

    artifacts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, candidate) for candidate in candidates]
        for done, future in enumerate(as_completed(futures), 1):
            artifact = future.result()
            artifacts.append(artifact)
            print(f"[tailor] {done}/{len(futures)} {artifact.get('custom_id')} -> {artifact.get('status')}", flush=True)

    artifacts.sort(key=lambda a: str(a.get("custom_id")))
    accepted = [accepted_record(a) for a in artifacts if a.get("status") == ACCEPTED_STATUS]
    tmp = out_dir / "accepted.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        for record in accepted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, out_dir / "accepted.jsonl")
    report = build_report(artifacts, selection, fingerprints)
    _atomic_write_json(out_dir / "report.json", report)
    return report
