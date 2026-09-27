"""Harness evolution loop (one harness per task class).

Evolution starts from the seed harness H0.  Every round:

  1. roll out the current harness on the evolution subset of D_train;
  2. the analysis agent diagnoses the successful and failed trajectories and the
     result is rendered into a structured evolution query;
  3. the proposal agent proposes one candidate harness per strategy axis;
  4. each candidate passes static validation and runtime validation (an end-to-end
     run on one evolution example) or receives a score of zero;
  5. the current harness and every valid candidate are evaluated on the held-out
     D_dev; the round winner becomes ``V{round}`` and the next parent.

The highest-scoring harness on D_dev across all rounds (``best_ever``) is the
evolved harness H*.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from evoin.evolve import analysis as A
from evoin.evolve import harness as H
from evoin.evolve.analysis_agent import analyze_rollout
from evoin.evolve.proposal import N_CANDIDATES, propose_candidates
from evoin.evolve.rollout import read_rollout, run_rollout
from evoin.evolve.validate import validate_harness_static
from evoin.models import EvolutionModels

_REQUIRED_HARNESS_PATHS = ("base_harness.yaml", "systemprompt.md", "tools", "tool_impl", "middleware")


@dataclass
class EvolutionConfig:
    rounds: int = 10
    correct_threshold: float = 1.0
    runtime_validation_tasks: int = 1
    analysis_batch: int = 10
    concurrency: int = 32
    train_samples: int = 1
    dev_samples: int = 1
    min_improvement: float = 0.0
    restart: bool = False


def _rel(path: Path, root: Path) -> str:
    """Path recorded in run artifacts: relative to ``root``, or just the directory name."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return Path(path).name


def _rollout(
    harness_dir: Path, tasks: list[dict], out_dir: Path, models: EvolutionModels, *,
    concurrency: int, samples: int,
) -> tuple[dict, list[dict]]:
    """Run one rollout; a completed ``out_dir`` from an earlier run is reused."""
    if not (out_dir / "_SUCCESS").exists():
        run_rollout(
            harness_dir=harness_dir, tasks=tasks, output_dir=out_dir,
            model=models.target, evaluator=models.evaluator,
            concurrency=concurrency, samples_per_task=samples,
        )
    return read_rollout(out_dir)


def _is_valid_version_dir(path: Path, expected_class: str) -> bool:
    if not path.is_dir() or not path.name.startswith("V") or not path.name[1:].isdigit():
        return False
    meta_path = path / H.META_FILE
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(meta, dict) or meta.get("round_valid", True) is False:
        return False
    if meta.get("round") != int(path.name[1:]) or meta.get("class") != expected_class:
        return False
    return all((path / relative).exists() for relative in _REQUIRED_HARNESS_PATHS)


def _existing_versions(class_out: Path) -> list[int]:
    if not class_out.exists():
        return []
    return sorted(
        int(path.name[1:])
        for path in class_out.iterdir()
        if _is_valid_version_dir(path, class_out.name)
    )


def _entry_pass_rates(entry: dict) -> dict[str, float]:
    per_task = entry.get("dev_per_task") or {}
    if per_task:
        return {
            str(task_id): float((value or {}).get("pass_rate", 0.0) if isinstance(value, dict) else (value or 0.0))
            for task_id, value in per_task.items()
        }
    return {
        str(task_id): float(float(score or 0.0) >= 1.0)
        for task_id, score in (entry.get("dev_scores") or {}).items()
    }


def _attribution(parent_entry: dict | None, winner_entry: dict) -> dict | None:
    """Per-task attribution of the winner against the parent on the held-out split.

    A task counts as fixed (regressed) when its pass rate rises (falls) by at
    least 0.5; the net effect gives the verdict.
    """
    if parent_entry is None or winner_entry.get("is_parent"):
        return None
    pp = _entry_pass_rates(parent_entry)
    wp = _entry_pass_rates(winner_entry)
    fixed, regressed = [], []
    for tid in sorted(set(pp) | set(wp)):
        a = float(pp.get(tid, 0.0) or 0.0)
        b = float(wp.get(tid, 0.0) or 0.0)
        if b - a >= 0.5:
            fixed.append(tid)
        elif a - b >= 0.5:
            regressed.append(tid)
    verdict = ("EFFECTIVE" if len(fixed) > len(regressed)
               else "HARMFUL" if len(regressed) > len(fixed) else "MIXED")
    return {
        "winner": winner_entry.get("label"), "strategy_axis": winner_entry.get("strategy_axis", ""),
        "predicted_fix": winner_entry.get("predicted_fix", ""),
        "n_fixed": len(fixed), "n_regressed": len(regressed), "verdict": verdict,
        "fixed_tasks": fixed[:10], "regressed_tasks": regressed[:10],
    }


def _format_attribution(attr: dict | None) -> str:
    if not attr:
        return ""
    return (
        f"Last round winner=`{attr.get('winner')}` (strategy axis {attr.get('strategy_axis', '?')}) "
        f"predicted fix: {str(attr.get('predicted_fix', ''))[:200]}\n"
        f"Measured vs parent: fixed {attr.get('n_fixed', 0)} task(s), regressed "
        f"{attr.get('n_regressed', 0)} task(s) -> verdict {attr.get('verdict', '?')}.\n"
        f"Guidance: EFFECTIVE -> deepen this direction; HARMFUL -> abandon / roll back this "
        f"direction and switch to another strategy axis; MIXED -> keep the effective part and fix "
        f"the change that introduced the regression."
    )


def _load_prev_attribution(current_dir: Path) -> dict | None:
    meta_path = Path(current_dir) / H.META_FILE
    if not meta_path.exists():
        return None
    try:
        return (json.loads(meta_path.read_text(encoding="utf-8")) or {}).get("attribution")
    except Exception:  # noqa: BLE001
        return None


def _select_winner(pool: list[dict], min_improvement: float) -> dict:
    """A candidate must beat the parent by ``min_improvement`` on the held-out split to win."""
    pool_sorted = sorted(
        pool, key=lambda e: (-e["dev_avg"], 0 if e.get("is_parent") else 1,
                             len(e.get("tools", [])), str(e.get("label"))))
    parent = next((e for e in pool if e.get("is_parent")), None)
    best = pool_sorted[0]
    if parent is None or best.get("is_parent"):
        return best
    if best["dev_avg"] >= parent["dev_avg"] + min_improvement:
        return best
    return parent


def _select_round_winner(pool: list[dict], min_improvement: float) -> dict:
    valid_pool = [p for p in pool if not p.get("invalid")]
    return _select_winner(valid_pool or pool, min_improvement)


def _update_best_ever(best: dict | None, pool: list[dict], min_improvement: float, round_idx: int) -> dict:
    pool = [p for p in pool if not p.get("invalid")] or pool
    ranked = sorted(
        pool, key=lambda e: (-float(e.get("dev_avg", 0.0)), 0 if e.get("is_parent") else 1, str(e.get("label")))
    )
    best_in_pool = ranked[0]
    parent = next((e for e in pool if e.get("is_parent")), None)
    if parent is not None and not best_in_pool.get("is_parent"):
        if float(best_in_pool["dev_avg"]) < float(parent["dev_avg"]) + min_improvement:
            best_in_pool = parent
    if best is None or float(best_in_pool["dev_avg"]) >= float(best["dev_avg"]) + min_improvement:
        return {
            "round": round_idx,
            "label": best_in_pool["label"],
            "harness_dir": best_in_pool["harness_dir"],
            "dev_avg": float(best_in_pool["dev_avg"]),
            "dev_pass": float(best_in_pool.get("dev_pass", 0.0)),
        }
    return best


def _best_summary(best: dict | None) -> str:
    if not best:
        return "NONE"
    return (
        f"best_round={best.get('round')} label={best.get('label')} "
        f"dev_avg={best.get('dev_avg'):.3f} harness={Path(best.get('harness_dir') or '').name}"
    )


def _runtime_validation(
    cand_dir: Path, tasks: list[dict], out_dir: Path, models: EvolutionModels, *, work_root: Path,
) -> dict:
    """End-to-end run on evolution examples; it must finish without errors, correctness is not required."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    summary, _ = _rollout(cand_dir, tasks, out_dir, models, concurrency=1, samples=1)
    ok = bool(summary.get("n")) and int(summary.get("n_errors") or 0) == 0
    return {"ok": ok, "summary": summary, "output_dir": _rel(out_dir, work_root)}


def _candidate_entry(spec, cand_dir: Path, validation: dict, tools: list[str]) -> dict:
    return {
        "label": spec.name, "harness_dir": cand_dir, "is_parent": False,
        "rationale": spec.rationale, "tools": tools,
        "strategy_axis": spec.strategy_axis, "failure_pattern": spec.failure_pattern,
        "root_cause": spec.root_cause, "predicted_fix": spec.predicted_fix,
        "why_this_change": spec.why_this_change, "component_choice": spec.component_choice,
        "validation": validation, "file_changes": validation.get("file_changes", []),
    }


def evolve_class(
    *,
    class_name: str,
    train_tasks: list[dict],
    dev_tasks: list[dict],
    seed_dir: Path,
    out_dir: Path,
    work_dir: Path,
    models: EvolutionModels,
    config: EvolutionConfig | None = None,
) -> dict:
    """Evolve the harness of one task class.

    ``train_tasks`` is the evolution subset of D_train (analysis and runtime
    validation) and ``dev_tasks`` is D_dev (candidate evaluation and selection).
    Harness versions are written to ``out_dir/V{round}``; a rerun resumes from the
    latest valid version unless ``config.restart`` is set.
    """
    cfg = config or EvolutionConfig()
    seed_dir = Path(seed_dir).resolve()
    class_out = Path(out_dir) / class_name
    class_work = Path(work_dir) / class_name
    class_out.mkdir(parents=True, exist_ok=True)
    class_work.mkdir(parents=True, exist_ok=True)
    seed_view = H.load_harness(seed_dir)

    existing = _existing_versions(class_out)
    if existing and not cfg.restart:
        start_round = existing[-1] + 1
        current_dir = class_out / f"V{existing[-1]}"
    else:
        start_round = 1
        current_dir = seed_dir

    log_path = class_out / "evolve_log.jsonl"
    best_ever: dict | None = None
    if log_path.exists() and existing and not cfg.restart:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if int(row.get("round", -1)) in existing and row.get("best_ever"):
                    best_ever = {**row["best_ever"], "harness_dir": class_out / row["best_ever"]["harness_dir"]}

    print(f"\n==== class={class_name} | start_round={start_round} | current={current_dir.name} ====", flush=True)

    for r in range(start_round, start_round + cfg.rounds):
        round_work = class_work / f"round_{r:02d}"
        round_work.mkdir(parents=True, exist_ok=True)
        view = H.load_harness(current_dir)
        prev_attr = _format_attribution(_load_prev_attribution(current_dir))

        print(f"\n-- [class={class_name}] round {r}: rollout on the evolution subset --", flush=True)
        train_sum, train_rows = _rollout(
            current_dir, train_tasks, round_work / "train_parent", models,
            concurrency=cfg.concurrency, samples=cfg.train_samples,
        )
        analysis_dir = round_work / "analysis"
        overview, _ = analyze_rollout(
            models.analyst,
            train_summary=train_sum, train_rows=train_rows, output_dir=analysis_dir,
            class_name=class_name, round_idx=r, correct_threshold=cfg.correct_threshold,
            batch_size=cfg.analysis_batch,
        )
        query = A.render_evolution_query(
            overview=overview,
            view=view,
            available_tools=seed_view.available_tools,
            mandatory_tools=seed_view.mandatory_tools,
            previous_attribution_text=prev_attr,
            best_ever_summary=_best_summary(best_ever),
            n_candidates=N_CANDIDATES,
        )
        (round_work / "evolution_query.md").write_text(query, encoding="utf-8")

        print(f"-- [class={class_name}] round {r}: propose {N_CANDIDATES} candidates --", flush=True)
        specs = propose_candidates(
            models.proposer,
            evolution_query=query,
            system_prompt=view.system_prompt,
            available_tools=seed_view.available_tools,
            current_tools=view.enabled_tools,
            mandatory_tools=seed_view.mandatory_tools,
        )

        p_sum, _ = _rollout(
            current_dir, dev_tasks, round_work / "dev_parent", models,
            concurrency=cfg.concurrency, samples=cfg.dev_samples,
        )
        pool: list[dict] = [{
            "label": "parent", "harness_dir": current_dir, "is_parent": True,
            "dev_avg": p_sum.get("avg_score", 0.0), "dev_pass": p_sum.get("pass_rate", 0.0),
            "dev_per_task": p_sum.get("per_task", {}), "dev_scores": p_sum.get("scores", {}),
            "rationale": "(unchanged parent baseline)", "tools": view.enabled_tools,
        }]

        for i, spec in enumerate(specs):
            cand_dir = round_work / f"cand_{i}"
            dev_dir = round_work / f"dev_cand_{i}"
            if dev_dir.exists():
                shutil.rmtree(dev_dir)
            try:
                H.materialize_harness(
                    base_dir=current_dir, dst_dir=cand_dir,
                    system_prompt=spec.system_prompt, tools=spec.tools,
                    max_iterations=spec.max_iterations, file_changes=spec.file_changes,
                )
                validation = validate_harness_static(cand_dir, write_json=False)
            except Exception as exc:  # noqa: BLE001
                cand_dir.mkdir(parents=True, exist_ok=True)
                validation = {"ok": False, "harness": cand_dir.name, "checks": {},
                              "errors": [f"materialize: {type(exc).__name__}: {exc}"]}
            validation["candidate"] = spec.name
            validation["file_changes"] = [
                {"path": str(ch.get("path", "")), "action": str(ch.get("action", "replace"))}
                for ch in (spec.file_changes or []) if isinstance(ch, dict)
            ]
            if validation["ok"] and cfg.runtime_validation_tasks > 0:
                runtime = _runtime_validation(
                    cand_dir, train_tasks[:cfg.runtime_validation_tasks],
                    round_work / f"runtime_validation_cand_{i}", models, work_root=class_work,
                )
                validation["checks"]["runtime"] = runtime
                validation["ok"] = runtime["ok"]
            (cand_dir / "validation.json").write_text(
                json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

            if not validation["ok"]:
                print(f"-- [class={class_name}] round {r}: cand_{i} invalid; "
                      f"errors={validation.get('errors')}", flush=True)
                pool.append({**_candidate_entry(spec, cand_dir, validation, spec.tools),
                             "invalid": True, "dev_avg": 0.0, "dev_pass": 0.0, "dev_per_task": {}})
                continue
            cand_view = H.load_harness(cand_dir)
            print(f"-- [class={class_name}] round {r}: evaluate cand_{i} on D_dev "
                  f"(axis={spec.strategy_axis}, tools={cand_view.enabled_tools}) --", flush=True)
            c_sum, _ = _rollout(
                cand_dir, dev_tasks, dev_dir, models,
                concurrency=cfg.concurrency, samples=cfg.dev_samples,
            )
            pool.append({**_candidate_entry(spec, cand_dir, validation, cand_view.enabled_tools),
                         "dev_avg": c_sum.get("avg_score", 0.0), "dev_pass": c_sum.get("pass_rate", 0.0),
                         "dev_per_task": c_sum.get("per_task", {}), "dev_scores": c_sum.get("scores", {})})

        winner = _select_round_winner(pool, cfg.min_improvement)
        best_ever = _update_best_ever(best_ever, pool, cfg.min_improvement, r)
        parent_entry = next((e for e in pool if e.get("is_parent")), None)
        attribution = _attribution(parent_entry, winner)
        print(f"== [class={class_name}] round {r} winner={winner['label']} "
              f"dev_avg={winner['dev_avg']:.3f} best_ever={best_ever['dev_avg']:.3f} ==", flush=True)

        v_dir = class_out / f"V{r}"
        if v_dir.exists():
            shutil.rmtree(v_dir)
        shutil.copytree(winner["harness_dir"], v_dir, ignore=shutil.ignore_patterns("__pycache__"))
        if best_ever and best_ever.get("round") == r and best_ever.get("label") == winner["label"]:
            best_ever["harness_dir"] = v_dir
        best_record = {**best_ever, "harness_dir": _rel(best_ever["harness_dir"], class_out)} if best_ever else None
        meta = {
            "schema_version": "evoin.evolve.full_harness.v1",
            "class": class_name, "round": r, "parent": _rel(current_dir, class_out),
            "winner": winner["label"], "winner_dev_avg_score": winner["dev_avg"],
            "winner_dev_pass_rate": winner["dev_pass"],
            "best_ever": best_record,
            "train_analysis": {
                "overview": _rel(analysis_dir / "overview.json", class_work),
                "detail": _rel(analysis_dir / "detail.jsonl", class_work),
                "evolution_query": _rel(round_work / "evolution_query.md", class_work),
                "n_tasks": overview.get("n_tasks"),
                "pass_rate": overview.get("pass_rate"),
                "failure_type_counts": overview.get("failure_type_counts"),
            },
            "attribution": attribution,
            "candidates": [
                {"label": e["label"], "dev_avg_score": e["dev_avg"],
                 "dev_pass_rate": e["dev_pass"], "tools": e["tools"],
                 "strategy_axis": e.get("strategy_axis", ""),
                 "failure_pattern": e.get("failure_pattern", ""),
                 "root_cause": str(e.get("root_cause", ""))[:300],
                 "predicted_fix": str(e.get("predicted_fix", ""))[:300],
                 "why_this_change": str(e.get("why_this_change", ""))[:300],
                 "component_choice": str(e.get("component_choice", ""))[:300],
                 "invalid": bool(e.get("invalid", False)),
                 "validation": e.get("validation", {}),
                 "file_changes": e.get("file_changes", []),
                 "rationale": str(e.get("rationale", ""))[:500]}
                for e in pool
            ],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (v_dir / H.META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        current_dir = v_dir

    return {
        "class": class_name,
        "final_harness": str(current_dir),
        "best_ever_harness": str(best_ever["harness_dir"]) if best_ever else str(current_dir),
        "best_ever_score": best_ever.get("dev_avg") if best_ever else None,
        "final_version": current_dir.name,
    }
