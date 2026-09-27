# EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning

Anonymous code release accompanying the submission. It contains the algorithmic framework of
the two agent-driven parts of EvoIn:

1. **Harness evolution** (`evoin/evolve`): procedure proposal and procedure evaluation.
2. **Rewriting with Tailor** (`evoin/tailor`): trajectories collected under the evolved harness
   are rewritten into self-contained trajectories under the original harness, replayed,
   checked, repaired at most once, and admitted by an independent checking model.

Procedure internalization (supervised fine-tuning under the original harness) uses a standard
SFT recipe described in the paper appendix and is not part of this release. The framework
ships no model client: every model-dependent step receives a callable (see
[Model roles](#model-roles)).

## Pipeline

```
seed harness H0
  procedure proposal + evaluation, per task class, for R rounds
      rollout of the current harness on the evolution subset of D_train
      -> analysis agent: failure diagnosis and clustering -> evolution query
      -> proposal agent: 3 candidates on the axes retrieval_flow / evidence_discipline / toolset_budget
      -> static validation + runtime validation on one evolution example (failure = score 0)
      -> evaluation of the current harness and the valid candidates on D_dev -> V{round}
      best harness on D_dev across rounds = H*
  procedure-guided sampling
      rollout of H* on D_train; keep full-score trajectories with a valid final answer
  rewriting (Tailor)
      preflight + seed replay -> rewrite -> apply edits + replay -> deterministic checks
      -> at most one targeted repair + replay -> independent checking -> accepted trajectories
  procedure internalization: SFT under H0 (not included)
```

| Paper component | Code |
|---|---|
| Procedure proposal; proposal system prompt (Appendix) | `evoin/evolve/proposal.py`, `evoin/evolve/prompts/proposal_system.md` |
| Analysis agent | `evoin/evolve/analysis_agent.py` |
| Evolution query template (Appendix) | `evoin/evolve/analysis.py` (`render_evolution_query`) |
| Candidate validation | `evoin/evolve/validate.py`, runtime validation in `evoin/evolve/evolve.py` |
| Procedure evaluation and selection of H* | `evoin/evolve/evolve.py` (`evolve_class`) |
| Rollout scored by the task evaluator | `evoin/evolve/rollout.py` (`run_rollout`), `evoin/runtime/judge.py` |
| Procedure-guided sampling | `run_rollout` with H* on D_train |
| Tailor | `evoin/tailor/` (`pipeline.run`) |
| Rewriting, repair, and checking prompts (Appendix) | `evoin/tailor/prompts/{rewrite,repair,quality}.md` |
| Seed harness H0 | `harness/seed/` |

The proposal system prompt and the three Tailor prompts are the texts reproduced in the paper
appendix; the evolution query follows the appendix template section by section.

## Layout

```
harness/seed/      seed harness: base_harness.yaml, systemprompt.md, tools/, tool_impl/, middleware/
evoin/models.py    model roles (abstract callables)
evoin/runtime/     reference runtime that executes a harness directory with a target model
evoin/evolve/      evolution loop, rollout, analysis, proposal, validation
evoin/tailor/      trajectory rewriting pipeline
```

## Model roles

`evoin/models.py` defines the interfaces; any model can be plugged in behind them.

| Role in the paper | Interface | Used by |
|---|---|---|
| Target agent `M_theta` | `AgentModel(messages, tools) -> {"reasoning", "text", "tool_calls"}` | `run_agent`, `run_rollout` |
| Task evaluator | `Evaluator(task, answer) -> {"score", "rationale"}` | `run_rollout` |
| Analysis agent, proposal agent | `TextModel(system, user) -> str` | `evolve_class` |
| Tailor rewriting, repair, checking | `TextModel(system, user) -> str` | `tailor.pipeline.run` |

`evoin.runtime.judge.rubric_judge(text_model)` turns a `TextModel` into an `Evaluator` that
grades an answer against the reference answer or rubrics of a task. In the paper, the target
model is Qwen3.5-35B-A3B, Claude Opus 4.7 is both the analysis and the proposal agent, and
GLM-5.2 is the rewriting, repair, and checking model.

## Data format

One task per line (JSONL):

```json
{"task_id": "...", "question": "...", "document": "...", "answer": "...", "rubrics": ["..."]}
```

`document` is hidden from the model and only reachable through the harness tools;
`context_messages` (a list of `{role, content}`) can be given instead.

## Usage

```python
from pathlib import Path

from evoin.evolve.evolve import EvolutionConfig, evolve_class
from evoin.evolve.rollout import load_tasks, run_rollout
from evoin.models import EvolutionModels, TailorModels
from evoin.tailor import pipeline as tailor

models = EvolutionModels(target=..., evaluator=..., analyst=..., proposer=...)

# Procedure proposal and evaluation (10 rounds, 3 candidates per round in the paper).
result = evolve_class(
    class_name="class_a",
    train_tasks=load_tasks(Path("data/class_a/evolution.jsonl"), 100),
    dev_tasks=load_tasks(Path("data/class_a/dev.jsonl"), 100),
    seed_dir=Path("harness/seed"),
    out_dir=Path("runs/harnesses"),
    work_dir=Path("runs/work"),
    models=models,
    config=EvolutionConfig(rounds=10),
)
h_star = Path(result["best_ever_harness"])

# Procedure-guided sampling with H*.
run_rollout(
    harness_dir=h_star, tasks=load_tasks(Path("data/class_a/train.jsonl"), 0),
    output_dir=Path("runs/collect/class_a"), model=models.target, evaluator=models.evaluator,
    max_obs_chars=0,
)

# Rewriting with Tailor.
report = tailor.run(
    rollouts=[Path("runs/collect/class_a/rollout.jsonl")],
    tasks=[Path("data/class_a/train.jsonl")],
    out_dir=Path("runs/tailor/class_a"),
    models=TailorModels(rewrite=..., repair=..., check=...),
)
```

### Evolution outputs

Each round writes `out_dir/<class>/V{round}` with the winning harness and `_evolve_meta.json`
(candidates, their causal accounts, validation results, attribution against the parent, and
`best_ever`, the best harness on D_dev so far). Round artifacts (rollouts, analysis, evolution
query, candidates) are kept under `work_dir/<class>/round_XX`. A rerun resumes from the latest
valid version.

### Tailor outputs

Only trajectories with a full score, status `ok`, and a final `complete_task` call are
rewritten. For each one, Tailor:

1. converts the rollout to the Agent Trajectory Interchange Format (ATIF), removes unsafe calls
   that provably do not affect the answer, and replays every call under the seed harness, so
   all observations are regenerated by the seed tools (`rewrite_request.py`, `replay.py`);
   replayed bash commands run under memory, CPU-time, file-size, and process-count limits with a
   timeout;
2. asks the rewriting model for a sparse edit list (`rephrase`, `tool_remap`, `reground_flag`,
   `drop`, or infeasibility), merges it, and replays the edited trajectory (`edits.py`);
3. runs the deterministic checks (`lint.py`), applies safe deterministic fixes, and makes at
   most one targeted repair call followed by another full replay and check (`repair.py`);
4. sends survivors to the independent checking model (`quality.py`).

The reference answer is given only to the rewriting call. Accepted trajectories are written to
`accepted.jsonl`; `artifacts/` keeps one audit record per trajectory with every intermediate
trajectory, check result, and model output; `report.json` counts final statuses and check
findings. A rerun skips finished artifacts.

## Agent runtime

The experiments executed harnesses on an internal agent framework that loads the same harness
format. `evoin/runtime` re-implements what EvoIn depends on: loading a harness directory (tool
bindings, middleware, Jinja system prompt, iteration budget), the tool-calling loop, the seed
tools, observation formatting, and head/tail truncation of long tool outputs.

## Not included

- Supervised fine-tuning code and configurations.
- Model clients and serving configurations.
- The in-domain training data and the Table-Longer benchmark, which cannot be released, and the
  evolved harnesses and run logs derived from them.
- Evaluation scripts for the out-of-domain benchmarks.
