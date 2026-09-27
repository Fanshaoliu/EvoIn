"""Proposal agent: turn a structured evolution query into candidate harness changes.

The proposal model receives the system prompt in ``prompts/proposal_system.md``
(reproduced in the paper appendix) and, as its user message, the evolution query
rendered by ``analysis.render_evolution_query``.  It returns a strict JSON array
with one candidate per strategy axis.  Each candidate carries a causal account
(failure_pattern / root_cause / predicted_fix / why_this_change /
component_choice) that the next round uses for change attribution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evoin.models import TextModel

PROPOSAL_SYSTEM_MD = Path(__file__).resolve().parent / "prompts" / "proposal_system.md"

# One candidate per axis, in the order assigned by the proposal system prompt.
STRATEGY_AXES = ("retrieval_flow", "evidence_discipline", "toolset_budget")
N_CANDIDATES = len(STRATEGY_AXES)

# Diagnosis vocabulary shared by the analysis agent.
FAILURE_TAXONOMY = [
    "no_scan_first: jumped to grep/answer without scan()-ing first, missing document structure",
    "evidence_miss: grep/retrieval missed the key evidence lines (pattern too narrow / bad "
    "escaping / search not broadened)",
    "no_verify_lines: answered from a scan() summary or guess without read_lines() verification",
    "no_citation: final answer lacks line-number / verbatim support (evidence not traceable)",
    "incomplete: multi-part question only partially answered (insufficient coverage)",
    "repeat_degenerate: repeated the same call / spun in place without making progress",
    "format_violation: emitted tool commands / prose instead of the answer, or wrong format",
    "truncated: hit max_iterations or was cut off before reaching an answer",
    "wrong_reasoning: evidence was sufficient but synthesis / reasoning was wrong",
]


@dataclass
class CandidateSpec:
    name: str
    system_prompt: str
    tools: list[str]
    max_iterations: int | None = None
    rationale: str = ""
    failure_pattern: str = ""
    root_cause: str = ""
    predicted_fix: str = ""
    why_this_change: str = ""
    strategy_axis: str = ""
    component_choice: str = ""
    file_changes: list[dict] = field(default_factory=list)


def proposal_system_prompt() -> str:
    return PROPOSAL_SYSTEM_MD.read_text(encoding="utf-8").strip()


def extract_json_array(text: str) -> list:
    """Strictly parse a JSON array; malformed output fails and the caller regenerates."""
    try:
        obj = json.loads((text or "").strip())
    except json.JSONDecodeError:
        return []
    return obj if isinstance(obj, list) else []


def extract_json_object(text: str) -> dict:
    try:
        obj = json.loads((text or "").strip())
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def coerce_candidate(
    raw: dict,
    *,
    idx: int,
    available_tools: list[str],
    mandatory_tools: list[str],
    fallback_prompt: str,
    fallback_tools: list[str],
    strategy_axis: str = "",
) -> CandidateSpec:
    sp = str(raw.get("system_prompt") or "").strip() or fallback_prompt
    tools_in = raw.get("tools")
    if isinstance(tools_in, list) and tools_in:
        avail = set(available_tools)
        tools = [str(t) for t in tools_in if str(t) in avail]
    else:
        tools = list(fallback_tools)
    for m in mandatory_tools:
        if m not in tools:
            tools.append(m)
    max_iter = raw.get("max_iterations")
    try:
        max_iter = int(max_iter) if max_iter is not None else None
    except (TypeError, ValueError):
        max_iter = None
    return CandidateSpec(
        name=str(raw.get("name") or f"cand_{idx}"),
        system_prompt=sp,
        tools=tools,
        max_iterations=max_iter,
        rationale=str(raw.get("rationale") or ""),
        failure_pattern=str(raw.get("failure_pattern") or ""),
        root_cause=str(raw.get("root_cause") or ""),
        predicted_fix=str(raw.get("predicted_fix") or ""),
        why_this_change=str(raw.get("why_this_change") or ""),
        strategy_axis=str(raw.get("strategy_axis") or strategy_axis),
        component_choice=str(raw.get("component_choice") or ""),
        file_changes=raw.get("file_changes") if isinstance(raw.get("file_changes"), list) else [],
    )


def propose_candidates(
    proposer: TextModel,
    *,
    evolution_query: str,
    system_prompt: str,
    available_tools: list[str],
    current_tools: list[str],
    mandatory_tools: list[str],
    attempts: int = 4,
) -> list[CandidateSpec]:
    """Ask the proposal agent for one candidate per strategy axis.

    A response that violates the JSON array contract is regenerated, up to
    ``attempts`` calls in total.
    """
    system = proposal_system_prompt()
    last_head = ""
    for attempt in range(1, attempts + 1):
        raw_text = proposer(system, evolution_query)
        raw_list = [raw for raw in extract_json_array(raw_text)[:N_CANDIDATES] if isinstance(raw, dict)]
        candidates = [
            coerce_candidate(
                raw,
                idx=i,
                available_tools=available_tools,
                mandatory_tools=mandatory_tools,
                fallback_prompt=system_prompt,
                fallback_tools=current_tools,
                strategy_axis=STRATEGY_AXES[i],
            )
            for i, raw in enumerate(raw_list)
        ]
        if candidates:
            if len(candidates) < N_CANDIDATES:
                print(f"[proposal] parsed only {len(candidates)}/{N_CANDIDATES} candidates", flush=True)
            return candidates
        last_head = (raw_text or "")[:500]
        print(f"[proposal] attempt {attempt}/{attempts}: output violates the JSON array contract", flush=True)
    raise RuntimeError(f"proposal failed after {attempts} attempts; last output head={last_head!r}")
