"""Model roles used by EvoIn.

The framework never talks to a model itself.  Every model-dependent step
receives one of the callables below, so any model (local or hosted, any
decoding setup) can be plugged in without touching the algorithm.

Roles in the paper:

* target agent ``M_theta``                      -> :class:`AgentModel`
* task evaluator                                -> :class:`Evaluator`
* analysis agent, proposal agent                -> :class:`TextModel`
* Tailor rewriting, repair, and checking calls  -> :class:`TextModel`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class AgentModel(Protocol):
    """One decision of the target agent.

    ``messages`` is the conversation so far::

        {"role": "system" | "user", "content": str}
        {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "input"}]}
        {"role": "tool", "tool_call_id": str, "content": str}

    ``tools`` lists the enabled harness tools as
    ``{"name", "description", "input_schema"}``.

    Returns ``{"reasoning": str, "text": str, "tool_calls": [{"id", "name", "input"}]}``;
    ``id`` is optional and ``input`` must be a JSON object (anything else is
    recorded as a malformed call).  An empty ``tool_calls`` list ends the episode
    with ``text`` as the answer.
    """

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


class TextModel(Protocol):
    """A single system + user turn that returns the model's text."""

    def __call__(self, system: str, user: str) -> str: ...


class Evaluator(Protocol):
    """Task evaluator: score a final answer for a task.

    Returns ``{"score": float in [0, 1], "rationale": str}``; a score of 1 is a
    full task score.
    """

    def __call__(self, task: dict[str, Any], answer: str) -> dict[str, Any]: ...


@dataclass
class EvolutionModels:
    target: AgentModel
    evaluator: Evaluator
    analyst: TextModel
    proposer: TextModel


@dataclass
class TailorModels:
    rewrite: TextModel
    repair: TextModel
    check: TextModel
