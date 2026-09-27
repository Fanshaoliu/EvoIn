"""Rubric judge: a task evaluator built from any text model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evoin.models import Evaluator, TextModel

JUDGE_PROMPT_MD = Path(__file__).resolve().parent / "judge_prompt.md"
_SCORE_RE = re.compile(r'"score"\s*:\s*([01])(?:\.0+)?\b')


def _strip_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return fenced.group(1) if fenced else text


def parse_judge_output(raw: str) -> dict[str, Any]:
    text = _strip_fences(raw or "")
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict) and "score" in parsed:
                return {
                    "score": float(parsed.get("score") or 0),
                    "rationale": str(parsed.get("rationale") or ""),
                    "requirement_status": list(parsed.get("requirement_status") or []),
                }
        except (TypeError, ValueError):
            pass
    match = _SCORE_RE.search(text)
    if match:
        return {"score": float(match.group(1)), "rationale": text[:500], "requirement_status": [],
                "judge_json_recovered": True}
    return {"score": 0.0, "rationale": f"unparseable judge output: {text[:300]}", "requirement_status": []}


def rubric_judge(model: TextModel, system_prompt: str | None = None) -> Evaluator:
    """Score an answer against the task's reference answer or rubrics with ``model``."""
    system = system_prompt or JUDGE_PROMPT_MD.read_text(encoding="utf-8")

    def evaluate(task: dict[str, Any], answer: str) -> dict[str, Any]:
        request = {
            "question": task.get("question") or "",
            "rubrics": task.get("rubrics") or [],
            "answer": task.get("answer") or "",
            "model_output": answer,
        }
        return parse_judge_output(model(system, json.dumps(request, ensure_ascii=False, indent=2)))

    return evaluate
