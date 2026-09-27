"""System prompts of the three Tailor calls, as reproduced in the paper appendix."""

from __future__ import annotations

from functools import lru_cache

from evoin.tailor import config


@lru_cache(maxsize=1)
def rewrite_system_prompt() -> str:
    return config.REWRITE_PROMPT_MD.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def repair_system_prompt() -> str:
    return config.REPAIR_PROMPT_MD.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def quality_system_prompt() -> str:
    return config.QUALITY_PROMPT_MD.read_text(encoding="utf-8").strip()
