"""Tailor constants: prompts, the seed harness, and the replay sandbox."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
PROMPT_DIR = PKG_DIR / "prompts"
REWRITE_PROMPT_MD = PROMPT_DIR / "rewrite.md"
REPAIR_PROMPT_MD = PROMPT_DIR / "repair.md"
QUALITY_PROMPT_MD = PROMPT_DIR / "quality.md"

SCHEMA_VERSION = "evoin.tailor.v1"

# The only target environment of rewriting; ``pipeline.run`` may point it elsewhere.
SEED_HARNESS_DIR = REPO_ROOT / "harness" / "seed"

# A trajectory must close with a native ``complete_task`` call.
COMPLETE_TASK_TERMINAL = "native_complete_task"


def allowed_terminal_modes() -> frozenset[str]:
    return frozenset({COMPLETE_TASK_TERMINAL})


# Replayed bash commands run in the task workspace under these limits.
BASH_REPLAY_TIMEOUT = 30
BASH_REPLAY_TIMEOUT_MAX = 180
BASH_REPLAY_MEM_BYTES = 2 * 1024**3
BASH_REPLAY_CPU_SECONDS = 30
BASH_REPLAY_FSIZE_BYTES = 64 * 1024**2
# ulimit -u is per-uid, not per-command; too low a value fails every replay on
# a busy host, so it only guards against fork bombs.
BASH_REPLAY_NPROC = 4096
BASH_REPLAY_OUTPUT_BYTES = 8 * 1024**2
# Commands that reach any of these capabilities are unsafe and never executed.
BASH_REPLAY_DENY = (
    "socket", "requests", "urllib", "http.client", "httplib", "ftplib",
    "subprocess", "os.system", "os.popen", "multiprocessing", "ctypes",
    "pty.spawn", "os.fork", "os.exec",
)

# Upper bound on one rendered observation during replay.
RUNTIME_OBSERVATION_LIMIT = 12000

# A stage response that violates its JSON contract is regenerated, up to this
# many calls per stage.
SCHEMA_ATTEMPTS = 3


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def bash_replay_policy() -> dict[str, Any]:
    return {
        "timeout": BASH_REPLAY_TIMEOUT,
        "timeout_max": BASH_REPLAY_TIMEOUT_MAX,
        "mem_bytes": BASH_REPLAY_MEM_BYTES,
        "cpu_seconds": BASH_REPLAY_CPU_SECONDS,
        "fsize_bytes": BASH_REPLAY_FSIZE_BYTES,
        "nproc": BASH_REPLAY_NPROC,
        "output_bytes": BASH_REPLAY_OUTPUT_BYTES,
        "deny": list(BASH_REPLAY_DENY),
    }


_SEMANTIC_MODULES = (
    "config.py", "prompts.py", "seed.py", "trajectory.py", "replay.py",
    "rewrite_request.py", "edits.py", "lint.py", "repair.py", "quality.py", "stages.py",
)


def implementation_hash() -> str:
    return canonical_hash({
        name: sha256_file(PKG_DIR / name) if (PKG_DIR / name).is_file() else "MISSING"
        for name in _SEMANTIC_MODULES
    })


def pipeline_fingerprint(seed_fingerprints: dict[str, str]) -> dict[str, str]:
    """Identity of everything that changes an artifact; used to validate resumes."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rewrite_prompt": sha256_file(REWRITE_PROMPT_MD),
        "repair_prompt": sha256_file(REPAIR_PROMPT_MD),
        "quality_prompt": sha256_file(QUALITY_PROMPT_MD),
        "runtime_observation_limit": RUNTIME_OBSERVATION_LIMIT,
        "bash_replay": bash_replay_policy(),
        "seed": seed_fingerprints,
        "implementation": implementation_hash(),
    }
    return {
        "pipeline_sha256": canonical_hash(payload),
        "rewrite_prompt_sha256": payload["rewrite_prompt"],
        "repair_prompt_sha256": payload["repair_prompt"],
        "quality_prompt_sha256": payload["quality_prompt"],
        "implementation_sha256": payload["implementation"],
        **seed_fingerprints,
    }
