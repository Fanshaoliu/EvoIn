"""Reading and materializing harness directories.

A harness is a directory that the agent runtime can load directly::

    <harness>/
      base_harness.yaml    # Base Harness config: system_prompt -> ./systemprompt.md, tool list, budget
      systemprompt.md      # the decision procedure written as instructions
      tools/*.tool.yaml    # tool schemas and descriptions
      tool_impl/*.py       # harness-local tool implementations
      middleware/*.py      # harness-local middleware

Candidates may change any of these files, but writes stay inside the candidate
directory; the runtime source and paths outside the harness are never touched.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

AGENT_YAML = "base_harness.yaml"
SYSTEM_PROMPT_FILE = "systemprompt.md"
META_FILE = "_evolve_meta.json"


@dataclass
class HarnessView:
    """Read-only view of a harness on disk."""

    harness_dir: Path
    system_prompt: str
    enabled_tools: list[str]
    available_tools: list[str]
    mandatory_tools: list[str]
    max_iterations: int
    base_harness_yaml: str
    tool_yaml_files: dict[str, str]
    tool_impl_files: dict[str, str]
    middleware_files: dict[str, str]


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _tool_names(cfg: dict) -> list[str]:
    names: list[str] = []
    for entry in cfg.get("tools") or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
    return names


def load_harness(harness_dir: str | Path) -> HarnessView:
    harness_dir = Path(harness_dir)
    cfg = _read_yaml(harness_dir / AGENT_YAML)
    agent_yaml_text = (harness_dir / AGENT_YAML).read_text(encoding="utf-8")
    sp_text = (harness_dir / SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
    names = _tool_names(cfg)
    stop = [str(t) for t in (cfg.get("stop_tools") or [])]
    return HarnessView(
        harness_dir=harness_dir,
        system_prompt=sp_text,
        enabled_tools=names,
        available_tools=names,
        mandatory_tools=stop,
        max_iterations=int(cfg.get("max_iterations") or 0),
        base_harness_yaml=agent_yaml_text,
        tool_yaml_files=_read_relative_texts(harness_dir, "tools", "*.tool.yaml"),
        tool_impl_files=_read_relative_texts(harness_dir, "tool_impl", "*.py"),
        middleware_files=_read_relative_texts(harness_dir, "middleware", "*.py"),
    )


def _read_relative_texts(root: Path, subdir: str, pattern: str) -> dict[str, str]:
    base = root / subdir
    if not base.exists():
        return {}
    out: dict[str, str] = {}
    for path in sorted(base.rglob(pattern)):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return out


_ALLOWED_EXACT = {SYSTEM_PROMPT_FILE, AGENT_YAML}
_ALLOWED_PREFIXES = ("tools/", "tool_impl/", "middleware/")


def _safe_change_path(path: str) -> Path:
    rel = Path(str(path).strip())
    if rel.is_absolute() or ".." in rel.parts or not str(rel):
        raise ValueError(f"unsafe file_changes path: {path!r}")
    s = rel.as_posix()
    if s in _ALLOWED_EXACT:
        return rel
    if s.startswith("tools/") and s.endswith(".tool.yaml"):
        return rel
    if s.startswith(_ALLOWED_PREFIXES[1:]) and s.endswith(".py"):
        return rel
    raise ValueError(
        "file_changes path must be systemprompt.md, base_harness.yaml, "
        "tools/*.tool.yaml, tool_impl/*.py, or middleware/*.py; got "
        f"{path!r}"
    )


def apply_file_changes(harness_dir: str | Path, file_changes: list[dict] | None) -> list[dict]:
    """Apply strict full-file add/replace changes inside a materialized harness."""
    root = Path(harness_dir)
    applied: list[dict] = []
    for raw in file_changes or []:
        if not isinstance(raw, dict):
            raise ValueError(f"file_changes item must be object, got {type(raw).__name__}")
        action = str(raw.get("action") or "replace").strip().lower()
        if action not in {"add", "replace"}:
            raise ValueError(f"unsupported file_changes action: {action!r}")
        rel = _safe_change_path(str(raw.get("path") or ""))
        if "content" not in raw:
            raise ValueError(f"file_changes item missing content for {rel}")
        target = root / rel
        if action == "replace" and not target.exists():
            raise FileNotFoundError(f"cannot replace missing harness file: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(raw.get("content") or ""), encoding="utf-8")
        applied.append({"path": rel.as_posix(), "action": action, "bytes": target.stat().st_size})
    return applied


def materialize_harness(
    *,
    base_dir: str | Path,
    dst_dir: str | Path,
    system_prompt: str,
    tools: list[str] | None = None,
    max_iterations: int | None = None,
    file_changes: list[dict] | None = None,
) -> Path:
    """Copy ``base_dir`` to ``dst_dir`` and apply a candidate's changes.

    The system prompt, enabled tool subset, and iteration budget are applied
    first; full-file ``file_changes`` are applied last.
    """
    base_dir = Path(base_dir)
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(base_dir, dst_dir, ignore=shutil.ignore_patterns("__pycache__"))
    stale_meta = dst_dir / META_FILE
    if stale_meta.exists():
        stale_meta.unlink()

    (dst_dir / SYSTEM_PROMPT_FILE).write_text(system_prompt, encoding="utf-8")

    cfg_path = dst_dir / AGENT_YAML
    cfg = _read_yaml(cfg_path)
    if tools is not None:
        keep = set(tools) | {str(t) for t in (cfg.get("stop_tools") or [])}
        cfg["tools"] = [
            entry
            for entry in (cfg.get("tools") or [])
            if isinstance(entry, dict) and entry.get("name") in keep
        ]
    if max_iterations is not None:
        cfg["max_iterations"] = int(max_iterations)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    apply_file_changes(dst_dir, file_changes)
    return dst_dir
