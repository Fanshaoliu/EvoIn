"""Deterministic evidence and hygiene checks for rewritten trajectories.

Text patterns cover English and Chinese phrasing; Chinese alternatives are
written as ``\\u`` escapes.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from evoin.tailor.replay import observation_is_truncated
from evoin.tailor.trajectory import ASSISTANT_TEXT_TERMINAL, terminal_mode_from_atif

_TEMPLATE_PATTERNS = (
    r"\bmandatory structural probe\b",
    r"\bevidence ledger\b",
    r"\boperative verb\b",
    r"\banalytical dimensions\b",
    r"\bdeliverable shape\b",
    r"\bsub-items\b",
    r"\bintent\s*:",
    r"\br\d+\s*::",
    r"\bpass\s*\d+\b",
)
_COMPLETED_FULL_READ_PATTERN = re.compile(
    r"\b(?:i|we)\s+(?:have|['’]ve)\s+(?:now\s+)?(?:read|reviewed|covered)\s+"
    r"(?:the\s+)?(?:entire|full|whole)\s+(?:document|text)\b"
    r"|\bnow\s+(?:i|we)\s+have\s+(?:the\s+)?full\s+(?:document|text)\b"
    r"|(?:\u5df2(?:\u7ecf)?|\u73b0\u5df2|\u8bfb\u5b8c|\u770b\u5b8c|\u5b8c\u6210\u4e86).{0,16}(?:\u5168\u6587|\u5b8c\u6574\u6587\u6863|\u5168\u7bc7)",
    re.IGNORECASE,
)
_UNSUPPORTED_SOURCE_ABSENCE_PATTERN = re.compile(
    r"\b(?:materials?|content|evidence)\b[^\n]{0,100}\b(?:is|are|was|were)\s+"
    r"not\s+in\s+(?:the\s+)?`?document(?:\.md)?`?(?:\s+file)?"
    r"|(?:\u6750\u6599|\u5185\u5bb9|\u8bc1\u636e)[^\n]{0,48}(?:\u4e0d\u5728|\u6ca1\u6709)[^\n]{0,24}(?:\u6587\u6863|document)",
    re.IGNORECASE,
)
_DOC_BOUNDARY_PATTERN = re.compile(r"\bL(?P<line>\d+):\s*doc(?P<doc>\d+):", re.IGNORECASE)
_KEY_LINE_PATTERN = re.compile(r"\bL(?P<line>\d+):\s*(?P<key>[A-Z][A-Z0-9]{4,}):")
_DOC_KEY_CLAIM_PATTERN = re.compile(
    r"\bdoc\s*(?P<doc>\d+)\b(?:(?!\bdoc\s*\d+\b)[^\n]){0,100}?"
    r"\b(?:starts?\s+with|has\s+(?:key\s+)?|key\s+(?:is|at))\s*"
    r"[\"'`{(]*(?P<key>[A-Z][A-Z0-9]{4,})\b",
    re.IGNORECASE,
)
_NO_MATCH_CLAIM_PATTERN = re.compile(
    r"\b(?:grep|search|query)[^\n.]{0,80}\b(?:did(?:n't| not)|does(?:n't| not)|"
    r"failed\s+to)\s+(?:find|return)[^\n.]{0,40}\b(?:match|result)"
    r"|\bno\s+matches\b|(?:\u6ca1\u6709|\u672a)[^\n\u3002]{0,24}(?:\u5339\u914d|\u7ed3\u679c)",
    re.IGNORECASE,
)
_RAW_LINE_PATTERN = re.compile(r"(?m)^L(?P<line>\d+):\s?(?P<text>[^\n]*)$")
_LINE_THEN_QUOTE_PATTERN = re.compile(
    r"\bL(?P<first>\d+)(?:\s*(?:-|–|—|to)\s*L?(?P<last>\d+))?"
    r"\s*(?:[\uff1a:]\s*|(?:states?|says|mentions?|shows?|contains?|indicates?|reads?)\s*)"
    r"[\"“](?P<quote>[^\"”\n]{4,240})[\"”]",
    re.IGNORECASE,
)
_QUOTE_THEN_LINE_PATTERN = re.compile(
    r"[\"“](?P<quote>[^\"”\n]{4,240})[\"”]\s*"
    r"(?:at|in|on|from|\u4f4d\u4e8e|\u5728)\s*L(?P<first>\d+)"
    r"(?:\s*(?:-|–|—|to)\s*L?(?P<last>\d+))?",
    re.IGNORECASE,
)
_ZH_DECIMAL_LIMIT_PATTERN = re.compile(
    r"(?:\u6240\u6709)?(?:\u56de\u7b54|\u7b54\u6848)[^\u3002\n]{0,24}?"
    r"(?:\u6700\u591a|\u81f3\u591a|\u4e0d\u8d85\u8fc7|\u4fdd\u7559)[^\u3002\n]{0,8}?"
    r"(?P<digits>[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]+)\s*\u4f4d\u5c0f\u6570"
)
_EN_DECIMAL_LIMIT_PATTERN = re.compile(
    r"(?:at\s+most|no\s+more\s+than|up\s+to|maximum)\s+"
    r"(?P<digits>\d+)\s+decimal\s+places?",
    re.IGNORECASE,
)
_DECIMAL_LITERAL_PATTERN = re.compile(r"(?<![\d.])[-+]?\d+\.(?P<fraction>\d+)(?!\d)")
_ZH_NUMERAL_DIGITS = {
    "\u96f6": 0, "\u4e00": 1, "\u4e8c": 2, "\u4e09": 3, "\u56db": 4, "\u4e94": 5,
    "\u516d": 6, "\u4e03": 7, "\u516b": 8, "\u4e5d": 9, "\u4e24": 2, "\u5341": 10,
}


def _issue(
    type_: str,
    detail: str,
    *,
    step_id: int | None = None,
    severity: str = "blocker",
) -> dict:
    out = {"type": type_, "detail": detail, "severity": severity}
    if step_id is not None:
        out["step_id"] = step_id
    return out


def _agent_texts(trajectory: dict) -> Iterator[tuple[int | None, str, str]]:
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        step_id = step.get("step_id")
        for field in ("reasoning_content", "message"):
            value = step.get(field)
            if isinstance(value, str):
                yield step_id, field, value
        for call in step.get("tool_calls") or []:
            if call.get("function_name") == "complete_task":
                result = (call.get("arguments") or {}).get("result")
                if isinstance(result, str):
                    yield step_id, "complete_task.result", result
            if call.get("function_name") == "notepad":
                content = (call.get("arguments") or {}).get("content")
                if isinstance(content, str):
                    yield step_id, "notepad.content", content


def _all_call_results(trajectory: dict) -> Iterator[tuple[dict, dict, dict]]:
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        calls = {
            str(call.get("tool_call_id")): call
            for call in (step.get("tool_calls") or [])
        }
        for result in ((step.get("observation") or {}).get("results") or []):
            yield step, calls.get(str(result.get("source_call_id")), {}), result


def _unsupported_source_absence_claims(trajectory: dict) -> list[dict]:
    findings: list[dict] = []
    has_prior_text_probe = False
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        step_id = step.get("step_id")
        for field in ("reasoning_content", "message"):
            text = step.get(field)
            if (
                isinstance(text, str)
                and not has_prior_text_probe
                and _UNSUPPORTED_SOURCE_ABSENCE_PATTERN.search(text)
            ):
                findings.append(_issue(
                    "unsupported_source_absence_claim",
                    f"{field} asserts that source material is absent before any successful "
                    "grep/read_lines observation could support that claim",
                    step_id=step_id,
                ))
        calls = {
            str(call.get("tool_call_id")): call
            for call in (step.get("tool_calls") or [])
            if isinstance(call, dict)
        }
        for result in ((step.get("observation") or {}).get("results") or []):
            call = calls.get(str(result.get("source_call_id"))) or {}
            if (
                call.get("function_name") in {"grep", "read_lines"}
                and not result.get("is_error")
                and result.get("replay_status") == "replayed"
            ):
                has_prior_text_probe = True
    return findings


def _document_key_mapping_violations(trajectory: dict) -> list[dict]:
    boundaries: list[tuple[int, int]] = []
    key_lines: list[tuple[int, str]] = []
    for _step, _call, result in _all_call_results(trajectory):
        if result.get("is_error") or result.get("replay_status") != "replayed":
            continue
        content = str(result.get("content") or "")
        boundaries.extend(
            (int(match.group("line")), int(match.group("doc")))
            for match in _DOC_BOUNDARY_PATTERN.finditer(content)
        )
        key_lines.extend(
            (int(match.group("line")), match.group("key"))
            for match in _KEY_LINE_PATTERN.finditer(content)
        )
    if not boundaries or not key_lines:
        return []
    starts = sorted(set(boundaries))
    keys_by_doc: dict[int, set[str]] = {}
    for line, key in key_lines:
        preceding = [entry for entry in starts if entry[0] <= line]
        if preceding:
            keys_by_doc.setdefault(preceding[-1][1], set()).add(key)
    findings: list[dict] = []
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        for field in ("reasoning_content", "message"):
            text = step.get(field)
            if not isinstance(text, str):
                continue
            for match in _DOC_KEY_CLAIM_PATTERN.finditer(text):
                doc = int(match.group("doc"))
                claimed_key = match.group("key")
                observed_keys = keys_by_doc.get(doc)
                if observed_keys and claimed_key not in observed_keys:
                    findings.append(_issue(
                        "contradictory_document_key_mapping",
                        f"{field} maps doc{doc} to {claimed_key}, but replayed evidence maps "
                        f"doc{doc} to {sorted(observed_keys)}",
                        step_id=step.get("step_id"),
                    ))
    return findings


def _is_empty_navigation_result(function_name: str, content: str) -> bool:
    if function_name != "grep":
        return False
    lowered = content.lower()
    return "`matches`: []" in lowered or "no matches for " in lowered


def _reasoning_search_result_contradictions(trajectory: dict) -> list[dict]:
    findings: list[dict] = []
    last_grep_had_matches: bool | None = None
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        claims_no_match = any(
            _NO_MATCH_CLAIM_PATTERN.search(str(step.get(field) or ""))
            for field in ("reasoning_content", "message")
        )
        calls = {
            str(call.get("tool_call_id")): call
            for call in (step.get("tool_calls") or [])
            if isinstance(call, dict)
        }
        current: list[bool] = []
        for result in ((step.get("observation") or {}).get("results") or []):
            call = calls.get(str(result.get("source_call_id"))) or {}
            if (
                call.get("function_name") == "grep"
                and not result.get("is_error")
                and result.get("replay_status") == "replayed"
            ):
                current.append(
                    not _is_empty_navigation_result("grep", str(result.get("content") or ""))
                )
        if claims_no_match:
            if last_grep_had_matches is True:
                findings.append(_issue(
                    "reasoning_observation_mismatch",
                    "reasoning claims the prior grep found no matches, but the preceding "
                    "replayed grep has matches",
                    step_id=step.get("step_id"),
                ))
            elif last_grep_had_matches is None and any(current):
                findings.append(_issue(
                    "reasoning_observation_mismatch",
                    "reasoning claims no grep matches even though its retained replayed grep has matches",
                    step_id=step.get("step_id"),
                ))
        if current:
            last_grep_had_matches = current[-1]
    return findings


def _normalise_line_text(text: str) -> str:
    return "".join(char.casefold() for char in text if not char.isspace())


def _line_reference_quote_mismatches(trajectory: dict) -> list[dict]:
    findings: list[dict] = []
    observed_lines: dict[int, set[str]] = {}
    seen: set[tuple[int | None, str, int, int]] = set()

    def add_observed_lines(step: dict) -> None:
        calls = {
            str(call.get("tool_call_id")): call
            for call in (step.get("tool_calls") or [])
            if isinstance(call, dict)
        }
        for result in ((step.get("observation") or {}).get("results") or []):
            call = calls.get(str(result.get("source_call_id"))) or {}
            if (
                call.get("function_name") not in {"scan", "grep", "read_lines"}
                or result.get("is_error")
                or result.get("replay_status") != "replayed"
            ):
                continue
            for raw_line in _RAW_LINE_PATTERN.finditer(str(result.get("content") or "")):
                observed_lines.setdefault(int(raw_line.group("line")), set()).add(
                    raw_line.group("text")
                )

    def check_text(step_id: int | None, field: str, text: str) -> None:
        for pattern in (_LINE_THEN_QUOTE_PATTERN, _QUOTE_THEN_LINE_PATTERN):
            for match in pattern.finditer(text):
                first = int(match.group("first"))
                last = int(match.group("last") or first)
                if last < first:
                    first, last = last, first
                quote = _normalise_line_text(match.group("quote"))
                if len(quote) < 4:
                    continue
                cited = {
                    line: values for line, values in observed_lines.items()
                    if first <= line <= last
                }
                all_matches = sorted(
                    line for line, values in observed_lines.items()
                    if any(quote in _normalise_line_text(value) for value in values)
                )
                if not all_matches or any(
                    quote in _normalise_line_text(value)
                    for values in cited.values() for value in values
                ):
                    continue
                key = (step_id, field, first, last)
                if key not in seen:
                    seen.add(key)
                    findings.append(_issue(
                        "reasoning_line_reference_mismatch",
                        f"{field} attributes quoted text to L{first}"
                        f"{'-L' + str(last) if last != first else ''}, but prior replayed "
                        f"evidence contains it at {all_matches[:8]} instead",
                        step_id=step_id,
                    ))

    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        for field in ("reasoning_content", "message"):
            text = step.get(field)
            if isinstance(text, str):
                check_text(step.get("step_id"), field, text)
        add_observed_lines(step)
    return findings


def _task_decimal_limit(question: str) -> int | None:
    for pattern in (_ZH_DECIMAL_LIMIT_PATTERN, _EN_DECIMAL_LIMIT_PATTERN):
        match = pattern.search(question)
        if not match:
            continue
        digits = match.group("digits")
        if digits.isdigit():
            return int(digits)
        if digits in _ZH_NUMERAL_DIGITS:
            return _ZH_NUMERAL_DIGITS[digits]
    return None


def _task_precision_violations(artifact: dict, trajectory: dict) -> list[dict]:
    max_digits = _task_decimal_limit(str(artifact.get("question") or ""))
    if max_digits is None:
        return []
    findings: list[dict] = []
    targets: list[tuple[Any, str, str]] = []
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        for call in step.get("tool_calls") or []:
            if call.get("function_name") != "complete_task":
                continue
            result = (call.get("arguments") or {}).get("result")
            if not isinstance(result, str):
                continue
            targets.append((step.get("step_id"), "complete_task.result", result))
    if terminal_mode_from_atif(trajectory) == ASSISTANT_TEXT_TERMINAL:
        final_agent = next(
            (
                step
                for step in reversed(trajectory.get("steps") or [])
                if step.get("source") == "agent"
            ),
            None,
        )
        if final_agent is not None and isinstance(final_agent.get("message"), str):
            targets.append((
                final_agent.get("step_id"),
                "terminal assistant message",
                final_agent["message"],
            ))
    for step_id, field, text in targets:
        for match in _DECIMAL_LITERAL_PATTERN.finditer(text):
            if len(match.group("fraction")) > max_digits:
                findings.append(_issue(
                    "decimal_precision_violation",
                    f"{field} contains {match.group(0)!r} with "
                    f"{len(match.group('fraction'))} decimal places, exceeding task limit {max_digits}",
                    step_id=step_id,
                ))
    return findings


def _derived_terms(artifact: dict) -> list[str]:
    values = (
        (artifact.get("rewrite_output") or {}).get("derived_diff", {}).get("ungrounded_terms")
        or []
    )
    terms: list[str] = []
    for value in values:
        if not isinstance(value, (str, int, float)):
            continue
        text = str(value).strip()
        if len(text) < 4:
            continue
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        if len(tokens) < 2 and not any(marker in text for marker in ("_", ":", "::", "/", "-")):
            continue
        terms.append(text)
    return terms


def lint(artifact: dict) -> dict:
    """Return explicit blockers/warnings from serialized replayed evidence."""
    trajectory = artifact.get("rewritten_trajectory") or {}
    blockers: list[dict] = []
    warnings: list[dict] = []
    if artifact.get("status") != "delabeled":
        blockers.append(_issue(
            "not_delabeled",
            f"artifact status is {artifact.get('status')!r}, not a replayed delabeled trajectory",
        ))
        return {
            "passed": False,
            "state": (
                "REROLLOUT_REQUIRED"
                if artifact.get("status") == "rerollout_required"
                else "REJECTED_BY_LINTER"
            ),
            "blockers": blockers,
            "warnings": warnings,
        }
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in _TEMPLATE_PATTERNS]
    patterns.extend(re.compile(re.escape(term), re.IGNORECASE) for term in _derived_terms(artifact))
    for step_id, field, text in _agent_texts(trajectory):
        for pattern in patterns:
            if pattern.search(text):
                blockers.append(_issue(
                    "residual_template",
                    f"{field} retains evolved/template term matching {pattern.pattern!r}",
                    step_id=step_id,
                ))
                break
        if "\\n" in text:
            blockers.append(_issue(
                "literal_newline_encoding",
                f"{field} contains literal \\\\n rather than a real line break",
                step_id=step_id,
            ))
    blockers.extend(_unsupported_source_absence_claims(trajectory))
    blockers.extend(_document_key_mapping_violations(trajectory))
    blockers.extend(_reasoning_search_result_contradictions(trajectory))
    blockers.extend(_line_reference_quote_mismatches(trajectory))
    blockers.extend(_task_precision_violations(artifact, trajectory))

    read_successes = 0
    navigation_successes = 0
    truncated_reads: list[tuple[int | None, str]] = []
    failed_signatures: dict[str, int] = {}
    for step, call, result in _all_call_results(trajectory):
        function_name = str(call.get("function_name") or "")
        step_id = step.get("step_id")
        status = str(result.get("replay_status") or "")
        content = str(result.get("content") or "")
        if status not in {"replayed", "not_required_stop_tool"}:
            blockers.append(_issue(
                "unresolved_replay",
                f"{function_name} call {result.get('source_call_id')} replay status is {status!r}",
                step_id=step_id,
            ))
        if result.get("is_error"):
            blockers.append(_issue(
                "errored_observation",
                f"{function_name} observation is_error=true and cannot support retained reasoning",
                step_id=step_id,
            ))
            signature = f"{function_name}:{call.get('arguments')}"
            failed_signatures[signature] = failed_signatures.get(signature, 0) + 1
        elif _is_empty_navigation_result(function_name, content):
            blockers.append(_issue(
                "empty_navigation_call",
                "grep returned no matches; retain it only when absence itself is a visible, "
                "load-bearing fact, otherwise drop it before SFT admission",
                step_id=step_id,
            ))
        if function_name == "read_lines" and not result.get("is_error") and status == "replayed":
            read_successes += 1
            if (
                observation_is_truncated(content)
                or (result.get("seed_replay") or {}).get("runtime_truncated")
            ):
                truncated_reads.append((step_id, str(result.get("source_call_id"))))
        if function_name in {"scan", "grep"} and not result.get("is_error") and status == "replayed":
            navigation_successes += 1
    for signature, count in failed_signatures.items():
        if count > 1:
            blockers.append(_issue(
                "repeated_failed_call",
                f"the same failed call appears {count} times: {signature[:160]}",
            ))
    if navigation_successes and not read_successes:
        blockers.append(_issue(
            "grep_or_scan_only_evidence",
            "trajectory has navigation output but no successful raw read_lines observation",
        ))
    if truncated_reads:
        for step_id, call_id in truncated_reads:
            warnings.append(_issue(
                "truncated_read_observation",
                f"read_lines call {call_id} remains truncated; it cannot support omitted facts",
                step_id=step_id,
                severity="warning",
            ))
        for step_id, field, text in _agent_texts(trajectory):
            if _COMPLETED_FULL_READ_PATTERN.search(text):
                blockers.append(_issue(
                    "truncated_full_read_claim",
                    f"{field} claims full-document reading despite truncated read_lines output",
                    step_id=step_id,
                ))
    quality_issues = (
        (artifact.get("rewrite_output") or {}).get("derived_diff", {})
        .get("trajectory_quality_issues") or []
    )
    for issue in quality_issues:
        if not isinstance(issue, dict):
            continue
        if issue.get("resolution") == "infeasible":
            blockers.append(_issue(
                "model_declared_quality_infeasible",
                str(issue.get("detail") or issue.get("type") or "model declared unresolved issue"),
                step_id=issue.get("step_id"),
            ))
        elif issue.get("resolution") == "reground":
            warnings.append(_issue(
                "model_requested_reground",
                str(issue.get("detail") or issue.get("type") or "reground requested"),
                step_id=issue.get("step_id"),
                severity="warning",
            ))
    provenance = (
        (artifact.get("rewrite_output") or {}).get("derived_diff", {})
        .get("evidence_provenance_gaps") or []
    )
    if provenance:
        warnings.append(_issue(
            "model_reported_provenance_gaps",
            f"model reported {len(provenance)} provenance gap(s); independent judge must inspect them",
            severity="warning",
        ))
    if artifact.get("sft_ready_after_edits") is False:
        blockers.append(_issue(
            "model_declared_not_ready",
            "the rewrite model marked sft_ready_after_edits=false, so unresolved work remains",
        ))

    def unique(items: list[dict]) -> list[dict]:
        seen: set[tuple] = set()
        output: list[dict] = []
        for item in items:
            key = (item.get("type"), item.get("step_id"), item.get("detail"))
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output

    blockers, warnings = unique(blockers), unique(warnings)
    unresolved_replay = any(item["type"] == "unresolved_replay" for item in blockers)
    return {
        "passed": not blockers,
        "state": (
            "REROLLOUT_REQUIRED" if unresolved_replay
            else ("LINT_PASSED" if not blockers else "REJECTED_BY_LINTER")
        ),
        "blockers": blockers,
        "warnings": warnings,
        "counts": {"blockers": len(blockers), "warnings": len(warnings)},
        "model_sft_ready_after_edits": artifact.get("sft_ready_after_edits"),
    }


def apply_lint(artifact: dict) -> dict:
    status_before_lint = artifact.get("status")
    result = lint(artifact)
    artifact["evidence_lint"] = result
    gates = artifact.setdefault("gates", {})
    gates["MODEL_SELF_REPORTED_READY"] = {
        "passed": artifact.get("sft_ready_after_edits") is True,
        "detail": "stage-2 self-assessment after sparse edits",
    }
    gates["EVIDENCE_VISIBLE_OK"] = {
        "passed": result["passed"],
        "detail": "deterministic evidence and hygiene linter",
        "blockers": result["blockers"],
    }
    if status_before_lint != "delabeled":
        return artifact
    if result["state"] == "REROLLOUT_REQUIRED":
        gates["REROLLOUT_REQUIRED"] = {
            "passed": True,
            "detail": "a retained call could not be replayed under the seed harness",
        }
        artifact["status"] = "rerollout_required"
    elif not result["passed"]:
        artifact["status"] = "rejected_by_linter"
    else:
        artifact["status"] = "lint_passed"
    return artifact
