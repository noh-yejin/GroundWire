from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.models import Issue


@dataclass(slots=True)
class GoldenExpectation:
    topic_contains: str
    expected_status: str
    min_grounded_ratio: float | None = None
    hold_reason_contains: str | None = None
    summary_terms_any: list[str] | None = None
    allowed_modes: list[str] | None = None


def load_golden_expectations(path: str | Path) -> list[GoldenExpectation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenExpectation(**item) for item in raw]


def evaluate_issues_against_goldens(issues: list[Issue], expectations: list[GoldenExpectation]) -> dict:
    results: list[dict] = []
    matched = 0
    passed = 0
    for expectation in expectations:
        issue = next((item for item in issues if expectation.topic_contains.lower() in item.topic.lower()), None)
        if issue is None:
            results.append(
                {
                    "topic_contains": expectation.topic_contains,
                    "matched": False,
                    "passed": False,
                    "checks": {"exists": False},
                }
            )
            continue
        matched += 1
        details = issue.analysis.grounding_details or {}
        grounding = details.get("grounding", {})
        mode = details.get("llm", {}).get("analysis_mode")
        checks = {
            "status": issue.status.value == expectation.expected_status,
            "grounded_ratio": True,
            "hold_reason": True,
            "summary_terms": True,
            "mode": True,
        }
        if expectation.min_grounded_ratio is not None:
            checks["grounded_ratio"] = float(grounding.get("grounded_ratio", 0.0) or 0.0) >= expectation.min_grounded_ratio
        if expectation.hold_reason_contains:
            checks["hold_reason"] = expectation.hold_reason_contains in (issue.analysis.hold_reason or "")
        if expectation.summary_terms_any:
            summary = issue.analysis.summary or ""
            checks["summary_terms"] = any(term in summary for term in expectation.summary_terms_any)
        if expectation.allowed_modes:
            checks["mode"] = mode in expectation.allowed_modes
        case_passed = all(checks.values())
        if case_passed:
            passed += 1
        results.append(
            {
                "topic_contains": expectation.topic_contains,
                "matched": True,
                "passed": case_passed,
                "topic": issue.topic,
                "status": issue.status.value,
                "mode": mode,
                "summary": issue.analysis.summary,
                "hold_reason": issue.analysis.hold_reason,
                "grounded_ratio": grounding.get("grounded_ratio"),
                "checks": checks,
            }
        )
    return {
        "matched": matched,
        "passed": passed,
        "total": len(expectations),
        "pass_rate": round(passed / max(len(expectations), 1), 3),
        "results": results,
    }
