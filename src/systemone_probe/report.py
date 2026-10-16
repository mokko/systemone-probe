"""Report rendering: markdown for humans, JSON for CI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

from systemone_probe.suites import FAIL, WARN, SuiteResult

STATUS_ICON = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP", "info": "INFO"}


def summary_line(results: Sequence[SuiteResult]) -> str:
    parts = [f"{r.name}={STATUS_ICON.get(r.status, r.status)}" for r in results]
    return "  ".join(parts)


def overall_status(results: Sequence[SuiteResult], *, enforce: bool = False) -> str:
    from systemone_probe.suites import QUALITY_SUITES

    for r in results:
        if r.status == FAIL and (enforce or r.name not in QUALITY_SUITES):
            return FAIL
    if any(r.status in (FAIL, WARN) for r in results):
        return "warn"
    if all(r.status == SKIP for r in results):
        return SKIP
    return "pass"


def to_json(
    results: Sequence[SuiteResult],
    *,
    target: str,
    model: str,
    dataset_size: int,
    enforce: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "model": model,
        "dataset_items": dataset_size,
        "enforce": enforce,
        "overall": overall_status(results, enforce=enforce),
        "suites": [
            {
                "name": r.name,
                "status": r.status,
                "calls": r.calls,
                "metrics": r.metrics,
                "checks": [
                    {"name": c.name, "status": c.status, "detail": c.detail} for c in r.checks
                ],
                "notes": r.notes,
            }
            for r in results
        ],
        **(extra or {}),
    }


def to_markdown(
    results: Sequence[SuiteResult],
    *,
    target: str,
    model: str,
    dataset_size: int,
    enforce: bool,
    extra: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"# System One probe report — {model}")
    lines.append("")
    lines.append(f"- target: `{target}`")
    lines.append(f"- model: `{model}`")
    lines.append(f"- dataset: {dataset_size} items (mechanically labelled)")
    lines.append(f"- generated: {now}")
    lines.append(f"- overall: **{overall_status(results, enforce=enforce).upper()}**"
                 + (" (quality suites enforced)" if enforce else " (quality suites advisory)"))
    lines.append("")
    lines.append("| suite | status | calls | headline |")
    lines.append("| --- | --- | --- | --- |")
    for r in results:
        lines.append(f"| {r.name} | {STATUS_ICON.get(r.status, r.status)} | {r.calls} | {_headline(r)} |")
    lines.append("")

    for r in results:
        lines.append(f"## {r.name} — {STATUS_ICON.get(r.status, r.status)}")
        lines.append("")
        for c in r.checks:
            lines.append(f"- [{STATUS_ICON.get(c.status, c.status)}] **{c.name}** — {c.detail}")
        if r.metrics:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(r.metrics, indent=2, sort_keys=True, default=str))
            lines.append("```")
        lines.append("")

    if extra:
        lines.append("## run metadata")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(extra, indent=2, sort_keys=True, default=str))
        lines.append("```")
    return "\n".join(lines) + "\n"


def _headline(r: SuiteResult) -> str:
    priority = (FAIL, "warn", "pass", "info", "skip")
    for status in priority:
        for c in r.checks:
            if c.status == status:
                return c.detail.replace("|", "/")[:160]
    return ""


def failures_only(results: Sequence[SuiteResult]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for r in results:
        for c in r.checks:
            if c.status == FAIL:
                out.append((r.name, c.name, c.detail))
    return out
