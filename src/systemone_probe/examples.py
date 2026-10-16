"""The example catalogue: where a System One model is the right tool, and why.

Each entry is a complete ``POST /v1/systemone`` request body plus the reasoning
that makes it System-One-shaped rather than LLM-shaped. Two of the fields matter
most:

* ``why`` — the property of the decision that makes a typed, calibrated answer
  better than generated text.
* ``code_then`` — what your code does with the answers. If you cannot write this
  line, the decision probably is not a System One decision.

``documented`` quotes the vendor's published measurement for the same pattern,
so an example can be judged against a real number rather than an opinion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from systemone_probe.client import CallResult, SystemOneClient
from systemone_probe.conformance import Violation, check_request

CATALOG_PATH = Path(__file__).resolve().parents[2] / "examples" / "catalog.json"


@dataclass
class Example:
    id: str
    title: str
    domain: str
    why: str
    state: Any
    questions: dict[str, Any]
    code_then: str = ""
    model: str = "jev-latest"
    documented: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Example":
        return cls(
            id=raw["id"],
            title=raw.get("title", raw["id"]),
            domain=raw.get("domain", ""),
            why=raw.get("why", ""),
            state=raw["state"],
            questions=raw["questions"],
            code_then=raw.get("code_then", ""),
            model=raw.get("model", "jev-latest"),
            documented=raw.get("documented") or {},
        )

    def payload(self, model: str | None = None) -> dict[str, Any]:
        return {"state": self.state, "model": model or self.model, "questions": self.questions}

    @property
    def question_types(self) -> list[str]:
        return sorted({q.get("type", "?") for q in self.questions.values()})

    def validate(self) -> list[Violation]:
        """Check the request locally — before it costs a request."""
        return check_request(self.payload())


def load_catalog(path: Path | str | None = None) -> list[Example]:
    target = Path(path) if path else CATALOG_PATH
    raw = json.loads(target.read_text())
    return [Example.from_dict(item) for item in raw["examples"]]


def get_example(example_id: str, path: Path | str | None = None) -> Example:
    for example in load_catalog(path):
        if example.id == example_id:
            return example
    known = ", ".join(e.id for e in load_catalog(path))
    raise KeyError(f"no example {example_id!r}; known: {known}")


def to_curl(example: Example, base_url: str, *, model: str | None = None,
            api_key_var: str = "$TYPESAFE_API_KEY") -> str:
    """Render the exact HTTP request, so the shape is visible and copy-pasteable."""
    body = json.dumps(example.payload(model), indent=2, ensure_ascii=False)
    return (
        f"curl {base_url.rstrip('/')}/v1/systemone \\\n"
        f'  -H "Authorization: Bearer {api_key_var}" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f"  -d '{body}'"
    )


def run_example(example: Example, client: SystemOneClient, *, model: str | None = None) -> dict[str, Any]:
    """Send one example and return the request, the raw response and the timings."""
    call: CallResult = client.evaluate(example.state, example.questions, model=model or example.model)
    return {
        "example": example.id,
        "title": example.title,
        "domain": example.domain,
        "endpoint": client.url(),
        "request": {"state": example.state, "model": model or example.model, "questions": example.questions},
        "status": call.status,
        "ok": call.ok,
        "latency_ms": round(call.latency_ms, 1),
        "attempts": call.attempts,
        "response_model": call.model,
        "answers": call.answers,
        "usage": call.usage,
        "error": call.error,
        "documented": example.documented,
        "code_then": example.code_then,
    }


def render_result(result: dict[str, Any]) -> str:
    """A compact human view: what came back, and what the code would do with it."""
    lines: list[str] = []
    header = f"{result['example']} — {result['title']}"
    lines.append(header)
    lines.append("-" * len(header))
    if not result.get("ok"):
        lines.append(f"  FAILED: status={result['status']} error={result.get('error')}")
        lines.append(f"  body: {str(result.get('answers'))[:200]}")
        return "\n".join(lines)

    doc = result.get("documented") or {}
    if doc:
        lines.append(f"  vendor-measured: {doc.get('result', '')}")
        lines.append(f"  source: {doc.get('source', '')}")

    for qid, answer in result["answers"].items():
        atype = answer.get("type")
        if atype == "noul":
            lines.append(f"  {qid:<24} noul={answer.get('noul'):.3f}")
        elif atype == "choice":
            probs = answer.get("probabilities") or {}
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            top = ", ".join(f"{k} {v:.2f}" for k, v in ranked[:4])
            lines.append(f"  {qid:<24} choice={answer.get('choice')!r} conf={answer.get('confidence')}  [{top}]")
        elif atype == "score":
            probs = answer.get("probabilities") or {}
            ranked = sorted(probs.items(), key=lambda kv: -float(kv[1]))
            legend = answer.get("legend") or {}
            top_key = ranked[0][0] if ranked else ""
            label = legend.get(str(top_key), "")
            top = ", ".join(f"L{k} {v:.2f}" for k, v in ranked[:4])
            lines.append(
                f"  {qid:<24} score={answer.get('score')} (most likely level: {label}) "
                f"conf={answer.get('confidence')}  [{top}]"
            )
        else:
            lines.append(f"  {qid:<24} {answer}")
    lines.append(f"  {result['latency_ms']} ms, model {result.get('response_model')}, usage {result.get('usage')}")
    return "\n".join(lines)


def render_catalog(examples: list[Example]) -> str:
    """The 'which of these is my problem?' view."""
    lines: list[str] = []
    width = max(len(e.id) for e in examples)
    for e in examples:
        lines.append(f"{e.id:<{width}}  [{e.domain}] {e.title}")
        lines.append(f"{'':<{width}}  why: {e.why}")
        if e.documented:
            lines.append(f"{'':<{width}}  measured: {e.documented.get('result', '')}")
        lines.append(f"{'':<{width}}  then: {e.code_then}")
        lines.append("")
    return "\n".join(lines)
