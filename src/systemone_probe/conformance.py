"""Contract checks for System One answers.

Everything here is derived from the published API reference and confidence
formulas, not from opinion:

* Answer fields per question type (``noul`` / ``choice`` / ``score``).
* ``probabilities`` / ``legend`` key sets, and that probabilities sum to 1.
* ``choice`` equals the arg-max option.
* ``score`` equals the probability-weighted level index, ``sum(i * p_i)``.
* ``confidence`` reproduced from the answer's own probabilities.

Tolerances exist because the vendor's own examples are printed rounded: their
Choice example shows ``confidence: 0.81`` where the formula gives 0.82.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: Tolerance for probability sums and the score identity.
PROB_TOL = 1e-3
#: Tolerance for confidence reproduced from probabilities.
CONF_TOL = 0.02

QUESTION_TYPES = {"noul", "choice", "score", "box"}


@dataclass
class Violation:
    code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.code}: {self.detail}"


# ---------------------------------------------------------------- confidence math


def noul_confidence(p: float) -> float:
    """``|2p - 1|`` — distance from an even split (docs: Confidence > How confidence is calculated)."""
    return abs(2.0 * p - 1.0)


def choice_confidence(probabilities: list[float]) -> float:
    """``(p_max - 1/n) / (1 - 1/n)``."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    return (max(probabilities) - 1.0 / n) / (1.0 - 1.0 / n)


def score_confidence(probabilities: list[float]) -> float:
    """``max(0, 1 - spread / MAD_uniform)`` over ordered levels."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    m = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    if mad_unif == 0:
        return 1.0
    return max(0.0, 1.0 - spread / mad_unif)


# ------------------------------------------------------------------- type checks


def _as_float_map(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        out[str(k)] = float(v)
    return out


def _level_order(probabilities: dict[str, float]) -> list[str] | None:
    """Return probability keys ordered by integer value, or None if not numeric."""
    try:
        return [k for _, k in sorted(((int(k), k) for k in probabilities))]
    except (TypeError, ValueError):
        return None


def check_answer(
    qid: str,
    question: dict[str, Any],
    answer: Any,
    *,
    prob_tol: float = PROB_TOL,
    conf_tol: float = CONF_TOL,
) -> list[Violation]:
    """Validate one answer against its question. Returns every violation found."""
    out: list[Violation] = []
    qtype = question.get("type")

    if not isinstance(answer, dict):
        return [Violation("answer.not_object", f"{qid}: answer is {type(answer).__name__}")]

    if answer.get("type") != qtype:
        out.append(
            Violation(
                "answer.type_mismatch",
                f"{qid}: answer type {answer.get('type')!r} != question type {qtype!r}",
            )
        )

    if qtype == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            out.append(Violation("noul.missing", f"{qid}: noul is {p!r}"))
        elif not (0.0 <= float(p) <= 1.0):
            out.append(Violation("noul.range", f"{qid}: noul={p} outside [0,1]"))

    elif qtype == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict):
            out.append(Violation("question.criteria", f"{qid}: choice criteria is not a map"))
            return out
        options = list(criteria.keys())
        probs = _as_float_map(answer.get("probabilities"))
        if probs is None:
            out.append(Violation("choice.probabilities", f"{qid}: probabilities is not a float map"))
            return out
        if set(probs) != set(options):
            out.append(
                Violation(
                    "choice.option_keys",
                    f"{qid}: probability keys {sorted(probs)} != criteria keys {sorted(options)}",
                )
            )
        total = sum(probs.values())
        if abs(total - 1.0) > prob_tol:
            out.append(Violation("choice.prob_sum", f"{qid}: probabilities sum to {total:.6f}"))
        if any(v < 0 or v > 1 for v in probs.values()):
            out.append(Violation("choice.prob_range", f"{qid}: probability outside [0,1]"))
        top = max(probs, key=lambda k: probs[k])
        if answer.get("choice") != top:
            out.append(
                Violation(
                    "choice.argmax",
                    f"{qid}: choice={answer.get('choice')!r} but arg-max is {top!r}",
                )
            )
        if set(probs) == set(options):
            expected = choice_confidence([probs[o] for o in options])
            _check_confidence(out, qid, answer.get("confidence"), expected, "choice", conf_tol)

    elif qtype == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list):
            out.append(Violation("question.criteria", f"{qid}: score criteria is not an array"))
            return out
        n = len(criteria)
        probs = _as_float_map(answer.get("probabilities"))
        if probs is None:
            out.append(Violation("score.probabilities", f"{qid}: probabilities is not a float map"))
            return out
        order = _level_order(probs)
        if order is None:
            out.append(Violation("score.level_keys", f"{qid}: level keys are not integers"))
            return out
        if len(order) != n:
            out.append(
                Violation("score.level_count", f"{qid}: {len(order)} levels scored, rubric has {n}")
            )
        legend = answer.get("legend")
        if not isinstance(legend, dict):
            out.append(Violation("score.legend", f"{qid}: legend missing or not a map"))
        else:
            expected_legend = {str(i): str(criteria[i]) for i in range(min(n, len(criteria)))}
            if {str(k): str(v) for k, v in legend.items()} != expected_legend:
                out.append(Violation("score.legend_mismatch", f"{qid}: legend != rubric"))
        total = sum(probs.values())
        if abs(total - 1.0) > prob_tol:
            out.append(Violation("score.prob_sum", f"{qid}: probabilities sum to {total:.6f}"))
        score = answer.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            out.append(Violation("score.missing", f"{qid}: score is {score!r}"))
        else:
            expected_score = sum(int(k) * v for k, v in probs.items())
            if abs(float(score) - expected_score) > max(conf_tol, prob_tol):
                out.append(
                    Violation(
                        "score.weighted_sum",
                        f"{qid}: score={score} but sum(i*p_i)={expected_score:.4f}",
                    )
                )
            if not (-prob_tol <= float(score) <= n - 1 + prob_tol):
                out.append(
                    Violation("score.range", f"{qid}: score={score} outside [0,{n - 1}]")
                )
        if order:
            expected_conf = score_confidence([probs[k] for k in order])
            _check_confidence(out, qid, answer.get("confidence"), expected_conf, "score", conf_tol)

    elif qtype == "box":
        box = answer.get("box")
        if not isinstance(box, dict) or set(box) != {"top", "bottom", "left", "right"}:
            out.append(Violation("box.shape", f"{qid}: box edges malformed"))
        else:
            if not all(0.0 <= float(v) <= 1.0 for v in box.values()):
                out.append(Violation("box.range", f"{qid}: box edge outside [0,1]"))
            if float(box["left"]) >= float(box["right"]):
                out.append(Violation("box.order", f"{qid}: left >= right"))
            if float(box["top"]) >= float(box["bottom"]):
                out.append(Violation("box.order", f"{qid}: top >= bottom"))

    else:
        out.append(Violation("question.type_unknown", f"{qid}: unknown type {qtype!r}"))

    return out


def _check_confidence(
    out: list[Violation],
    qid: str,
    got: Any,
    expected: float,
    kind: str,
    tol: float,
) -> None:
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        out.append(Violation(f"{kind}.confidence_missing", f"{qid}: confidence is {got!r}"))
        return
    if not (0.0 <= float(got) <= 1.0):
        out.append(Violation(f"{kind}.confidence_range", f"{qid}: confidence={got}"))
    if abs(float(got) - expected) > tol:
        out.append(
            Violation(
                f"{kind}.confidence_formula",
                f"{qid}: confidence={got} but formula gives {expected:.4f}",
            )
        )


def check_response_shape(response: dict[str, Any], questions: dict[str, Any]) -> list[Violation]:
    """Top-level response checks: model echoed, one answer per question, usage present."""
    out: list[Violation] = []
    if not isinstance(response.get("model"), str) or not response.get("model"):
        out.append(Violation("response.model", "response has no model string"))
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return out + [Violation("response.answers", "response has no answers map")]
    missing = set(questions) - set(answers)
    extra = set(answers) - set(questions)
    if missing:
        out.append(Violation("response.missing_answers", f"no answer for {sorted(missing)}"))
    if extra:
        out.append(Violation("response.extra_answers", f"unasked keys {sorted(extra)}"))
    usage = response.get("usage")
    if not isinstance(usage, dict):
        out.append(Violation("response.usage", "usage missing"))
    else:
        for key in ("input_tokens", "output_tokens"):
            if key in usage and not isinstance(usage[key], int):
                out.append(Violation("response.usage", f"usage.{key} is not an integer"))
    return out


def prob_vector(answer: dict[str, Any]) -> dict[str, float]:
    """Normalize any answer to a probability vector for cross-call comparison."""
    if answer.get("type") == "choice":
        return {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    if answer.get("type") == "score":
        return {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    if answer.get("type") == "noul":
        p = float(answer.get("noul", 0.0))
        return {"true": p, "false": 1.0 - p}
    return {}


def l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """L1 distance between two probability vectors over the union of their keys."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def is_finite_number(v: Any) -> bool:
    return (
        not isinstance(v, bool)
        and isinstance(v, (int, float))
        and math.isfinite(float(v))
    )


# ------------------------------------------------------------- request checking
# Catching payload bugs before the request costs money. These mirror the
# documented request rules; they are not a substitute for the server's 400/422.

MAX_NESTING = 64
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


def _depth(node: Any, limit: int = MAX_NESTING) -> int:
    """Deepest object/array nesting, cheaply capped."""
    if isinstance(node, dict):
        return 1 + max((_depth(v, limit - 1) for v in node.values()), default=0)
    if isinstance(node, list):
        return 1 + max((_depth(v, limit - 1) for v in node), default=0)
    return 0


def _nonfinite_paths(node: Any, prefix: str = "state") -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _nonfinite_paths(v, f"{prefix}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _nonfinite_paths(v, f"{prefix}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        out.append(prefix)
    return out


def check_question(qid: str, question: Any) -> list[Violation]:
    """Validate one question against the documented shape, before sending it."""
    out: list[Violation] = []
    if not isinstance(question, dict):
        return [Violation("question.not_object", f"{qid}: question is {type(question).__name__}")]
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        out.append(Violation("question.type_unknown", f"{qid}: type {qtype!r} not in {sorted(QUESTION_TYPES)}"))
    instr = question.get("instructions")
    if instr is None:
        out.append(Violation("question.instructions", f"{qid}: instructions is required"))
    elif not isinstance(instr, (str, dict, list)):
        out.append(Violation("question.instructions", f"{qid}: instructions must be string/object/array"))
    elif _depth(instr) > MAX_NESTING:
        out.append(Violation("question.nesting", f"{qid}: instructions nest deeper than {MAX_NESTING}"))

    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            out.append(Violation("question.criteria", f"{qid}: choice needs a non-empty criteria map"))
        elif len(criteria) > MAX_CHOICE_OPTIONS:
            out.append(Violation("question.criteria", f"{qid}: {len(criteria)} options > {MAX_CHOICE_OPTIONS}"))
    elif qtype == "score":
        if not isinstance(criteria, list):
            out.append(Violation("question.criteria", f"{qid}: score needs a criteria array of levels"))
        elif not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            out.append(
                Violation(
                    "question.criteria",
                    f"{qid}: {len(criteria)} levels; docs allow 2..{MAX_SCORE_LEVELS}",
                )
            )
    elif qtype == "noul" and criteria is not None:
        if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
            out.append(
                Violation("question.criteria", f"{qid}: noul criteria keys must be 'true' and/or 'false'")
            )
    elif qtype == "box" and criteria is not None:
        out.append(Violation("question.criteria", f"{qid}: box takes no criteria"))
    return out


def check_request(payload: dict[str, Any]) -> list[Violation]:
    """Validate a whole request body before it leaves the machine."""
    out: list[Violation] = []
    if not isinstance(payload, dict):
        return [Violation("request.not_object", "payload is not an object")]
    state = payload.get("state")
    if state is None:
        out.append(Violation("request.state", "state is required"))
    elif not isinstance(state, (str, dict, list)):
        out.append(Violation("request.state", f"state is {type(state).__name__}, expected string/object/array"))
    elif isinstance(state, (dict, list)):
        if _depth(state) > MAX_NESTING:
            out.append(Violation("request.nesting", f"state nests deeper than {MAX_NESTING}"))
        for path in _nonfinite_paths(state):
            out.append(Violation("request.non_finite", f"{path} is not a finite number"))
    if not isinstance(payload.get("model"), str) or not payload.get("model"):
        out.append(Violation("request.model", "model is required"))
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        out.append(Violation("request.questions", "questions must be a non-empty map"))
    else:
        for qid, q in questions.items():
            out += check_question(qid, q)
    return out
