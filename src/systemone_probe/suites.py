"""The probe suites.

Each suite is an independent experiment against a live endpoint and returns a
:class:`SuiteResult`. Suites are split by *kind of claim* they test, because the
thresholds are not equally trustworthy:

* **Protocol claims** (contract, determinism, isolation, limits) are hard: the
  docs state them, so a failure is a failure.
* **Quality claims** (calibration, robustness, accuracy) are heuristics. They run
  and report by default; they only gate the exit code with ``--enforce``, because
  the right thresholds depend on your workload, not on a doc example.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from systemone_probe.client import CallResult, SystemOneClient
from systemone_probe.conformance import (
    check_answer,
    check_response_shape,
    l1_distance,
    prob_vector,
)
from systemone_probe.datasets import ChoiceItem, Dataset, NoulItem, ScoreItem, monotone_score_pairs

PASS, FAIL, WARN, SKIP, INFO = "pass", "fail", "warn", "skip", "info"

#: Quality-suite thresholds. Deliberately loose: they exist to catch a broken
#: or dishonest model, not to rank two working ones. ``ece_max`` is looser than
#: the literature's usual 0.10 because ECE is noisy on small labelled sets
#: (below ~100 items, one misbinned answer moves it by several points) and a
#: model with discrete beliefs has an intrinsic floor.
DEFAULT_THRESHOLDS = {
    "brier_max": 0.25,
    "ece_max": 0.20,
    "accuracy_min": 0.60,
    "determinism_min": 0.90,
    "batch_latency_ratio_fail": 10.0,
    "batch_latency_ratio_warn": 3.0,
    "robustness_agreement_min": 0.90,
}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class SuiteResult:
    name: str
    status: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    calls: int = 0

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]


def _status_of(checks: Sequence[Check]) -> str:
    if any(c.status == FAIL for c in checks):
        return FAIL
    if any(c.status == WARN for c in checks):
        return WARN
    if any(c.status == PASS for c in checks):
        return PASS
    if checks:
        return INFO
    return SKIP


def _questions_of(dataset: Dataset) -> dict[str, dict[str, Any]]:
    qs: dict[str, dict[str, Any]] = {}
    for item in [*dataset.nouls, *dataset.choices, *dataset.scores]:
        qs[item.id] = item.question
    return qs


def _state_of(dataset: Dataset) -> Any:
    """Single state carrying every item's fields under its own key.

    Real System One usage puts one coherent state in the request; the probe set
    needs many independent states, so each item's state is nested under its id
    and its instructions already point at those paths via backticks.
    """
    merged: dict[str, Any] = {}
    for item in [*dataset.nouls, *dataset.choices, *dataset.scores]:
        merged[item.id] = item.state
    return merged


def _nested_questions(dataset: Dataset) -> dict[str, dict[str, Any]]:
    """Rewrite instructions so backticked field names resolve inside the merged state."""
    out: dict[str, dict[str, Any]] = {}
    for item in [*dataset.nouls, *dataset.choices, *dataset.scores]:
        q = dict(item.question)
        instr = q["instructions"]
        if isinstance(instr, str):
            q["instructions"] = f"Within `{item.id}`: {instr.replace('`', '`')}"
        out[item.id] = q
    return out


# ------------------------------------------------------------------ 1. contract


def suite_contract(client: SystemOneClient, dataset: Dataset, *, thresholds: dict[str, float] | None = None) -> SuiteResult:
    """Answer shapes, probability sums, arg-max, and the confidence formulas."""
    res = SuiteResult("contract", SKIP)
    state = _state_of(dataset)
    questions = _nested_questions(dataset)
    call = client.evaluate(state, questions)
    res.calls = 1
    if not call.ok:
        res.add("http.ok", FAIL, f"status={call.status} body={call.body_text[:200]}")
        res.status = FAIL
        return res
    res.add("http.ok", PASS, f"{call.status} in {call.latency_ms:.0f} ms")

    shape = check_response_shape(call.response or {}, questions)
    res.add(
        "response.shape",
        FAIL if shape else PASS,
        "; ".join(str(v) for v in shape) if shape else "model, answers and usage all present",
    )
    res.metrics["response_model"] = (call.response or {}).get("model")

    codes: dict[str, int] = {}
    for qid, question in questions.items():
        for v in check_answer(qid, question, call.answers.get(qid)):
            codes[v.code] = codes.get(v.code, 0) + 1
    res.metrics["violations_by_code"] = codes
    if codes:
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(codes.items(), key=lambda kv: -kv[1]))
        res.add("answer.conformance", FAIL, detail)
    else:
        res.add("answer.conformance", PASS, f"{len(questions)} answers conform")

    res.metrics["usage"] = (call.response or {}).get("usage") or {}
    res.status = _status_of(res.checks)
    return res


# --------------------------------------------------------------- 2. correctness


def suite_correctness(client: SystemOneClient, dataset: Dataset, *, thresholds: dict[str, float] | None = None) -> SuiteResult:
    """Accuracy against Python-computed ground truth, overall and per group."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    res = SuiteResult("correctness", SKIP)
    state = _state_of(dataset)
    questions = _nested_questions(dataset)
    call = client.evaluate(state, questions)
    res.calls = 1
    if not call.ok:
        res.add("http.ok", FAIL, f"status={call.status}")
        res.status = FAIL
        return res

    per_group: dict[str, list[bool]] = {}
    for item in dataset.nouls:
        ans = call.answers.get(item.id) or {}
        p = ans.get("noul")
        if p is None:
            continue
        per_group.setdefault(item.group, []).append((float(p) >= 0.5) == item.truth)
    for item in dataset.choices:
        ans = call.answers.get(item.id) or {}
        if "choice" not in ans:
            continue
        per_group.setdefault(item.group, []).append(ans["choice"] == item.truth)
    for item in dataset.scores:
        ans = call.answers.get(item.id) or {}
        if "score" not in ans:
            continue
        per_group.setdefault(item.group, []).append(round(float(ans["score"])) == item.truth_level)

    all_hits = [hit for hits in per_group.values() for hit in hits]
    accuracy = sum(all_hits) / len(all_hits) if all_hits else 0.0
    res.metrics["accuracy"] = round(accuracy, 4)
    res.metrics["n_scored"] = len(all_hits)
    res.metrics["per_group"] = {
        g: round(sum(h) / len(h), 4) for g, h in sorted(per_group.items()) if h
    }
    for group, hits in sorted(per_group.items()):
        group_acc = sum(hits) / len(hits)
        res.add(
            f"accuracy.{group}",
            INFO,
            f"{group_acc:.2%} over {len(hits)} items",
        )

    res.add(
        "accuracy.overall",
        PASS if accuracy >= th["accuracy_min"] else FAIL,
        f"{accuracy:.2%} over {len(all_hits)} items (floor {th['accuracy_min']:.0%})",
    )

    pairs = monotone_score_pairs(dataset.scores)
    inversions = 0
    for lo, hi in pairs:
        a = call.answers.get(lo.id) or {}
        b = call.answers.get(hi.id) or {}
        if "score" in a and "score" in b and float(a["score"]) > float(b["score"]) + 0.5:
            inversions += 1
    res.metrics["score_inversions"] = inversions
    res.add(
        "score.monotone",
        PASS if inversions == 0 else WARN,
        f"{inversions} order inversions across {len(pairs)} ordered pairs",
    )
    res.status = _status_of(res.checks)
    return res


# --------------------------------------------------------------- 3. calibration


def suite_calibration(client: SystemOneClient, dataset: Dataset, *, thresholds: dict[str, float] | None = None) -> SuiteResult:
    """Brier score, ECE and a reliability table over the labelled noul probes."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    res = SuiteResult("calibration", SKIP)
    state = _state_of(dataset)
    questions = {item.id: item.question for item in dataset.nouls}
    call = client.evaluate(state, {k: _with_prefix(k, v) for k, v in questions.items()})
    res.calls = 1
    if not call.ok:
        res.add("http.ok", FAIL, f"status={call.status}")
        res.status = FAIL
        return res

    pairs: list[tuple[float, int]] = []
    for item in dataset.nouls:
        ans = call.answers.get(item.id) or {}
        if ans.get("noul") is None:
            continue
        pairs.append((float(ans["noul"]), 1 if item.truth else 0))
    if not pairs:
        res.add("data", FAIL, "no noul answers returned")
        res.status = FAIL
        return res

    brier = brier_score(pairs)
    ece, bins = expected_calibration_error(pairs)
    res.metrics["brier"] = round(brier, 4)
    res.metrics["ece"] = round(ece, 4)
    res.metrics["reliability"] = bins
    res.metrics["n"] = len(pairs)

    res.add("brier", PASS if brier <= th["brier_max"] else FAIL,
            f"{brier:.4f} (max {th['brier_max']})")
    res.add("ece", PASS if ece <= th["ece_max"] else FAIL,
            f"{ece:.4f} (max {th['ece_max']})")
    res.add("reliability", INFO, "bin / n / mean predicted / observed")
    for b in bins:
        if b["n"]:
            res.add(
                f"reliability[{b['lo']:.1f},{b['hi']:.1f})",
                INFO,
                f"n={b['n']} predicted={b['mean_pred']:.3f} observed={b['observed']:.3f}",
            )
    res.status = _status_of(res.checks)
    return res


def brier_score(pairs: Iterable[tuple[float, int]]) -> float:
    """Mean squared error of probabilities against 0/1 outcomes."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def expected_calibration_error(pairs: Sequence[tuple[float, int]], n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    """Standard equal-width ECE, plus the reliability table behind it."""
    pairs = list(pairs)
    bins: list[dict[str, Any]] = []
    total = len(pairs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        members = [(p, y) for p, y in pairs if (lo <= p < hi) or (i == n_bins - 1 and p == 1.0)]
        n = len(members)
        if n:
            mean_pred = sum(p for p, _ in members) / n
            observed = sum(y for _, y in members) / n
            ece += (n / total) * abs(mean_pred - observed)
        else:
            mean_pred = observed = 0.0
        bins.append({"lo": lo, "hi": hi, "n": n, "mean_pred": mean_pred, "observed": observed})
    return ece, bins


def confidence_accuracy_split(
    samples: Sequence[tuple[float, bool]],
) -> dict[str, Any]:
    """Does low confidence actually mean more errors? Split at the median."""
    if len(samples) < 4:
        return {}
    ordered = sorted(samples, key=lambda s: s[0])
    half = len(ordered) // 2
    low, high = ordered[:half], ordered[half:]
    return {
        "low_confidence_accuracy": round(sum(h for _, h in low) / len(low), 4),
        "high_confidence_accuracy": round(sum(h for _, h in high) / len(high), 4),
        "low_confidence_mean": round(statistics.fmean(p for p, _ in low), 4),
        "high_confidence_mean": round(statistics.fmean(p for p, _ in high), 4),
    }


# --------------------------------------------------------------- 4. determinism


def suite_determinism(client: SystemOneClient, dataset: Dataset, *, thresholds: dict[str, float] | None = None, repeat: int = 3) -> SuiteResult:
    """Identical requests, repeated: do the answers hold still?"""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    res = SuiteResult("determinism", SKIP)
    state = _state_of(dataset)
    questions = _nested_questions(dataset)
    calls: list[CallResult] = [client.evaluate(state, questions) for _ in range(max(2, repeat))]
    res.calls = len(calls)
    bad = [c for c in calls if not c.ok]
    if bad:
        res.add("http.ok", FAIL, f"{len(bad)}/{len(calls)} calls failed (status {bad[0].status})")
        res.status = FAIL
        return res

    base = calls[0].answers
    exact = 0
    worst_drift = 0.0
    worst_qid = ""
    for qid in questions:
        b = base.get(qid)
        if b is None:
            continue
        identical = True
        qid_drift = 0.0
        for other in calls[1:]:
            o = other.answers.get(qid)
            if o is None:
                identical = False
                continue
            d = l1_distance(prob_vector(b), prob_vector(o))
            qid_drift = max(qid_drift, d)
            if d > 1e-6:
                identical = False
        if identical:
            exact += 1
        if qid_drift > worst_drift:
            worst_drift, worst_qid = qid_drift, qid
    total = len(questions)
    rate = exact / total if total else 0.0
    res.metrics.update(
        {"repeat": len(calls), "identical_rate": round(rate, 4),
         "max_probability_drift_l1": round(worst_drift, 6), "worst_question": worst_qid}
    )
    res.add(
        "answers.identical",
        PASS if rate >= th["determinism_min"] else FAIL,
        f"{exact}/{total} answers byte-identical across {len(calls)} identical requests "
        f"(floor {th['determinism_min']:.0%}); worst probability drift {worst_drift:.4f} on {worst_qid or '-'}",
    )
    if 1e-6 < worst_drift <= 0.05 and rate >= th["determinism_min"]:
        res.add("drift.tolerable", WARN, f"small drift {worst_drift:.4f} — sampling noise, not a contract break")
    res.status = _status_of(res.checks)
    return res


# ------------------------------------------------------------------ 5. batching


def suite_batching(
    client: SystemOneClient,
    dataset: Dataset,
    *,
    thresholds: dict[str, float] | None = None,
    repeat: int = 3,
    single_call_cap: int = 12,
) -> SuiteResult:
    """Two documented claims: many questions are ~free, and questions do not interact."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    res = SuiteResult("batching", SKIP)
    state = _state_of(dataset)
    everything = _nested_questions(dataset)
    probe_id = dataset.nouls[0].id
    single_q = {probe_id: everything[probe_id]}

    singles = [client.evaluate(state, single_q) for _ in range(repeat)]
    batch = client.evaluate(state, everything)
    res.calls = len(singles) + 1
    if any(not c.ok for c in singles) or not batch.ok:
        res.add("http.ok", FAIL, "call failed")
        res.status = FAIL
        return res

    single_ms = statistics.median(c.latency_ms for c in singles)
    ratio = batch.latency_ms / single_ms if single_ms > 0 else float("inf")
    res.metrics.update(
        {
            "single_question_ms": round(single_ms, 1),
            "batch_ms": round(batch.latency_ms, 1),
            "batch_questions": len(everything),
            "latency_ratio": round(ratio, 2),
        }
    )
    status = PASS
    if ratio >= th["batch_latency_ratio_fail"]:
        status = FAIL
    elif ratio >= th["batch_latency_ratio_warn"]:
        status = WARN
    res.add(
        "latency.near_constant",
        status,
        f"1 question {single_ms:.0f} ms vs {len(everything)} questions {batch.latency_ms:.0f} ms "
        f"(ratio {ratio:.2f}; warn ≥{th['batch_latency_ratio_warn']}, fail ≥{th['batch_latency_ratio_fail']})",
    )

    # Isolation: the same questions asked alone must give the same answers.
    sample = [*dataset.nouls[:single_call_cap], *dataset.choices[:3], *dataset.scores[:3]]
    mismatches: list[str] = []
    for item in sample:
        q = {item.id: everything[item.id]}
        one = client.evaluate(state, q)
        res.calls += 1
        if not one.ok:
            mismatches.append(f"{item.id}: single call failed ({one.status})")
            continue
        d = l1_distance(prob_vector(batch.answers.get(item.id) or {}), prob_vector(one.answers.get(item.id) or {}))
        if d > 1e-6:
            mismatches.append(f"{item.id}: L1 {d:.4f}")
    res.metrics["isolation_mismatches"] = len(mismatches)
    res.add(
        "questions.isolated",
        PASS if not mismatches else FAIL,
        f"{len(sample) - len(mismatches)}/{len(sample)} answers unchanged when asked alone"
        + (f"; first mismatch {mismatches[0]}" if mismatches else ""),
    )
    res.status = _status_of(res.checks)
    return res


# ---------------------------------------------------------------- 6. robustness


def suite_robustness(client: SystemOneClient, dataset: Dataset, *, thresholds: dict[str, float] | None = None) -> SuiteResult:
    """Would a rename, a re-encoding, or a rephrasing change the answer?"""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    res = SuiteResult("robustness", SKIP)
    state = _state_of(dataset)
    questions = _nested_questions(dataset)
    base = client.evaluate(state, questions)
    res.calls = 1
    if not base.ok:
        res.add("http.ok", FAIL, f"status={base.status}")
        res.status = FAIL
        return res

    # (a) question ids are documented as not reaching the model.
    renamed = {f"probe_{i:03d}": q for i, q in enumerate(questions.values())}
    mapping = dict(zip(questions, renamed))
    ren_call = client.evaluate(state, renamed)
    res.calls += 1
    if ren_call.ok:
        agree = sum(
            1 for old, new in mapping.items()
            if l1_distance(prob_vector(base.answers.get(old) or {}), prob_vector(ren_call.answers.get(new) or {})) <= 1e-6
        )
        rate = agree / len(mapping)
        res.metrics["id_invariance_agreement"] = round(rate, 4)
        res.add(
            "ids.invariant",
            PASS if rate >= th["robustness_agreement_min"] else FAIL,
            f"{agree}/{len(mapping)} answers unchanged after renaming question ids",
        )
    else:
        res.add("ids.invariant", FAIL, f"renamed-id call failed (status {ren_call.status})")

    # (b) the same data as a JSON string vs an object.
    str_call = client.evaluate(json.dumps(state), questions)
    res.calls += 1
    if str_call.ok:
        agree = sum(
            1 for qid in questions
            if l1_distance(prob_vector(base.answers.get(qid) or {}), prob_vector(str_call.answers.get(qid) or {})) <= 1e-6
        )
        rate = agree / len(questions)
        res.metrics["state_form_agreement"] = round(rate, 4)
        res.add(
            "state.form_invariant",
            PASS if rate >= th["robustness_agreement_min"] else FAIL,
            f"{agree}/{len(questions)} answers unchanged when state is a JSON string instead of an object",
        )
    else:
        res.add("state.form_invariant", FAIL, f"string-state call failed (status {str_call.status})")

    # (c) paraphrased instructions — informational, wording is the model's business.
    para = [item for item in dataset.nouls if getattr(item, "paraphrase", None)]
    if para:
        qs = {f"para_{i}": {"type": "noul", "instructions": item.paraphrase} for i, item in enumerate(para)}
        para_state = {f"para_{i}": item.state for i, item in enumerate(para)}
        para_call = client.evaluate(para_state, qs)
        res.calls += 1
        if para_call.ok:
            flips = 0
            for i, item in enumerate(para):
                got = para_call.answers.get(f"para_{i}") or {}
                orig = base.answers.get(item.id) or {}
                if got.get("noul") is None or orig.get("noul") is None:
                    continue
                if (float(got["noul"]) >= 0.5) != (float(orig["noul"]) >= 0.5):
                    flips += 1
            res.metrics["paraphrase_flips"] = flips
            res.add("instructions.paraphrase", INFO, f"{flips}/{len(para)} verdicts flipped on a rewording")
    res.status = _status_of(res.checks)
    return res


# -------------------------------------------------------------------- 7. limits


def suite_limits(client: SystemOneClient, dataset: Dataset, *, thresholds: dict[str, float] | None = None) -> SuiteResult:
    """The documented error surface: 400 / 401 / 422."""
    res = SuiteResult("limits", SKIP)
    state = {"left": 5, "right": 3}
    q = {"cmp": {"type": "noul", "instructions": "Is `left` greater than `right`?"}}

    nan_body = json.dumps({"state": {"value": float("nan")}, "model": client.model,
                           "questions": {"q": {"type": "noul", "instructions": "Is `value` finite?"}}},
                          allow_nan=True)
    nan_call = client.evaluate(None, {}, raw_body=nan_body)
    res.calls += 1
    res.add("400.non_finite", PASS if nan_call.status == 400 else FAIL,
            f"NaN in state returned {nan_call.status} (expected 400)")

    bad_q = json.dumps({"state": state, "model": client.model, "questions": {"q": {"type": "noul"}}})
    bad_call = client.evaluate(None, {}, raw_body=bad_q)
    res.calls += 1
    res.add("422.malformed_question", PASS if bad_call.status == 422 else FAIL,
            f"question without instructions returned {bad_call.status} (expected 422)")

    unkeyed = SystemOneClient(base_url=client.base_url, api_key=None, model=client.model,
                              retries=0, timeout=client.timeout)
    unauth = unkeyed.evaluate(state, q)
    res.calls += 1
    res.add("401.missing_key", PASS if unauth.status == 401 else FAIL,
            f"request without Authorization returned {unauth.status} (expected 401)")

    ok = client.evaluate(state, q)
    res.calls += 1
    res.add("200.well_formed", PASS if ok.ok else FAIL,
            f"{ok.status} in {ok.latency_ms:.0f} ms; usage={ok.usage}")
    res.status = _status_of(res.checks)
    return res


# ------------------------------------------------------------------ driver


SUITES = {
    "contract": suite_contract,
    "correctness": suite_correctness,
    "calibration": suite_calibration,
    "determinism": suite_determinism,
    "batching": suite_batching,
    "robustness": suite_robustness,
    "limits": suite_limits,
}

#: Suites whose thresholds are heuristics, so they only gate with ``--enforce``.
QUALITY_SUITES = frozenset({"calibration", "correctness", "robustness"})


def _with_prefix(qid: str, question: dict[str, Any]) -> dict[str, Any]:
    q = dict(question)
    instr = q.get("instructions")
    if isinstance(instr, str):
        q["instructions"] = f"Within `{qid}`: {instr}"
    return q


def run_all(
    client: SystemOneClient,
    dataset: Dataset,
    *,
    names: Sequence[str] | None = None,
    thresholds: dict[str, float] | None = None,
    repeat: int = 3,
    single_call_cap: int = 12,
) -> list[SuiteResult]:
    """Run the requested suites in a stable order."""
    wanted = list(names) if names else list(SUITES)
    results: list[SuiteResult] = []
    for name in wanted:
        fn = SUITES.get(name)
        if fn is None:
            results.append(SuiteResult(name, SKIP, [Check("suite.unknown", SKIP, f"no suite named {name!r}")]))
            continue
        kwargs: dict[str, Any] = {"thresholds": thresholds}
        if name in {"determinism", "batching"}:
            kwargs["repeat"] = repeat
        if name == "batching":
            kwargs["single_call_cap"] = single_call_cap
        results.append(fn(client, dataset, **kwargs))
    return results
