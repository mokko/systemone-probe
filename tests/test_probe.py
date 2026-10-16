"""Tests for the harness itself: formulas, detection, dataset truth, end-to-end."""

from __future__ import annotations

import json

import pytest

from systemone_probe.client import SystemOneClient
from systemone_probe.conformance import (
    check_answer,
    choice_confidence,
    score_confidence,
)
from systemone_probe.datasets import build_dataset, monotone_score_pairs
from systemone_probe.mock import MockServer
from systemone_probe.suites import (
    brier_score,
    expected_calibration_error,
    run_all,
    suite_batching,
    suite_calibration,
    suite_contract,
    suite_determinism,
)

ALL_SUITES = ["contract", "correctness", "calibration", "determinism", "batching", "robustness", "limits"]


# ------------------------------------------------------------------ doc fixtures
# The examples printed in the published API reference. If these stop passing, the
# conformance rules have drifted from the vendor's own contract.

DOC_NOUL = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
}
DOC_NOUL_ANSWER = {"type": "noul", "noul": 0.95}

DOC_CHOICE = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        }
    },
}
DOC_CHOICE_ANSWER = {
    "type": "choice",
    "choice": "billing",
    "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
    "confidence": 0.81,
}

DOC_SCORE = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        }
    },
}
DOC_SCORE_ANSWER = {
    "type": "score",
    "score": 1.05,
    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
    "confidence": 0.92,
}


def test_doc_noul_example_conforms():
    q = DOC_NOUL["questions"]["is_urgent"]
    assert check_answer("is_urgent", q, DOC_NOUL_ANSWER) == []


def test_doc_choice_example_conforms():
    q = DOC_CHOICE["questions"]["department"]
    assert check_answer("department", q, DOC_CHOICE_ANSWER) == []


def test_doc_score_example_conforms():
    q = DOC_SCORE["questions"]["frustration"]
    assert check_answer("frustration", q, DOC_SCORE_ANSWER) == []


# ------------------------------------------------------------- confidence formulas


def test_choice_confidence_matches_docs():
    # (0.6, 0.3, 0.1) has confidence 0.4 in the docs' own words.
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    # Docs example: probabilities (0.88, 0.12, 0.0) printed as confidence 0.81.
    assert choice_confidence([0.88, 0.12, 0.0]) == pytest.approx(0.81, abs=0.02)


def test_score_confidence_matches_docs():
    # Three levels: (0, 0.5, 0.5) -> 0.25; (0.5, 0, 0.5) -> 0.
    assert score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    # Docs example (0, 0.95, 0.05) printed as 0.92; formula gives 0.925.
    assert score_confidence([0.0, 0.95, 0.05]) == pytest.approx(0.92, abs=0.02)


# ------------------------------------------------------------------- detection


def test_probability_sum_violation_is_detected():
    q = DOC_CHOICE["questions"]["department"]
    bad = dict(DOC_CHOICE_ANSWER, probabilities={"billing": 0.9, "technical": 0.2, "sales": 0.1})
    codes = {v.code for v in check_answer("department", q, bad)}
    assert "choice.prob_sum" in codes


def test_argmax_violation_is_detected():
    q = DOC_CHOICE["questions"]["department"]
    bad = dict(DOC_CHOICE_ANSWER, choice="sales")
    codes = {v.code for v in check_answer("department", q, bad)}
    assert "choice.argmax" in codes


def test_confidence_formula_violation_is_detected():
    q = DOC_SCORE["questions"]["frustration"]
    bad = dict(DOC_SCORE_ANSWER, confidence=0.05)
    codes = {v.code for v in check_answer("frustration", q, bad)}
    assert "score.confidence_formula" in codes


def test_score_weighted_sum_violation_is_detected():
    q = DOC_SCORE["questions"]["frustration"]
    bad = dict(DOC_SCORE_ANSWER, score=2.0)
    codes = {v.code for v in check_answer("frustration", q, bad)}
    assert "score.weighted_sum" in codes


def test_legend_mismatch_is_detected():
    q = DOC_SCORE["questions"]["frustration"]
    bad = dict(DOC_SCORE_ANSWER, legend={"0": "Calm", "1": "Upset", "2": "Furious"})
    codes = {v.code for v in check_answer("frustration", q, bad)}
    assert "score.legend_mismatch" in codes


# ------------------------------------------------------------ calibration math


def test_brier_score_known_values():
    assert brier_score([(1.0, 1), (0.0, 1)]) == pytest.approx(0.5)
    assert brier_score([(0.5, 1), (0.5, 0)]) == pytest.approx(0.25)
    assert brier_score([(0.9, 1), (0.9, 1), (0.9, 1), (0.9, 1), (0.9, 0)]) == pytest.approx(
        (4 * 0.01 + 0.81) / 5
    )


def test_ece_known_values():
    # Perfectly calibrated pair inside one bin cancels out.
    ece, bins = expected_calibration_error([(0.5, 1), (0.5, 0)], n_bins=10)
    assert ece == pytest.approx(0.0)
    assert sum(b["n"] for b in bins) == 2
    # Always saying 0.0 while always being wrong: maximum miscalibration here.
    ece2, _ = expected_calibration_error([(0.0, 1), (0.0, 1)], n_bins=10)
    assert ece2 == pytest.approx(1.0)
    # Bins partition the samples.
    ece3, bins3 = expected_calibration_error([(0.99, 1), (0.5, 1), (0.5, 0), (0.01, 0)], n_bins=10)
    assert sum(b["n"] for b in bins3) == 4
    assert ece3 > 0


def test_confidence_accuracy_split_needs_enough_samples():
    from systemone_probe.suites import confidence_accuracy_split

    assert confidence_accuracy_split([(0.9, True), (0.2, False)]) == {}
    split = confidence_accuracy_split(
        [(0.95, True), (0.9, True), (0.55, False), (0.1, False)]
    )
    assert split["high_confidence_accuracy"] == 1.0
    assert split["low_confidence_accuracy"] == 0.0


# --------------------------------------------------------------- dataset truth


def test_dataset_truth_is_recomputed_correctly():
    ds = build_dataset(seed=1234, per_group=3)
    for item in ds.nouls:
        st = item.state
        if "left" in st:
            assert item.truth == (st["left"] > st["right"])
        elif "number" in st:
            assert item.truth == (st["number"] % 2 == 0)
        elif "haystack" in st:
            assert item.truth == (st["needle"] in st["haystack"])
        elif "text" in st:
            assert item.truth == (len(st["text"].split()) > st["threshold"])
        else:  # pragma: no cover
            pytest.fail(f"unknown noul item shape: {st}")


def test_dataset_choice_and_score_truth():
    ds = build_dataset(seed=99, per_group=2)
    for item in ds.choices:
        st = item.state
        if "number" in st:
            n = st["number"]
            expect = "below zero" if n < 0 else ("zero to nine" if n <= 9 else "ten or more")
            assert item.truth == expect
        elif "a" in st:
            assert item.truth == max(("a", "b", "c"), key=lambda k: st[k])
        else:
            vowel = st["word"][0].lower() in "aeiou"
            assert item.truth == ("yes" if vowel else "no")
    assert all(item.truth in item.criteria for item in ds.choices)
    for item in ds.scores:
        assert 0 <= item.truth_level < len(item.levels)
    for lo, hi in monotone_score_pairs(ds.scores):
        assert lo.truth_level <= hi.truth_level


def test_dataset_is_deterministic_for_a_seed():
    assert [i.id for i in build_dataset(seed=5).nouls] == [i.id for i in build_dataset(seed=5).nouls]
    assert build_dataset(seed=5).nouls[0].state == build_dataset(seed=5).nouls[0].state


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def good_endpoint():
    with MockServer(mode="good") as server:
        yield server


def test_good_endpoint_passes_every_suite(good_endpoint):
    client = SystemOneClient(base_url=good_endpoint.base_url, api_key="test-key")
    results = run_all(client, build_dataset(seed=11, per_group=4), names=ALL_SUITES,
                      repeat=3, single_call_cap=4)
    statuses = {r.name: r.status for r in results}
    assert statuses == {name: "pass" for name in ALL_SUITES}, statuses


def test_contract_suite_fails_on_broken_probabilities():
    with MockServer(mode="badprob") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        res = suite_contract(client, build_dataset(seed=3, per_group=2))
        assert res.status == "fail"
        assert res.metrics["violations_by_code"].get("choice.prob_sum", 0) > 0


def test_determinism_suite_fails_on_unstable_answers():
    with MockServer(mode="unstable") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        res = suite_determinism(client, build_dataset(seed=3, per_group=2), repeat=3)
        assert res.status == "fail"
        assert res.metrics["identical_rate"] < 1.0


def test_batching_suite_catches_contamination_and_slowdown():
    ds = build_dataset(seed=3, per_group=3)
    with MockServer(mode="leaky") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        res = suite_batching(client, ds, repeat=2, single_call_cap=4)
        assert res.metrics["isolation_mismatches"] > 0
        assert res.status == "fail"
    with MockServer(mode="slowbatch") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        res = suite_batching(client, ds, repeat=2, single_call_cap=2)
        assert res.metrics["latency_ratio"] >= 10
        assert res.status == "fail"


def test_calibration_suite_fails_on_overconfident_model():
    # Same dataset the self-check uses: 24 labelled nouls keeps ECE/Brier stable.
    ds = build_dataset(seed=20261016, per_group=6)
    with MockServer(mode="overconfident") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        res = suite_calibration(client, ds)
        assert res.status == "fail"
        assert res.metrics["brier"] > 0.25 or res.metrics["ece"] > 0.20, res.metrics


def test_good_endpoint_is_calibrated_enough_to_pass():
    ds = build_dataset(seed=3, per_group=6)
    with MockServer(mode="good") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        res = suite_calibration(client, ds)
        assert res.status == "pass", res.metrics


def test_client_retries_rate_limited_requests():
    with MockServer(mode="flaky") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key", retries=2, backoff=0.05)
        r = client.evaluate({"left": 9, "right": 1},
                            {"cmp": {"type": "noul", "instructions": "Is `left` greater than `right`?"}})
        assert r.ok and r.attempts >= 2


def test_client_reports_transport_failure_without_raising():
    client = SystemOneClient(base_url="http://127.0.0.1:9", api_key="x", retries=0, timeout=1.0)
    r = client.evaluate({"a": 1}, {"q": {"type": "noul", "instructions": "Is `a` one?"}})
    assert not r.ok and r.status == 0 and r.error


def test_unauthenticated_request_is_rejected():
    with MockServer(mode="good") as server:
        client = SystemOneClient(base_url=server.base_url, api_key=None, retries=0)
        r = client.evaluate({"a": 1}, {"q": {"type": "noul", "instructions": "Is `a` one?"}})
        assert r.status == 401


def test_nan_body_is_rejected_by_the_strict_reference_server():
    with MockServer(mode="good") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="k", retries=0)
        body = json.dumps({"state": {"v": float("nan")}, "model": "m",
                           "questions": {"q": {"type": "noul", "instructions": "Is `v` finite?"}}},
                          allow_nan=True)
        r = client.evaluate(None, {}, raw_body=body)
        assert r.status == 400


def test_selfcheck_command_passes():
    from systemone_probe.cli import main

    assert main(["selfcheck"]) == 0


# ------------------------------------------------------------------ examples


def test_catalog_loads_and_is_consistent():
    from systemone_probe.examples import load_catalog

    catalog = load_catalog()
    assert len(catalog) >= 10
    ids = [e.id for e in catalog]
    assert len(ids) == len(set(ids))
    for example in catalog:
        assert example.why and example.code_then, f"{example.id} needs why and code_then"
        assert example.questions, f"{example.id} has no questions"


def test_every_example_payload_is_valid():
    from systemone_probe.examples import load_catalog

    for example in load_catalog():
        assert example.validate() == [], f"{example.id}: {example.validate()}"


def test_request_validation_catches_planted_bugs():
    from systemone_probe.conformance import check_request

    good = {
        "state": {"n": 1},
        "model": "jev-latest",
        "questions": {"q": {"type": "noul", "instructions": "Is `n` one?"}},
    }
    assert check_request(good) == []

    missing_state = {"model": "jev-latest", "questions": good["questions"]}
    assert {v.code for v in check_request(missing_state)} == {"request.state"}

    non_finite = {"state": {"n": float("inf")}, "model": "m", "questions": good["questions"]}
    assert "request.non_finite" in {v.code for v in check_request(non_finite)}

    bad_choice = {"state": {}, "model": "m",
                  "questions": {"q": {"type": "choice", "instructions": "x"}}}
    assert "question.criteria" in {v.code for v in check_request(bad_choice)}

    too_many_levels = {
        "state": {},
        "model": "m",
        "questions": {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}},
    }
    assert "question.criteria" in {v.code for v in check_request(too_many_levels)}

    one_level = {"state": {}, "model": "m",
                 "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}}
    assert "question.criteria" in {v.code for v in check_request(one_level)}

    bad_noul_criteria = {
        "state": {},
        "model": "m",
        "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}},
    }
    assert "question.criteria" in {v.code for v in check_request(bad_noul_criteria)}


def test_deep_nesting_is_rejected():
    from systemone_probe.conformance import check_request

    deep: dict[str, object] = {}
    node = deep
    for _ in range(70):
        node["n"] = {}
        node = node["n"]  # type: ignore[assignment]
    payload = {"state": deep, "model": "m",
               "questions": {"q": {"type": "noul", "instructions": "x"}}}
    assert "request.nesting" in {v.code for v in check_request(payload)}


def test_to_curl_targets_the_endpoint():
    from systemone_probe.examples import get_example, to_curl

    curl = to_curl(get_example("support_triage"), "https://api.typesafe.ai")
    assert "https://api.typesafe.ai/v1/systemone" in curl
    assert "TYPESAFE_API_KEY" in curl
    assert '"is_urgent"' in curl


def test_run_example_against_the_stub_returns_one_answer_per_question():
    from systemone_probe.examples import get_example, render_result, run_example

    example = get_example("support_triage")
    with MockServer(mode="good") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key")
        result = run_example(example, client)
    assert result["ok"], result
    assert set(result["answers"]) == set(example.questions)
    assert "support_triage" in render_result(result)
    assert render_result(result).count("choice=") == 1


def test_examples_cli_check_and_list():
    from systemone_probe.cli import main

    assert main(["examples", "--check"]) == 0
    assert main(["examples"]) == 0
    assert main(["examples", "--show", "museum_record_match"]) == 0
    assert main(["examples", "--show", "does_not_exist"]) == 2

