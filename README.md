# systemone-probe

A conformance and behaviour test harness for **System One** decision models —
Jev (TypeSafe's flagship, the first of the class) and the open Jev-like models
that speak the same wire protocol (`POST /v1/systemone`).

Point it at an endpoint, get a report on what the endpoint actually does:
does it honour the answer contract, does it ground on the state, do its
probabilities hold still, is its confidence calibrated, and are the documented
behaviours (many questions per request, no cross-question contamination, the
error surface) real.

```
contract=PASS  correctness=PASS  calibration=PASS  determinism=PASS  batching=PASS  robustness=PASS  limits=PASS
```

## Why this exists

A System One model returns typed answers with calibrated probabilities instead of
text — *"the answer is `billing`, p=0.88, confidence 0.81"*. That makes it
testable in a way an LLM is not: there is a documented contract, exact formulas
behind every field, and no free text to grade. Three things follow:

* Most of the contract can be checked **without labels** — probability sums,
  arg-max agreement, `score == Σ i·pᵢ`, and `confidence` reproduced from the
  answer's own probabilities.
* Ground truth for accuracy/calibration can be **computed in Python** (integer
  comparison, parity, substring containment, word counts) instead of human- or
  judge-labelled. Same items, every endpoint, no opinion.
* Claims in the docs ("adding questions barely changes the response time") are
  measurable, so they get measured rather than trusted.

## Quickstart

```bash
uv venv .venv && uv pip install -p .venv -e . pytest   # stdlib-only runtime; pytest for the tests
source .venv/bin/activate

pytest tests                                          # 27 tests: formulas, detection, end-to-end
systemone-probe selfcheck                             # proves the harness catches each injected fault
```

Probe a real endpoint:

```bash
export TYPESAFE_API_KEY=sk-...
systemone-probe run --base-url https://api.typesafe.ai --model jev-latest --out runs/jev
```

Probe an open Jev-like model served locally (same protocol, different base URL):

```bash
systemone-probe run --base-url http://localhost:8000 --model circuit-8b --out runs/circuit
```

No key, no network — the offline reference endpoint:

```bash
systemone-probe mock --port 8111 &                    # conformant stub
systemone-probe run --base-url http://127.0.0.1:8111 --api-key test --out runs/mock-good
```

Reports land in `runs/<name>/report.md` (human) and `report.json` (CI). Exit code
is `1` on a hard failure, `0` otherwise.

## The suites

| Suite | Question it answers | Severity |
| --- | --- | --- |
| `contract` | Answer shapes, option/level key sets, `Σp = 1`, `choice` = arg-max, `score = Σ i·pᵢ`, `confidence` matches the documented formula | hard |
| `correctness` | Accuracy against Python-computed truth, per task family; score monotonicity across ordered states | quality |
| `calibration` | Brier score, ECE, and a reliability table over the labelled noul probes | quality |
| `determinism` | Same request, repeated: identical answers, worst L1 probability drift | hard |
| `batching` | 1 question vs N questions latency (the "nearly free" claim) and **isolation** — the same question asked alone must give the same answer | hard |
| `robustness` | Renaming question ids, re-encoding the state as a JSON string, paraphrasing instructions | quality |
| `limits` | The documented error surface: 400 on non-finite numbers, 401 without a key, 422 on a malformed question | hard |

**Severity matters.** Protocol claims are hard failures. Quality thresholds are
heuristics that depend on your workload: they report by default and only gate the
exit code with `--enforce`. Defaults (`brier ≤ 0.25`, `ECE ≤ 0.20`,
accuracy ≥ 0.60, determinism ≥ 90%) are deliberately loose — they catch a broken
or dishonest model, not rank two working ones. ECE is looser than the usual 0.10
because it is noisy below ~100 labelled items.

## The offline reference endpoint (`systemone-probe mock`)

A stub that implements the wire protocol and pattern-matches the built-in probe
instructions, so the harness is testable with no key and no network. It answers
correctly at a configurable rate and, when wrong, says "I don't know" (p≈0.5) —
which makes it calibrated by construction and a fair reference.

Each fault mode violates exactly one thing, and `systemone-probe selfcheck`
asserts the harness notices:

| Mode | Injected fault | Suite that must catch it |
| --- | --- | --- |
| `badprob` | probabilities scaled so they no longer sum to 1 | `contract` |
| `badconf` | `confidence` is a constant, not a function of the probabilities | `contract` |
| `unstable` | answers change between identical requests | `determinism` |
| `leaky` | batched questions contaminate each other | `batching` |
| `slowbatch` | latency grows linearly with question count | `batching` |
| `overconfident` | always 0.99 at ~62% accuracy | `calibration` |
| `refuse` | 401 on everything | `limits` |
| `flaky` | 429 on the first call, then fine | client retry logic |

`selfcheck` is the part that makes the rest trustworthy: a test harness nobody
tests is a liability.

## Worked examples: where a System One model is the right tool

```bash
systemone-probe examples                    # the catalogue: decision, why, what your code does
systemone-probe examples --check            # validate every payload locally, no network, no cost
systemone-probe examples --show rag_gate    # the exact HTTP request, ready to paste
systemone-probe examples --run --api-key-env TYPESAFE_API_KEY --out examples/results
```

`examples/catalog.json` holds twelve complete requests. Six are the vendor's own
high-volume patterns (chat moderation, passage re-ranking, confidence-gated
classification, citation checking, RAG gating, agent tool routing); six are
problems from this machine's own projects (catalogue-record matching, metadata
triage, character-trainer grading).

Each entry carries three fields that decide whether it is a System One problem at
all:

- `why` — the property of the decision that makes a typed, calibrated answer
  better than generated text.
- `code_then` — what your code does with the numbers. If you cannot write this
  line, the decision is not a System One decision.
- `documented` — the vendor's published measurement for that pattern, so an
  example can be judged against a real number instead of an opinion.

### The test in one sentence

**A decision belongs on a System One model when a closed, enumerable answer set is
consumed by code, the same judgement is repeated many times, and you need a
probability to decide whether to act on it.** Everything else — generation,
multi-step reasoning, open-ended analysis, writing prose — stays with an LLM, or
with your own deterministic code.

Corollaries worth internalising:

* **Atomic questions, weighted in code.** "Rate this startup pitch" is not a
  question; market size, technical feasibility and differentiation are. Combine
  them yourself so priorities live in a coefficient, not a prompt.
* **Confidence is a second axis.** The answer tells you *what*, confidence tells
  you *whether to act*. Low confidence is a routing decision to a human, not a
  failure.
* **The model is not the perception layer and not the actor.** Render the state,
  ask the narrow question, keep the veto — the vendor's own drone example runs
  classical CV for the scene and Jev only for the ~2.5 Hz tactical call.

## What this harness deliberately does not do

* **No LLM-judge grading.** Nothing here evaluates text, because a System One
  model does not emit text. The prose quality of an *LLM* is out of scope.
* **No ranking.** It tells you whether an endpoint is broken, ungrounded,
  unstable, or dishonest about its confidence. It does not tell you which of two
  working models is better; that needs your workload's data.
* **The probe tasks are easy on purpose.** They measure whether a model grounds
  on the state at all. A model that fails them is broken, not merely weak.
* **`--difficulty close` is the only difficulty knob** (near-tie comparisons). A
  serious evaluation needs your domain's tasks, not these.

## Known doc/behaviour caveats worth carrying

* Vendor docs are inconsistent about modality: the System One concept page says
  "text input only … images, audio and video are not supported (yet)", while the
  Box and Images pages ship box questions on images behind a
  `TypeSafe-Danger-Zone-Images: true` header with `model: jev-preview`, labelled
  "dreamer-ready" (not production). Box answers are in this harness's contract
  checks, but no image suite exists yet.
* `score` is documented as "the probability-weighted answer"; the doc example
  confirms it is exactly `Σ i·pᵢ`, which is what the contract suite enforces.
* Confidence tolerances are ±0.02 because the vendor's own printed examples are
  rounded (their Choice example shows 0.81 where the formula gives 0.82).

## Cost and egress

A full run against a hosted endpoint is roughly `5 + N` requests (N = the
isolation sample, `--single-call-cap`, default 12) and sends your probe content
off-machine. Use `--suite contract,determinism` for a cheap smoke test, or the
`mock` for anything free.

## Layout

```
src/systemone_probe/
  client.py       # POST /v1/systemone, timing, retries, usage
  conformance.py  # contract checks, request validation, the documented formulas
  datasets.py     # mechanically-labelled probe items
  suites.py       # the seven suites
  mock.py         # offline reference endpoint + fault injection
  examples.py     # the worked-example catalogue (load, render, run)
  report.py       # markdown / JSON reports
  cli.py          # run | mock | selfcheck | examples
examples/
  catalog.json    # twelve requests: state, questions, why, code_then, documented result
tests/            # formulas, detection, dataset truth, request validation, end-to-end
```
