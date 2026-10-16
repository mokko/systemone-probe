"""Offline reference endpoint: a Jev-like System One server with fault injection.

Two jobs:

1. **Run the harness with no API key and no network.** The ``good`` mode
   implements the wire protocol and answers the probe dataset's task families
   correctly at a configurable accuracy, with confidence derived from the
   documented formulas.
2. **Prove the harness detects breakage.** Each fault mode violates exactly one
   contract and ``systemone-probe selfcheck`` asserts the matching suite fails.

It is a *stub model*, not a decision model: it pattern-matches the built-in probe
instructions. Real endpoints (TypeSafe, self-hosted circuit/von/Jev-Vision) are
tested by pointing the harness at their URL — the mock exists so the harness
itself is testable offline.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from systemone_probe.conformance import choice_confidence, score_confidence

MODES = (
    "good",         # conformant, calibrated-ish, deterministic, isolated
    "unstable",     # answers change on identical requests
    "badprob",      # choice/score probabilities do not sum to 1
    "badconf",      # confidence not a function of the probabilities
    "leaky",        # batched questions contaminate each other
    "slowbatch",    # latency grows linearly with question count
    "overconfident",  # always p=0.99, regardless of whether it is right
    "refuse",       # 401 on every request (auth behaviour probe)
    "flaky",        # 429 on the first request, then fine (retry probe)
)

DENSITY_LEVELS = ["empty", "sparse", "moderate", "dense"]


# ------------------------------------------------------------------- the fake model


def _stable_seed(state: Any, instructions: Any) -> int:
    blob = json.dumps(state, sort_keys=True, default=str) + "|" + json.dumps(instructions, sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(blob.encode()).digest()[:8], "big")


def _text_of(instructions: Any) -> str:
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, sort_keys=True, default=str)


_SCOPE_RE = re.compile(r"\s*Within `([^`]+)`:")
_DIRECT_FIELDS = ("left", "right", "number", "haystack", "needle", "text", "a", "b", "c", "word")


def _resolve_scope(state: Any, instructions: Any) -> dict[str, Any]:
    """Find the slice of the state the instruction is talking about.

    Mirrors how the probe set is built: a request carries many independent probe
    states, each nested under its own key, and the instruction names that key in
    backticks. Falls back to the top level, then to the first nested object.
    """
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return {}
    if not isinstance(state, dict):
        return {}
    text = _text_of(instructions)
    match = _SCOPE_RE.match(text)
    if match and isinstance(state.get(match.group(1)), dict):
        return state[match.group(1)]
    if any(k in state for k in _DIRECT_FIELDS):
        return state
    for value in state.values():
        if isinstance(value, dict):
            return value
    return state


def _density_level(n_words: int) -> int:
    if n_words == 0:
        return 0
    if n_words <= 3:
        return 1
    if n_words <= 8:
        return 2
    return 3


def _distribute(options: list[str], best: str, best_p: float) -> dict[str, float]:
    """Put ``best_p`` on ``best`` and spread the rest evenly."""
    others = [o for o in options if o != best]
    if not others:
        return {best: 1.0}
    rest = (1.0 - best_p) / len(others)
    return {o: (best_p if o == best else rest) for o in options}


class MockModel:
    """Answers the probe task families; wrong with probability ``1 - accuracy``."""

    def __init__(
        self,
        mode: str = "good",
        accuracy: float = 0.88,
        confident_p: float = 0.92,
        unconfident_p: float = 0.6,
        seed: int = 7,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.accuracy = accuracy
        # The "overconfident" fault is a model that stopped discriminating but
        # still claims certainty: 0.99 every time at ~62% accuracy.
        if mode == "overconfident":
            self.accuracy = 0.62
        self.confident_p = confident_p
        self.unconfident_p = unconfident_p
        self._seed = seed
        self._counter = 0
        self._lock = threading.Lock()

    # -- helpers

    def _rng(self, state: Any, instructions: Any) -> random.Random:
        if self.mode == "unstable":
            with self._lock:
                self._counter += 1
                n = self._counter
            return random.Random((time.time_ns() ^ n) & 0xFFFFFFFF)
        return random.Random(_stable_seed(state, instructions) ^ self._seed)

    def _is_right(self, rng: random.Random) -> bool:
        return rng.random() < self.accuracy

    # -- task families

    def _noul_truth(self, state: Any, instructions: Any) -> bool | None:
        text = _text_of(instructions).lower()
        scope = _resolve_scope(state, instructions)
        if not scope:
            return None
        if "greater than" in text and {"left", "right"} <= set(scope):
            return scope["left"] > scope["right"]
        if "even" in text and "number" in scope:
            return scope["number"] % 2 == 0
        if "contain" in text and {"haystack", "needle"} <= set(scope):
            return str(scope["needle"]) in str(scope["haystack"])
        if "more than" in text and {"text", "threshold"} <= set(scope):
            return len(str(scope["text"]).split()) > scope["threshold"]
        return None

    def _choice_truth(self, state: Any, question: dict[str, Any]) -> str | None:
        criteria = question.get("criteria")
        scope = _resolve_scope(state, question.get("instructions", ""))
        if not isinstance(criteria, dict) or not scope:
            return None
        text = _text_of(question.get("instructions", "")).lower()
        options = list(criteria)
        if "largest" in text and {"a", "b", "c"} <= set(scope):
            best = max(("a", "b", "c"), key=lambda k: scope[k])
            return best if best in options else None
        if "range" in text and "number" in scope:
            n = scope["number"]
            want = "below zero" if n < 0 else ("zero to nine" if n <= 9 else "ten or more")
            return want if want in options else None
        if "vowel" in text and "word" in scope:
            is_vowel = str(scope["word"])[:1].lower() in "aeiou"
            return ("yes" if is_vowel else "no") if {"yes", "no"} <= set(options) else None
        return None

    def _score_truth(self, state: Any, question: dict[str, Any]) -> int | None:
        scope = _resolve_scope(state, question.get("instructions", ""))
        if not scope or "text" not in scope:
            return None
        levels = question.get("criteria")
        if not isinstance(levels, list) or len(levels) != len(DENSITY_LEVELS):
            return None
        if [str(x).lower() for x in levels] != DENSITY_LEVELS:
            return None
        return _density_level(len(str(scope["text"]).split()))

    # -- public

    def answer(self, state: Any, qid: str, question: dict[str, Any], *, batched_qids: tuple[str, ...] = ()) -> dict[str, Any]:
        qtype = question.get("type")
        # Seed on the *resolved* scope, not the raw request: the same data sent as
        # an object or as a JSON string must produce the same answer.
        scope = _resolve_scope(state, question.get("instructions"))
        rng = self._rng(scope, question.get("instructions"))

        if qtype == "noul":
            truth = self._noul_truth(state, question.get("instructions"))
            if truth is None:
                p = round(rng.random(), 4)
            else:
                right = self._is_right(rng)
                if self.mode == "overconfident":
                    hi = 0.99
                else:
                    # Confident when right; an honest "no idea" when wrong. That
                    # keeps the stub calibrated by construction: p≈0.5 buckets
                    # contain a 50/50 truth mix, so they do not wreck ECE.
                    hi = self.confident_p if right else 0.5
                # p is P(yes): a wrong answer must flip the verdict side of 0.5.
                p = hi if (truth == right) else 1.0 - hi
                p = round(p + rng.uniform(-0.02, 0.02), 4)
                p = min(max(p, 0.0), 1.0)
            if self.mode == "leaky" and len(batched_qids) > 1 and rng.random() < 0.5:
                p = round(min(1.0, p + 0.2), 4)
            return {"type": "noul", "noul": p}

        if qtype == "choice":
            criteria = question.get("criteria") or {}
            options = list(criteria)
            truth = self._choice_truth(state, question)
            if not options:
                return {"type": "choice", "choice": "", "probabilities": {}, "confidence": 0.0}
            if truth is None:
                probs = {o: round(1.0 / len(options), 6) for o in options}
                top = max(probs, key=lambda k: probs[k])
            else:
                right = self._is_right(rng)
                chosen = truth if right else rng.choice([o for o in options if o != truth] or [truth])
                best_p = self.confident_p if self.mode != "overconfident" else 0.99
                probs = {o: round(v, 6) for o, v in _distribute(options, chosen, best_p).items()}
                top = chosen
            if self.mode == "badprob":
                probs = {o: round(v * 1.08, 6) for o, v in probs.items()}
            conf = (
                0.99 if self.mode == "badconf"
                else round(choice_confidence([probs[o] for o in options]), 4) if options
                else 0.5
            )
            return {"type": "choice", "choice": top, "probabilities": probs, "confidence": conf}

        if qtype == "score":
            levels = question.get("criteria") or []
            n = len(levels)
            truth = self._score_truth(state, question)
            if truth is None or n < 2:
                weights = [round(1.0 / max(n, 1), 6)] * n
                probs = {str(i): weights[i] for i in range(n)}
            else:
                right = self._is_right(rng)
                level = truth if right else rng.choice([i for i in range(n) if i != truth] or [truth])
                remaining = [i for i in range(n) if i != level]
                tail = 0.08 / max(len(remaining), 1)
                probs = {
                    str(i): round((1.0 - 0.08) if i == level else tail, 6) for i in range(n)
                }
            if self.mode == "badprob":
                probs = {k: round(v * 1.08, 6) for k, v in probs.items()}
            order = sorted(probs, key=int)
            score = round(sum(int(k) * probs[k] for k in order), 4)
            conf = (
                0.99 if self.mode == "badconf"
                else round(score_confidence([probs[k] for k in order]), 4)
            )
            legend = {str(i): str(levels[i]) for i in range(n)} if levels else {}
            return {
                "type": "score",
                "score": score,
                "legend": legend,
                "probabilities": probs,
                "confidence": conf,
            }

        return {"type": str(qtype), "error": f"unsupported question type {qtype!r}"}


# ------------------------------------------------------------------------ server


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockSystemOne/0.1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence stderr spam
        return

    @property
    def model(self) -> MockModel:
        return self.server.model  # type: ignore[attr-defined]

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {"data": [{"id": "mock-jev-1.0", "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        model: MockModel = self.model

        if model.mode == "refuse" or not self.headers.get("Authorization"):
            self._send(401, {"error": {"message": "missing or invalid API key"}})
            return

        # Docs: "Finite numbers only. NaN, Infinity ... are rejected." Mirror it.
        if any(tok in raw for tok in ("NaN", "Infinity", "-Infinity")):
            self._send(400, {"error": {"message": "numbers must be finite"}})
            return

        if model.mode == "flaky":
            with model._lock:  # noqa: SLF001 - deliberate one-shot fault injection
                first = model._counter == 0
                model._counter += 1
            if first:
                self._send(429, {"error": {"message": "rate limited"}})
                return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send(400, {"error": {"message": f"invalid JSON: {exc}"}})
            return

        state = payload.get("state")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            self._send(422, {"error": {"message": "questions must be a non-empty map"}})
            return
        for qid, q in questions.items():
            if not isinstance(q, dict) or "type" not in q or "instructions" not in q:
                self._send(422, {"error": {"message": f"question {qid!r} needs type and instructions"}})
                return
            if q["type"] == "score" and not isinstance(q.get("criteria"), list):
                self._send(422, {"error": {"message": f"score question {qid!r} needs a criteria array"}})
                return
            if q["type"] == "choice" and not isinstance(q.get("criteria"), dict):
                self._send(422, {"error": {"message": f"choice question {qid!r} needs a criteria map"}})
                return

        if model.mode == "slowbatch":
            time.sleep(0.05 * len(questions))
        else:
            time.sleep(0.01)

        qids = tuple(questions)
        answers = {
            qid: model.answer(state, qid, q, batched_qids=qids)
            for qid, q in questions.items()
        }
        self._send(
            200,
            {
                "model": "mock-jev-1.0",
                "answers": answers,
                "usage": {"input_tokens": max(1, len(raw) // 4), "output_tokens": 8 * len(questions)},
            },
        )


class MockServer:
    """Context manager wrapping a threaded HTTP server in a background thread."""

    def __init__(self, mode: str = "good", host: str = "127.0.0.1", port: int = 0, **model_kwargs: Any) -> None:
        self.model = MockModel(mode=mode, **model_kwargs)
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.model = self.model  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.host, self.port = self._server.server_address[0], self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "MockServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "MockServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve_forever(mode: str = "good", host: str = "127.0.0.1", port: int = 8111) -> None:
    """Blocking variant used by ``systemone-probe mock``."""
    server = MockServer(mode=mode, host=host, port=port)
    print(f"mock System One endpoint ({mode}) on {server.base_url}/v1/systemone", flush=True)
    try:
        server._thread.start()  # noqa: SLF001 - long-lived CLI process
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:  # pragma: no cover
        server.stop()
