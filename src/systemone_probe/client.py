"""HTTP client for the System One evaluation endpoint (``POST /v1/systemone``).

Deliberately stdlib-only: the harness has to be able to point at a local mock, a
self-hosted Jev-like model, or the hosted API without dragging a dependency tree
along. Timing is measured here because latency claims ("adding questions barely
changes response time") are part of what the harness checks.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"

#: Statuses worth retrying with backoff (docs: 429 rate limit, 529 overloaded).
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})

IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"


class SystemOneError(RuntimeError):
    """Raised for transport-level failures (connection, TLS, timeout)."""


@dataclass
class CallResult:
    """One evaluation round-trip, successful or not."""

    status: int
    latency_ms: float
    total_ms: float = 0.0
    attempts: int = 1
    response: dict[str, Any] | None = None
    body_text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.response is not None

    @property
    def answers(self) -> dict[str, Any]:
        if not self.response:
            return {}
        return self.response.get("answers") or {}

    @property
    def usage(self) -> dict[str, Any]:
        if not self.response:
            return {}
        return self.response.get("usage") or {}

    @property
    def model(self) -> str | None:
        if not self.response:
            return None
        return self.response.get("model")

    def answer(self, qid: str) -> dict[str, Any] | None:
        return self.answers.get(qid)


@dataclass
class SystemOneClient:
    """Minimal client for a System One (Jev-like) endpoint."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    model: str = "jev-latest"
    timeout: float = 60.0
    retries: int = 2
    backoff: float = 0.5
    extra_headers: dict[str, str] = field(default_factory=dict)
    last_request: dict[str, Any] | None = field(default=None, init=False)

    def url(self, path: str = SYSTEMONE_PATH) -> str:
        return self.base_url.rstrip("/") + path

    def evaluate(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        model: str | None = None,
        raw_body: bytes | str | None = None,
        extra_headers: dict[str, str] | None = None,
        allow_nan: bool = False,
    ) -> CallResult:
        """Send one evaluation request.

        ``raw_body`` bypasses payload construction so the limits suite can send
        bodies that ``json.dumps`` would refuse (``NaN``, malformed questions).
        """
        if raw_body is not None:
            body = raw_body.encode() if isinstance(raw_body, str) else raw_body
        else:
            payload = {
                "state": state,
                "model": model or self.model,
                "questions": questions,
            }
            self.last_request = payload
            body = json.dumps(payload, allow_nan=allow_nan).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "systemone-probe/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        if extra_headers:
            headers.update(extra_headers)

        return self._post(body, headers)

    def models(self) -> CallResult:
        return self._post(b"", {"Accept": "application/json"}, method="GET",
                          url=self.url(MODELS_PATH))

    # ------------------------------------------------------------------ internals

    def _post(
        self,
        body: bytes,
        headers: dict[str, str],
        *,
        method: str = "POST",
        url: str | None = None,
    ) -> CallResult:
        target = url or self.url()
        started = time.perf_counter()
        attempt = 0
        last_error: str | None = None
        latency_ms = 0.0

        while attempt <= self.retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target, data=body if method == "POST" else None,
                    headers=headers, method=method,
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    latency_ms = (time.perf_counter() - t0) * 1000
                    status = resp.status
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                latency_ms = (time.perf_counter() - t0) * 1000
                status = exc.code
                if status in RETRYABLE_STATUS and attempt <= self.retries:
                    time.sleep(self.backoff * (2 ** (attempt - 1)))
                    continue
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.retries:
                    time.sleep(self.backoff * (2 ** (attempt - 1)))
                    continue
                return CallResult(
                    status=0, latency_ms=latency_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                    attempts=attempt, error=last_error,
                )

            parsed: dict[str, Any] | None = None
            if raw.strip():
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    pass

            if status in RETRYABLE_STATUS and attempt <= self.retries:
                time.sleep(self.backoff * (2 ** (attempt - 1)))
                continue

            return CallResult(
                status=status,
                latency_ms=latency_ms,
                total_ms=(time.perf_counter() - started) * 1000,
                attempts=attempt,
                response=parsed,
                body_text=raw,
                error=last_error,
            )

        return CallResult(
            status=0, latency_ms=latency_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            attempts=attempt, error=last_error or "exhausted retries",
        )
