"""systemone-probe: a conformance and behavior test harness for System One models.

A *System One* model (Jev and the open Jev-like models that speak the same wire
protocol) takes a ``state`` plus typed ``questions`` and returns typed ``answers``
with probabilities. ``POST /v1/systemone`` is the whole surface area.

This package tests any endpoint that claims to speak that contract:

    from systemone_probe import SystemOneClient
    c = SystemOneClient(base_url="http://localhost:8000")
    r = c.evaluate("Help! My payouts failed.", {"u": {"type": "noul",
                  "instructions": "Does this convey urgency?"}})
    r.answers["u"]["noul"]
"""

from systemone_probe.client import (
    DEFAULT_BASE_URL,
    CallResult,
    SystemOneClient,
    SystemOneError,
)
from systemone_probe.conformance import (
    choice_confidence,
    check_answer,
    noul_confidence,
    score_confidence,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "CallResult",
    "SystemOneClient",
    "SystemOneError",
    "check_answer",
    "choice_confidence",
    "score_confidence",
    "noul_confidence",
]

__version__ = "0.1.0"
