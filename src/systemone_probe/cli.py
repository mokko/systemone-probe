"""CLI: ``systemone-probe run``, ``mock``, and ``selfcheck``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from systemone_probe import report as report_mod
from systemone_probe.client import DEFAULT_BASE_URL, SystemOneClient
from systemone_probe.datasets import build_dataset
from systemone_probe.mock import MODES, MockServer, serve_forever
from systemone_probe.suites import FAIL, QUALITY_SUITES, SUITES, run_all

OUT_DIR_DEFAULT = Path("runs")


def _build_client(args: argparse.Namespace) -> SystemOneClient:
    api_key = args.api_key
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    return SystemOneClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
    )


def cmd_run(args: argparse.Namespace) -> int:
    client = _build_client(args)
    dataset = build_dataset(seed=args.seed, per_group=args.per_group, difficulty=args.difficulty)
    names = [s.strip() for s in args.suite.split(",") if s.strip()] if args.suite else list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}; known: {', '.join(SUITES)}", file=sys.stderr)
        return 2

    if args.api_key is None and args.api_key_env and not os.environ.get(args.api_key_env):
        print(f"warning: ${args.api_key_env} is empty — the endpoint will likely answer 401", file=sys.stderr)

    thresholds = None
    if args.enforce:
        thresholds = None  # defaults already live in DEFAULT_THRESHOLDS

    results = run_all(
        client,
        dataset,
        names=names,
        thresholds=thresholds,
        repeat=args.repeat,
        single_call_cap=args.single_call_cap,
    )

    # Quality suites report but do not gate unless --enforce.
    if not args.enforce:
        for r in results:
            if r.name in QUALITY_SUITES and r.status == FAIL:
                r.notes.append("quality suite failed but --enforce was not set: advisory only")
                r.status = "warn"

    extra = {
        "seed": args.seed,
        "per_group": args.per_group,
        "difficulty": args.difficulty,
        "repeat": args.repeat,
        "single_call_cap": args.single_call_cap,
    }
    overall = report_mod.overall_status(results, enforce=args.enforce)

    stamp = report_mod.to_json(
        results, target=client.base_url, model=client.model,
        dataset_size=dataset.size, enforce=args.enforce, extra=extra,
    )
    md = report_mod.to_markdown(
        results, target=client.base_url, model=client.model,
        dataset_size=dataset.size, enforce=args.enforce, extra=extra,
    )

    print(report_mod.summary_line(results))
    for suite, check, detail in report_mod.failures_only(results):
        print(f"  FAIL {suite}.{check}: {detail}")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(report_mod.json.dumps(stamp, indent=2, sort_keys=True, default=str))
        (out_dir / "report.md").write_text(md)
        print(f"wrote {out_dir/'report.md'} and {out_dir/'report.json'}")

    if args.print_markdown:
        print(md)

    return 1 if overall == FAIL else 0


def cmd_mock(args: argparse.Namespace) -> int:
    serve_forever(mode=args.mode, host=args.host, port=args.port)
    return 0


# ------------------------------------------------------------------ selfcheck

#: mode -> (suite that must fail, or None if the mode must produce a clean run)
SELFCHECK_EXPECTATIONS: list[tuple[str, str | None, str]] = [
    ("good", None, "conformant stub passes every suite"),
    ("badprob", "contract", "probabilities that do not sum to 1 are caught"),
    ("badconf", "contract", "confidence not derived from probabilities is caught"),
    ("unstable", "determinism", "unstable answers are caught"),
    ("leaky", "batching", "cross-question contamination is caught"),
    ("slowbatch", "batching", "batch latency growing with question count is caught"),
    ("overconfident", "calibration", "an always-0.99 model fails calibration when enforced"),
    ("refuse", "limits", "a 401-only endpoint is caught by the limits suite"),
]


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Run the harness against the mock in every mode and assert it notices."""
    dataset = build_dataset(seed=args.seed, per_group=6)
    rows: list[tuple[str, str, str, str]] = []
    ok = True
    print(f"self-check: {len(SELFCHECK_EXPECTATIONS)} modes × suites {','.join(SUITES)}\n")
    for mode, must_fail, why in SELFCHECK_EXPECTATIONS:
        with MockServer(mode=mode) as server:
            client = SystemOneClient(base_url=server.base_url, api_key="test-key", retries=2, backoff=0.05)
            results = run_all(
                client, dataset,
                names=list(SUITES),
                repeat=3,
                single_call_cap=4,
            )
        failed = {r.name for r in results if r.status == FAIL}
        if must_fail is None:
            good = not failed
            detail = "clean run" if good else f"unexpected failures: {sorted(failed)}"
        else:
            good = must_fail in failed
            detail = (
                f"{must_fail} failed as expected"
                if good
                else f"{must_fail} did NOT fail (failed={sorted(failed) or 'none'})"
            )
        rows.append((mode, "ok" if good else "MISBEHAVED", must_fail or "-", detail))
        if not good:
            ok = False
        print(f"[{'ok' if good else 'XX'}] {mode:14s} expect fail={must_fail or '-':12s} {detail}")

    # The retry path needs its own assertion: a 429 on the first call must be retried.
    with MockServer(mode="flaky") as server:
        client = SystemOneClient(base_url=server.base_url, api_key="test-key", retries=2, backoff=0.05)
        r = client.evaluate({"left": 5, "right": 3},
                            {"cmp": {"type": "noul", "instructions": "Is `left` greater than `right`?"}})
    retry_ok = r.ok and r.attempts >= 2
    print(f"[{'ok' if retry_ok else 'XX'}] flaky          expect retry         "
          f"status={r.status} attempts={r.attempts}")
    ok = ok and retry_ok
    rows.append(("flaky", "ok" if retry_ok else "MISBEHAVED", "retry", f"attempts={r.attempts}"))

    print("\nself-check:", "all expectations met" if ok else "HARNESS MISBEHAVED")
    return 0 if ok else 1


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="systemone-probe",
        description="Conformance and behaviour harness for Jev / System One decision models.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="probe a live System One endpoint")
    run.add_argument("--base-url", default=os.environ.get("SYSTEMONE_BASE_URL", DEFAULT_BASE_URL))
    run.add_argument("--model", default=os.environ.get("SYSTEMONE_MODEL", "jev-latest"))
    run.add_argument("--api-key", default=None, help="API key (prefer --api-key-env)")
    run.add_argument("--api-key-env", default="TYPESAFE_API_KEY", help="env var holding the key")
    run.add_argument("--suite", default=None, help=f"comma-separated subset of: {','.join(SUITES)}")
    run.add_argument("--out", default=str(OUT_DIR_DEFAULT), help="directory for report.md/report.json")
    run.add_argument("--no-out", action="store_const", const="", dest="out", help="do not write files")
    run.add_argument("--print-markdown", action="store_true")
    run.add_argument("--enforce", action="store_true",
                     help="let quality suites (accuracy/calibration/robustness) gate the exit code")
    run.add_argument("--repeat", type=int, default=3, help="repeats for determinism/batching suites")
    run.add_argument("--single-call-cap", type=int, default=12,
                     help="how many questions to re-ask individually for the isolation check")
    run.add_argument("--per-group", type=int, default=6, help="noul items per task family")
    run.add_argument("--difficulty", choices=["mixed", "close"], default="mixed",
                     help="'close' narrows the margin on comparison items")
    run.add_argument("--seed", type=int, default=20261016)
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--retries", type=int, default=2)
    run.set_defaults(func=cmd_run)

    mock = sub.add_parser("mock", help="serve an offline Jev-like stub")
    mock.add_argument("--mode", choices=MODES, default="good")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, default=8111)
    mock.set_defaults(func=cmd_mock)

    self_check = sub.add_parser("selfcheck", help="verify the harness detects each injected fault")
    self_check.add_argument("--seed", type=int, default=20261016)
    self_check.set_defaults(func=cmd_selfcheck)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
