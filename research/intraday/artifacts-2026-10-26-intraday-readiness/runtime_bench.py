"""Offline year-size runtime benchmark for the intraday observer (synthetic data, no network).

Drives every registered session through the timer wrapper's dispatch (``run_job``: recover,
inputs, decide, archive) with a simulated clock and a fresh Observer per stage, as a real
timer-launched process would, timing real wall clock per stage including durable writes. Then
measures the final session's decision cold (new interpreter) on a copy of the full-year state.
Bounded: ``--sessions`` is capped at 252 and nothing touches the network or credentials.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.lab import intraday_observer as obs  # noqa: E402

STAGE_TIMES = (
    ("recover", "15:15:05"),
    ("inputs", "15:20:05"),
    ("decide", "15:25:05"),
    ("archive", "16:20:05"),
)


class RealisticTransport(obs.SyntheticTransport):
    """Synthetic bars shaped like Alpaca's: compact JSON with the n and vw fields per bar."""

    def get(self, url: str, params: dict) -> obs.Response:
        reply = super().get(url, params)
        body = json.loads(reply.body)
        for bar in body["bars"]["SPY"]:
            bar.update(n=1234, vw=round(bar["c"], 4))
        return obs.Response(200, {}, json.dumps(body, separators=(",", ":")).encode())


COLD = """
import sys
sys.path.insert(0, {src!r})
from ggTrader.lab import intraday_observer as obs
clock = obs.SimClock(obs.utc_at({day!r}, "15:25:00"))
deps = obs.Deps(clock=clock, sleep=clock.sleep, transport=obs.SyntheticTransport(),
                clock_sync=lambda: obs.CLOCK_STATUS_OK)
rec = obs.run_job(obs.Observer({state!r}, deps), "decide")
print(rec["status"], rec["faults"])
"""


def prepare(work: Path, activation: str) -> tuple[Path, obs.SimClock, obs.Deps, dict]:
    (work / "inputs").mkdir(parents=True)
    protocol = work / "inputs" / "protocol_stub.txt"
    protocol.write_text("synthetic benchmark protocol stub\n")
    snapshot = work / "inputs" / "official_calendar.json"
    after = date.fromisoformat(activation)
    snapshot.write_text(
        json.dumps(
            obs.offline_sessions(
                activation, (after + timedelta(days=obs.LOOKAHEAD_DAYS + 40)).isoformat()
            )
        )
    )
    clock = obs.SimClock(obs.utc_at(activation, "09:00:00"))
    state = work / "state"
    manifest = obs.init_state(
        state,
        mode="rehearsal",
        activation_date=activation,
        official_calendar_file=snapshot,
        protocol_file=protocol,
        protocol_sha256=obs.file_sha256(protocol),
        source_delay_seconds=900,
        now=clock(),
    )
    cal = manifest["calendar"]
    early = {r["date"]: r["close"] for r in cal["sessions"] if r["kind"] == "early_close"}
    deps = obs.Deps(
        clock=clock,
        sleep=clock.sleep,
        transport=RealisticTransport(obs.RehearsalPrices(cal["sessions"]), early),
        clock_sync=lambda: obs.CLOCK_STATUS_OK,
    )
    clock.set(activation, "10:00:00")
    obs.Observer(state, deps).activate("benchmark rehearsal; synthetic data; not an owner approval")
    return state, clock, deps, cal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=252, help="registered slots to run (<=252)")
    parser.add_argument("--cold-runs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path.home() / ".cache")
    parser.add_argument("--activation-date", default="2026-11-23")
    args = parser.parse_args()
    if not 1 <= args.sessions <= obs.SLOTS:
        parser.error(f"--sessions must be in 1..{obs.SLOTS}")
    work = Path(tempfile.mkdtemp(prefix="ggt-observer-bench-", dir=args.work_root))
    try:
        state, clock, deps, cal = prepare(work / "w", args.activation_date)
        timings: list[dict] = []
        last_day = next(r["date"] for r in cal["sessions"] if r["slot"] == args.sessions)
        rows = [r for r in cal["sessions"] if r["date"] <= last_day]
        started = time.perf_counter()
        for row in rows:
            day = row["date"]
            for stage, text in STAGE_TIMES:
                if day == last_day and stage in ("decide", "archive"):
                    break  # the final decision is measured separately, cold and warm
                clock.set(day, text)
                t0 = time.perf_counter()
                ob = obs.Observer(state, deps)  # a fresh process re-validates and re-reads
                init = time.perf_counter() - t0
                result = obs.run_job(ob, stage)
                timings.append(
                    {
                        "day": day,
                        "slot": row["slot"],
                        "stage": stage,
                        "noop": bool(result.get("noop")),
                        "init_s": init,
                        "total_s": time.perf_counter() - t0,
                    }
                )
        built_s = time.perf_counter() - started
        records = sorted((state / "records").glob("0*.json"))
        sizes = {
            "records": len(records),
            "record_bytes": sum(p.stat().st_size for p in records),
            "raw_files": len(list((state / "raw").iterdir())),
            "raw_bytes": sum(p.stat().st_size for p in (state / "raw").iterdir()),
            "budget_files": len(list((state / "budget").glob("*.json"))),
        }
        snap = work / "snapshot"
        shutil.copytree(state, snap)
        warm, cold = [], []
        for _ in range(3):
            shutil.rmtree(state)
            shutil.copytree(snap, state)
            clock.set(last_day, "15:25:00")
            t0 = time.perf_counter()
            body = obs.run_job(obs.Observer(state, deps), "decide")
            warm.append(time.perf_counter() - t0)
            assert body["status"] == "ok", body
        for _ in range(args.cold_runs):
            shutil.rmtree(state)
            shutil.copytree(snap, state)
            code = COLD.format(src=str(ROOT / "src"), day=last_day, state=str(state))
            t0 = time.perf_counter()
            done = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            )
            cold.append(time.perf_counter() - t0)
            assert done.stdout.split()[0] == "ok", done.stdout + done.stderr
        shutil.rmtree(state)
        shutil.copytree(snap, state)
        clock.set(last_day, "15:25:00")
        obs.run_job(obs.Observer(state, deps), "decide")
        clock.set(last_day, "17:00:00")
        ob = obs.Observer(state, deps)
        t0 = time.perf_counter()
        problems = ob.verify_chain()
        verify_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ob.report()
        report_s = time.perf_counter() - t0
        by_stage: dict[str, list[float]] = {}
        for row in timings:
            if not row["noop"]:
                by_stage.setdefault(row["stage"], []).append(row["total_s"])
        report = {
            "schema": "intraday_observer_runtime_bench",
            "sessions": args.sessions,
            "activation_date": args.activation_date,
            "last_session": last_day,
            "host": {"python": platform.python_version(), "machine": platform.machine()},
            "build_seconds": built_s,
            "state_size_before_final_decision": sizes,
            "decision_window_seconds": 60,
            "verify_chain_seconds_full_year": verify_s,
            "verify_chain_problems": problems,
            "report_seconds_full_year": report_s,
            "final_decision_warm_seconds": warm,
            "final_decision_cold_process_seconds": cold,
            "stage_seconds_over_whole_build": {
                k: {
                    "n": len(v),
                    "median": statistics.median(v),
                    "p95": sorted(v)[int(0.95 * (len(v) - 1))],
                    "max": max(v),
                    "last": v[-1],
                }
                for k, v in by_stage.items()
            },
            "decide_curve": [
                r["total_s"] for r in timings if r["stage"] == "decide" and not r["noop"]
            ][:: max(1, args.sessions // 12)],
        }
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "decide_curve"}, indent=1))
        print("decide curve", [round(x, 3) for x in report["decide_curve"]])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
