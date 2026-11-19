"""Backtest harness for forecast #2 (dev tool, not part of the entry).

Runs main.py as a subprocess on constructed cases and scores the
outputs with dev_scorer2 against the two coordinator snapshots:

  A  p01 as-is (flat, 90-min horizon)        vs answer_1043 -> 0.1547 / 0.96
  B  synthetic non-flat (03:00 -> 10:43)     vs answer_2335 -> 0.1778 / 0.95
  F  p01 retargeted to 23:35 (1205 min)      vs answer_2335 -> 0.3084 / 0.93

Plus the three synthetic checks (refresh jump, rising earners, flat
history vs house baseline) and the format checks on p00/p01.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import dev_scorer2 as scorer

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.py"


def run_model(cases):
    """cases: {stem: case dict} -> {stem: (rows_by_id, raw_bytes, seconds)}."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        for stem, case in cases.items():
            (in_dir / f"{stem}.in").write_text(json.dumps(case), encoding="utf-8")
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = str(in_dir)
        env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
        start = time.time()
        proc = subprocess.run(
            [sys.executable, str(MAIN)], env=env, capture_output=True, text=True
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(f"main.py failed: {proc.stderr}")
        for stem in cases:
            raw = (out_dir / f"{stem}.out").read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            rows = {row["id"]: row["q"] for row in payload["trophies"]}
            results[stem] = (rows, raw, elapsed)
    return results


def build_case_b(p01, answer_1043):
    """Two-read non-flat history: launch page, then the 10:43 refresh."""
    launch_rates = p01["history"][-1]["rates"]
    case = copy.deepcopy(p01)
    case["id"] = "b01"
    case["at"] = "2026-11-19T10:43:00Z"
    case["target"] = "2026-11-19T23:35:00Z"
    case["horizon_minutes"] = 772
    case["history"] = [
        {
            "slot": "2026-11-19T03:00:00Z",
            "at": "2026-11-19T03:00:00Z",
            "players_tracked": 638,
            "trophies_earned": 943,
            "rates": list(launch_rates),
        },
        {
            "slot": "2026-11-19T10:43:00Z",
            "at": "2026-11-19T10:43:00Z",
            "players_tracked": 3868,
            "trophies_earned": 0,
            "rates": list(answer_1043),
        },
    ]
    return case


def build_case_f(p01):
    """p01's flat history with a long horizon to the 23:35 snapshot."""
    case = copy.deepcopy(p01)
    case["id"] = "f01"
    case["target"] = "2026-11-19T23:35:00Z"
    case["horizon_minutes"] = 1205
    return case


def synthetic_read(at, players, rates):
    return {
        "slot": at,
        "at": at,
        "players_tracked": players,
        "trophies_earned": 0,
        "rates": list(rates),
    }


def build_refresh_case(p01):
    """E flat, N jumps 6x in the last read (rates divided by 6)."""
    base = p01["history"][-1]["rates"]
    case = copy.deepcopy(p01)
    case["id"] = "r01"
    case["at"] = "2026-11-19T03:00:00Z"
    case["target"] = "2026-11-19T04:30:00Z"
    case["horizon_minutes"] = 90
    case["history"] = [
        synthetic_read("2026-11-19T00:00:00Z", 638, base),
        synthetic_read("2026-11-19T01:00:00Z", 638, base),
        synthetic_read("2026-11-19T02:00:00Z", 638, base),
        synthetic_read("2026-11-19T03:00:00Z", 3828, [r / 6.0 for r in base]),
    ]
    return case


def build_rising_case(p01):
    """N flat, every rate (hence earners) rising steadily."""
    base = p01["history"][-1]["rates"]
    case = copy.deepcopy(p01)
    case["id"] = "u01"
    case["at"] = "2026-11-19T03:00:00Z"
    case["target"] = "2026-11-19T04:30:00Z"
    case["horizon_minutes"] = 90
    factors = [1.0, 1.4, 1.8, 2.2]
    stamps = [
        "2026-11-19T00:00:00Z",
        "2026-11-19T01:00:00Z",
        "2026-11-19T02:00:00Z",
        "2026-11-19T03:00:00Z",
    ]
    case["history"] = [
        synthetic_read(ts, 638, [min(r * f, 100.0) for r in base])
        for ts, f in zip(stamps, factors)
    ]
    return case


def report(tag, rows, answers, want_loss, want_cov):
    loss, miss, cov = scorer.score_rows(rows, answers)
    print(
        f"{tag}: loss={loss:.4f} (want {want_loss})  "
        f"miss={miss:.4f}pp  cov={cov:.2f} (want {want_cov})  "
        f"d_loss={loss - want_loss:+.4f}"
    )
    return loss, cov


def main():
    p01 = scorer.load_case(scorer.KIT_INPUT / "p01.in")
    p00 = scorer.load_case(scorer.KIT_INPUT / "p00.in")
    ans43 = scorer.load_answer("answer_1043.json")
    ans35 = scorer.load_answer("answer_2335.json")

    print("== A / B / F backtests ==")
    out = run_model(
        {
            "a01": p01,
            "b01": build_case_b(p01, ans43),
            "f01": build_case_f(p01),
        }
    )
    report("A p01 vs 1043 ", out["a01"][0], ans43, 0.1547, 0.96)
    report("B synth vs 2335", out["b01"][0], ans35, 0.1778, 0.95)
    report("F long vs 2335 ", out["f01"][0], ans35, 0.3084, 0.93)

    print("\n== synthetic checks ==")
    syn = run_model(
        {"r01": build_refresh_case(p01), "u01": build_rising_case(p01)}
    )
    base = p01["history"][-1]["rates"]
    refresh_rows = syn["r01"][0]
    ratios = [
        refresh_rows[i + 1][2] / (base[i] / 6.0)
        for i in range(55)
        if base[i] > 0
    ]
    print(
        f"refresh jump: mean median/last-rate = {sum(ratios)/len(ratios):.3f} "
        f"(want well below 1)  max = {max(ratios):.3f}"
    )
    rising_rows = syn["u01"][0]
    last_rising = [min(r * 2.2, 100.0) for r in base]
    ratios_up = [
        rising_rows[i + 1][2] / last_rising[i]
        for i in range(55)
        if last_rising[i] < 99.99
    ]
    print(
        f"rising earners: mean median/last-rate = {sum(ratios_up)/len(ratios_up):.3f} "
        f"(want above 1)  min = {min(ratios_up):.3f}"
    )
    house = scorer.house_baseline_rows(p01)
    flat_rows = out["a01"][0]
    moved = sum(1 for i in range(1, 56) if abs(flat_rows[i][2] - house[i][2]) > 1e-9)
    # Spread comparison on the logit scale (where the scorer lives):
    # in raw pp the diluted medians are so small that pp-width shrinks
    # even though the logit spread is much wider than the house's.
    def logit_width(q):
        return scorer.logit_of_rate(q[4]) - scorer.logit_of_rate(q[0])

    wider = sum(1 for i in range(1, 56) if logit_width(flat_rows[i]) > logit_width(house[i]))
    print(
        f"flat history: medians moved {moved}/55 (id 2 pinned at 99.99 by design), "
        f"logit spreads wider {wider}/55 (want 55/55)"
    )

    print("\n== format checks (p00 + p01, two runs) ==")
    first = run_model({"p00": p00, "p01": p01})
    second = run_model({"p00": p00, "p01": p01})
    for stem in ("p00", "p01"):
        rows, raw, secs = first[stem]
        raw2 = second[stem][1]
        ids = sorted(rows)
        monotone = all(
            all(q[k] <= q[k + 1] for k in range(4)) for q in rows.values()
        )
        in_range = all(0.0 <= v <= 100.0 for q in rows.values() for v in q)
        print(
            f"{stem}: rows={len(rows)} ids_ok={ids == list(range(1, 56))} "
            f"monotone={monotone} in_range={in_range} bytes={len(raw)} "
            f"identical={raw == raw2} batch_time={secs:.2f}s"
        )


if __name__ == "__main__":
    main()
