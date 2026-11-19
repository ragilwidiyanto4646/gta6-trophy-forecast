#!/usr/bin/env python3
"""Dev-only backtest for the trophy forecast entry.

- Runs main.py on p00/p01 and scores it against answer_1043.json with
  the dev scorer (which reproduced house baseline A at 1.4022).
- Synthetic checks: refresh jump (median must fall), rising earners
  (median must rise), flat history (must NOT equal the house baseline).
- Format checks: 55 rows, non-decreasing, <=64KB, byte-identical reruns.
Not part of the entry; never committed.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = "/home/hatch/workspace/makefun-account/gta6-trophy-rarity-forecast-01-kit"
WORK = "/home/hatch/workspace/bounties/gta6-trophy-forecast"
sys.path.insert(0, HERE)
import dev_score

spec = importlib.util.spec_from_file_location("entry", os.path.join(HERE, "main.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def run_entry(input_dir, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=input_dir,
               MAKEFUN_OUTPUT_DIR=out_dir)
    r = subprocess.run([sys.executable, os.path.join(HERE, "main.py")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return {f[:-4]: json.load(open(os.path.join(out_dir, f)))
            for f in sorted(os.listdir(out_dir))}


def as_map(forecast):
    return {row["id"]: row["q"] for row in forecast["trophies"]}


def format_checks(forecast, raw_bytes, label):
    rows = forecast["trophies"]
    assert len(rows) == 55, f"{label}: {len(rows)} rows"
    assert [r["id"] for r in rows] == list(range(1, 56)), f"{label}: ids"
    for r in rows:
        q = r["q"]
        assert len(q) == 5 and all(q[i] <= q[i + 1] for i in range(4)), \
            f"{label}: quantiles not non-decreasing for id {r['id']}"
        assert all(isinstance(v, (int, float)) and 0 <= v <= 100 for v in q), \
            f"{label}: quantile out of range for id {r['id']}"
    assert len(raw_bytes) <= 65536, f"{label}: {len(raw_bytes)} bytes"
    print(f"{label}: format OK ({len(raw_bytes)} bytes)")


def main():
    answer = json.load(open(os.path.join(WORK, "answer_1043.json")))
    rates = answer["rates"]
    in_dir = os.path.join(KIT, "input")
    out1 = run_entry(in_dir, "/tmp/trophy-out-1")
    out2 = run_entry(in_dir, "/tmp/trophy-out-2")
    for cid in ("p00", "p01"):
        raw1 = open(f"/tmp/trophy-out-1/{cid}.out", "rb").read()
        raw2 = open(f"/tmp/trophy-out-2/{cid}.out", "rb").read()
        assert raw1 == raw2, f"{cid}: runs differ!"
        format_checks(out1[cid], raw1, cid)
        loss, miss, cov = dev_score.score_forecast(as_map(out1[cid]), rates)
        print(f"{cid} MODEL: loss={loss:.4f} miss={miss:.4f}pp cov90={cov:.2f}")
        # House baseline A for comparison, same answer horizon.
        case = json.load(open(os.path.join(in_dir, cid + ".in")))
        mins = (dev_score.parse_ts(answer["at"])
                - dev_score.parse_ts(case["at"])).total_seconds() / 60.0
        for name, fc in (("A", dev_score.baseline_a(case, mins)),
                         ("B", dev_score.baseline_b(case, mins))):
            l2, m2, c2 = dev_score.score_forecast(fc, rates)
            print(f"{cid} baseline {name}: loss={l2:.4f} miss={m2:.4f}pp cov90={c2:.2f}")

    # ---- Synthetic checks ----
    base = json.load(open(os.path.join(in_dir, "p01.in")))
    last_rates = base["history"][-1]["rates"]

    def synth(mod_history, label, check):
        case = json.loads(json.dumps(base))
        case["history"] = mod_history
        fc = entry._forecast_case(case)
        m = as_map(fc)
        check(m)
        print(f"synthetic {label}: OK")

    # (a) Refresh jump: N jumps 6x in the last read, rates fall 6x, E flat.
    # Median must fall well below the last rate.
    import copy
    hist = copy.deepcopy(base["history"])
    jumped = copy.deepcopy(hist[-1])
    jumped["players_tracked"] = 638 * 6
    jumped["rates"] = [round(r / 6, 2) if r is not None else None
                       for r in last_rates]
    hist.append(jumped)
    def check_refresh(m):
        med = m[1][2]
        assert med < last_rates[0] * 0.5, f"refresh: median {med} not below half of {last_rates[0]}"
        assert med > 0.01, "refresh: median collapsed to floor"
    synth(hist, "refresh-jump", check_refresh)

    # (b) Rising earners: N flat, rates rising steadily each read.
    hist2 = []
    for k, read in enumerate(base["history"]):
        r2 = copy.deepcopy(read)
        f = 1.0 + 0.10 * k
        r2["rates"] = [round(min(r * f, 99.0), 2) if r is not None else None
                       for r in last_rates]
        hist2.append(r2)
    def check_growth(m):
        med = m[20][2]  # fishing trophy, last rate 2.04
        assert med > last_rates[19], f"growth: median {med} not above last {last_rates[19]}"
    synth(hist2, "rising-earners", check_growth)

    # (c) Flat history: must NOT equal the house baseline.
    fc = entry._forecast_case(base)
    m = as_map(fc)
    house = dev_score.baseline_a(base, base["horizon_minutes"])
    diffs = [abs(m[i][2] - house[i][2]) for i in range(1, 56)]
    assert max(diffs) > 0.2, f"flat: median barely moved (max diff {max(diffs)})"
    spreads = [m[i][4] - m[i][0] for i in range(1, 56)]
    house_spreads = [house[i][4] - house[i][0] for i in range(1, 56)]
    assert sum(spreads) / 55 > sum(house_spreads) / 55, "flat: spread not wider than house"
    print(f"synthetic flat-history: OK (max median move {max(diffs):.2f}pp, "
          f"mean spread {sum(spreads)/55:.2f}pp vs house {sum(house_spreads)/55:.2f}pp)")


if __name__ == "__main__":
    main()
