#!/usr/bin/env python3
"""Dev-only scorer + baseline reference for the trophy forecast build.

Implements the pinball scorer from RULES.txt section Scoring and the two
house baselines from RULES.txt section Baseline and prizes, so the builder
can check its own model against the coordinator's prep numbers:
  house baseline A on (p01 -> 10:43 live answer): loss ~1.4022,
  typical miss ~0.68 pp, coverage_90 0.00.
Not part of the entry; never committed.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

KIT = "/home/hatch/workspace/makefun-account/gta6-trophy-rarity-forecast-01-kit"
WORK = "/home/hatch/workspace/bounties/gta6-trophy-forecast"
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def q4(x):
    """Round to 4 decimals, half up (JS Math.round style)."""
    scaled = math.floor(x * 10000 + 0.5)
    v = scaled / 10000
    return int(v) if float(v).is_integer() else v


def pinball(y, q, tau):
    d = logit_of(y) - logit_of(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_forecast(forecast, answer_rates):
    """forecast: {id: [q05..q95]}; answer_rates: list aligned with ids 1..55.
    Returns (mean loss, typical miss in pp, coverage_90)."""
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    tot = 0.0
    n = 0
    miss = 0.0
    covered = 0
    trophies = 0
    for tid, qs in sorted(forecast.items()):
        y = answer_rates[tid - 1]
        for tau, q in zip(levels, qs):
            tot += pinball(y, q, tau)
            n += 1
        miss += abs(y - qs[2])
        if qs[0] <= y <= qs[4]:
            covered += 1
        trophies += 1
    return tot / n, miss / trophies, covered / trophies


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(ts):
    return (parse_ts(ts) - RELEASE).total_seconds() / 3600.0


def baseline_a(case, horizon_minutes):
    """House baseline: median = last read's rate, random-walk spread."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(horizon_minutes / 1440.0)
    out = {}
    for i, tr in enumerate(case["trophies"]):
        r = last["rates"][i]
        lg = logit_of(r if r is not None else 0.0)
        out[tr["id"]] = [q4(inv_logit(lg + z * spread)) for z in Z]
    return out


def baseline_b(case, horizon_minutes):
    """House curve baseline: per-trophy OLS of logit(rate) on ln(1+h) over
    the case's last 24h of reads; line read at the target; spread =
    random walk combined with the fit residual (root-sum-square)."""
    target_h = hours_since_release(case["target"])
    last = case["history"][-1]
    rw = 0.5 * math.sqrt(horizon_minutes / 1440.0)
    cutoff = parse_ts(case["at"]).timestamp() - 24 * 3600
    reads = [h for h in case["history"]
             if parse_ts(h["at"]).timestamp() >= cutoff]
    out = {}
    for i, tr in enumerate(case["trophies"]):
        xs, ys = [], []
        for h in reads:
            r = h["rates"][i]
            if r is None:
                continue
            xs.append(math.log1p(hours_since_release(h["at"])))
            ys.append(logit_of(r))
        if len(xs) < 3:
            lg = logit_of(last["rates"][i] if last["rates"][i] is not None else 0.0)
            out[tr["id"]] = [q4(inv_logit(lg + z * rw)) for z in Z]
            continue
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 1e-12:
            lg = my
            resid = 0.0
        else:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            a = my - b * mx
            lg = a + b * math.log1p(target_h)
            resid = math.sqrt(sum((y - (a + b * x)) ** 2
                                  for x, y in zip(xs, ys)) / n)
        spread = math.sqrt(rw * rw + resid * resid)
        out[tr["id"]] = [q4(inv_logit(lg + z * spread)) for z in Z]
    return out


def main():
    answer = json.load(open(os.path.join(WORK, "answer_1043.json")))
    rates = answer["rates"]
    answer_at = parse_ts(answer["at"])
    for cid in ("p00", "p01"):
        case = json.load(open(os.path.join(KIT, "input", cid + ".in")))
        mins_to_answer = (answer_at - parse_ts(case["at"])).total_seconds() / 60.0
        for name, fc in (("A@answer-horizon", baseline_a(case, mins_to_answer)),
                         ("A@case-horizon", baseline_a(case, case["horizon_minutes"])),
                         ("B@answer-horizon", baseline_b(case, mins_to_answer))):
            loss, miss, cov = score_forecast(fc, rates)
            print(f"{cid} baseline {name}: loss={loss:.4f} miss={miss:.4f}pp cov90={cov:.2f}")
    # Per-tier median logit drop, launch-night last rates -> 10:43 answer.
    case = json.load(open(os.path.join(KIT, "input", "p01.in")))
    last = case["history"][-1]["rates"]
    drops = {}
    for i, tr in enumerate(case["trophies"]):
        d = logit_of(last[i]) - logit_of(rates[i])
        drops.setdefault(tr["tier"], []).append(d)
    for tier, ds in sorted(drops.items()):
        ds.sort()
        med = ds[len(ds) // 2] if len(ds) % 2 else 0.5 * (ds[len(ds) // 2 - 1] + ds[len(ds) // 2])
        print(f"tier {tier}: n={len(ds)} median logit drop={med:.2f} "
              f"min={ds[0]:.2f} max={ds[-1]:.2f}")
    all_drops = sorted(d for ds in drops.values() for d in ds)
    n = len(all_drops)
    med = all_drops[n // 2] if n % 2 else 0.5 * (all_drops[n // 2 - 1] + all_drops[n // 2])
    print(f"overall median logit drop={med:.2f}")


if __name__ == "__main__":
    main()
