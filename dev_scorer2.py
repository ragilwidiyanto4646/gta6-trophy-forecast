"""Dev scorer for the GTA VI trophy forecast entry (forecast #2).

Implements the pinball scorer from RULES.txt section "Scoring" plus the
house baseline A generator, so the coordinator's sanity number can be
reproduced locally:

  house baseline A on p01 vs answer_1043.json
  -> loss ~= 1.4022, typical miss ~= 0.68 pp, coverage_90 = 0.00

Scoring recap (RULES.txt): for each trophy and each level tau in
[0.05, 0.25, 0.5, 0.75, 0.95], d = logit(y) - logit(q) with rates
clamped to [0.01, 99.99] before the logit; item loss =
2 * max(tau*d, (tau-1)*d); loss = mean over all 55*5 items.
typical_miss_pp = mean |y - q50|; coverage_90 = share of trophies with
q05 <= y <= q95.
"""

import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
]

HERE = Path(__file__).resolve().parent
KIT_INPUT = (
    Path.home()
    / "workspace/makefun-account/gta6-trophy-rarity-forecast-01-kit/input"
)
ANSWER_DIR = Path.home() / "workspace/bounties/gta6-trophy-forecast"


def clamp_rate(x):
    """Clamp a percent rate into the scorer's logit-safe band."""
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def logit_of_rate(x):
    p = clamp_rate(x) / 100.0
    return math.log(p / (1.0 - p))


def rate_from_logit(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def round4_half_up(x):
    """Round to 4 decimals, half-up (the house baseline's rounding)."""
    return float(Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def score_rows(rows_by_id, answer_rates):
    """Score {id: [q05..q95]} against answer rates in trophy-id order 1..N.

    Returns (loss, typical_miss_pp, coverage_90).
    """
    total = 0.0
    items = 0
    miss_total = 0.0
    covered = 0
    n = len(answer_rates)
    for idx, y in enumerate(answer_rates):
        tid = idx + 1
        qs = rows_by_id[tid]
        ly = logit_of_rate(y)
        for tau, q in zip(LEVELS, qs):
            d = ly - logit_of_rate(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            items += 1
        miss_total += abs(y - qs[2])
        if qs[0] <= y <= qs[4]:
            covered += 1
    return total / items, miss_total / n, covered / n


def load_case(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def last_rates_of(case):
    """Per-trophy rate in the case's last read, keyed by trophy id."""
    trophies = case["trophies"]
    last = case["history"][-1]
    out = {}
    for pos, tro in enumerate(trophies):
        out[tro["id"]] = last["rates"][pos]
    return out


def house_baseline_rows(case):
    """House baseline A: median = last rate, sigma = 0.5*sqrt(horizon_days)."""
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = 0.5 * math.sqrt(horizon_days)
    rows = {}
    for tid, r in last_rates_of(case).items():
        centre = logit_of_rate(r)
        qs = [round4_half_up(rate_from_logit(centre + z * sigma)) for z in Z_SCORES]
        # keep the row non-decreasing after rounding
        fixed = []
        running = 0.0
        for q in qs:
            running = max(running, q)
            fixed.append(running)
        rows[tid] = fixed
    return rows


def load_answer(name):
    with open(ANSWER_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)["rates"]


def main():
    case = load_case(KIT_INPUT / "p01.in")
    answers = load_answer("answer_1043.json")
    # The coordinator's backtest scores p01's last read against the rates
    # actually observed at 10:43 UTC, i.e. a 433-minute horizon from the
    # 03:30 checkpoint (DESIGN.md section 3, sigma = 0.274) -- not the
    # case file's own 90-minute practice horizon.
    case["horizon_minutes"] = 433
    loss, miss, cov = score_rows(house_baseline_rows(case), answers)
    print(f"house baseline A on p01 (horizon 433) vs answer_1043:")
    print(f"  loss           = {loss:.4f}  (target ~1.4022)")
    print(f"  typical miss   = {miss:.4f} pp  (target ~0.68)")
    print(f"  coverage_90    = {cov:.2f}  (target 0.00)")


if __name__ == "__main__":
    main()
