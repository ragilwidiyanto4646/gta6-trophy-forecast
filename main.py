"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Top-down hierarchical forecast: each trophy's earn rate is a race between
its earners E(t) and the tracked-player base N(t), rate = 100 * E / N.
Earners are fit in log space against log(1 + hours since release), and
each trophy's slope is shrunk toward its tier's mean slope, so a trophy
is predicted from what happened to similar trophies, not from its own
short noisy slope alone. Players N(t) get one global power-law fit.
A refresh jumps N (rates fall) but leaves E untouched, so fitting E
instead of the rate makes the model refresh-robust where a fit of the
rate itself is fooled.

When a case's history is completely flat (every read identical, as on
launch night), no trajectory exists to fit. The model then falls back
to a dilution prior measured from the launch-night -> morning refresh
Exophase performed (tracked players 638 -> 3,868, every rate but the
100% one down): the median rate drops by a per-tier logit step, with
earners held flat and the player base jumping instead. The prior is
shrunk halfway toward the pooled drop so no tier is trusted fully.

Standard library only, deterministic, no network, no clock. Reads
every *.in in the input directory and writes one *.out per case.
A case that fails for any reason still gets a valid output built from
the last read, so the program never crashes and never leaves a case
unwritten.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
SIGMA_RW = 0.5  # logit units per sqrt(day): the house random-walk scale

# --- Calibration constants (priors, not fitted to any one case) ---
# Median logit drop of the launch-night -> morning refresh, pooled over
# all 55 trophies, and the per-tier medians measured from the same event
# (public 10:43 UTC snapshot vs the launch-night reads). Tier figures
# are shrunk 50% toward the pooled figure before use.
DILUTION_POOLED = 1.59
DILUTION_TIER = {"platinum": 1.85, "gold": 1.71, "silver": 1.85,
                 "bronze": 1.59}
SHRINK_TIER_DROP = 0.5
# How much observed movement in log-earners it takes to trust a trophy's
# own slope halfway over its tier mean.
SHRINK_SCALE = 0.20
# Spread floors (logit units). The flat-history floor is wide because a
# refresh-sized move is the honest uncertainty when no trajectory exists.
VOL_FLOOR = 0.15
VOL_FLOOR_FLAT = 0.50
DISP_FLOOR = 0.10
DISP_FLOOR_FLAT = 0.30
N_CAP_MULTIPLE = 50.0  # sanity cap: N(target) never exceeds 50x N(last)


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def _round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    whole = math.floor(x * 10000 + 0.5)
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _stdev(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (slope, intercept, resid_sd)
    or None when the fit is degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icept = my - slope * mx
    resid = math.sqrt(sum((y - (icept + slope * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return slope, icept, resid


def _release_instant(case):
    """Midnight UTC on the game's release date, from the case itself."""
    try:
        day = case["game"]["release_date"]
        return datetime.fromisoformat(day + "T00:00:00+00:00")
    except (KeyError, TypeError, ValueError):
        return datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def _hours_since(release, stamp):
    return (_parse_time(stamp) - release).total_seconds() / 3600.0


def _safe_log1p_hours(h):
    """log(1 + h) with h floored so pre-release reads stay defined."""
    return math.log1p(max(h, -0.9))


def _valid_reads(case):
    """Reads with a usable players-tracked count, oldest first."""
    reads = []
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        if isinstance(players, (int, float)) and players > 0:
            reads.append(read)
    return reads


def _history_is_flat(case, reads):
    """True when every read is identical to the last one: same player
    count and same rate for every trophy. No trajectory, no signal."""
    if len(reads) < 2:
        return True
    last = reads[-1]
    for read in reads[:-1]:
        if read.get("players_tracked") != last.get("players_tracked"):
            return False
        if read.get("rates") != last.get("rates"):
            return False
    return True


def _fit_players(case, reads, release, target_h):
    """Global power-law fit of log N against log(1 + h), extrapolated
    to the target with the growth rate capped at the recent observed
    rate and the level capped at a sane multiple of the last read."""
    last_n = reads[-1]["players_tracked"]
    if len(reads) < 2:
        return float(last_n)
    case_at = _parse_time(case["at"])
    window_start = case_at.timestamp() - 24 * 3600
    window = [r for r in reads
              if _parse_time(r["at"]).timestamp() >= window_start]
    if len(window) < 2:
        window = reads
    xs = [_safe_log1p_hours(_hours_since(release, r["at"])) for r in window]
    ys = [math.log(r["players_tracked"]) for r in window]
    fit = _ols(xs, ys)
    if fit is None:
        return float(last_n)
    slope, icept, _ = fit
    n_target = math.exp(icept + slope * _safe_log1p_hours(target_h))
    # Cap the implied growth rate at the recent observed rate. Recent
    # means the last 6 hours of the window, not the whole 24: launch
    # growth decelerates, so a 24-hour average overstates the rate the
    # player base is growing at right now.
    recent_window = [r for r in window
                     if _parse_time(r["at"]).timestamp()
                     >= _parse_time(window[-1]["at"]).timestamp() - 6 * 3600]
    if len(recent_window) < 2:
        recent_window = window
    h_first = _hours_since(release, recent_window[0]["at"])
    h_last = _hours_since(release, recent_window[-1]["at"])
    span = h_last - h_first
    if span > 0:
        recent = (math.log(recent_window[-1]["players_tracked"])
                  - math.log(recent_window[0]["players_tracked"])) / span
        if recent <= 0:
            n_target = min(n_target, float(last_n))
        else:
            cap = float(last_n) * math.exp(recent * (target_h - h_last))
            n_target = min(n_target, cap)
    n_target = max(n_target, float(last_n))
    n_target = min(n_target, N_CAP_MULTIPLE * float(last_n))
    return n_target


def _earner_series(case, reads, release, index):
    """(x, log E, E_last) for one trophy: x is log(1 + h), E is the
    earner count implied by the rate and the player base. A zero rate
    becomes half a pseudo-earner so the log stays defined."""
    xs, log_e, e_last = [], [], 0.5
    for read in reads:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        players = read["players_tracked"]
        earners = rates[index] / 100.0 * players
        e_last = earners
        xs.append(_safe_log1p_hours(_hours_since(release, read["at"])))
        log_e.append(math.log(max(earners, 0.5)))
    return xs, log_e, e_last


def _dilution_step(tier, horizon_minutes):
    """Logit drop of the dilution prior for one tier, scaled by how
    much of a refresh window the horizon covers (a 90-minute horizon
    covers a whole refresh; a shorter one only part of it)."""
    pooled = DILUTION_POOLED
    tier_drop = DILUTION_TIER.get(tier, pooled)
    shrunk = pooled + SHRINK_TIER_DROP * (tier_drop - pooled)
    return shrunk * min(1.0, horizon_minutes / 90.0)


def _forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the five scored quantiles")
    reads = _valid_reads(case)
    if not reads:
        raise ValueError("case has no usable history read")
    trophies = case["trophies"]
    release = _release_instant(case)
    target_h = _hours_since(release, case["target"])
    last = reads[-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma_hor = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    flat = _history_is_flat(case, reads)

    if flat:
        n_target = float(last["players_tracked"])
    else:
        n_target = _fit_players(case, reads, release, target_h)
    h_last = _hours_since(release, last["at"])

    # Per-trophy log-earner fits and the tier pools they shrink toward.
    fits = []
    for i, trophy in enumerate(trophies):
        xs, log_e, e_last = _earner_series(case, reads, release, i)
        fit = _ols(xs, log_e) if len(xs) >= 2 else None
        slope = fit[0] if fit else 0.0
        resid = fit[2] if fit else 0.0
        spread = _stdev(log_e)
        fits.append({"tier": trophy["tier"], "slope": slope,
                     "resid": resid, "spread": spread, "e_last": e_last})
    tiers = {}
    for entry in fits:
        tiers.setdefault(entry["tier"], []).append(entry)
    tier_mean = {t: _mean([e["slope"] for e in es])
                 for t, es in tiers.items()}
    tier_disp = {t: _stdev([e["slope"] for e in es])
                 for t, es in tiers.items()}
    tier_resid = {t: _median([e["resid"] for e in es])
                  for t, es in tiers.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        entry = fits[i]
        tier = trophy["tier"]
        last_rate = last["rates"][i] if i < len(last.get("rates") or []) \
            else None
        if last_rate is None:
            last_rate = 0.0
        # The trophy everyone already has stays (nearly) universal.
        if last_rate >= 99.995:
            median = 99.99
            sigma = math.sqrt(max(VOL_FLOOR, 0.15) ** 2
                              + sigma_hor ** 2 + DISP_FLOOR ** 2)
            rows.append((trophy["id"], median, sigma))
            continue
        if flat:
            # Dilution prior: earners flat, players jump by the tier's
            # refresh step, so the rate falls by that step in logit.
            jump = math.exp(_dilution_step(tier, case["horizon_minutes"]))
            n_t = float(last["players_tracked"]) * jump
            e_t = max(entry["e_last"], 0.5)
            median = 100.0 * e_t / n_t
            sigma_vol = max(tier_resid.get(tier, 0.0), VOL_FLOOR_FLAT)
            sigma_disp = max(tier_disp.get(tier, 0.0), DISP_FLOOR_FLAT)
        else:
            weight = entry["spread"] / (entry["spread"] + SHRINK_SCALE)
            slope = (weight * entry["slope"]
                     + (1.0 - weight) * tier_mean.get(tier, 0.0))
            growth = ((1.0 + target_h) / (1.0 + h_last)) ** slope \
                if target_h > h_last else 1.0
            e_base = max(entry["e_last"], 0.5)
            e_t = e_base * growth
            e_t = max(e_t, e_base)
            e_t = min(e_t, n_target)
            median = 100.0 * e_t / n_target
            sigma_vol = max(entry["resid"], VOL_FLOOR)
            sigma_disp = max(tier_disp.get(tier, 0.0), DISP_FLOOR)
        median = _clamp_pct(median)
        sigma = math.sqrt(sigma_vol ** 2 + sigma_hor ** 2 + sigma_disp ** 2)
        rows.append((trophy["id"], median, sigma))

    out_rows = []
    for tid, median, sigma in sorted(rows):
        center = _logit(median)
        quantiles = []
        prev = None
        for z in Z:
            q = _round4(_inv_logit(center + z * sigma))
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        out_rows.append({"id": tid, "q": quantiles})
    return {"trophies": out_rows}


def _baseline_rows(case):
    """House-baseline-shaped fallback: median = last rate, random-walk
    spread. Used only when the model itself fails on a case, so the
    output is still valid."""
    last = case["history"][-1]
    spread = SIGMA_RW * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        center = _logit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"],
                     "q": [_round4(_inv_logit(center + z * spread))
                           for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # Unparseable input: no trophy list to forecast, nothing
            # valid can be written for this case.
            continue
        try:
            forecast = _forecast_case(case)
        except Exception:
            try:
                forecast = _baseline_rows(case)
            except Exception:
                continue
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
