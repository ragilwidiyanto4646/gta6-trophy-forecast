"""GTA VI trophy rarity forecast (forecast #2).

Hierarchical analogue growth model. A trophy's earn rate is the ratio
of its earners E(t) to the tracked-player base N(t): rate = 100*E/N.
The model fits earners and players separately in log-log space
(log E and log N against log(1 + hours since release)), shrinks each
trophy's earner slope toward its tier's mean slope, extrapolates both
series to the target, and reports five quantiles around the implied
median rate with a spread combining fit volatility, tier slope
dispersion and a horizon random-walk scale.

Forecast #2 changes relative to probe #1:
  (a) SHRINK_SCALE 0.20 -> 0.05: a trophy's own earner slope is
      trusted sooner (crossover trophies were under-predicted).
  (b) SHRINK_TIER_DROP 0.5 -> 1.0: per-tier flat-history dilution
      drops are used as measured, not shrunk toward the pooled drop.
  (c) The flat-history dilution step keeps growing past 90 minutes:
      step = drop * (1 + 0.3*ln(horizon_minutes/90)) for longer
      horizons, instead of capping at the 90-minute refresh size.

Stdlib only, offline, deterministic (no randomness, no clock reads).
Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes
one <stem>.out per case in $MAKEFUN_OUTPUT_DIR (else /output). A case
that fails for any reason still receives a valid baseline-shaped
output; the program never crashes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

# --- fixed model constants -------------------------------------------------

RELEASE_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

LEVEL_Z = [
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
]

SIGMA_RW = 0.5            # horizon random-walk scale (per sqrt(day))
VOL_FLOOR = 0.15          # min residual volatility, growth branch
VOL_FLOOR_FLAT = 0.50     # min residual volatility, flat branch
DISP_FLOOR = 0.10         # min tier slope dispersion, growth branch
DISP_FLOOR_FLAT = 0.30    # min tier slope dispersion, flat branch
N_CAP_MULTIPLE = 50.0     # N(target) may not exceed 50x N(last)
N_FIT_WINDOW_H = 24.0     # player fit uses at most the last 24 hours
N_RECENT_WINDOW_H = 6.0   # growth-rate cap looks at the last 6 hours

SHRINK_SCALE = 0.05       # forecast #2 (a): was 0.20 in probe #1
SHRINK_TIER_DROP = 1.0    # forecast #2 (b): was 0.5 in probe #1

# Measured launch-night -> morning dilution (median logit drop per tier).
POOLED_DROP = 1.59
TIER_DROP = {
    "platinum": 1.85,
    "gold": 1.71,
    "silver": 1.85,
    "bronze": 1.59,
}

PSEUDO_EARNERS = 0.5      # log-fit stand-in when a trophy has 0 earners


# --- small numeric helpers -------------------------------------------------

def parse_timestamp(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    cleaned = str(text).strip()
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_since_release(moment):
    return (moment - RELEASE_EPOCH).total_seconds() / 3600.0


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def logit_of_rate(rate):
    p = clamp(rate, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def rate_from_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimal places, half-up."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def mean_of(values):
    return sum(values) / len(values)


def pstdev_of(values):
    if len(values) < 2:
        return 0.0
    centre = mean_of(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / len(values))


def log_time_axis(hours):
    """x-axis of every growth fit: log(1 + hours since release)."""
    return math.log(max(1.0 + hours, 0.05))


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (slope, intercept, rmse)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = mean_of(xs)
    my = mean_of(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        slope = 0.0
    else:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    rmse = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return slope, intercept, rmse


def quantiles_around(median_rate, sigma):
    """Five quantiles for a median rate and a logit-scale spread."""
    centre = logit_of_rate(median_rate)
    out = []
    running = 0.0
    for z in LEVEL_Z:
        q = round4(clamp(rate_from_logit(centre + z * sigma), 0.0, 100.0))
        running = max(running, q)
        out.append(running)
    return out


# --- case unpacking --------------------------------------------------------

class Read:
    __slots__ = ("hours", "players", "rates")

    def __init__(self, hours, players, rates):
        self.hours = hours
        self.players = players
        self.rates = rates


def unpack_case(case):
    """Pull (trophies, valid reads, horizon_minutes, target_hours) out."""
    trophies = case["trophies"]
    reads = []
    for raw in case.get("history", []):
        players = raw.get("players_tracked")
        if not players:  # missing or 0 -> unusable read
            continue
        reads.append(
            Read(
                hours_since_release(parse_timestamp(raw["at"])),
                float(players),
                list(raw.get("rates", [])),
            )
        )
    horizon = case.get("horizon_minutes")
    if horizon is None:
        start = parse_timestamp(case["at"])
        finish = parse_timestamp(case["target"])
        horizon = (finish - start).total_seconds() / 60.0
    target_hours = hours_since_release(parse_timestamp(case["target"]))
    return trophies, reads, float(horizon), target_hours


def latest_rate(reads, pos):
    """Most recent non-null rate for the trophy at position pos."""
    for read in reversed(reads):
        if pos < len(read.rates) and read.rates[pos] is not None:
            return float(read.rates[pos])
    return None


def history_is_flat(reads):
    """True when every usable read shows the identical page."""
    if len(reads) <= 1:
        return True
    first = reads[0]
    for read in reads[1:]:
        if read.players != first.players or read.rates != first.rates:
            return False
    return True


# --- flat-history branch ---------------------------------------------------

def flat_step(tier, horizon_minutes):
    """Dilution step (logit units) for a flat history. Forecast #2 (b)(c)."""
    tier_drop = TIER_DROP.get(tier, POOLED_DROP)
    drop = POOLED_DROP + SHRINK_TIER_DROP * (tier_drop - POOLED_DROP)
    if horizon_minutes <= 90.0:
        return drop * (horizon_minutes / 90.0)
    return drop * (1.0 + 0.3 * math.log(horizon_minutes / 90.0))


def forecast_flat(trophies, reads, horizon_minutes):
    sigma_h = SIGMA_RW * math.sqrt(horizon_minutes / 1440.0)
    sigma = math.sqrt(VOL_FLOOR_FLAT ** 2 + sigma_h ** 2 + DISP_FLOOR_FLAT ** 2)
    rows = []
    for pos, tro in enumerate(trophies):
        last = latest_rate(reads, pos)
        if last is None:
            last = 0.01
        if last >= 99.99:  # everyone has it: median pinned, spread kept
            rows.append({"id": tro["id"], "q": quantiles_around(99.99, sigma)})
            continue
        step = flat_step(tro.get("tier", "bronze"), horizon_minutes)
        median = rate_from_logit(logit_of_rate(last) - step)
        median = clamp(median, 0.01, 99.99)
        rows.append({"id": tro["id"], "q": quantiles_around(median, sigma)})
    return rows


# --- growth branch ---------------------------------------------------------

def forecast_players(reads, target_hours):
    """Extrapolate the tracked-player base to the target (power law)."""
    last = reads[-1]
    window = [r for r in reads if r.hours >= last.hours - N_FIT_WINDOW_H]
    if len(window) < 2:
        window = reads
    xs = [log_time_axis(r.hours) for r in window]
    ys = [math.log(r.players) for r in window]
    slope, _intercept, _rmse = fit_line(xs, ys)

    # Cap the growth rate at what the most recent hours actually did.
    recent = [r for r in reads if r.hours >= last.hours - N_RECENT_WINDOW_H]
    if len(recent) >= 2:
        rxs = [log_time_axis(r.hours) for r in recent]
        rys = [math.log(r.players) for r in recent]
        recent_slope, _ri, _rr = fit_line(rxs, rys)
        slope = min(slope, recent_slope)

    ratio = (1.0 + target_hours) / max(1.0 + last.hours, 0.05)
    projected = last.players * (ratio ** slope) if ratio > 0 else last.players
    projected = min(projected, N_CAP_MULTIPLE * last.players)
    return max(projected, last.players)


def earner_series(reads, pos):
    """(x, log E) points for one trophy; refresh jumps leave E intact."""
    xs, ys = [], []
    for read in reads:
        if pos >= len(read.rates) or read.rates[pos] is None:
            continue
        earners = float(read.rates[pos]) * read.players / 100.0
        if earners <= 0.0:
            earners = PSEUDO_EARNERS
        xs.append(log_time_axis(read.hours))
        ys.append(math.log(earners))
    return xs, ys


def forecast_growth(trophies, reads, horizon_minutes, target_hours):
    players_target = forecast_players(reads, target_hours)
    last_hours = reads[-1].hours
    time_ratio = (1.0 + target_hours) / max(1.0 + last_hours, 0.05)
    sigma_h = SIGMA_RW * math.sqrt(horizon_minutes / 1440.0)

    # Raw per-trophy earner slopes, spreads and residual volatilities.
    raw_slope = {}
    spread = {}
    vol = {}
    earners_last = {}
    for pos, tro in enumerate(trophies):
        tid = tro["id"]
        xs, ys = earner_series(reads, pos)
        if ys:
            slope, _i, rmse = fit_line(xs, ys)
            raw_slope[tid] = slope
            spread[tid] = pstdev_of(ys)
            vol[tid] = max(rmse, VOL_FLOOR)
            earners_last[tid] = math.exp(ys[-1])
        else:
            raw_slope[tid] = 0.0
            spread[tid] = 0.0
            vol[tid] = VOL_FLOOR
            earners_last[tid] = PSEUDO_EARNERS

    # Tier means (the analogue pool) plus the global mean.
    global_mean = mean_of(list(raw_slope.values())) if raw_slope else 0.0
    tier_members = {}
    for tro in trophies:
        tier_members.setdefault(tro.get("tier", "bronze"), []).append(tro["id"])
    tier_mean = {}
    tier_disp = {}
    for tier, ids in tier_members.items():
        slopes = [raw_slope[i] for i in ids]
        tier_mean[tier] = mean_of(slopes) if slopes else global_mean
        tier_disp[tier] = max(pstdev_of(slopes), DISP_FLOOR)

    rows = []
    for pos, tro in enumerate(trophies):
        tid = tro["id"]
        tier = tro.get("tier", "bronze")
        last = latest_rate(reads, pos)
        if last is not None and last >= 99.99:
            sigma_full = math.sqrt(
                vol[tid] ** 2 + sigma_h ** 2 + tier_disp.get(tier, DISP_FLOOR) ** 2
            )
            rows.append({"id": tid, "q": quantiles_around(99.99, sigma_full)})
            continue
        # Hierarchical shrinkage: trust the trophy's own slope in
        # proportion to how much its earners actually moved.
        w = spread[tid] / (spread[tid] + SHRINK_SCALE)
        slope_star = w * raw_slope[tid] + (1.0 - w) * tier_mean.get(tier, global_mean)
        earners_target = earners_last[tid] * (time_ratio ** slope_star)
        earners_target = clamp(earners_target, earners_last[tid], players_target)
        median = clamp(100.0 * earners_target / players_target, 0.01, 99.99)
        sigma = math.sqrt(vol[tid] ** 2 + sigma_h ** 2 + tier_disp.get(tier, DISP_FLOOR) ** 2)
        rows.append({"id": tid, "q": quantiles_around(median, sigma)})
    return rows


# --- fallback + driver -----------------------------------------------------

def forecast_baseline(case):
    """House-baseline-shaped output: median = last rate, RW spread."""
    trophies, reads, horizon_minutes, _th = unpack_case(case)
    sigma = SIGMA_RW * math.sqrt(horizon_minutes / 1440.0)
    rows = []
    for pos, tro in enumerate(trophies):
        last = latest_rate(reads, pos)
        if last is None:
            last = 0.01
        rows.append({"id": tro["id"], "q": quantiles_around(clamp(last, 0.01, 99.99), sigma)})
    return rows


def forecast_case(case):
    trophies, reads, horizon_minutes, target_hours = unpack_case(case)
    if history_is_flat(reads):
        rows = forecast_flat(trophies, reads, horizon_minutes)
    else:
        rows = forecast_growth(trophies, reads, horizon_minutes, target_hours)
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def process_file(in_path, out_path):
    try:
        with open(in_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            payload = forecast_case(case)
        except Exception:
            payload = {"trophies": forecast_baseline(case)}
    except Exception:
        payload = {"trophies": []}
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        names = sorted(name for name in os.listdir(in_dir) if name.endswith(".in"))
    except OSError:
        return 0
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    for name in names:
        stem = name[:-3]
        process_file(os.path.join(in_dir, name), os.path.join(out_dir, stem + ".out"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
