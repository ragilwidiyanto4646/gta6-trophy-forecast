# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ragilwidiyanto4646/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ragilwidiyanto4646/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts all 55 Grand Theft Auto VI PS5 trophies' Exophase earn rates at 2026-11-26T00:00:00Z, as five quantiles each. Scores by mean pinball loss on the logit scale — the goal is to beat both house baselines and land as close to the real day-7 rates as possible.

## How it works

A trophy's rate is a race between its earners E(t) and the tracked-player base N(t): rate = 100 * E / N. The model fits earners, not rates: a page refresh jumps N (rates fall) but leaves E untouched, so fitting E makes the forecast refresh-robust where fitting the rate itself is fooled by the jump.

- **Players N(t):** one global power-law fit of log N against log(1 + hours since release) over the last 24 hours of reads, extrapolated to the target with the growth rate capped at the most recent 6 hours' observed rate and the total capped at 50x the last count — launch growth decelerates, so the latest hours are the honest guide.
- **Earners E(t):** per trophy, a log-log fit of earners against log(1 + hours since release). Each trophy's slope is shrunk toward its tier's mean slope (platinum / gold / silver / bronze), weighted by how much its own earners actually moved, so a trophy is predicted from what happened to similar trophies, not from its own short noisy slope alone. A trophy whose earners never moved takes the tier slope entirely. Extrapolated earners are clamped to never fall below their last count and never exceed the player base.
- **Spread:** residual volatility of the earner fit, the tier's slope dispersion, and the horizon random-walk scale (0.5 * sqrt(horizon days)), combined in quadrature — honest about refresh-sized moves, never narrower than the horizon allows. The trophy everyone already has keeps its median pinned at 99.99% with the same honest spread.
- **Flat history:** when every read is identical (launch night), there is no trajectory to fit. The model then falls back to a dilution prior measured from Exophase's launch-night-to-morning refresh (tracked players 638 → 3,868, every rate but the 100% one down): the median drops by a per-tier logit step (platinum 1.85, gold 1.71, silver 1.85, bronze 1.59). It never outputs the house baseline's flat median on a flat history — that is exactly the failure the practice punishes.

## Forecast #2 changes

Three changes over probe #1, each validated against two scored backtests (the 10:43 and 23:35 UTC Exophase snapshots of launch day):

1. **Less slope shrinkage** (`SHRINK_SCALE` 0.20 → 0.05): a trophy's own earner slope is trusted sooner. Out-of-sample checks showed trophies whose earners are already outgrowing dilution — the crossover trophies — were systematically under-predicted when the tier mean dominated.
2. **Per-tier dilution drops used as measured** (`SHRINK_TIER_DROP` 0.5 → 1.0): the flat-history step no longer shrinks each tier's drop halfway toward the pooled 1.59; both backtest snapshots prefer the unshrunk tiers.
3. **The dilution step grows with horizon:** past 90 minutes the step is `drop * (1 + 0.3 * ln(horizon_minutes / 90))` instead of capping at the 90-minute refresh size. Dilution keeps running over longer horizons (median logit drop ≈1.59 by ~7 h, ≈2.94 by ~20 h), and a capped step badly under-predicts them. At 90 minutes the step is unchanged; at 1,205 minutes it is ~1.78x the 90-minute step.

The program is deterministic, stdlib-only, and never crashes: any case that fails is retried with a baseline-shaped forecast so a valid output is always written.

## Run it

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one `<stem>.out` forecast per case into the output directory.

<sub>Entered on [make.fun](https://make.fun/?ref=ztnjpsw2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
