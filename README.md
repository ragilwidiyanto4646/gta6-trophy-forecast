# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ragilwidiyanto4646/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ragilwidiyanto4646/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts all 55 Grand Theft Auto VI PS5 trophies' Exophase earn rates at 2026-11-26T00:00:00Z, as five quantiles each. Scores by mean pinball loss on the logit scale — the goal is to beat both house baselines and land as close to the real day-7 rates as possible.

## How it works

A trophy's rate is a race between its earners E(t) and the tracked-player base N(t): rate = 100 * E / N. The model fits earners, not rates: a page refresh jumps N (rates fall) but leaves E untouched, so fitting E makes the forecast refresh-robust where fitting the rate itself is fooled by the jump.

- **Players N(t):** one global power-law fit of log N against log(1 + hours since release), extrapolated to the target with the growth rate capped at the most recent 6 hours' observed rate — launch growth decelerates, so the latest hours are the honest guide.
- **Earners E(t):** per trophy, a log-log fit of earners against log(1 + hours since release). Each trophy's slope is shrunk toward its tier's mean slope (platinum / gold / silver / bronze), so a trophy is predicted from what happened to similar trophies, not from its own short noisy slope alone. A trophy whose earners never moved takes the tier slope entirely.
- **Spread:** residual volatility of the earner fit, the tier's slope dispersion, and the horizon random-walk scale, combined in quadrature — honest about refresh-sized moves, never narrower than the horizon allows.
- **Flat history:** when every read is identical (launch night), there is no trajectory to fit. The model then falls back to a dilution prior measured from Exophase's launch-night-to-morning refresh (tracked players 638 → 3,868, every rate but the 100% one down): earners held flat, players jumping by a per-tier logit step shrunk halfway toward the pooled drop. It never outputs the house baseline's flat median on a flat history — that is exactly the failure the practice punishes.

The program is deterministic, stdlib-only, and never crashes: any case that fails is retried with a baseline-shaped forecast so a valid output is always written.

## Run it

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one `<stem>.out` forecast per case into the output directory.

<sub>Entered on [make.fun](https://make.fun/?ref=ztnjpsw2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
