"""Minute-by-minute quantile forecasts for Polymarket's FAA outage
attribution family (Iran / foreign actor / hack / AI sandbox escape).

Regime, from the family's own public price history (Nov 13-14, 2026):
the four Yes prices opened at 0.15-0.52 and collapsed to 0.006-0.04 within
~5 hours, then went quiet. By the final's checkpoints the prices sit a
few cents above zero, so the game is different from a mid-price event:
the downside is bounded by zero (a wide low leg is just clipped to 0 and
pays pinball every minute the price stays put), the upside is a rare
news jump, and the last hour's move is noise, not signal -- in the
collapse window every variant that chased the last hour's move or
widened the upside leg lost to a flat anchor with narrow legs.

So: anchor at the last price, no drift, narrow asymmetric legs, and a
per-market scale from how much the market actually moved lately (mean
absolute 1-minute change, last 6h blended with the last 24h, floored
and capped). The spread grows with the square root of the horizon and
is damped at the far end, because jumps in this family fade instead of
compounding.

Backtest (local pinball scorer, price units):
- Real FAA history, one full 240-min case + seven rolling 60-min cases:
  this entry 0.01060 / 0.00386, vs the NVIDIA-tuned vol model
  0.01110 / 0.00398 and a wide-upside variant 0.01240 / 0.00433.
- Kit practice (NVIDIA CEO event, p00-p04): 0.00313, vs 0.00305 for
  the NVIDIA-tuned model -- within 3%, same shape.
All constants are round, coarse numbers fitted to the shape of the
history, not to any single case.

Runs offline on Python 3.13, standard library only. Reads every *.in
file in $MAKEFUN_INPUT_DIR (else /input) and writes one <id>.out per
case to $MAKEFUN_OUTPUT_DIR (else /output). Deterministic: same input
bytes give the same output bytes, every run.
"""

import json
import math
import os
from os import listdir, makedirs
from os.path import join

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Quantile levels, in the order each output row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Spread per unit of market scale per square-root minute, one per level.
# Narrow on purpose: at prices of a few cents the downside is bounded by
# zero, so a wide low leg is clipped to 0 and pays pinball on every quiet
# minute; the upside leg stays modest because the realised window shows
# upside jumps are rare and the wide-upside variants lost to this shape.
# The middle legs sit close to the median because most minutes the price
# does not move at all.
SPREAD = (-1.5, -0.4, 0.0, 0.4, 1.5)

# Long-horizon damping: the spread's square-root growth is multiplied by
# (1 - DAMP * h / horizon), so at the far end it grows about 30% slower
# than a random walk -- jumps fade instead of compounding.
DAMP = 0.3

# Market scale bounds, in price units per minute. The floor keeps a market
# that has not moved at all lately from collapsing to a zero-width forecast
# (a jump can still arrive); the cap is high because this family's early
# hours include 0.08-0.22 one-minute jumps, and a cap fitted to a quiet
# event would strangle the spread just when the market is live.
SCALE_FLOOR = 0.0003
SCALE_CAP = 0.02


def clip01(value):
    """Clamp a price to the [0, 1] band the market lives in."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def six(value):
    """Round to 6 decimals, normalising -0.0 to 0.0."""
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


def mean_abs(values):
    """Mean of absolute values; 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(abs(v) for v in values) / len(values)


def market_scale(prices):
    """A market's current choppiness: mean absolute 1-minute change.

    Blends the last 6 hours (weight 0.6, tracks the live regime) with the
    last 24 (weight 0.4, steadies it), then floors and caps the result.
    A market that moved a lot lately forecasts wide; a quiet one narrow.
    Young markets simply have fewer than 1440 diffs; the blend uses
    whatever exists.
    """
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    recent = mean_abs(diffs[-360:])
    day = mean_abs(diffs[-1440:])
    scale = 0.6 * recent + 0.4 * day
    if scale < SCALE_FLOOR:
        return SCALE_FLOOR
    if scale > SCALE_CAP:
        return SCALE_CAP
    return scale


def horizon_rows(prices, horizon):
    """All 240 forecast rows for one market.

    The median stays at the last price for the whole horizon (no drift:
    the last hour's move in this family is noise, and chasing it lost in
    backtest). Each other quantile sits at the median plus its spread
    leg, the spread widening with the square root of the horizon and
    damped at the far end. Everything is clipped to [0, 1] and each row
    is forced non-decreasing after rounding.
    """
    if prices:
        anchor = prices[-1]
    else:
        anchor = 0.5
    scale = market_scale(prices) if prices else SCALE_FLOOR
    rows = []
    for h in range(1, horizon + 1):
        growth = math.sqrt(h) * (1.0 - DAMP * h / horizon)
        row = [six(clip01(anchor + SPREAD[j] * scale * growth)) for j in range(5)]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def run_case(case):
    """Forecast every market of one parsed case file."""
    horizon = case["horizon_minutes"]
    forecasts = []
    for market in case["markets"]:
        forecasts.append({
            "id": market["id"],
            "q": horizon_rows(market["prices"], horizon),
        })
    return {"forecasts": forecasts}


def main():
    makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(join(INPUT_DIR, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = run_case(case)
        text = json.dumps(result, separators=(",", ":"), sort_keys=False) + "\n"
        with open(join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
