# FAA Outage Attribution Forecaster

Entry for make.fun `polymarket-faa-attribution-02`: forecast the Yes price
of every market in Polymarket's FAA outage attribution family (Iran /
foreign actor / hack / AI sandbox escape) at every minute of the 4 hours
after each checkpoint — five quantiles per minute, scored by mean pinball
loss.

## Approach

The family's own public price history (Nov 13–14, 2026) shows the regime:
the four Yes prices opened at 0.15–0.52, collapsed to 0.006–0.04 within
~5 hours, then went quiet. At prices a few cents above zero the downside
is bounded by zero, the upside is a rare news jump, and the last hour's
move is noise.

- **Median anchored at the last price, no drift.** Chasing the last
  hour's move lost in backtest.
- **Narrow asymmetric legs** `(-1.5, -0.4, 0, 0.4, 1.5)` per unit of
  market scale per sqrt-minute. A wide low leg is clipped to 0 and pays
  pinball on every quiet minute; a wide upside leg lost to this shape on
  the realised window.
- **Per-market scale** = mean absolute 1-minute change, 0.6 × last 6h +
  0.4 × last 24h, floored at 0.0003 and capped at 0.02 (the family's
  early hours include 0.08–0.22 one-minute jumps).
- **Spread grows with sqrt(horizon), damped 30% at the far end** —
  jumps fade instead of compounding.

## Backtest (local pinball scorer, price units)

- Real FAA history — one full 240-min case + seven rolling 60-min
  cases: **0.01060 / 0.00386**, vs 0.01110 / 0.00398 for the
  NVIDIA-tuned vol model and 0.01240 / 0.00433 for a wide-upside
  variant.
- Kit practice (NVIDIA CEO event, p00–p04): **0.3127** (×100) vs house
  baseline 0.3636 — beats the baseline by ~14%.

## Run

Python 3.13, standard library only, no network:

```
MAKEFUN_INPUT_DIR=<cases> MAKEFUN_OUTPUT_DIR=<out> python3 main.py
```

Reads every `*.in` case file, writes one `<id>.out` per case.
Deterministic; ~0.2 s per case.
