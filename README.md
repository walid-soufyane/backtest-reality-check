# Backtest Reality Check

[![tests](https://github.com/Walid-Soufyane/backtest-reality-check/actions/workflows/tests.yml/badge.svg)](https://github.com/Walid-Soufyane/backtest-reality-check/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

**A small Python toolkit that shows how a backtest can look great and still be wrong.**

Run one command and the same synthetic market produces:

| Strategy | Gross Sharpe | Net Sharpe (5 bps) | What the checks say |
|---|---|---|---|
| Short-term reversal (honest) | 1.83 | **−0.21** | Break-even cost is 4.5 bps; one extra bar of delay turns it to −2.00 |
| "Next-day return" signal | 55.29 | 52.60 | **FAIL**: past signal values change when future rows are removed |

Those numbers come from `brc demo` with its default seed. The data is synthetic and deterministic, so you will see the same figures. No API keys, downloads or paid data are needed.

> **Scope.** This is an educational and research-hygiene tool. It is **not investment advice**, not a trading system, and not a guarantee that any backtest is realistic.

---

## Why this exists

Most misleading backtests aren't fraud. They come from small, invisible assumptions:

- **Look-ahead.** The signal uses data that was not available yet: a `shift(-1)`, a centered rolling window, or a scaler fit on the whole sample.
- **Optimistic execution.** You assume you trade at the same close the signal was computed from.
- **Ignored costs.** A high-turnover edge of a few basis points a day goes away once realistic costs are applied.
- **Selection bias.** You try 50 parameter values and report the best one.

Backtest Reality Check makes each of these explicit, and tests the ones that can be tested mechanically.

## What it does (v0.1)

| Component | What it does |
|---|---|
| `generate_prices` | Deterministic, seeded synthetic multi-asset prices (market factor + idiosyncratic AR(1)), so you can plant a *known* weak edge |
| `run_backtest` | Vectorized backtester with an explicit `execution_lag` and linear transaction costs; reports gross/net returns, turnover, and costs |
| `metrics` | Total return, CAGR, volatility, Sharpe, drawdown series, max drawdown, hit rate |
| `checks` | The diagnostics below, each labelled **verified** or **illustrative** |
| `reality_check` | Runs everything on one signal function and prints a plain-text report |
| `brc demo` | CLI demo comparing honest and deliberately leaky strategies |

### Verified vs. illustrative

The tool keeps two kinds of check separate.

**Verified checks** test a concrete property of *your inputs*:

| Check | What a FAIL means |
|---|---|
| `price_data` | Unsorted or duplicate dates, gaps, or non-positive prices |
| `signal_alignment` | Signal index or columns don't match prices, or there are infinite weights (WARN when gross leverage exceeds a limit) |
| `execution_lag` | Lag 0: positions earn the return that produced them. Lag 1 is a WARN that states the same-close assumption |
| `determinism` | Two calls on identical data give different signals (unseeded randomness) |
| `truncation_invariance` | Recomputing the signal on truncated history changes past values, which **proves** that the function reads future price rows |

**Illustrative checks** measure fragility. They can't prove or disprove bias:

| Check | What it shows |
|---|---|
| `cost_sensitivity` | Break-even cost (bps per unit turnover) compared with the cost you think is realistic |
| `lag_sensitivity` | How much Sharpe survives one extra bar of delay |
| `selection_bias_on_noise` | The best Sharpe you get by trying N lookbacks on pure noise |

### What it does *not* do

- It **cannot** detect every form of look-ahead. A PASS on `truncation_invariance` only rules out leaks that come from future rows of the `prices` you pass in. It cannot see leakage from other data (restated fundamentals, revised macro data, point-in-time index membership), from globals, or from parameters you chose after seeing the full sample. Known gap: a leak confined to the first few warm-up rows, such as `.bfill()` after a rolling window, currently passes because every cutoff falls after the warm-up period. This is tracked in [#2](https://github.com/Walid-Soufyane/backtest-reality-check/issues/2).
- It does not detect survivorship bias in a universe you supply.
- The cost model is linear. It has no market impact, spread, borrow, or financing.
- The synthetic data is a controlled lab and is not a realistic market model.
- There is no live data, no broker connection, and no order routing.

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/Walid-Soufyane/backtest-reality-check.git
cd backtest-reality-check
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

brc demo                       # or: python -m backtest_reality_check demo
python examples/quickstart.py  # check your own signal function
```

## Example: check your own signal

A signal function takes a price DataFrame and returns target weights with the same shape. The weights on row `t` may use only data up to row `t`.

```python
import pandas as pd
from backtest_reality_check import generate_prices, reality_check


def my_signal(prices: pd.DataFrame) -> pd.DataFrame:
    past = prices.pct_change(20, fill_method=None)
    w = past.sub(past.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)


prices = generate_prices(seed=7, autocorr=-0.05)
report = reality_check(prices, my_signal, execution_lag=1, cost_bps=5)
print(report.to_text())
```

The report lists every assumption, the headline metrics net of costs, and one line per check:

```
[PASS] (verified) truncation_invariance: past values unchanged at 5 truncation cutoffs (rules out leaks from future price rows only)
[WARN] (illustrative) cost_sensitivity: no positive gross edge even before costs
```

20-day momentum has no edge in this synthetic market, so seeing that result is correct and expected.

You can also use the parts on their own: `run_backtest`, `cost_sweep`, `lag_sweep`, `check_truncation_invariance(my_signal, prices)`, and so on.

### Timing convention

The signal is computed at the close of bar `t`. With `execution_lag=L`, the position earns the return from close `t+L-1` to close `t+L`.

- `L=0`: earns the return from `t-1` to `t`, which was already known. This is look-ahead.
- `L=1`: trades at the close the signal was computed on. Standard in research, but optimistic.
- `L=2`: trades at the next close.

## Running tests and lint

```bash
pytest
ruff check .
ruff format --check .
```

CI runs the same commands on Python 3.10–3.13.

## Project layout

```
src/backtest_reality_check/
  data.py      synthetic prices
  backtest.py  run_backtest, cost/lag sweeps, break-even cost
  metrics.py   return and drawdown metrics
  checks.py    verified and illustrative diagnostics
  signals.py   baseline signals + deliberately leaky counterexamples
  report.py    reality_check() and text report
  cli.py       `brc demo`
examples/quickstart.py
tests/
```

## Contributing

Contributions of every size are welcome, including documentation fixes, new leaky counterexamples, new checks, and better explanations. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and the issues labelled [`good first issue`](https://github.com/Walid-Soufyane/backtest-reality-check/labels/good%20first%20issue). If you know a backtest pitfall that isn't covered, open an issue describing it. A minimal code example that reproduces it helps most.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) © 2026 Walid Soufyane

## Disclaimer

This software is for education and research hygiene only. It is not investment, financial, legal, or tax advice, and it is not intended for live trading. Results on synthetic data say nothing about future returns on real markets. Use at your own risk.
