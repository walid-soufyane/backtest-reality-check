# Changelog

## Unreleased

- New deliberately leaky counterexample: `leaky_calendar_group_mean` (calendar-group averages that include future days).
- `examples/compare_strategies.py`: one table comparing every honest and leaky strategy.
- pre-commit hooks and Dependabot configuration.

## 0.1.0 (2026-11-22)

Initial release.

- Deterministic synthetic multi-asset price generator with an optional planted AR(1) edge
- Vectorized backtester with an explicit execution lag and linear transaction costs
- Metrics: total return, CAGR, volatility, Sharpe, drawdown, hit rate
- Verified checks: price data, signal alignment, execution lag, determinism, truncation invariance
- Illustrative checks: cost sensitivity / break-even cost, lag sensitivity, selection bias on noise
- Deliberately leaky example signals and a `brc demo` CLI
