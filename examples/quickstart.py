"""Quickstart: check your own signal function against synthetic data.

Run with:  python examples/quickstart.py
"""

import pandas as pd

from backtest_reality_check import cost_sweep, generate_prices, lag_sweep, reality_check


def my_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Long the 2 assets with the lowest 5-day return, short the 2 highest. Weights sum to 0, gross 1."""
    past = prices.pct_change(5, fill_method=None)
    ranks = past.rank(axis=1)
    n = prices.shape[1]
    weights = (ranks <= 2).astype(float) * 0.25 - (ranks > n - 2).astype(float) * 0.25
    return weights.where(past.notna(), 0.0)


def my_leaky_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Same idea, but ranks are computed on a window that ends 2 days in the future.

    Leaks do not always flatter a backtest: this one makes it look terrible.
    Either way the result is invalid, and the truncation check flags it.
    """
    return my_signal(prices.shift(-2))


if __name__ == "__main__":
    prices = generate_prices(seed=7, autocorr=-0.05)

    report = reality_check(prices, my_signal, cost_bps=5)
    print(report.to_text())
    print()
    print("Net Sharpe vs. cost assumption (bps per unit turnover):")
    print(cost_sweep(prices, my_signal(prices)).round(2).to_string())
    print()
    print("Net Sharpe vs. execution lag (5 bps costs):")
    print(lag_sweep(prices, my_signal(prices), cost_bps=5).round(2).to_string())
    print()
    print(reality_check(prices, my_leaky_signal, cost_bps=5).to_text())
