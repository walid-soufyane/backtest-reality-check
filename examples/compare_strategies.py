"""Compare every bundled honest and deliberately leaky strategy in one table.

Run with:  python examples/compare_strategies.py

For each strategy the table shows gross and net Sharpe, how much survives one
extra bar of execution delay, the break-even cost, and whether the truncation
check finds future-data leakage. Leaky strategies are invalid however good (or
bad) their numbers look.
"""

import pandas as pd

from backtest_reality_check import (
    breakeven_cost_bps,
    check_truncation_invariance,
    generate_prices,
    run_backtest,
)
from backtest_reality_check.metrics import sharpe_ratio
from backtest_reality_check.signals import HONEST_SIGNALS, LEAKY_SIGNALS


def compare(prices: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    rows = []
    for kind, signals in (("honest", HONEST_SIGNALS), ("leaky", LEAKY_SIGNALS)):
        for name, fn in signals.items():
            weights = fn(prices)
            lag1 = run_backtest(prices, weights, execution_lag=1, cost_bps=cost_bps)
            lag2 = run_backtest(prices, weights, execution_lag=2, cost_bps=cost_bps)
            rows.append(
                {
                    "strategy": name,
                    "kind": kind,
                    "gross_sharpe": sharpe_ratio(lag1.gross_returns),
                    "net_sharpe": sharpe_ratio(lag1.net_returns),
                    "net_sharpe_lag2": sharpe_ratio(lag2.net_returns),
                    "breakeven_bps": breakeven_cost_bps(lag1),
                    "truncation_check": check_truncation_invariance(fn, prices).status,
                }
            )
    return pd.DataFrame(rows).set_index("strategy")


if __name__ == "__main__":
    prices = generate_prices(seed=42, autocorr=-0.05)
    table = compare(prices)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 160, "display.max_columns", None):
        print(table)
    print("\nEducational example on synthetic data. Not investment advice.")
