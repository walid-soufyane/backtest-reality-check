"""A small, explicit vectorized backtester.

Timing convention (read this once, it matters):

* A *signal* (target portfolio weights) is computed at the close of bar ``t``
  using information available up to and including that close.
* ``execution_lag`` is the number of bars between the signal bar and the first
  bar whose return the position earns:

  - ``execution_lag=0`` -- the weights decided at the close of ``t`` earn the
    return *from close t-1 to close t*. That return was already known when the
    weights were chosen: this is look-ahead and the result is meaningless.
  - ``execution_lag=1`` -- the weights earn the return from close ``t`` to
    close ``t+1``. This assumes you can observe the close and trade at that same
    close price. Common in research, often optimistic in practice.
  - ``execution_lag=2`` -- you trade at the next close (``t+1``) and earn from
    ``t+1`` to ``t+2``. More conservative.

* The portfolio is assumed to be rebalanced to the target weights every bar.
  Intra-bar weight drift between rebalances is ignored (a documented
  simplification).
* Transaction costs are linear: ``cost_bps / 10_000`` times turnover, where
  turnover is the sum of absolute weight changes. No market impact, borrow fees,
  financing, or slippage models beyond this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import metrics
from .data import simple_returns


@dataclass
class BacktestResult:
    """Outputs of :func:`run_backtest`. All series share the price index."""

    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame
    execution_lag: int
    cost_bps: float
    assumptions: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        out = {f"net_{k}": v for k, v in metrics.summarize(self.net_returns).items()}
        out["gross_sharpe"] = metrics.sharpe_ratio(self.gross_returns)
        out["gross_cagr"] = metrics.cagr(self.gross_returns)
        out["avg_daily_turnover"] = float(self.turnover.mean())
        out["annualized_cost_drag"] = float(self.costs.mean() * 252)
        return out


def align_signal(prices: pd.DataFrame, signal: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Coerce a signal to a DataFrame aligned to ``prices`` (missing -> NaN)."""
    if isinstance(signal, pd.Series):
        if prices.shape[1] != 1:
            raise ValueError("A Series signal is only allowed for single-asset prices")
        signal = signal.to_frame(prices.columns[0])
    return signal.reindex(index=prices.index, columns=prices.columns)


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame | pd.Series,
    execution_lag: int = 1,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Backtest target weights against close prices.

    Parameters
    ----------
    prices:
        Close prices, one column per asset.
    signal:
        Target weights decided at each bar's close (same index/columns as
        ``prices``). NaN weights are treated as 0 (flat).
    execution_lag:
        Bars between the signal bar and the first return earned. See module
        docstring. Must be >= 0. ``0`` is allowed only so the look-ahead
        failure mode can be demonstrated.
    cost_bps:
        Linear transaction cost in basis points per unit of turnover.
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")

    weights = align_signal(prices, signal).fillna(0.0)
    asset_returns = simple_returns(prices).fillna(0.0)

    # held[t] earns the return from close t-1 to close t. Weights decided at close s
    # are first held at bar s + lag, so held = weights shifted forward by `lag` bars.
    # lag 0 means the weights earn the very return that was used to compute them.
    held = weights.shift(execution_lag).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    costs = turnover * (cost_bps / 10_000.0)
    net = gross - costs

    assumptions = [
        f"execution_lag={execution_lag}: weights decided at close t earn returns from "
        + (
            "close t-1 to close t (LOOK-AHEAD)"
            if execution_lag == 0
            else "close t to close t+1 (trade at the signal bar's close)"
            if execution_lag == 1
            else f"close t+{execution_lag - 1} to close t+{execution_lag}"
        ),
        "rebalanced to target weights every bar; intra-bar drift ignored",
        f"linear costs of {cost_bps:g} bps per unit turnover; no impact, borrow or financing costs",
        "close prices only; no bid/ask, gaps, halts, or partial fills",
    ]
    return BacktestResult(
        gross_returns=gross.rename("gross"),
        net_returns=net.rename("net"),
        costs=costs.rename("costs"),
        turnover=turnover.rename("turnover"),
        positions=held,
        execution_lag=execution_lag,
        cost_bps=float(cost_bps),
        assumptions=assumptions,
    )


def breakeven_cost_bps(result: BacktestResult) -> float:
    """Linear cost (bps per unit turnover) at which mean net return is zero.

    Returns ``inf`` if the strategy never trades and ``nan`` if gross mean
    return is not positive (there is no edge to break even on).
    """
    mean_turnover = float(result.turnover.mean())
    mean_gross = float(result.gross_returns.mean())
    if mean_turnover == 0:
        return float("inf")
    if mean_gross <= 0:
        return float("nan")
    return mean_gross / mean_turnover * 10_000.0


def cost_sweep(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps_grid: tuple[float, ...] = (0, 2, 5, 10, 20, 50),
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Net Sharpe and CAGR across a grid of linear cost assumptions."""
    rows = []
    for c in cost_bps_grid:
        res = run_backtest(prices, signal, execution_lag=execution_lag, cost_bps=c)
        rows.append(
            {
                "cost_bps": c,
                "net_sharpe": metrics.sharpe_ratio(res.net_returns),
                "net_cagr": metrics.cagr(res.net_returns),
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def lag_sweep(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 5),
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Net Sharpe across execution lags."""
    rows = []
    for lag in lags:
        res = run_backtest(prices, signal, execution_lag=lag, cost_bps=cost_bps)
        rows.append({"execution_lag": lag, "net_sharpe": metrics.sharpe_ratio(res.net_returns)})
    return pd.DataFrame(rows).set_index("execution_lag")


__all__ = [
    "BacktestResult",
    "align_signal",
    "breakeven_cost_bps",
    "cost_sweep",
    "lag_sweep",
    "run_backtest",
]
