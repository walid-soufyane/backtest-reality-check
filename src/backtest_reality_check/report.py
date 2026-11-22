"""Run all diagnostics on one strategy and print a plain-text report."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import checks
from .backtest import BacktestResult, run_backtest
from .checks import CheckResult, SignalFn

DISCLAIMER = (
    "Educational tool. Not investment advice and not a live-trading system. "
    "A PASS means a specific test found nothing, not that the backtest is realistic."
)


@dataclass
class RealityCheckReport:
    strategy_name: str
    result: BacktestResult
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(c.status == "FAIL" for c in self.checks)

    def to_text(self) -> str:
        s = self.result.summary()
        lines = [
            f"=== Backtest Reality Check: {self.strategy_name} ===",
            "",
            "Assumptions:",
            *[f"  - {a}" for a in self.result.assumptions],
            "",
            "Headline metrics (net of costs):",
            f"  CAGR {s['net_cagr']:+.2%} | vol {s['net_annualized_volatility']:.2%} | "
            f"Sharpe {s['net_sharpe']:.2f} | max drawdown {s['net_max_drawdown']:.2%}",
            f"  gross Sharpe {s['gross_sharpe']:.2f} | avg daily turnover {s['avg_daily_turnover']:.2f} | "
            f"annual cost drag {s['annualized_cost_drag']:.2%}",
            "",
            "Checks:",
            *[f"  {c}" for c in self.checks],
            "",
            f"Note: {DISCLAIMER}",
        ]
        return "\n".join(lines)


def reality_check(
    prices: pd.DataFrame,
    signal_fn: SignalFn,
    strategy_name: str | None = None,
    execution_lag: int = 1,
    cost_bps: float = 5.0,
    assumed_cost_bps: float | None = None,
) -> RealityCheckReport:
    """Backtest ``signal_fn`` on ``prices`` and run every diagnostic.

    ``assumed_cost_bps`` is the cost level you consider realistic for the
    break-even comparison; it defaults to ``cost_bps``.
    """
    name = strategy_name or getattr(signal_fn, "__name__", "strategy")
    signal = signal_fn(prices)
    result = run_backtest(prices, signal, execution_lag=execution_lag, cost_bps=cost_bps)
    results = [
        checks.check_price_data(prices),
        checks.check_signal_alignment(prices, signal),
        checks.check_execution_lag(execution_lag),
        checks.check_determinism(signal_fn, prices),
        checks.check_truncation_invariance(signal_fn, prices),
        checks.check_cost_sensitivity(
            prices,
            signal,
            execution_lag=max(execution_lag, 1),
            assumed_cost_bps=cost_bps if assumed_cost_bps is None else assumed_cost_bps,
        ),
        checks.check_lag_sensitivity(prices, signal, cost_bps=cost_bps),
    ]
    return RealityCheckReport(strategy_name=name, result=result, checks=results)
