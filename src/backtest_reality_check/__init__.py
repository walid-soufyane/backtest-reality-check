"""Backtest Reality Check: expose common ways quantitative backtests mislead.

Educational toolkit. Not investment advice and not a live-trading system.
"""

from .backtest import BacktestResult, breakeven_cost_bps, cost_sweep, lag_sweep, run_backtest
from .checks import (
    CheckResult,
    check_cost_sensitivity,
    check_determinism,
    check_execution_lag,
    check_lag_sensitivity,
    check_price_data,
    check_signal_alignment,
    check_truncation_invariance,
    selection_bias_on_noise,
)
from .data import SyntheticMarketConfig, generate_prices, simple_returns
from .report import RealityCheckReport, reality_check

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "CheckResult",
    "RealityCheckReport",
    "SyntheticMarketConfig",
    "__version__",
    "breakeven_cost_bps",
    "check_cost_sensitivity",
    "check_determinism",
    "check_execution_lag",
    "check_lag_sensitivity",
    "check_price_data",
    "check_signal_alignment",
    "check_truncation_invariance",
    "cost_sweep",
    "generate_prices",
    "lag_sweep",
    "reality_check",
    "run_backtest",
    "selection_bias_on_noise",
    "simple_returns",
]
