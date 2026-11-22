"""Return and risk metrics for a daily return series.

All functions accept a pandas Series of *simple* periodic returns (e.g. 0.01 for
+1%). NaNs are dropped. Annualization assumes 252 periods per year unless
``periods_per_year`` is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS_PER_YEAR


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Compounded growth of ``start_value``."""
    r = _clean(returns)
    return start_value * (1.0 + r).cumprod()


def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    if r.empty:
        return 0.0
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    years = len(r) / periods_per_year
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio using arithmetic mean excess return.

    ``risk_free_rate`` is annual and converted to a per-period rate by simple
    division. Returns 0.0 for empty or zero-volatility series.
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    # Treat volatility indistinguishable from floating-point noise as zero.
    if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, float(excess.abs().max())):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (0 at peaks, negative below)."""
    curve = equity_curve(returns)
    if curve.empty:
        return curve
    # Include the starting value of 1.0 as the initial peak.
    peak = np.maximum.accumulate(np.concatenate([[1.0], curve.to_numpy()]))[1:]
    return pd.Series(curve.to_numpy() / peak - 1.0, index=curve.index)


def max_drawdown(returns: pd.Series) -> float:
    """Most negative drawdown, e.g. -0.25 for a 25% peak-to-trough loss."""
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else 0.0


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-zero periods with a positive return."""
    r = _clean(returns)
    r = r[r != 0]
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def summarize(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict[str, float]:
    """Dictionary of the headline metrics."""
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }
