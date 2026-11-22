import numpy as np
import pandas as pd
import pytest

from backtest_reality_check import metrics


def test_total_return_and_equity_curve():
    r = pd.Series([0.10, -0.10, 0.05])
    assert metrics.total_return(r) == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert metrics.equity_curve(r).iloc[-1] == pytest.approx(1.1 * 0.9 * 1.05)


def test_cagr_one_year_of_constant_returns():
    daily = (1.10) ** (1 / 252) - 1
    r = pd.Series([daily] * 252)
    assert metrics.cagr(r) == pytest.approx(0.10, rel=1e-9)


def test_max_drawdown_known_path():
    # equity: 1.0 -> 1.2 -> 0.9 -> 1.0 -> 1.3 ; worst drawdown 0.9 / 1.2 - 1 = -25%
    r = pd.Series([0.2, 0.9 / 1.2 - 1, 1.0 / 0.9 - 1, 1.3 - 1])
    assert metrics.max_drawdown(r) == pytest.approx(-0.25)
    dd = metrics.drawdown_series(r)
    assert dd.iloc[0] == 0.0 and dd.iloc[-1] == 0.0


def test_drawdown_counts_initial_loss():
    assert metrics.max_drawdown(pd.Series([-0.1, 0.0])) == pytest.approx(-0.1)


def test_sharpe_matches_formula_and_edge_cases():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)
    assert metrics.sharpe_ratio(pd.Series([0.01] * 10)) == 0.0  # zero volatility
    assert metrics.sharpe_ratio(pd.Series([], dtype=float)) == 0.0


def test_hit_rate_ignores_flat_days_and_nans():
    r = pd.Series([0.01, -0.01, 0.0, np.nan, 0.02])
    assert metrics.hit_rate(r) == pytest.approx(2 / 3)


def test_summarize_keys():
    s = metrics.summarize(pd.Series([0.01, -0.005, 0.002]))
    assert set(s) == {"total_return", "cagr", "annualized_volatility", "sharpe", "max_drawdown", "hit_rate"}
