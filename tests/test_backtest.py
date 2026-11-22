import numpy as np
import pandas as pd
import pytest

from backtest_reality_check import breakeven_cost_bps, cost_sweep, generate_prices, lag_sweep, run_backtest


@pytest.fixture
def tiny_prices():
    idx = pd.bdate_range("2020-01-01", periods=5)
    # returns: day1 +10%, day2 -10%, day3 +20%, day4 0%
    return pd.DataFrame({"A": [100.0, 110.0, 99.0, 118.8, 118.8]}, index=idx)


def test_lag1_hand_computed(tiny_prices):
    signal = pd.DataFrame({"A": [1.0, 0.0, 1.0, 1.0, 0.0]}, index=tiny_prices.index)
    res = run_backtest(tiny_prices, signal, execution_lag=1)
    # held = signal shifted by one bar: [0, 1, 0, 1, 1]
    assert res.positions["A"].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert res.gross_returns.tolist() == pytest.approx([0.0, 0.10, 0.0, 0.20, 0.0])


def test_lag0_is_lookahead_and_lag2_is_later(tiny_prices):
    signal = pd.DataFrame({"A": [1.0, 0.0, 1.0, 1.0, 0.0]}, index=tiny_prices.index)
    assert run_backtest(tiny_prices, signal, 0).positions["A"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]
    assert run_backtest(tiny_prices, signal, 2).positions["A"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]


def test_costs_equal_turnover_times_bps(tiny_prices):
    signal = pd.DataFrame({"A": [1.0, -1.0, 1.0, 1.0, 0.0]}, index=tiny_prices.index)
    res = run_backtest(tiny_prices, signal, execution_lag=1, cost_bps=10)
    # held = [0, 1, -1, 1, 1] -> turnover [0, 1, 2, 2, 0]
    assert res.turnover.tolist() == pytest.approx([0, 1, 2, 2, 0])
    assert res.costs.tolist() == pytest.approx([0, 0.001, 0.002, 0.002, 0])
    assert (res.gross_returns - res.net_returns).tolist() == pytest.approx(res.costs.tolist())


def test_nan_signal_means_flat(tiny_prices):
    signal = pd.DataFrame({"A": [np.nan] * 5}, index=tiny_prices.index)
    res = run_backtest(tiny_prices, signal)
    assert (res.net_returns == 0).all()


def test_series_signal_single_asset(tiny_prices):
    res = run_backtest(tiny_prices, pd.Series(1.0, index=tiny_prices.index))
    assert res.gross_returns.iloc[1] == pytest.approx(0.10)


def test_invalid_arguments(tiny_prices):
    signal = pd.DataFrame({"A": [1.0] * 5}, index=tiny_prices.index)
    with pytest.raises(ValueError):
        run_backtest(tiny_prices, signal, execution_lag=-1)
    with pytest.raises(ValueError):
        run_backtest(tiny_prices, signal, cost_bps=-1)


def test_perfect_foresight_at_lag0_beats_lag1():
    prices = generate_prices(n_days=500, seed=3)
    rets = prices.pct_change(fill_method=None)
    oracle = np.sign(rets.sub(rets.mean(axis=1), axis=0)) / prices.shape[1]
    lag0 = run_backtest(prices, oracle, execution_lag=0).net_returns.mean()
    lag1 = run_backtest(prices, oracle, execution_lag=1).net_returns.mean()
    assert lag0 > 0.005 and abs(lag1) < 0.002


def test_breakeven_cost_zeroes_mean_net_return():
    prices = generate_prices(n_days=1000, seed=4, autocorr=-0.1)
    from backtest_reality_check.signals import short_term_reversal

    signal = short_term_reversal(prices)
    be = breakeven_cost_bps(run_backtest(prices, signal))
    assert be > 0
    assert run_backtest(prices, signal, cost_bps=be).net_returns.mean() == pytest.approx(0.0, abs=1e-12)


def test_breakeven_edge_cases(tiny_prices):
    flat = pd.DataFrame({"A": [0.0] * 5}, index=tiny_prices.index)
    assert breakeven_cost_bps(run_backtest(tiny_prices, flat)) == float("inf")
    short = pd.DataFrame({"A": [-1.0] * 5}, index=tiny_prices.index)
    assert np.isnan(breakeven_cost_bps(run_backtest(tiny_prices, short)))


def test_sweeps_are_monotone_in_cost():
    prices = generate_prices(n_days=800, seed=6)
    from backtest_reality_check.signals import cross_sectional_momentum

    signal = cross_sectional_momentum(prices, 5)
    sweep = cost_sweep(prices, signal)
    assert sweep["net_sharpe"].is_monotonic_decreasing
    assert list(lag_sweep(prices, signal).index) == [1, 2, 3, 5]
