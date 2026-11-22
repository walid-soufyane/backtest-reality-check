import numpy as np
import pandas as pd
import pytest

from backtest_reality_check import SyntheticMarketConfig, generate_prices, simple_returns


def test_same_seed_is_bit_identical():
    a = generate_prices(seed=123, n_days=300)
    b = generate_prices(seed=123, n_days=300)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = generate_prices(seed=1, n_days=300)
    b = generate_prices(seed=2, n_days=300)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_shape_index_and_positivity():
    prices = generate_prices(n_days=500, n_assets=5)
    assert prices.shape == (500, 5)
    assert list(prices.columns) == [f"ASSET_{i:02d}" for i in range(5)]
    assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
    assert (prices > 0).all().all()
    assert np.allclose(prices.iloc[0], 100.0)
    assert prices.attrs["config"]["n_assets"] == 5


def test_config_object_and_overrides():
    cfg = SyntheticMarketConfig(n_days=50, n_assets=2, seed=9)
    a = generate_prices(cfg)
    b = generate_prices(cfg, seed=10)
    assert a.shape == (50, 2)
    assert b.attrs["config"]["seed"] == 10


def test_planted_autocorrelation_is_recoverable():
    # With no market factor, idiosyncratic AR(1) should show up in lag-1 autocorrelation.
    prices = generate_prices(n_days=20000, n_assets=4, autocorr=0.3, annual_market_vol=0.0, seed=5)
    r = simple_returns(prices).dropna()
    rho = np.mean([r[c].autocorr(1) for c in r.columns])
    assert 0.25 < rho < 0.35


@pytest.mark.parametrize(
    "kwargs",
    [{"n_days": 1}, {"n_assets": 0}, {"autocorr": 1.0}, {"annual_idio_vol": -0.1}, {"initial_price": 0}],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_prices(**kwargs)
