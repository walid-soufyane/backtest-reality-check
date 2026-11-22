"""Deterministic synthetic market data.

Everything here is generated from a seeded NumPy random generator, so the same
arguments always produce exactly the same prices on any machine. No API keys,
downloads, or external services are involved.

The model is deliberately simple and fully transparent:

* daily log returns = beta * market_factor + idiosyncratic_component + drift
* the idiosyncratic component optionally follows an AR(1) process with
  coefficient ``autocorr`` so that a weak, *known* short-horizon predictability
  can be planted in the data. This lets the toolkit show how a real (but tiny)
  edge reacts to execution delay and transaction costs.

This is not a realistic model of any market. It is a controlled laboratory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Parameters of the synthetic market. All rates are annualized."""

    n_days: int = 2520
    n_assets: int = 8
    seed: int = 42
    start: str = "2015-01-01"
    annual_drift: float = 0.05
    annual_market_vol: float = 0.15
    annual_idio_vol: float = 0.20
    autocorr: float = 0.0
    initial_price: float = 100.0

    def validate(self) -> None:
        if self.n_days < 2:
            raise ValueError("n_days must be at least 2")
        if self.n_assets < 1:
            raise ValueError("n_assets must be at least 1")
        if not -1.0 < self.autocorr < 1.0:
            raise ValueError("autocorr must be strictly between -1 and 1")
        if self.annual_market_vol < 0 or self.annual_idio_vol < 0:
            raise ValueError("volatilities must be non-negative")
        if self.initial_price <= 0:
            raise ValueError("initial_price must be positive")


def generate_prices(config: SyntheticMarketConfig | None = None, **overrides) -> pd.DataFrame:
    """Generate a deterministic panel of synthetic close prices.

    Parameters
    ----------
    config:
        A :class:`SyntheticMarketConfig`. Defaults are used when omitted.
    **overrides:
        Field overrides, e.g. ``generate_prices(seed=7, autocorr=0.05)``.

    Returns
    -------
    pandas.DataFrame
        Business-day indexed close prices, one column per asset
        (``ASSET_00``, ``ASSET_01``, ...). The configuration used is stored in
        ``df.attrs["config"]`` for reproducibility.
    """
    base = config or SyntheticMarketConfig()
    cfg = SyntheticMarketConfig(**{**asdict(base), **overrides})
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)
    n, k = cfg.n_days, cfg.n_assets
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    market_vol = cfg.annual_market_vol * np.sqrt(dt)
    idio_vol = cfg.annual_idio_vol * np.sqrt(dt)

    betas = rng.uniform(0.6, 1.4, size=k)
    market = rng.standard_normal(n) * market_vol
    shocks = rng.standard_normal((n, k))

    # AR(1) idiosyncratic returns, scaled so the stationary volatility equals idio_vol.
    phi = cfg.autocorr
    innovation_scale = idio_vol * np.sqrt(1.0 - phi**2)
    idio = np.empty((n, k))
    idio[0] = shocks[0] * idio_vol
    for t in range(1, n):
        idio[t] = phi * idio[t - 1] + shocks[t] * innovation_scale

    total_var = (betas**2) * market_vol**2 + idio_vol**2
    drift = cfg.annual_drift * dt - 0.5 * total_var  # log drift so E[simple return] ~ annual_drift
    log_returns = market[:, None] * betas[None, :] + idio + drift[None, :]
    log_returns[0] = 0.0  # first row is the starting price

    prices = cfg.initial_price * np.exp(np.cumsum(log_returns, axis=0))
    index = pd.bdate_range(start=cfg.start, periods=n, name="date")
    columns = [f"ASSET_{i:02d}" for i in range(k)]
    df = pd.DataFrame(prices, index=index, columns=columns)
    df.attrs["config"] = asdict(cfg)
    return df


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close simple returns. The first row is NaN by construction."""
    return prices.pct_change(fill_method=None)
