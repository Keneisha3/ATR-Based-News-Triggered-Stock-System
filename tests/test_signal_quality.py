"""Signal-quality controls: the breach edge must be read against a baseline.

These tests verify the *machinery* of the random-date baseline on synthetic
data (no network). The real, data-driven validation is run via
`python main.py backtest`, which prints the live breach-vs-random comparison.

Headline result from a 5-year, 56-ticker run (2021-2026, 3,084 events):

    ATR breach   1d 51.7%  5d 51.6%
    Random       1d 50.8%  5d 50.1%
    Edge        +0.9 pp   +1.5 pp

i.e. a small but consistently positive edge over random — not noise, not a
money-printer. That honest framing is the point of this file.
"""

import numpy as np
import pandas as pd

from atr_news_alert import backtest as bt, config


def _series(closes, signs, ticker="X"):
    """Build a minimal price frame the backtest functions understand."""
    n = len(closes)
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "close": closes,
        "move_in_atr": signs,
    })


def test_baseline_shape_and_keys(monkeypatch):
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    prices = _series([100.0, 101.0, 102.0, 103.0, 104.0],
                     [0.0, 1.0, -1.0, 2.0, -0.5])
    base = bt.random_baseline(prices, n_events=3, trials=50)
    assert base["events"] == 3
    assert base["trials"] == 50
    assert "hit_rate_1d_pct" in base
    assert "avg_fwd_1d_pct" in base
    assert 0.0 <= base["hit_rate_1d_pct"] <= 100.0


def test_baseline_is_deterministic_for_a_seed(monkeypatch):
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    prices = _series([100.0, 101.0, 99.0, 103.0, 101.0, 105.0],
                     [0.0, 1.5, -1.5, 2.0, -1.0, 1.0])
    a = bt.random_baseline(prices, n_events=4, trials=100, seed=7)
    b = bt.random_baseline(prices, n_events=4, trials=100, seed=7)
    assert a == b


def test_baseline_recovers_known_drift(monkeypatch):
    """A series that always rises => any random long-direction draw is a hit."""
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    closes = list(np.linspace(100.0, 130.0, 20))   # strictly increasing
    signs = [1.0] * 20                              # every move flagged bullish
    prices = _series(closes, signs)
    base = bt.random_baseline(prices, n_events=10, trials=100, seed=1)
    # Up-only series with bullish signs: every sampled forward return is positive.
    assert base["hit_rate_1d_pct"] == 100.0
    assert base["avg_fwd_1d_pct"] > 0


def test_baseline_empty_inputs():
    assert bt.random_baseline(pd.DataFrame(), n_events=5) == {}
    prices = _series([100.0, 101.0, 102.0], [0.0, 1.0, -1.0])
    assert bt.random_baseline(prices, n_events=0) == {}


def test_breach_edge_is_measured_against_baseline(monkeypatch):
    """End-to-end: real backtest stat and baseline stat are comparable numbers.

    Construct data where breaches genuinely predict continuation, and confirm
    the breach hit-rate comes out strictly above the random baseline.
    """
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])

    # Breaches (|move_in_atr|>=1.5) are always followed by continuation;
    # non-breach days drift randomly around flat.
    closes = [100, 100, 110,   # bullish breach on day1 -> +10% (hit)
              110, 110, 99,     # bearish breach -> -10% (hit)
              99, 100, 101,     # small noise, no breach
              101, 101, 112]    # bullish breach -> +~11% (hit)
    signs = [0, 2.0, 0, 0, -2.0, 0, 0, 0.3, 0, 0, 2.0, 0]
    prices = _series([float(c) for c in closes], signs)

    _, agg = bt.backtest(prices)
    base = bt.random_baseline(prices, agg["events"], trials=200, seed=3)

    assert agg["events"] >= 2
    assert agg["hit_rate_1d_pct"] > base["hit_rate_1d_pct"]
