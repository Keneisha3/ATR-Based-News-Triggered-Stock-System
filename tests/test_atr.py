"""ATR / volatility math — verifies the core threshold engine (check #2's basis)."""

import numpy as np
import pandas as pd

from atr_news_alert import prices as price_mod


def _ohlc():
    return pd.DataFrame({
        "open":  [10.0, 11.0, 10.5, 12.0],
        "high":  [10.5, 11.5, 11.0, 12.5],
        "low":   [9.5, 10.5, 10.0, 11.5],
        "close": [10.0, 11.0, 10.5, 12.0],
        "volume": [100, 100, 100, 100],
    })


def test_true_range_matches_hand_calc():
    tr = price_mod._true_range(_ohlc())
    # bar0: only H-L (no prev close). then max of the three TR components.
    assert tr.tolist() == [1.0, 1.5, 1.0, 2.0]


def test_compute_atr_columns():
    df = price_mod.compute_atr(_ohlc(), period=2)
    # ATR is a positive smoothed average of True Range.
    assert (df["atr"].dropna() > 0).all()
    # atr_pct = atr / close.
    np.testing.assert_allclose(df["atr_pct"], df["atr"] / df["close"])
    # move_pct = close-over-close return; bar1 = 11/10 - 1 = 0.10.
    assert df["move_pct"].iloc[1] == pytest_approx(0.10)


def test_move_in_atr_uses_prior_atr_and_sign():
    df = price_mod.compute_atr(_ohlc(), period=2)
    # last bar moved up (10.5 -> 12.0) so move_in_atr must be positive.
    assert df["move_in_atr"].iloc[-1] > 0
    expected = (12.0 - 10.5) / df["atr"].shift(1).iloc[-1]
    assert df["move_in_atr"].iloc[-1] == pytest_approx(expected)


def test_latest_bar_breach_and_direction():
    prices = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02",
                                "2026-01-01", "2026-01-02"]),
        "close": [100, 105, 50, 49],
        "atr": [2.0, 2.0, 1.0, 1.0],
        "atr_pct": [0.02, 0.02, 0.02, 0.02],
        "move_pct": [0.0, 0.05, 0.0, -0.02],
        "move_in_atr": [0.0, 2.5, 0.0, 0.5],  # AAA breaches 1.5, BBB does not
    })
    last = price_mod.latest_bar(prices)
    aaa = last[last["ticker"] == "AAA"].iloc[0]
    bbb = last[last["ticker"] == "BBB"].iloc[0]
    assert aaa["breach"] and aaa["direction"] == "bullish"
    assert not bbb["breach"] and bbb["direction"] == "bearish"


# tiny local approx helper to avoid importing pytest at module top repeatedly
def pytest_approx(v, tol=1e-9):
    import pytest
    return pytest.approx(v, abs=tol)
