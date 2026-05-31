"""Event study: return convention, non-leakage, regime stratification.

Offline / synthetic. The critical guarantees: the curve uses the same
breach-direction convention as the backtest, day 0 is always zero (no
look-ahead into the entry), and regime labels are trailing-only.
"""

import numpy as np
import pandas as pd

from atr_news_alert import signal_analyzer as sa, config


def _one_ticker(closes, signs, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"ticker": "X", "date": dates,
                         "close": [float(c) for c in closes],
                         "move_in_atr": signs})


def test_day0_is_zero_and_convention_matches_backtest(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Bullish breach on day1, price rises 100 -> 110 -> 121 (compounding).
    prices = _one_ticker([100, 100, 110, 121], [0, 2.0, 0, 0])
    study = sa.event_study(prices, max_days=2, by_regime=False)
    ov = study["overall"].set_index("event_day")
    assert ov.loc[0, "avg_cum_return_pct"] == 0.0          # no look-ahead at entry
    assert round(ov.loc[1, "avg_cum_return_pct"], 1) == 10.0   # +10% by day 1
    assert round(ov.loc[2, "avg_cum_return_pct"], 1) == 21.0   # +21% cumulative by day 2


def test_bearish_breach_counts_continuation_as_positive(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Bearish breach, price falls 100 -> 90: measured in breach direction that is +10%.
    prices = _one_ticker([100, 100, 90, 90], [0, -2.0, 0, 0])
    ov = sa.event_study(prices, max_days=1, by_regime=False)["overall"].set_index("event_day")
    assert round(ov.loc[1, "avg_cum_return_pct"], 1) == 10.0
    assert ov.loc[1, "win_rate_pct"] == 100.0


def test_win_rate_nan_at_day0(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _one_ticker([100, 100, 110, 110], [0, 2.0, 0, 0])
    ov = sa.event_study(prices, max_days=1, by_regime=False)["overall"].set_index("event_day")
    assert pd.isna(ov.loc[0, "win_rate_pct"])


def test_no_lookahead_near_series_end(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Breach on the last usable bar -> future offsets are NaN, not fabricated.
    prices = _one_ticker([100, 100, 110], [0, 0, 2.0])
    ov = sa.event_study(prices, max_days=3, by_regime=False)["overall"].set_index("event_day")
    assert ov.loc[1, "n_events"] == 0      # no data after the final bar


def test_market_regime_is_trailing_only(monkeypatch):
    # First `ma_window` days cannot be classified (no trailing window yet).
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = pd.DataFrame({"ticker": "X", "date": dates,
                           "close": np.linspace(100, 160, 60),
                           "move_in_atr": 0.0})
    regime = sa.market_regime(prices, ma_window=20)
    assert (regime.iloc[:19] == "unknown").all()
    # Rising index sits above its trailing MA -> bull once classifiable.
    assert regime.iloc[-1] == "bull"


def test_regime_stratification_splits_events(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    closes = np.linspace(100, 200, 80)
    signs = np.zeros(80); signs[70] = 2.0          # a breach late in an uptrend
    prices = pd.DataFrame({"ticker": "X", "date": dates,
                           "close": closes, "move_in_atr": signs})
    study = sa.event_study(prices, max_days=3, by_regime=True)
    assert "regime_counts" in study
    assert study["n_events"] == 1


def test_empty_prices():
    assert sa.event_study(pd.DataFrame()) == {}
    assert "No breach events" in sa.format_summary({})
