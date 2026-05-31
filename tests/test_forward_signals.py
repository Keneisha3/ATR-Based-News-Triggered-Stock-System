"""Decision layer: forward signals surface today's breaches with regime context.

The key guarantees: only latest-bar breaches are reported, the expected drift is
regime-conditioned (not the ticker's own future), and news confirmation flows
through to the action text.
"""

import numpy as np
import pandas as pd

from atr_news_alert import signal_analyzer as sa, config


def _series(closes, signs, ticker="X", start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"ticker": ticker, "date": dates,
                         "close": [float(c) for c in closes], "move_in_atr": signs})


def test_only_latest_bar_breaches_reported(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Breach is in the middle, not on the last bar -> no signal today.
    prices = _series([100, 100, 110, 110, 111], [0, 2.0, 0, 0, 0.2])
    res = sa.forward_signals(prices, horizon=2)
    assert res["signals"] == []

    # Now the last bar itself breaches.
    prices2 = _series([100, 100, 110, 110, 130], [0, 2.0, 0, 0, 2.0])
    res2 = sa.forward_signals(prices2, horizon=2)
    assert [s["ticker"] for s in res2["signals"]] == ["X"]
    assert res2["signals"][0]["breach"] == "bullish"


def test_news_confirmation_flows_to_action(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _series([100, 100, 110, 110, 130], [0, 2.0, 0, 0, 2.0])
    summary = pd.DataFrame({"ticker": ["X"]})
    res = sa.forward_signals(prices, horizon=2, news_summary=summary)
    assert res["signals"][0]["has_news"] is True
    assert "news-confirmed" in res["signals"][0]["action"]

    res_none = sa.forward_signals(prices, horizon=2)
    assert "no fresh news" in res_none["signals"][0]["action"]


def test_regime_reported_and_drift_band_present(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    rng = np.random.default_rng(1)
    n = 120
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    signs = np.zeros(n)
    signs[::7] = 2.0          # periodic breaches to populate the regime pool
    signs[-1] = 2.0           # a breach today
    prices = _series(close, signs)
    res = sa.forward_signals(prices, horizon=3)
    assert res["regime"] in {"bull", "bear", "unknown"}
    assert "continuation" in res
    assert len(res["signals"]) >= 1


def test_format_signals_handles_no_breaches(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _series([100, 101, 102, 103], [0, 0.1, 0.1, 0.1])
    out = sa.format_signals(sa.forward_signals(prices))
    assert "Nothing to act on" in out


def test_empty_prices():
    res = sa.forward_signals(pd.DataFrame())
    assert res["signals"] == [] and res["regime"] == "unknown"
