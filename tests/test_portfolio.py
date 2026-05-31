"""Event-driven portfolio simulation: timing, costs, overlap, and metrics.

All synthetic / offline. Verifies the *mechanics* are correct (no look-ahead,
costs bite, overlap diversifies, metrics compute) so the numbers from the real
`python main.py portfolio` run can be trusted.
"""

import numpy as np
import pandas as pd

from atr_news_alert import portfolio as pf, config


def _frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "close", "move_in_atr"])


def _ramp(ticker, closes, signs, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"ticker": ticker, "date": dates,
                         "close": [float(c) for c in closes], "move_in_atr": signs})


def test_enters_day_after_breach_no_lookahead(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Breach is flagged on day1 (index1). Price only rises *after* it (day1->day2).
    # The breach-day return itself must NOT be captured (that would be look-ahead).
    prices = _ramp("X", [100, 100, 110, 110], [0, 2.0, 0, 0])
    m = pf.simulate(prices, hold=1, cost_bps=0)
    # Entry at close of day1 (=100), exit after 1 day at day2 close (=110): +10%.
    assert round(m["total_return_pct"], 1) == 10.0


def test_costs_reduce_return(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _ramp("X", [100, 100, 110, 110], [0, 2.0, 0, 0])
    free = pf.simulate(prices, hold=1, cost_bps=0)["total_return_pct"]
    costed = pf.simulate(prices, hold=1, cost_bps=50)["total_return_pct"]
    assert costed < free


def test_short_position_profits_when_price_falls(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Bearish breach on day1, price then falls 100 -> 90: a short gains +10%.
    prices = _ramp("X", [100, 100, 90, 90], [0, -2.0, 0, 0])
    m = pf.simulate(prices, hold=1, cost_bps=0)
    assert round(m["total_return_pct"], 1) == 10.0


def test_overlap_is_equal_weighted_not_leveraged(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Two names both breach bullish on day1; both rise 10% on day2.
    # Equal weight => portfolio earns 10% (the average), NOT 20% (leverage).
    a = _ramp("A", [100, 100, 110, 110], [0, 2.0, 0, 0])
    b = _ramp("B", [100, 100, 110, 110], [0, 2.0, 0, 0])
    m = pf.simulate(pd.concat([a, b], ignore_index=True), hold=1, cost_bps=0)
    assert round(m["total_return_pct"], 1) == 10.0


def test_metrics_present_and_typed(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _ramp("X", [100, 100, 110, 105, 108, 108], [0, 2.0, 0, -2.0, 0, 0])
    m = pf.simulate(prices, hold=2, cost_bps=5)
    for k in ("total_return_pct", "benchmark_return_pct", "cagr_pct",
              "sharpe", "max_drawdown_pct", "trades", "pct_days_invested"):
        assert k in m
    assert m["max_drawdown_pct"] <= 0          # drawdown is non-positive
    assert m["trades"] >= 1
    assert isinstance(pf.format_summary(m), str)


def test_long_only_skips_bearish_breaches(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # A bullish breach (day1) and a bearish breach (day3); price falls after the
    # bearish one. Long/short would *profit* from the short; long-only ignores it.
    prices = _ramp("X", [100, 100, 110, 110, 99, 99], [0, 2.0, 0, -2.0, 0, 0])
    ls = pf.simulate(prices, hold=1, cost_bps=0, long_only=False)
    lo = pf.simulate(prices, hold=1, cost_bps=0, long_only=True)
    assert ls["long_only"] is False and lo["long_only"] is True
    # Long-only takes fewer trades (the bearish breach is skipped).
    assert lo["trades"] < ls["trades"]
    # Long-only never holds a short, so it cannot earn the +10% from the fall.
    assert lo["total_return_pct"] < ls["total_return_pct"]


def test_long_only_appears_in_summary(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _ramp("X", [100, 100, 110, 110], [0, 2.0, 0, 0])
    assert "long-only" in pf.format_summary(pf.simulate(prices, hold=1, long_only=True))
    assert "long/short" in pf.format_summary(pf.simulate(prices, hold=1, long_only=False))


def test_trade_log_reconstructs_discrete_trades(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _ramp("X", [100, 100, 110, 110, 99, 99], [0, 2.0, 0, -2.0, 0, 0])
    m = pf.simulate(prices, hold=1, cost_bps=0)
    trades = pf.trade_log(m["frame"])
    assert len(trades) == 2                       # one long, one short
    assert set(trades["direction"]) == {"long", "short"}
    assert (trades["days_held"] == 1).all()


def test_positions_timeline_counts_exposure(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    a = _ramp("A", [100, 100, 110, 110], [0, 2.0, 0, 0])
    b = _ramp("B", [100, 100, 90, 90], [0, -2.0, 0, 0])
    m = pf.simulate(pd.concat([a, b], ignore_index=True), hold=1, cost_bps=0)
    tl = pf.positions_timeline(m["frame"])
    # On the entry day both are open: one long, one short.
    assert tl["n_long"].max() == 1
    assert tl["n_short"].max() == 1
    assert tl["n_active"].max() == 2


def test_sweep_grid_and_stability(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = _ramp("X", [100, 100, 110, 108, 112, 110, 115, 113],
                   [0, 2.0, 0, -2.0, 0, 2.0, 0, 0])
    grid, summary = pf.sweep(prices, holds=[1, 2], costs=[0.0, 5.0])
    assert len(grid) == 4                          # 2 holds x 2 costs
    assert set(grid.columns) >= {"hold_days", "cost_bps", "sharpe", "cagr_pct"}
    assert summary["cells"] == 4
    assert "stability_score" in summary
    assert "best" in summary


def test_sweep_empty_prices():
    grid, summary = pf.sweep(pd.DataFrame(), holds=[1], costs=[0.0])
    assert grid.empty and summary == {}


def test_empty_prices_returns_empty():
    assert pf.simulate(pd.DataFrame()) == {}
    assert "No portfolio" in pf.format_summary({})
