"""Backtest forward-return / hit-rate math on a hand-built price series."""

from atr_news_alert import backtest as bt, config
import pandas as pd


def test_forward_returns_and_hit_rate(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])

    # One upward breach on day 1 (move_in_atr 2.0), price then rises 100->110.
    # That is a "hit" (price continued in the breach direction) of +10%.
    prices = pd.DataFrame({
        "ticker": ["X"] * 3,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "close": [100.0, 100.0, 110.0],
        "move_in_atr": [0.0, 2.0, 0.0],
    })
    per_ticker, agg = bt.backtest(prices)
    assert agg["events"] == 1
    assert agg["hit_rate_1d_pct"] == 100.0
    assert agg["avg_fwd_1d_pct"] == 10.0
    assert per_ticker.iloc[0]["ticker"] == "X"


def test_downward_breach_hit_when_price_falls(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])

    # Downward breach (move_in_atr -2.0), price then falls 100->90: a hit.
    prices = pd.DataFrame({
        "ticker": ["Y"] * 3,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "close": [100.0, 100.0, 90.0],
        "move_in_atr": [0.0, -2.0, 0.0],
    })
    _, agg = bt.backtest(prices)
    assert agg["events"] == 1
    assert agg["hit_rate_1d_pct"] == 100.0     # fell after a bearish breach
    assert agg["avg_fwd_1d_pct"] == 10.0       # +10% measured in breach direction


def test_no_events_below_threshold(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    prices = pd.DataFrame({
        "ticker": ["Z"] * 3,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "close": [100.0, 101.0, 102.0],
        "move_in_atr": [0.0, 0.5, 0.4],
    })
    per_ticker, agg = bt.backtest(prices)
    assert agg.get("events") == 0
    assert per_ticker.empty


def _two_ticker_frame():
    """A (bullish breach) on day2, B (bearish breach) on day2, both continue."""
    return pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "B"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
        "close": [100.0, 100.0, 110.0, 100.0, 100.0, 90.0],
        "move_in_atr": [0.0, 2.0, 0.0, 0.0, -2.0, 0.0],
    })


def test_by_sector_splits_and_maps(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    sec = bt.by_sector(_two_ticker_frame(), {"A": "Tech", "B": "Energy"})
    assert set(sec["sector"]) == {"Tech", "Energy"}
    # Both breaches "hit" (price continued in breach direction).
    assert (sec["hit_rate_1d"] == 100.0).all()


def test_by_sector_unmapped_ticker_falls_back_to_other(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    sec = bt.by_sector(_two_ticker_frame(), {"A": "Tech"})  # B unmapped
    assert "Other" in set(sec["sector"])


def test_by_year_groups_by_calendar_year(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    df = _two_ticker_frame()
    df.loc[df["ticker"] == "B", "date"] = pd.to_datetime(
        ["2025-01-01", "2025-01-02", "2025-01-03"])
    yr = bt.by_year(df)
    assert list(yr["year"]) == [2024, 2025]   # sorted ascending
    assert (yr["events"] == 1).all()


def test_breach_events_has_signed_columns(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    ev = bt.breach_events(_two_ticker_frame())
    assert len(ev) == 2
    assert "signed_1d" in ev.columns
    # signed return is positive for both (continuation in breach direction)
    assert (ev["signed_1d"] > 0).all()
