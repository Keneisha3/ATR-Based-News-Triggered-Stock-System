"""Portfolio comparison: each basket is evaluated and presented as a row."""

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import portfolios, config


@pytest.fixture(autouse=True)
def _atr(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)


def _prices(tickers, n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    out = []
    for i, t in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(0.0008, 0.02, n))
        atr = pd.Series(close).rolling(14).std().bfill().to_numpy() + 1e-6
        move = np.concatenate([[0], np.diff(close)]) / atr
        out.append(pd.DataFrame({"ticker": t, "date": dates, "close": close,
                                 "move_in_atr": move}))
    return pd.concat(out, ignore_index=True)


def test_compare_returns_row_per_basket():
    presets = {"A": ["AAPL", "MSFT"], "B": ["TSLA"], "All": []}
    prices = _prices(["AAPL", "MSFT", "TSLA"])
    df = portfolios.compare(prices, presets)
    assert set(df["portfolio"]) == {"A", "B", "All"}
    # "All" basket spans every ticker present.
    assert df[df["portfolio"] == "All"]["tickers"].iloc[0] == 3


def test_compare_has_headline_metrics():
    prices = _prices(["AAPL", "MSFT", "NVDA"])
    df = portfolios.compare(prices, {"Tech": ["AAPL", "MSFT", "NVDA"]})
    row = df.iloc[0]
    for col in ("events", "hit_5d_pct", "edge_pp", "longonly_return_pct",
                "longonly_sharpe"):
        assert col in row.index


def test_format_table_renders():
    prices = _prices(["AAPL", "MSFT"])
    out = portfolios.format_table(portfolios.compare(prices, {"X": ["AAPL", "MSFT"]}))
    assert "portfolio" in out and "X" in out


def test_empty_prices_table():
    assert "No data" in portfolios.format_table(pd.DataFrame())
