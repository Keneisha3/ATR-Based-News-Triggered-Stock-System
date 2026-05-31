"""Price ingestion and ATR computation.

Pulls daily OHLC from Yahoo Finance (yfinance, no API key) and computes
Wilder's Average True Range, the core volatility measure for the alert engine.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config

warnings.filterwarnings("ignore")  # silence yfinance/urllib3 noise


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H-prevClose|, |L-prevClose|)."""
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _wilder_atr(tr: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) of True Range."""
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.DataFrame:
    """Add tr, atr, atr_pct, move_pct and move_in_atr columns to an OHLC frame."""
    df = df.copy()
    df["tr"] = _true_range(df)
    df["atr"] = _wilder_atr(df["tr"], period)
    df["atr_pct"] = df["atr"] / df["close"]

    prev_close = df["close"].shift(1)
    df["move_pct"] = (df["close"] - prev_close) / prev_close
    # Today's close-to-close move measured in units of *yesterday's* ATR.
    df["move_in_atr"] = (df["close"] - prev_close) / df["atr"].shift(1)
    return df


def _flatten_yf(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance returns MultiIndex columns for multi-ticker pulls; normalize."""
    if isinstance(raw.columns, pd.MultiIndex):
        # columns like ('Close', 'AAPL') -> select this ticker's level
        raw = raw.xs(ticker, axis=1, level=1)
    raw = raw.rename(columns=str.lower)
    keep = ["open", "high", "low", "close", "volume"]
    raw = raw[[c for c in keep if c in raw.columns]]
    return raw


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Download OHLC for each ticker and attach ATR columns.

    Returns a long (tidy) frame: one row per ticker/date.
    """
    import yfinance as yf

    frames = []
    for t in tickers:
        try:
            raw = yf.download(
                t,
                period=config.PRICE_LOOKBACK,
                interval=config.PRICE_INTERVAL,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:  # network / symbol issues shouldn't kill the run
            print(f"  ! price fetch failed for {t}: {exc}")
            continue
        if raw is None or raw.empty:
            print(f"  ! no price data for {t}")
            continue

        df = _flatten_yf(raw, t)
        if df.empty or "close" not in df:
            continue
        df = compute_atr(df)
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        df["ticker"] = t
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    cols = ["ticker", "date", "open", "high", "low", "close", "volume",
            "tr", "atr", "atr_pct", "move_pct", "move_in_atr"]
    return out[[c for c in cols if c in out.columns]]


def latest_bar(prices: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per ticker, with a `breach` flag set."""
    if prices.empty:
        return prices
    last = prices.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()
    last["abs_move_in_atr"] = last["move_in_atr"].abs()
    last["breach"] = last["abs_move_in_atr"] >= config.ATR_BREACH_MULT
    last["direction"] = np.where(last["move_pct"] >= 0, "bullish", "bearish")
    return last


def save_prices(prices: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(config.PRICES_CSV, index=False)
