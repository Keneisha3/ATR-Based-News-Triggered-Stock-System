"""Do ATR breaches precede meaningful moves?

For every historical bar that breaches the ATR threshold, measure forward returns
over the configured horizons. Reports the hit-rate (did price keep moving in the
breach direction) and the average forward return, in aggregate and broken down by
ticker, sector and year, alongside a random-date baseline so the edge can be read
against chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add fwd_<h>d (forward % return) columns, computed per ticker."""
    prices = prices.sort_values(["ticker", "date"]).copy()
    closes = prices.groupby("ticker")["close"]
    for h in config.FORWARD_HORIZONS:
        prices[f"fwd_{h}d"] = closes.shift(-h) / prices["close"] - 1.0
    return prices


def breach_events(prices: pd.DataFrame) -> pd.DataFrame:
    """Every historical ATR-breach bar, with sign and forward-return columns.

    The shared substrate for every breakdown below (overall, per-ticker, per
    sector, per year): one row per breach event, `sign` = breach direction,
    `fwd_<h>d` = raw forward return, `signed_<h>d` = return in breach direction.
    """
    if prices.empty:
        return pd.DataFrame()
    prices = _forward_returns(prices)
    breaches = prices[prices["move_in_atr"].abs() >= config.ATR_BREACH_MULT].copy()
    if breaches.empty:
        return breaches
    breaches["sign"] = np.sign(breaches["move_in_atr"])
    for h in config.FORWARD_HORIZONS:
        breaches[f"signed_{h}d"] = breaches["sign"] * breaches[f"fwd_{h}d"]
    return breaches


def _signed_stats(signed: pd.Series) -> tuple[float, float]:
    """(avg forward return %, hit-rate %) for a series of breach-direction returns."""
    signed = signed.dropna()
    if signed.empty:
        return float("nan"), float("nan")
    return round(float(signed.mean()) * 100, 3), round(float((signed > 0).mean()) * 100, 1)


def _grouped_stats(breaches: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    """Hit-rate / avg-return per group (ticker, sector, year, ...)."""
    rows = []
    for key, grp in breaches.groupby(by):
        rec = {label: key, "events": len(grp)}
        for h in config.FORWARD_HORIZONS:
            avg, hit = _signed_stats(grp[f"signed_{h}d"])
            rec[f"avg_fwd_{h}d"] = avg
            rec[f"hit_rate_{h}d"] = hit
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("events", ascending=False)


def backtest(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (per-ticker stats frame, aggregate stats dict)."""
    if prices.empty:
        return pd.DataFrame(), {}
    breaches = breach_events(prices)
    if breaches.empty:
        return pd.DataFrame(), {"events": 0}

    per_ticker = _grouped_stats(breaches, "ticker", "ticker")

    agg = {"events": int(len(breaches)), "tickers": int(breaches["ticker"].nunique())}
    for h in config.FORWARD_HORIZONS:
        avg, hit = _signed_stats(breaches[f"signed_{h}d"])
        agg[f"avg_fwd_{h}d_pct"] = avg
        agg[f"hit_rate_{h}d_pct"] = hit
    return per_ticker, agg


def by_sector(prices: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Hit-rate and average return per sector, to show where the signal works."""
    breaches = breach_events(prices)
    if breaches.empty:
        return pd.DataFrame()
    breaches = breaches.copy()
    breaches["sector"] = breaches["ticker"].map(sectors).fillna("Other")
    return _grouped_stats(breaches, "sector", "sector")


def by_year(prices: pd.DataFrame) -> pd.DataFrame:
    """Hit-rate and average return per calendar year, to expose instability.

    A rough check on durability: if the edge only shows up in one or two years it
    is fragile rather than a lasting signal.
    """
    breaches = breach_events(prices)
    if breaches.empty:
        return pd.DataFrame()
    breaches = breaches.copy()
    breaches["year"] = pd.to_datetime(breaches["date"]).dt.year
    return _grouped_stats(breaches, "year", "year").sort_values("year")


def random_baseline(prices: pd.DataFrame, n_events: int, *,
                    trials: int = 200, seed: int | None = None) -> dict:
    """Control experiment: is the breach edge better than picking random days?

    Draws `n_events` random (ticker, date) rows `trials` times and applies the
    exact same "return measured in the move's direction" rule the real backtest
    uses. Returns the mean hit-rate / avg forward return across trials, so the
    breach numbers can be read against a like-for-like null hypothesis.
    """
    if prices.empty or n_events <= 0:
        return {}
    if seed is None:
        seed = config.BASELINE_SEED

    prices = _forward_returns(prices)
    prices = prices.dropna(subset=["move_in_atr"])
    if prices.empty:
        return {}

    rng = np.random.default_rng(seed)
    out = {"events": n_events, "trials": trials}
    for h in config.FORWARD_HORIZONS:
        col = f"fwd_{h}d"
        pool = prices.dropna(subset=[col])
        if pool.empty:
            out[f"avg_fwd_{h}d_pct"] = np.nan
            out[f"hit_rate_{h}d_pct"] = np.nan
            continue
        signed = (np.sign(pool["move_in_atr"]) * pool[col]).to_numpy()
        size = min(n_events, len(signed))
        hits, avgs = [], []
        for _ in range(trials):
            sample = signed[rng.integers(0, len(signed), size=size)]
            hits.append(float((sample > 0).mean()))
            avgs.append(float(sample.mean()))
        out[f"avg_fwd_{h}d_pct"] = round(float(np.mean(avgs)) * 100, 3)
        out[f"hit_rate_{h}d_pct"] = round(float(np.mean(hits)) * 100, 1)
    return out


def save_backtest(per_ticker: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_ticker.to_csv(config.BACKTEST_CSV, index=False)
