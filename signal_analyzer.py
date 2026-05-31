"""Event study of price behaviour after a breach, independent of any strategy.

This module measures what price does in the days after a breach, tracing the
average cumulative return from the breach (t=0) out to +N days and splitting it
by market regime.

It works at the signal level rather than the strategy level. It does not import
the portfolio code and makes no hold-period, cost, weighting or execution
assumption, so it cannot restate what a strategy already assumes. A breach is
defined only from data up to t (move_in_atr is a close-to-close value known at
the close of t), and the regime label for an event uses a trailing moving average
as of the entry day. Returns use the same convention as the backtest, measured in
the breach direction as sign(move) x (close[t+d]/close[t] - 1), so a hit means
the move continued.

Regime is derived from an equal-weight index of the universe (its cross-sectional
mean daily return), so the study is self-contained and reproducible from cached
prices. Swapping in SPY would be a one-line change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def market_regime(prices: pd.DataFrame, ma_window: int | None = None) -> pd.Series:
    """Per-date 'bull'/'bear' label from an equal-weight universe index.

    Index = cumulative equal-weight mean daily return; bull when the index is at
    or above its trailing `ma_window` moving average. Trailing-only => non-leaky.
    Returns a Series indexed by date.
    """
    ma_window = ma_window or config.REGIME_MA_WINDOW
    if prices.empty:
        return pd.Series(dtype=object)

    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    mkt = df.groupby("date")["ret"].mean()
    index = (1.0 + mkt.fillna(0.0)).cumprod()
    ma = index.rolling(ma_window, min_periods=ma_window).mean()
    regime = np.where(index >= ma, "bull", "bear")
    regime = pd.Series(regime, index=index.index, name="regime")
    regime[ma.isna()] = "unknown"  # not enough history to classify yet
    return regime


def _event_offsets(prices: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """One row per breach event with signed cumulative return at each offset 0..N."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["sign"] = np.sign(df["move_in_atr"])
    closes = df.groupby("ticker")["close"]
    for d in range(0, max_days + 1):
        fwd = closes.shift(-d)
        df[f"sret_{d}"] = df["sign"] * (fwd / df["close"] - 1.0)
    return df[df["move_in_atr"].abs() >= config.ATR_BREACH_MULT].copy()


def _curve(events: pd.DataFrame, offsets: range) -> pd.DataFrame:
    """Aggregate signed cumulative return + win-rate at each event day."""
    rows = []
    for d in offsets:
        s = events[f"sret_{d}"].dropna()
        avg = round(float(s.mean()) * 100, 4) if len(s) else float("nan")
        win = (round(float((s > 0).mean()) * 100, 1)
               if len(s) and d > 0 else float("nan"))
        rows.append({"event_day": d, "n_events": int(len(s)),
                     "avg_cum_return_pct": avg, "win_rate_pct": win})
    return pd.DataFrame(rows)


def event_study(prices: pd.DataFrame, *, max_days: int | None = None,
                by_regime: bool = True) -> dict:
    """Compute the event-time response, overall and (optionally) by regime.

    Returns {"max_days", "n_events", "overall": df, "bull": df, "bear": df,
    "regime_counts": {...}} where each df has columns
    event_day / n_events / avg_cum_return_pct / win_rate_pct.
    """
    max_days = max_days or config.EVENT_STUDY_MAX_DAYS
    if prices.empty:
        return {}
    events = _event_offsets(prices, max_days)
    if events.empty:
        return {"max_days": max_days, "n_events": 0, "overall": pd.DataFrame()}

    offsets = range(0, max_days + 1)
    out = {"max_days": max_days, "n_events": int(len(events)),
           "overall": _curve(events, offsets)}

    if by_regime:
        regime = market_regime(prices)
        events = events.copy()
        events["regime"] = events["date"].map(regime)
        counts = events["regime"].value_counts().to_dict()
        out["regime_counts"] = {k: int(v) for k, v in counts.items()}
        for label in ("bull", "bear"):
            sub = events[events["regime"] == label]
            out[label] = _curve(sub, offsets) if not sub.empty else pd.DataFrame()
    return out


def _bootstrap_ci(samples: np.ndarray, *, n_boot: int = 1000,
                  seed: int | None = None, lo: float = 5, hi: float = 95):
    """Percentile confidence interval for the mean, via bootstrap. Seeded."""
    if len(samples) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(config.BASELINE_SEED if seed is None else seed)
    means = np.array([samples[rng.integers(0, len(samples), len(samples))].mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def forward_signals(prices: pd.DataFrame, *, horizon: int = 3,
                    news_summary: pd.DataFrame | None = None,
                    seed: int | None = None) -> dict:
    """For each ticker that breached on the latest bar, return the regime-
    conditioned expected drift (with a bootstrap band) and an action.

    This turns the retrospective event study into a daily decision aid. Expected
    drift is read from the current market regime's post-breach distribution, not
    from this ticker's own future.
    """
    if prices.empty:
        return {"regime": "unknown", "horizon": horizon, "signals": []}

    regime = market_regime(prices)
    regime_now = regime.iloc[-1] if len(regime) else "unknown"

    # Regime-conditioned post-breach distribution at the chosen horizon.
    events = _event_offsets(prices, horizon)
    events = events.copy()
    events["regime"] = events["date"].map(regime)
    pool = events.loc[events["regime"] == regime_now, f"sret_{horizon}"].dropna().to_numpy()
    drift_pct = float(np.mean(pool)) * 100 if len(pool) else float("nan")
    hit_rate = float((pool > 0).mean()) * 100 if len(pool) else float("nan")
    lo, hi = _bootstrap_ci(pool, seed=seed)
    band = (round(lo * 100, 3), round(hi * 100, 3)) if len(pool) else (None, None)
    # Continuation if breaches tend to keep going in this regime; else reversion.
    continuation = bool(drift_pct > 0) if not np.isnan(drift_pct) else True

    news_tickers = set(news_summary["ticker"]) if (
        news_summary is not None and not news_summary.empty) else set()

    # Latest bar per ticker -> who breached today, and which way.
    last = prices.sort_values("date").groupby("ticker").tail(1)
    signals = []
    for _, r in last.iterrows():
        if abs(r["move_in_atr"]) < config.ATR_BREACH_MULT:
            continue
        direction = "bullish" if r["move_in_atr"] > 0 else "bearish"
        d = 1 if r["move_in_atr"] > 0 else -1
        # Expected price-direction move for THIS ticker = d * breach-direction drift.
        exp_move = d * drift_pct if not np.isnan(drift_pct) else float("nan")
        has_news = r["ticker"] in news_tickers
        signals.append({
            "ticker": r["ticker"],
            "breach": direction,
            "move_in_atr": round(float(r["move_in_atr"]), 2),
            "expected_move_pct": round(exp_move, 3) if not np.isnan(exp_move) else None,
            "has_news": has_news,
            "action": _action(direction, continuation, has_news),
        })
    signals.sort(key=lambda s: abs(s["move_in_atr"]), reverse=True)
    return {"regime": regime_now, "horizon": horizon,
            "drift_pct": round(drift_pct, 3) if not np.isnan(drift_pct) else None,
            "drift_band": band, "n_regime_events": int(len(pool)),
            "regime_hit_rate": round(hit_rate, 1) if not np.isnan(hit_rate) else None,
            "continuation": continuation, "signals": signals}


def _action(direction: str, continuation: bool, has_news: bool) -> str:
    """Plain-language recommendation from regime behaviour and news confirmation."""
    if continuation:
        base = ("watch for upside follow-through" if direction == "bullish"
                else "watch for downside follow-through, caution on longs")
    else:  # regime favours reversion
        base = ("reversion risk, fade the pop or take profits" if direction == "bullish"
                else "reversion bounce possible, avoid chasing the drop")
    return base + (" [news-confirmed]" if has_news else " [no fresh news]")


def format_signals(res: dict) -> str:
    """Daily decision-aid block."""
    if not res or not res.get("signals"):
        return (f"No ATR breaches on the latest bar "
                f"(market regime: {res.get('regime', 'unknown')}). Nothing to act on.")
    band = res.get("drift_band") or (None, None)
    mech = "momentum/continuation" if res["continuation"] else "mean-reversion"
    L = [f"Market regime: {res['regime'].upper()}.  Post-breach behaviour: {mech}",
         f"Expected {res['horizon']}-day drift in breach direction: "
         f"{res['drift_pct']:+.2f}%  (90% band {band[0]:+.2f}%..{band[1]:+.2f}%, "
         f"n={res['n_regime_events']:,})"
         if res.get("drift_pct") is not None else "Expected drift: n/a",
         "",
         f"  {'ticker':8}{'breach':9}{'ATR':>6}{'exp move':>10}  action"]
    for s in res["signals"]:
        em = f"{s['expected_move_pct']:+.2f}%" if s["expected_move_pct"] is not None else "  n/a"
        L.append(f"  {s['ticker']:8}{s['breach']:9}{s['move_in_atr']:>6.1f}{em:>10}  {s['action']}")
    return "\n".join(L)


def format_summary(study: dict) -> str:
    """Human-readable event-study table with a plain-language decay read."""
    if not study or study.get("n_events", 0) == 0:
        return "No breach events to study."
    ov = study["overall"]
    L = [f"Event study: {study['n_events']:,} breaches, "
         f"return measured in breach direction:",
         "",
         f"  {'day':>4}{'avg cum %':>12}{'win-rate':>10}{'n':>8}"]
    for _, r in ov.iterrows():
        win = "" if pd.isna(r["win_rate_pct"]) else f"{r['win_rate_pct']:.1f}%"
        L.append(f"  {int(r['event_day']):>4}{r['avg_cum_return_pct']:>11.3f}%"
                 f"{win:>10}{int(r['n_events']):>8}")

    # Note where most of the move accrues.
    final = ov.iloc[-1]["avg_cum_return_pct"]
    if final and not pd.isna(final) and final != 0:
        half = ov[ov["avg_cum_return_pct"].abs() >= abs(final) * 0.5]
        if not half.empty:
            d_half = int(half.iloc[0]["event_day"])
            L += ["", f"Half of the {study['max_days']}-day move is captured by "
                  f"day {d_half}, consistent with short-lived post-breach drift "
                  f"and with shorter holds working better in the sweep."]
    if "regime_counts" in study:
        L.append(f"Regime split: {study['regime_counts']}")
    return "\n".join(L)
