"""Alert trigger rule + severity scoring (the heart of the system)."""

import pandas as pd

from atr_news_alert import engine


def _latest(breach_vals, moves):
    n = len(breach_vals)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "date": pd.to_datetime(["2026-01-02"] * n),
        "close": [100.0] * n,
        "atr": [2.0] * n,
        "atr_pct": [0.02] * n,
        "move_pct": moves,
        "move_in_atr": [m / 0.02 / 100 for m in moves],  # not used by engine directly
        "abs_move_in_atr": [abs(b) for b in breach_vals],
        "breach": [abs(b) >= 1.5 for b in breach_vals],
        "direction": ["bullish" if m >= 0 else "bearish" for m in moves],
    }).assign(move_in_atr=breach_vals)


def _summary(tickers_counts):
    return pd.DataFrame([
        {"ticker": t, "recent_count": c, "net_sentiment": 1.0 * c,
         "headline_sentiment": "bullish", "top_category": "Earnings",
         "top_headline": f"{t} good news", "top_link": "L"}
        for t, c in tickers_counts.items()
    ])


def test_requires_both_news_and_breach():
    # T0: breach + news -> alert.  T1: breach + NO news -> no alert.
    # T2: news + NO breach -> no alert.
    latest = _latest(breach_vals=[3.0, 3.0, 0.5], moves=[0.06, 0.06, 0.01])
    summary = _summary({"T0": 5, "T2": 5})  # T1 has no news
    alerts = engine.build_alerts(latest, summary)
    assert set(alerts["ticker"]) == {"T0"}
    assert alerts.iloc[0]["alert_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_severity_increases_with_move_size():
    small = engine.build_alerts(_latest([1.6], [0.03]), _summary({"T0": 3}))
    large = engine.build_alerts(_latest([3.9], [0.08]), _summary({"T0": 3}))
    assert large.iloc[0]["severity"] > small.iloc[0]["severity"]


def test_direction_agreement_bonus():
    # bullish move + bullish headlines should out-score the same move with
    # bearish headlines (agreement bonus).
    latest = _latest([2.0], [0.04])
    agree = engine.build_alerts(latest, _summary({"T0": 3}))
    disagree_summary = _summary({"T0": 3})
    disagree_summary.loc[0, "headline_sentiment"] = "bearish"
    disagree = engine.build_alerts(latest, disagree_summary)
    assert agree.iloc[0]["severity"] > disagree.iloc[0]["severity"]


def test_no_alerts_when_empty():
    out = engine.build_alerts(pd.DataFrame(), pd.DataFrame())
    assert out.empty
