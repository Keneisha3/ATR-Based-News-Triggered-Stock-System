"""Alert engine: fuse ATR volatility breaches with recent RSS news.

An alert is "news-triggered" when a ticker has recent headlines, and
"ATR-based" because the price move is measured in ATR units so the bar is
held to a volatility-normalized threshold that is comparable across all
50+ equities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _severity(row) -> float:
    """0-10 score blending volatility, news volume and sentiment agreement."""
    # Volatility component: 0 at the breach threshold, saturates ~4 ATR.
    vol = np.clip(row["abs_move_in_atr"] / 4.0, 0, 1) * 6.0

    # News component: more recent headlines -> stronger, saturates at 6.
    news = np.clip(row["recent_count"] / 6.0, 0, 1) * 3.0

    # Agreement bonus: price move direction matches headline sentiment.
    agree = 0.0
    if row["recent_count"] > 0 and row["headline_sentiment"] != "neutral":
        if row["direction"] == row["headline_sentiment"]:
            agree = 1.0
    return float(round(min(vol + news + agree, 10.0), 2))


def build_alerts(latest: pd.DataFrame, news_summary: pd.DataFrame) -> pd.DataFrame:
    """Join latest bars with recent-news summary and score alerts.

    Trigger rule: recent news present AND ATR breach on the latest bar.
    """
    cols = ["ticker", "date", "close", "move_pct", "move_in_atr", "atr",
            "atr_pct", "direction", "recent_count", "headline_sentiment",
            "net_sentiment", "top_category", "severity", "alert_level",
            "top_headline", "top_link"]
    if latest.empty:
        return pd.DataFrame(columns=cols)

    df = latest.merge(news_summary, on="ticker", how="left")
    df["recent_count"] = df["recent_count"].fillna(0).astype(int)
    df["net_sentiment"] = df["net_sentiment"].fillna(0.0).astype(float)
    df["headline_sentiment"] = df["headline_sentiment"].fillna("none")
    df["top_category"] = df["top_category"].fillna("-")
    df["top_headline"] = df["top_headline"].fillna("")
    df["top_link"] = df["top_link"].fillna("")

    # News-triggered + ATR-based: both conditions must hold.
    triggered = df[(df["recent_count"] > 0) & (df["breach"])].copy()
    if triggered.empty:
        return pd.DataFrame(columns=cols)

    triggered["severity"] = triggered.apply(_severity, axis=1)
    triggered = triggered[triggered["severity"] >= config.MIN_ALERT_SEVERITY]
    if triggered.empty:
        return pd.DataFrame(columns=cols)

    triggered["alert_level"] = np.where(
        triggered["severity"] >= 7.5, "HIGH",
        np.where(triggered["severity"] >= 5.5, "MEDIUM", "LOW"),
    )
    triggered = triggered.sort_values("severity", ascending=False)
    return triggered[cols].reset_index(drop=True)


def save_alerts(alerts: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(config.ALERTS_CSV, index=False)


def format_alert(row) -> str:
    arrow = "▲" if row["direction"] == "bullish" else "▼"
    cat = row.get("top_category", "-") or "-"
    return (
        f"[{row['alert_level']:^6}] {row['ticker']:<6} {arrow} "
        f"{row['move_pct']*100:+5.1f}%  "
        f"({row['move_in_atr']:+.2f} ATR)  "
        f"sev {row['severity']:.1f}/10  "
        f"news x{int(row['recent_count'])} [{row['headline_sentiment']}/{cat}]\n"
        f"         \"{row['top_headline'][:90]}\""
    )
