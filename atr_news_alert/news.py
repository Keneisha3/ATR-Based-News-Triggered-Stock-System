"""RSS news ingestion, de-duplication and lightweight headline sentiment.

Two free, no-key sources per ticker:
  * Yahoo Finance headline RSS
  * Google News RSS (query = ticker + company name)
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import pandas as pd

from . import config, llm


def _yahoo_url(ticker: str) -> str:
    return (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={quote_plus(ticker)}&region=US&lang=en-US"
    )


def _google_url(ticker: str, name: str) -> str:
    query = f'"{name}" OR {ticker} stock'
    return (
        f"https://news.google.com/rss/search?q={quote_plus(query)}"
        "&hl=en-US&gl=US&ceid=US:en"
    )


def _entry_time(entry) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _hash(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def score_headline(title: str) -> tuple[str, int]:
    """Return (sentiment_label, signed_word_score) from the finance lexicon."""
    words = re.findall(r"[a-z']+", title.lower())
    bull = sum(w in config.BULLISH_WORDS for w in words)
    bear = sum(w in config.BEARISH_WORDS for w in words)
    score = bull - bear
    if score > 0:
        return "bullish", score
    if score < 0:
        return "bearish", score
    return "neutral", 0


def fetch_news(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Ingest RSS for every ticker; de-duplicate; tag sentiment + recency."""
    import feedparser

    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.NEWS_LOOKBACK_HOURS)
    rows = []
    seen: set[str] = set()

    for _, r in watchlist.iterrows():
        ticker, name = r["ticker"], r["name"]
        for url in (_yahoo_url(ticker), _google_url(ticker, name)):
            try:
                feed = feedparser.parse(url)
            except Exception as exc:
                print(f"  ! news fetch failed for {ticker}: {exc}")
                continue

            for entry in feed.entries[: config.MAX_HEADLINES_PER_TICKER]:
                title = _clean(entry.get("title", ""))
                if not title:
                    continue
                key = _hash(ticker, title.lower())
                if key in seen:
                    continue
                seen.add(key)

                published = _entry_time(entry)
                sentiment, word_score = score_headline(title)
                rows.append({
                    "id": key,
                    "ticker": ticker,
                    "title": title,
                    "link": entry.get("link", ""),
                    "source": _clean(entry.get("source", {}).get("title", ""))
                    or ("Yahoo" if "yahoo" in url else "GoogleNews"),
                    "published": published.replace(tzinfo=None),
                    "is_recent": published >= cutoff,
                    "sentiment": sentiment,
                    "category": "",
                    "confidence": 0,
                    "word_score": word_score,
                    # signed strength used by the alert engine; lexicon default
                    "signal": float(word_score),
                    "sentiment_source": "lexicon",
                })

    cols = ["id", "ticker", "title", "link", "source", "published",
            "is_recent", "sentiment", "category", "confidence",
            "word_score", "signal", "sentiment_source"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols].sort_values("published", ascending=False)
    df = _apply_llm(df)
    return df


def _apply_llm(df: pd.DataFrame) -> pd.DataFrame:
    """Upgrade recent-headline sentiment with the LLM when one is configured."""
    if df.empty or not llm.available():
        return df

    recent_idx = df.index[df["is_recent"]][: config.LLM_MAX_HEADLINES]
    if len(recent_idx) == 0:
        return df

    titles = df.loc[recent_idx, "title"].tolist()
    print(f"      classifying {len(titles)} recent headlines with "
          f"{llm.provider()} LLM ...")
    results = llm.classify(titles)
    if not results:
        return df

    for idx, res in zip(recent_idx, results):
        if res["confidence"] == 0:  # batch failed -> keep lexicon
            continue
        senti = res["sentiment"]
        conf = res["confidence"]
        df.at[idx, "sentiment"] = senti
        df.at[idx, "category"] = res["category"]
        df.at[idx, "confidence"] = conf
        sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}[senti]
        df.at[idx, "signal"] = sign * conf / 100.0
        df.at[idx, "sentiment_source"] = "llm"
    return df


def summarize_by_ticker(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recent-news signal per ticker for the alert engine."""
    cols = ["ticker", "recent_count", "net_sentiment", "headline_sentiment",
            "top_category", "top_headline", "top_link"]
    if news.empty:
        return pd.DataFrame(columns=cols)

    recent = news[news["is_recent"]]
    if recent.empty:
        return pd.DataFrame(columns=cols)

    out = []
    for ticker, grp in recent.groupby("ticker"):
        net = float(grp["signal"].sum())
        label = "bullish" if net > 0 else "bearish" if net < 0 else "neutral"
        # Prefer the highest-confidence/strongest headline as the representative.
        ranked = grp.sort_values(
            ["confidence", "published"], ascending=[False, False])
        top = ranked.iloc[0]
        out.append({
            "ticker": ticker,
            "recent_count": len(grp),
            "net_sentiment": round(net, 2),
            "headline_sentiment": label,
            "top_category": top["category"] or "-",
            "top_headline": top["title"],
            "top_link": top["link"],
        })
    return pd.DataFrame(out)[cols]


def save_news(news: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    news.to_csv(config.NEWS_CSV, index=False)
