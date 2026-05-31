"""Historical news store for testing whether news adds value.

RSS feeds only return recent headlines, so there is no way out of the box to
backtest breach-only versus breach-plus-news. The fix is data rather than code:
each scan appends its headlines to an append-only archive, so a historical news
dataset builds up over time. Once enough days exist, breach events can be split by
whether news was present and compared.

Workflow:
    python main.py scan            # fetch and archive today's headlines (automatic)
    ...let it run daily for weeks or months...
    python main.py news-value      # once the archive spans history, run the test

Until the archive covers a date, `compare_news_vs_breach` reports that rather than
producing a result from data it does not have.
"""

from __future__ import annotations

import pandas as pd

from . import config, backtest as bt

# Columns persisted per archived headline (a stable subset of news.fetch_news).
ARCHIVE_COLS = ["id", "ticker", "title", "published", "sentiment",
                "signal", "sentiment_source", "archived_date"]


def append_news(news: pd.DataFrame, *, today: pd.Timestamp | None = None) -> int:
    """Append today's headlines to the archive, de-duped by id. Returns #new rows."""
    if news is None or news.empty:
        return 0
    today = pd.Timestamp(today or pd.Timestamp.now().normalize())

    add = news.copy()
    add["archived_date"] = today
    for col in ARCHIVE_COLS:
        if col not in add.columns:
            add[col] = ""
    add = add[ARCHIVE_COLS]

    existing = load_archive()
    before = len(existing)
    combined = add if existing.empty else pd.concat([existing, add], ignore_index=True)
    # An id can recur across days; keep the first time we saw each headline.
    combined = combined.drop_duplicates(subset=["id"], keep="first")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.NEWS_ARCHIVE_CSV, index=False)
    return len(combined) - before


def load_archive() -> pd.DataFrame:
    """Load the persisted archive (empty, correctly-typed frame if none yet)."""
    if not config.NEWS_ARCHIVE_CSV.exists():
        return pd.DataFrame(columns=ARCHIVE_COLS)
    df = pd.read_csv(config.NEWS_ARCHIVE_CSV, parse_dates=["published", "archived_date"])
    return df


def coverage(archive: pd.DataFrame | None = None) -> dict:
    """Summarize how much history the archive spans (so users know when it's ready)."""
    archive = load_archive() if archive is None else archive
    if archive.empty:
        return {"rows": 0, "tickers": 0, "days": 0, "start": None, "end": None}
    dates = pd.to_datetime(archive["archived_date"])
    return {
        "rows": int(len(archive)),
        "tickers": int(archive["ticker"].nunique()),
        "days": int(dates.dt.normalize().nunique()),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }


def tag_breaches_with_news(prices: pd.DataFrame, archive: pd.DataFrame | None = None,
                           *, lookback_hours: int | None = None) -> pd.DataFrame:
    """Label every breach event with whether news existed in the window before it.

    Returns the breach-events frame (see backtest.breach_events) plus a boolean
    `had_news` column. Only breaches whose date is covered by the archive are
    returned, since news presence cannot be known for days that were never recorded.
    """
    lookback_hours = lookback_hours or config.NEWS_LOOKBACK_HOURS
    archive = load_archive() if archive is None else archive
    events = bt.breach_events(prices)
    if events.empty or archive.empty:
        events = events.copy()
        events["had_news"] = pd.Series(dtype=bool)
        return events

    # Restrict to the window the archive actually covers (no look-ahead, no guessing).
    covered_start = pd.to_datetime(archive["archived_date"]).min().normalize()
    covered_end = pd.to_datetime(archive["archived_date"]).max().normalize()
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    events = events[(events["date"] >= covered_start) & (events["date"] <= covered_end)]
    if events.empty:
        events["had_news"] = pd.Series(dtype=bool)
        return events

    window = pd.Timedelta(hours=lookback_hours)
    pub = pd.to_datetime(archive["published"])
    by_ticker = {t: pub[archive["ticker"] == t].to_numpy() for t in archive["ticker"].unique()}

    def _has_news(row) -> bool:
        times = by_ticker.get(row["ticker"])
        if times is None or len(times) == 0:
            return False
        end = row["date"] + pd.Timedelta(days=1)  # news up to the breach day
        start = end - window
        return bool(((times >= start.to_datetime64()) & (times <= end.to_datetime64())).any())

    events["had_news"] = events.apply(_has_news, axis=1)
    return events


def compare_news_vs_breach(prices: pd.DataFrame, archive: pd.DataFrame | None = None,
                           *, horizon: int | None = None) -> dict:
    """The A/B/C test: breach-only vs breach+news, once the archive can support it.

    Returns a dict with per-bucket hit-rate / avg-return, plus a `ready` flag and
    a human-readable `note` explaining whether there is enough data yet.
    """
    horizon = horizon or config.FORWARD_HORIZONS[-1]
    tagged = tag_breaches_with_news(prices, archive)
    cov = coverage(archive)

    if tagged.empty or "had_news" not in tagged or tagged["had_news"].isna().all():
        return {"ready": False,
                "note": (f"News archive too thin to test yet "
                         f"({cov['rows']} headlines over {cov['days']} day(s)). "
                         f"Keep running `scan` daily; results appear once breaches "
                         f"fall inside the archived window."),
                "coverage": cov}

    col = f"signed_{horizon}d"
    out = {"ready": True, "coverage": cov, "horizon": horizon, "buckets": {}}
    for name, mask in (("breach+news", tagged["had_news"]),
                       ("breach-only", ~tagged["had_news"])):
        sub = tagged[mask]
        avg, hit = bt._signed_stats(sub[col]) if not sub.empty else (float("nan"), float("nan"))
        out["buckets"][name] = {"events": int(len(sub)),
                                "hit_rate_pct": hit, "avg_fwd_pct": avg}
    return out
