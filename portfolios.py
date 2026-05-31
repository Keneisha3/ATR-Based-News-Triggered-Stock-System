"""Compare the signal across different baskets of tickers.

The edge is not uniform across sectors, so this evaluates each named basket on the
same measures (hit-rate vs random baseline, long-only equity curve vs buy and
hold). The baskets are sliced from the cached prices, so it runs offline and fast.
"""

from __future__ import annotations

import pandas as pd

from . import backtest as bt, portfolio as pf

# Named baskets (subsets of the default watchlist universe).
PRESETS: dict[str, list[str]] = {
    "Mega-Cap Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO",
                      "ORCL", "ADBE", "CRM"],
    "High-Beta Momentum": ["TSLA", "NVDA", "AMD", "META", "NFLX", "UBER", "MU",
                           "AVGO"],
    "Defensive / Dividend": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "COST",
                             "MRK", "ABBV"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK"],
    "Energy + Industrials": ["XOM", "CVX", "COP", "BA", "CAT", "GE"],
    "Diversified (all)": [],   # empty => use every ticker present
}


def _evaluate(prices: pd.DataFrame) -> dict:
    """Headline numbers for one basket."""
    _, agg = bt.backtest(prices)
    events = agg.get("events", 0)
    base = bt.random_baseline(prices, events) if events else {}
    lo = pf.simulate(prices, long_only=True)
    hit = agg.get("hit_rate_5d_pct")
    rand = base.get("hit_rate_5d_pct")
    return {
        "tickers": int(prices["ticker"].nunique()),
        "events": events,
        "hit_5d_pct": hit,
        "edge_pp": (round(hit - rand, 1) if hit is not None and rand is not None
                    else None),
        "longonly_return_pct": lo.get("total_return_pct"),
        "buyhold_return_pct": lo.get("benchmark_return_pct"),
        "longonly_sharpe": lo.get("sharpe"),
        "buyhold_sharpe": lo.get("benchmark_sharpe"),
    }


def compare(prices: pd.DataFrame, presets: dict | None = None) -> pd.DataFrame:
    """Evaluate every preset basket; return a tidy comparison frame."""
    presets = presets or PRESETS
    rows = []
    for name, tickers in presets.items():
        sub = prices if not tickers else prices[prices["ticker"].isin(tickers)]
        if sub.empty:
            continue
        rec = {"portfolio": name}
        rec.update(_evaluate(sub))
        rows.append(rec)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data to compare (run a scan/backtest first to cache prices)."
    L = [f"  {'portfolio':22}{'tkrs':>5}{'events':>7}{'hit5d':>7}{'edge':>7}"
         f"{'LO ret':>9}{'B&H ret':>9}{'LO Shrp':>8}",
         "  " + "-" * 74]
    for _, r in df.iterrows():
        def g(k, suf="", w=0, dec=1):
            v = r[k]
            return (f"{v:.{dec}f}{suf}" if isinstance(v, (int, float)) and pd.notna(v)
                    else "n/a")
        L.append(f"  {r['portfolio']:22}{r['tickers']:>5}{r['events']:>7}"
                 f"{g('hit_5d_pct','%'):>7}{g('edge_pp','pp'):>7}"
                 f"{g('longonly_return_pct','%'):>9}{g('buyhold_return_pct','%'):>9}"
                 f"{g('longonly_sharpe','',dec=2):>8}")
    return "\n".join(L)
