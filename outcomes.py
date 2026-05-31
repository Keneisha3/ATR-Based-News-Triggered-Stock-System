"""Outcome tracking: the feedback loop from prediction to calibration.

Every decision is logged here. Once enough days have passed, its realized forward
return is filled in and scored correct or incorrect, which turns the backtests
into a running record of how often the system is right over time.

  * `record`      append today's decisions (predicted side only) to outcomes.csv
  * `update`      fill realized 1/3/5-day returns for decisions old enough to resolve
  * `backfill`    apply the decision rule across all history for an immediate
                  in-sample calibration read instead of waiting weeks
  * `calibration` summarize hit-rate and realized return by action and regime

The `date` and `ticker` fields join predictions back to prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, signal_analyzer as sa, decision as decision_mod

HORIZONS = (1, 3, 5)
COLS = ["date", "ticker", "signal", "regime", "action", "confidence", "has_news",
        "realized_1d", "realized_3d", "realized_5d", "resolved", "correct"]


def _load() -> pd.DataFrame:
    if config.OUTCOMES_CSV.exists():
        return pd.read_csv(config.OUTCOMES_CSV, parse_dates=["date"])
    return pd.DataFrame(columns=COLS)


def _save(df: pd.DataFrame) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OUTCOMES_CSV, index=False)


def record(decisions: list[dict]) -> int:
    """Append today's decisions (predicted side only), de-duped by date+ticker."""
    if not decisions:
        return 0
    rows = [{"date": d["date"], "ticker": d["ticker"], "signal": d["signal"],
             "regime": d["regime"], "action": d["action"],
             "confidence": d["confidence"], "has_news": d["has_news"],
             "realized_1d": np.nan, "realized_3d": np.nan, "realized_5d": np.nan,
             "resolved": False, "correct": np.nan} for d in decisions]
    add = pd.DataFrame(rows)
    add["date"] = pd.to_datetime(add["date"])
    existing = _load()
    before = len(existing)
    combined = add if existing.empty else pd.concat([existing, add], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="first")
    _save(combined)
    return len(combined) - before


def _breach_dir_returns(prices: pd.DataFrame) -> dict:
    """Per ticker: date-indexed close + sign, for realized-return lookup."""
    out = {}
    for t, g in prices.sort_values("date").groupby("ticker"):
        g = g.reset_index(drop=True)
        out[t] = (g["date"].to_numpy(), g["close"].to_numpy(),
                  np.sign(g["move_in_atr"].to_numpy()))
    return out


def update(prices: pd.DataFrame, log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill realized breach-direction returns for decisions old enough to resolve."""
    log = _load() if log is None else log
    if log.empty or prices.empty:
        return log
    book = _breach_dir_returns(prices)
    log = log.copy()
    log["correct"] = log["correct"].astype("object")   # holds bool/NaN without warning
    for i, row in log.iterrows():
        if row["resolved"]:
            continue
        rec = book.get(row["ticker"])
        if rec is None:
            continue
        dates, closes, signs = rec
        idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(row["date"])))
        if idx >= len(dates) or dates[idx] != np.datetime64(pd.Timestamp(row["date"])):
            continue
        sgn = 1.0 if row["signal"] == "BREACH_UP" else -1.0
        for h in HORIZONS:
            if idx + h < len(closes):
                log.at[i, f"realized_{h}d"] = round(
                    float(sgn * (closes[idx + h] / closes[idx] - 1.0)) * 100, 4)
        # Resolve once the longest horizon is available; correct = continuation held.
        if idx + max(HORIZONS) < len(closes):
            log.at[i, "resolved"] = True
            log.at[i, "correct"] = bool(log.at[i, "realized_3d"] > 0)
    _save(log)
    return log


def backfill(prices: pd.DataFrame) -> int:
    """Apply the decision rule across all historical breaches for an immediate read.

    This is in-sample because it uses the fixed bull=momentum/bear=reversion rule,
    so treat it as a check that the decision logic is internally sound rather than
    as out-of-sample proof.
    """
    if prices.empty:
        return 0
    regime = sa.market_regime(prices)
    events = sa._event_offsets(prices, max(HORIZONS))
    events = events.copy()
    events["regime"] = events["date"].map(regime)

    rows = []
    for _, e in events.iterrows():
        sig = "BREACH_UP" if e["move_in_atr"] > 0 else "BREACH_DOWN"
        reg = e.get("regime", "unknown")
        continuation = (reg == "bull")
        action, _ = decision_mod._decide_action(sig, continuation)
        row = {"date": e["date"], "ticker": e["ticker"], "signal": sig,
               "regime": str(reg).upper(), "action": action,
               "confidence": np.nan, "has_news": False, "resolved": True}
        for h in HORIZONS:
            v = e.get(f"sret_{h}")   # event-study column is sret_<h> (no 'd')
            row[f"realized_{h}d"] = round(float(v) * 100, 4) if pd.notna(v) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["correct"] = df["realized_3d"] > 0
    df = df.dropna(subset=["realized_3d"])
    _save(df[COLS])
    return len(df)


def calibration(log: pd.DataFrame | None = None) -> dict:
    """Summarize resolved decisions: hit-rate + realized return, by action/regime."""
    log = _load() if log is None else log
    resolved = log[log["resolved"] == True] if not log.empty else log  # noqa: E712
    if resolved.empty:
        return {"resolved": 0}

    def _hit(df):
        return round(float((df["realized_3d"] > 0).mean()) * 100, 1) if len(df) else None

    out = {"resolved": int(len(resolved)),
           "overall_hit_3d_pct": _hit(resolved),
           "by_action": {}, "by_regime": {}}
    longs = resolved[resolved["action"] == "LONG"]
    if len(longs):
        out["long_n"] = int(len(longs))
        out["long_hit_3d_pct"] = _hit(longs)
        out["long_avg_3d_pct"] = round(float(longs["realized_3d"].mean()), 3)
    for a, g in resolved.groupby("action"):
        out["by_action"][a] = {"n": int(len(g)), "hit_3d_pct": _hit(g),
                               "avg_3d_pct": round(float(g["realized_3d"].mean()), 3)}
    for r, g in resolved.groupby("regime"):
        out["by_regime"][r] = {"n": int(len(g)), "hit_3d_pct": _hit(g)}
    return out


def format_calibration(cal: dict) -> str:
    if not cal or cal.get("resolved", 0) == 0:
        return ("No resolved decisions yet. Run the system daily (or "
                "`outcomes --backfill` for an in-sample read) to build calibration.")
    L = [f"Calibration: {cal['resolved']:,} resolved decisions "
         f"(realized 3-day, breach direction):",
         f"  overall continuation hit-rate: {cal['overall_hit_3d_pct']}%"]
    if "long_n" in cal:
        L.append(f"  LONG calls: {cal['long_n']:,} | hit-rate {cal['long_hit_3d_pct']}% "
                 f"| avg realized {cal['long_avg_3d_pct']:+.3f}%")
    if cal["by_regime"]:
        L.append("  by regime: " + ", ".join(
            f"{r} {v['hit_3d_pct']}% (n={v['n']:,})" for r, v in cal["by_regime"].items()))
    return "\n".join(L)
