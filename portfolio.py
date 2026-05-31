"""Event-driven portfolio simulation for the ATR breach signal.

The backtest module measures whether a breach predicts continuation. This module
measures what trading it would have produced, net of costs.

The simulation is kept conservative. There is no look-ahead: a breach is observed
at the close of day t (move_in_atr is a close-to-close statistic), the position is
entered at that close, and it earns returns from day t+1. Each name holds one unit
at most; overlapping breaches on the same ticker refresh the holding window rather
than stacking, so exposure per name stays within +/-1, and a later breach in the
opposite direction flips the position. Open names are equal-weighted each day,
with cash on days when nothing is open, so concurrent signals add diversification
rather than leverage. Every unit of turnover (open, close, or flip) is charged
`cost_bps`, and a flip costs twice because it is an exit and an entry.

Results are reported against an equal-weight buy-and-hold of the same universe, so
the signal is measured against owning the stocks rather than against zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _position_path(grp: pd.DataFrame, hold: int, long_only: bool) -> pd.DataFrame:
    """Per-ticker daily position direction (+1/0/-1) and turnover, from breaches.

    `grp` is one ticker, date-sorted, with `ret` (daily return) and `move_in_atr`.
    When `long_only`, bearish breaches are skipped entirely (no short leg).
    """
    grp = grp.sort_values("date").reset_index(drop=True)
    n = len(grp)
    pos = np.zeros(n)  # direction held *during* day i (earns ret[i])

    breach = grp["move_in_atr"].abs() >= config.ATR_BREACH_MULT
    sign = np.sign(grp["move_in_atr"].to_numpy())
    for i in np.flatnonzero(breach.to_numpy()):
        if long_only and sign[i] < 0:
            continue  # skip bearish breaches, no short leg
        # Breach at close of day i -> hold days i+1 .. i+hold (overwrite/refresh).
        lo, hi = i + 1, min(i + hold + 1, n)
        pos[lo:hi] = sign[i]

    grp["pos"] = pos
    # Turnover on day i = |pos_i - pos_{i-1}|: 1 on open/close, 2 on a flip.
    grp["turnover"] = np.abs(np.diff(pos, prepend=0.0))
    return grp


def simulate(prices: pd.DataFrame, *, hold: int | None = None,
             cost_bps: float | None = None, long_only: bool = False) -> dict:
    """Run the event-driven simulation. Returns metrics and the equity curve.

    `long_only=True` trades only bullish breaches (no shorts), which is closer to a
    retail account and avoids the short-leg drag in a bull market.
    """
    hold = hold or config.PORTFOLIO_HOLD_DAYS
    cost = (config.PORTFOLIO_COST_BPS if cost_bps is None else cost_bps) / 1e4
    if prices.empty:
        return {}

    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()

    df = pd.concat([_position_path(g, hold, long_only) for _, g in df.groupby("ticker")],
                   ignore_index=True)
    df["gross"] = df["pos"] * df["ret"]
    df["net"] = df["gross"] - df["turnover"] * cost
    # A name contributes to a given day if it is held OR transacting that day.
    df["active"] = (df["pos"] != 0) | (df["turnover"] != 0)

    # --- strategy: equal weight across active names each day ---
    daily = df[df["active"]].groupby("date")["net"].mean()
    all_days = pd.Index(sorted(df["date"].unique()), name="date")
    strat = daily.reindex(all_days).fillna(0.0)  # flat (cash) on idle days

    # --- benchmark: equal-weight buy & hold of the whole universe ---
    bench = df.dropna(subset=["ret"]).groupby("date")["ret"].mean()
    bench = bench.reindex(all_days).fillna(0.0)

    equity = (1.0 + strat).cumprod()
    bench_equity = (1.0 + bench).cumprod()

    n_trades = int((df["turnover"] > 0).sum())
    days_invested = float((df.groupby("date")["pos"].apply(lambda s: (s != 0).any())).mean())

    metrics = {
        "hold_days": hold,
        "long_only": long_only,
        "cost_bps": cost * 1e4,
        "trades": n_trades,
        "days": int(len(all_days)),
        "pct_days_invested": round(days_invested * 100, 1),
        "total_return_pct": round((float(equity.iloc[-1]) - 1) * 100, 2),
        "benchmark_return_pct": round((float(bench_equity.iloc[-1]) - 1) * 100, 2),
        "cagr_pct": round(_cagr(equity) * 100, 2),
        "benchmark_cagr_pct": round(_cagr(bench_equity) * 100, 2),
        "sharpe": round(_sharpe(strat), 2),
        "benchmark_sharpe": round(_sharpe(bench), 2),
        "max_drawdown_pct": round(_max_drawdown(equity) * 100, 2),
        "benchmark_max_drawdown_pct": round(_max_drawdown(bench_equity) * 100, 2),
        "equity": equity,
        "benchmark_equity": bench_equity,
        "strat_returns": strat,
        "bench_returns": bench,
        "frame": df,            # per-ticker/day pos, ret, turnover, net (for snapshots)
    }
    return metrics


def trade_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct discrete trades from the per-day position frame.

    A trade is a contiguous run of constant non-zero position for one ticker.
    Returns one row per trade with entry/exit, direction, days held, and the
    compounded gross return over the holding window.
    """
    cols = ["ticker", "entry_date", "exit_date", "direction", "days_held",
            "gross_return_pct"]
    rows = []
    for ticker, g in frame.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        pos = g["pos"].to_numpy()
        # Run boundaries: where the position value changes.
        change = np.flatnonzero(np.diff(pos, prepend=0.0) != 0)
        for start in change:
            d = pos[start]
            if d == 0:
                continue
            end = start
            while end + 1 < len(pos) and pos[end + 1] == d:
                end += 1
            window = g.iloc[start:end + 1]
            gross = float((1.0 + d * window["ret"].fillna(0.0)).prod() - 1.0)
            rows.append({
                "ticker": ticker,
                "entry_date": window["date"].iloc[0],
                "exit_date": window["date"].iloc[-1],
                "direction": "long" if d > 0 else "short",
                "days_held": len(window),
                "gross_return_pct": round(gross * 100, 4),
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("entry_date").reset_index(drop=True)


def positions_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily exposure: how many names long / short and net direction over time."""
    g = frame.groupby("date")["pos"]
    out = pd.DataFrame({
        "n_long": g.apply(lambda s: int((s > 0).sum())),
        "n_short": g.apply(lambda s: int((s < 0).sum())),
    })
    out["n_active"] = out["n_long"] + out["n_short"]
    out["net_exposure"] = out["n_long"] - out["n_short"]
    return out.reset_index()


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / config.TRADING_DAYS_PER_YEAR
    if years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float(equity.iloc[-1]) ** (1 / years) - 1


def _sharpe(daily_returns: pd.Series) -> float:
    sd = daily_returns.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(daily_returns.mean() / sd * np.sqrt(config.TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sweep(prices: pd.DataFrame, holds: list[int], costs: list[float], *,
          long_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Run the simulation across a grid of (hold, cost) and score robustness.

    Stability across parameters matters more than a single run. Returns a grid
    frame and a summary dict, where the summary reports how consistent performance
    is across the grid. A Sharpe that only holds at one hold-period is fragile; one
    that holds across the grid is more robust.
    """
    rows = []
    for h in holds:
        for c in costs:
            m = simulate(prices, hold=h, cost_bps=c, long_only=long_only)
            if not m:
                continue
            rows.append({"hold_days": h, "cost_bps": c,
                         "total_return_pct": m["total_return_pct"],
                         "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"],
                         "max_drawdown_pct": m["max_drawdown_pct"],
                         "trades": m["trades"]})
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid, {}

    sharpes = grid["sharpe"]
    mean_sh, std_sh = float(sharpes.mean()), float(sharpes.std() or 0.0)
    summary = {
        "cells": int(len(grid)),
        "long_only": long_only,
        "sharpe_mean": round(mean_sh, 3),
        "sharpe_std": round(std_sh, 3),
        # Stability: mean/std of Sharpe across the grid (higher = more robust).
        "stability_score": round(mean_sh / std_sh, 2) if std_sh > 0 else float("inf"),
        "sharpe_positive_pct": round(float((sharpes > 0).mean()) * 100, 1),
        "best": grid.loc[sharpes.idxmax()].to_dict(),
    }
    return grid, summary


def format_summary(m: dict) -> str:
    """Pretty multi-line report comparing the strategy to buy-and-hold."""
    if not m:
        return "No portfolio result (empty price history)."
    mode = "long-only" if m.get("long_only") else "long/short"
    L = [
        f"{mode} | hold {m['hold_days']}d | cost {m['cost_bps']:.0f} bps/turnover | "
        f"{m['trades']:,} trades over {m['days']:,} days | "
        f"invested {m['pct_days_invested']:.0f}% of days",
        "",
        f"{'':22}{'ATR strategy':>14}{'Buy & hold':>14}",
        f"{'Total return':22}{m['total_return_pct']:>13.1f}%{m['benchmark_return_pct']:>13.1f}%",
        f"{'CAGR':22}{m['cagr_pct']:>13.1f}%{m['benchmark_cagr_pct']:>13.1f}%",
        f"{'Sharpe':22}{m['sharpe']:>14.2f}{m['benchmark_sharpe']:>14.2f}",
        f"{'Max drawdown':22}{m['max_drawdown_pct']:>13.1f}%{m['benchmark_max_drawdown_pct']:>13.1f}%",
    ]
    return "\n".join(L)
