"""Walk-forward evaluation: out-of-sample testing on an expanding window.

The other analytics measure the signal on the full history. This module instead
learns the rule as it goes and tests it on data it has not seen:

    train 2021            -> test 2022
    train 2021-2022       -> test 2023
    train 2021-2023       -> test 2024
    ...

On each fold the decision rule is learned from the training years only (does each
market regime's breach continue or revert?), frozen, and then applied to the next
year. Pooling the test folds gives an out-of-sample track record.

The regime label is a trailing moving-average classification, so it is causal,
and the continuation/reversion mapping per regime is fixed on training data
before the test fold is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, signal_analyzer as sa


def _learn_rule(train: pd.DataFrame, ret_col: str) -> dict:
    """Per regime: +1 if breaches continued in training, -1 if they reverted."""
    rule = {}
    for regime, g in train.groupby("regime"):
        rule[regime] = 1.0 if g[ret_col].mean() > 0 else -1.0
    return rule


def walk_forward(prices: pd.DataFrame, *, horizon: int | None = None,
                 min_train_years: int = 1) -> dict:
    """Expanding-window out-of-sample evaluation. Returns folds + pooled metrics."""
    horizon = horizon or config.PORTFOLIO_HOLD_DAYS
    if prices.empty:
        return {}

    regime = sa.market_regime(prices)                 # causal (trailing MA)
    events = sa._event_offsets(prices, horizon)
    ret_col = f"sret_{horizon}"
    events = events.copy()
    events["regime"] = events["date"].map(regime)
    events["year"] = pd.to_datetime(events["date"]).dt.year
    events = events.dropna(subset=[ret_col])
    if events.empty:
        return {}

    years = sorted(events["year"].unique())
    if len(years) <= min_train_years:
        return {"folds": [], "pooled": {}, "note": "not enough years to walk forward"}

    fold_rows, pooled_ret = [], []
    for i in range(min_train_years, len(years)):
        train = events[events["year"].isin(years[:i])]
        test = events[events["year"] == years[i]]
        if train.empty or test.empty:
            continue

        rule = _learn_rule(train, ret_col)            # frozen from training only
        # Apply to unseen test fold. Position sign in breach-return terms is the
        # learned continuation factor; unseen regime defaults to continuation.
        c = test["regime"].map(rule).fillna(1.0).to_numpy()
        strat = c * test[ret_col].to_numpy()          # realized OOS per-trade return
        pooled_ret.extend(strat.tolist())

        fold_rows.append({
            "train": f"{years[0]}-{years[i-1]}",
            "test": str(years[i]),
            "n_events": int(len(test)),
            "rule": {k: ("cont" if v > 0 else "rev") for k, v in rule.items()},
            "oos_hit_pct": round(float((strat > 0).mean()) * 100, 1),
            "oos_avg_ret_pct": round(float(strat.mean()) * 100, 3),
        })

    pooled = np.array(pooled_ret)
    pooled_stats = {}
    if len(pooled):
        sd = pooled.std()
        pooled_stats = {
            "n_events": int(len(pooled)),
            "oos_hit_pct": round(float((pooled > 0).mean()) * 100, 1),
            "oos_avg_ret_pct": round(float(pooled.mean()) * 100, 3),
            # Per-trade information ratio (mean/std), a unitless OOS quality measure.
            "info_ratio": round(float(pooled.mean() / sd), 3) if sd > 0 else None,
        }
    return {"horizon": horizon, "folds": fold_rows, "pooled": pooled_stats}


def format_summary(res: dict) -> str:
    if not res or not res.get("folds"):
        return "Not enough history for walk-forward evaluation."
    L = [f"Walk-forward (expanding window, {res['horizon']}-day horizon, "
         f"rule learned on train years only):", "",
         f"  {'train':>12}{'test':>6}{'events':>8}{'OOS hit':>9}{'OOS avg':>9}  rule"]
    for f in res["folds"]:
        rule = ", ".join(f"{k.lower()}:{v}" for k, v in f["rule"].items())
        L.append(f"  {f['train']:>12}{f['test']:>6}{f['n_events']:>8}"
                 f"{f['oos_hit_pct']:>8.1f}%{f['oos_avg_ret_pct']:>8.3f}%  {rule}")
    p = res.get("pooled", {})
    if p:
        L += ["", f"  Pooled OOS: {p['n_events']:,} trades | hit-rate "
              f"{p['oos_hit_pct']}% | avg {p['oos_avg_ret_pct']:+.3f}% | "
              f"info-ratio {p['info_ratio']}"]
    return "\n".join(L)
