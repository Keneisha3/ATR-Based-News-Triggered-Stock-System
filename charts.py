"""Render the results CSVs into PNG charts.

Reads the same in-memory objects the results layer snapshots (equity series, trade
log, position frame, sweep grid) and writes PNGs to `results/`. Uses the Agg
backend so it runs headless, without a display.

Charts:
  * equity_and_drawdown  strategy vs buy and hold, with a drawdown panel below
  * trade_distribution   holding-period histogram and win/loss return distribution
  * exposure_timeline    how many names are long vs short over time
  * sweep_heatmap        a chosen metric across the hold x cost parameter grid
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: render to file, never to a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, portfolio as pf  # noqa: E402

_STRAT = "#1f77b4"
_BENCH = "#888888"
_POS = "#2ca02c"
_NEG = "#d62728"


def _style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def equity_and_drawdown(metrics: dict, path) -> str:
    """Two-panel: equity curves (strategy vs buy & hold) over a drawdown panel."""
    eq = metrics["equity"]
    bench = metrics["benchmark_equity"]
    dd = eq / eq.cummax() - 1.0
    mode = "long-only" if metrics.get("long_only") else "long/short"

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    a1.plot(eq.index, eq.to_numpy(), color=_STRAT, lw=1.6,
            label=f"ATR strategy ({mode})  +{metrics['total_return_pct']:.0f}%")
    a1.plot(bench.index, bench.to_numpy(), color=_BENCH, lw=1.4, ls="--",
            label=f"Buy & hold  +{metrics['benchmark_return_pct']:.0f}%")
    a1.set_ylabel("Growth of $1")
    a1.set_title(f"Equity curve. Sharpe {metrics['sharpe']:.2f} vs "
                 f"{metrics['benchmark_sharpe']:.2f}, "
                 f"max DD {metrics['max_drawdown_pct']:.0f}% vs "
                 f"{metrics['benchmark_max_drawdown_pct']:.0f}%")
    a1.legend(frameon=False, loc="upper left")
    _style(a1)

    a2.fill_between(dd.index, dd.to_numpy() * 100, 0, color=_NEG, alpha=0.4)
    a2.set_ylabel("Drawdown %")
    a2.set_xlabel("Date")
    _style(a2)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def trade_distribution(trades: pd.DataFrame, path) -> str:
    """Holding-period histogram + per-trade return distribution (wins vs losses)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    if not trades.empty:
        a1.hist(trades["days_held"], bins=range(1, int(trades["days_held"].max()) + 2),
                color=_STRAT, alpha=0.8, align="left")
    a1.set_title("Holding period")
    a1.set_xlabel("Days held")
    a1.set_ylabel("Trades")
    _style(a1)

    if not trades.empty:
        r = trades["gross_return_pct"]
        bins = np.linspace(r.min(), r.max(), 40) if len(r) > 1 else 10
        a2.hist(r[r > 0], bins=bins, color=_POS, alpha=0.7, label="winners")
        a2.hist(r[r <= 0], bins=bins, color=_NEG, alpha=0.7, label="losers")
        a2.axvline(float(r.mean()), color="black", lw=1.2, ls="--",
                   label=f"mean {r.mean():+.2f}%")
        win = (r > 0).mean() * 100
        a2.set_title(f"Per-trade return  (win-rate {win:.0f}%, n={len(r):,})")
        a2.legend(frameon=False)
    a2.set_xlabel("Trade return %")
    a2.set_ylabel("Trades")
    _style(a2)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def exposure_timeline(timeline: pd.DataFrame, path) -> str:
    """Daily count of names held long vs short over time."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    d = pd.to_datetime(timeline["date"])
    ax.fill_between(d, timeline["n_long"], 0, color=_POS, alpha=0.6, label="long names")
    ax.fill_between(d, -timeline["n_short"], 0, color=_NEG, alpha=0.6, label="short names")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Exposure over time (number of open positions)")
    ax.set_ylabel("Short  ◄  names  ►  Long")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def sweep_heatmap(grid: pd.DataFrame, path, *, metric: str = "sharpe") -> str:
    """Heatmap of a metric across the hold-period x cost parameter grid."""
    pivot = grid.pivot(index="hold_days", columns="cost_bps", values=metric)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    ax.set_xlabel("Cost (bps / turnover)")
    ax.set_ylabel("Hold period (days)")
    ax.set_title(f"Robustness surface: {metric}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def event_study_curve(study: dict, path) -> str:
    """The core signal-signature plot: avg cumulative return vs days after breach.

    Overall plus bull/bear regime lines. A flattening curve = decaying signal; a
    bull/bear gap = regime dependence. This is the chart that *explains* the rest.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    def _plot(df, color, label, lw=1.8, ls="-"):
        if df is None or df.empty:
            return
        ax.plot(df["event_day"], df["avg_cum_return_pct"], color=color,
                lw=lw, ls=ls, marker="o", ms=3, label=label)

    _plot(study.get("overall"), _STRAT, f"all events (n={study.get('n_events', 0):,})", lw=2.2)
    rc = study.get("regime_counts", {})
    _plot(study.get("bull"), _POS, f"bull regime (n={rc.get('bull', 0):,})", ls="--")
    _plot(study.get("bear"), _NEG, f"bear regime (n={rc.get('bear', 0):,})", ls="--")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trading days after breach")
    ax.set_ylabel("Avg cumulative return % (breach direction)")
    ax.set_title("Event study: cumulative drift after a breach")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def ml_diagnostics(res: dict, path) -> str:
    """Two-panel ML diagnostic: feature importance + reliability (calibration) curve."""
    from sklearn.calibration import calibration_curve

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    imp = res.get("importance")
    if imp is not None and not imp.empty:
        d = imp.iloc[::-1]   # largest at top
        a1.barh(d["feature"], d["importance"], color=_STRAT, alpha=0.85)
    a1.set_title("Permutation feature importance (OOS AUC)")
    a1.set_xlabel("Δ AUC when shuffled")
    _style(a1)

    # Reliability curve for the gradient-boosting model.
    gb = res.get("models", {}).get("gradient_boosting")
    if gb is not None and "_proba" in gb:
        frac, mean_pred = calibration_curve(gb["_y"], gb["_proba"], n_bins=10,
                                            strategy="quantile")
        a2.plot([0, 1], [0, 1], ls="--", color=_BENCH, label="perfectly calibrated")
        a2.plot(mean_pred, frac, marker="o", color=_STRAT, label="gradient boosting")
        a2.set_xlabel("Predicted P(continue)")
        a2.set_ylabel("Observed frequency")
        a2.set_title(f"Calibration (OOS AUC {gb['auc']:.3f})")
        a2.legend(frameon=False, loc="upper left")
    _style(a2)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def render_ml(res: dict, outdir=None) -> dict:
    outdir = outdir or config.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return {"ml": ml_diagnostics(res, outdir / "ml_diagnostics.png")}


def render_event_study(study: dict, outdir=None) -> dict:
    outdir = outdir or config.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return {"event_study": event_study_curve(study, outdir / "event_study_curve.png")}


def render_portfolio(metrics: dict, outdir=None) -> dict:
    """Render the full portfolio chart set. Returns {name: path}."""
    outdir = outdir or config.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    trades = pf.trade_log(metrics["frame"])
    timeline = pf.positions_timeline(metrics["frame"])
    return {
        "equity": equity_and_drawdown(metrics, outdir / "equity_curve.png"),
        "trades": trade_distribution(trades, outdir / "trade_distribution.png"),
        "exposure": exposure_timeline(timeline, outdir / "exposure_timeline.png"),
    }


def render_sweep(grid: pd.DataFrame, outdir=None, *, metric: str = "sharpe") -> dict:
    outdir = outdir or config.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return {"heatmap": sweep_heatmap(grid, outdir / "sweep_heatmap.png", metric=metric)}
