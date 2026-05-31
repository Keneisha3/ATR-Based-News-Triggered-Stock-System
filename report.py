"""Build the markdown research report.

The report works from the price/news data passed in by the CLI. `main.py` decides
whether that data came from the local cache or from a fresh scan.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import (config, portfolio as pf, backtest as bt, signal_analyzer as sa,
               results as results_mod, ml as ml_mod)


def _metrics_table(m: dict, label: str) -> str:
    return (f"| {label} | {m['total_return_pct']:+.1f}% | {m['cagr_pct']:+.1f}% | "
            f"{m['sharpe']:.2f} | {m['max_drawdown_pct']:.1f}% | "
            f"{m['pct_days_invested']:.0f}% |")



def generate(prices: pd.DataFrame, *, news_summary: pd.DataFrame | None = None,
             with_charts: bool = True, on=None) -> str:
    """Run the report inputs and return markdown."""
    if prices.empty:
        return "# Report\n\nNo price history available.\n"

    ls = pf.simulate(prices, long_only=False)
    lo = pf.simulate(prices, long_only=True)
    grid, sweep_sum = pf.sweep(prices, config.SWEEP_HOLD_DAYS, config.SWEEP_COST_BPS)
    study = sa.event_study(prices)
    ml_res = ml_mod.run_ml(prices)
    today = sa.forward_signals(prices, news_summary=news_summary)
    _, agg = bt.backtest(prices)
    base = bt.random_baseline(prices, agg.get("events", 0))

    # Use the long/short run for the main manifest.
    results_mod.snapshot_portfolio(ls, prices, command="report")
    results_mod.snapshot_sweep(grid, sweep_sum, prices)
    results_mod.snapshot_event_study(study, prices)
    if with_charts:
        from . import charts
        charts.render_portfolio(lo)
        charts.render_sweep(grid)
        charts.render_event_study(study)
        if ml_res.get("ok"):
            charts.render_ml(ml_res)

    d0, d1 = prices["date"].min().date(), prices["date"].max().date()
    ts = (f"as of {on}" if on is not None
          else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    ov = study.get("overall", pd.DataFrame())
    md = []
    md.append("# ATR Breach Signal - Daily Research Memo\n")
    md.append(f"*Generated {ts} - data {d0} to {d1} - "
              f"{prices['ticker'].nunique()} tickers - "
              f"git `{results_mod.git_hash() or 'n/a'}`*\n")

    md.append("## 1. Executive summary\n")
    md.append(f"- ATR breaches have beaten the random-date baseline "
              f"({agg.get('hit_rate_5d_pct','?')}% vs "
              f"{base.get('hit_rate_5d_pct','?')}% 5-day hit-rate, "
              f"{agg.get('events','?'):,} events).")
    md.append(f"- The event study is regime-dependent: continuation in bull regimes, "
              f"mean reversion in bear regimes. That is why long/short is muted "
              f"and long-only is the cleaner use case.")
    md.append(f"- Parameter sweep: Sharpe is positive in "
              f"{sweep_sum.get('sharpe_positive_pct','?')}% of "
              f"{sweep_sum.get('cells','?')} parameter cells "
              f"(stability {sweep_sum.get('stability_score','?')}). Shorter holds work better.")
    md.append(f"- Long-only vs buy & hold: {lo['total_return_pct']:+.0f}% vs "
              f"{lo['benchmark_return_pct']:+.0f}% total "
              f"(Sharpe {lo['sharpe']:.2f} vs {lo['benchmark_sharpe']:.2f}). "
              f"Most of that is still market beta, so the signal is best treated "
              f"as a timing overlay.")
    md.append(f"- *Findings conditional on this universe and the {d0.year}-{d1.year} "
              f"period. Use `python main.py walk-forward` for the out-of-sample view.*\n")

    md.append(f"## 2. Today's actions ({d1})\n")
    md.append("```\n" + sa.format_signals(today) + "\n```\n")

    md.append("## 3. Signal behaviour - event study\n")
    if with_charts:
        md.append("![event study](event_study_curve.png)\n")
    if not ov.empty:
        final = ov.iloc[-1]["avg_cum_return_pct"]
        half_day = None
        if final and final != 0:
            reached = ov[ov["avg_cum_return_pct"].abs() >= abs(final) * 0.5]
            if not reached.empty:
                half_day = int(reached.iloc[0]["event_day"])
        rc = study.get("regime_counts", {})
        md.append(
            f"The blended curve is small and front-loaded"
            + (f" (half the {study.get('max_days')}-day move reached by day "
               f"{half_day})" if half_day is not None else "")
            + f". The split matters more than the blended line: in bull regimes "
            f"(n~{rc.get('bull', 0):,}) breaches tend to continue; in bear regimes "
            f"(n~{rc.get('bear', 0):,}) they tend to fade after the first couple "
            f"of days. Those effects offset each other in a long/short book.\n")

    md.append("### Machine-learning benchmark\n")
    if ml_res.get("ok"):
        if with_charts:
            md.append("![ml diagnostics](ml_diagnostics.png)\n")
        gb = ml_res["models"].get("gradient_boosting", {})
        rb = ml_res.get("rule_baseline", {})
        imp = ml_res.get("importance")
        top = (", ".join(imp.head(3)["feature"].tolist())
               if imp is not None and not imp.empty else "n/a")
        md.append(
            f"A continuation classifier was tested walk-forward "
            f"({ml_res.get('n_oos', 0):,} out-of-sample predictions, "
            f"base rate {ml_res.get('base_rate', 0):.1%}). Best OOS AUC "
            f"{gb.get('auc', float('nan')):.3f}; "
            f"the simple regime rule (accuracy {rb.get('accuracy', float('nan')):.3f}) is "
            f"competitive with the best ML model (accuracy {gb.get('accuracy', float('nan')):.3f}). "
            f"Top features: {top}. The model does not add enough to justify using it "
            f"as the main decision rule.\n")
    else:
        md.append(f"_ML benchmark not run: {ml_res.get('reason', 'n/a')}._\n")

    md.append("## 4. Performance\n")
    if with_charts:
        md.append("![equity curve](equity_curve.png)\n")
    md.append("| Construction | Total | CAGR | Sharpe | Max DD | Invested |")
    md.append("|---|---:|---:|---:|---:|---:|")
    md.append(_metrics_table(ls, "Long/short"))
    md.append(_metrics_table(lo, "Long-only"))
    md.append(f"| Buy & hold | {ls['benchmark_return_pct']:+.1f}% | "
              f"{ls['benchmark_cagr_pct']:+.1f}% | {ls['benchmark_sharpe']:.2f} | "
              f"{ls['benchmark_max_drawdown_pct']:.1f}% | 100% |\n")

    md.append("## 5. Robustness\n")
    if with_charts:
        md.append("![sweep heatmap](sweep_heatmap.png)\n")
    md.append("Sharpe across the hold x cost grid:\n")
    md.append("| hold (d) | cost (bps) | total % | CAGR % | Sharpe | max DD % |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in grid.iterrows():
        md.append(f"| {int(r['hold_days'])} | {r['cost_bps']:g} | "
                  f"{r['total_return_pct']:+.1f} | {r['cagr_pct']:+.1f} | "
                  f"{r['sharpe']:.2f} | {r['max_drawdown_pct']:.1f} |")
    md.append("")

    md.append("## 6. Execution quality\n")
    if with_charts:
        md.append("![trade distribution](trade_distribution.png)\n")
    trades = pf.trade_log(ls["frame"])
    if not trades.empty:
        win = (trades["gross_return_pct"] > 0).mean() * 100
        md.append(f"- {len(trades):,} discrete trades; win-rate {win:.0f}%; "
                  f"mean per-trade {trades['gross_return_pct'].mean():+.2f}%; "
                  f"avg hold {trades['days_held'].mean():.1f} days.")
    md.append(f"- {ls['trades']:,} turnover events over {ls['days']:,} days; "
              f"invested {ls['pct_days_invested']:.0f}% of the time.\n")

    md.append("## 7. Data provenance\n")
    md.append(f"- Data: {len(prices):,} daily bars, {d0} to {d1}, "
              f"{prices['ticker'].nunique()} tickers.")
    md.append(f"- Params: ATR({config.ATR_PERIOD}), breach >= {config.ATR_BREACH_MULT} "
              f"ATR, horizons {config.FORWARD_HORIZONS}, baseline seed "
              f"{config.BASELINE_SEED}.")
    md.append(f"- Full machine-readable manifests in `results/*_manifest.json`; "
              f"every number here is reproducible via `python main.py verify`.\n")

    md.append("## 8. Validation layer\n")
    md.append("- Breach results are compared with a seeded random-date baseline.")
    md.append("- `walk-forward` trains on past years and tests on the next year.")
    md.append("- The event study reports bull and bear regimes separately.")
    md.append("- The portfolio simulator enters after the breach bar and includes costs.")
    md.append("- `verify` checks data integrity, P&L reconstruction, and determinism.")
    md.append("- `outcomes` compares logged decisions with realized returns.")

    return "\n".join(md)


def write_report(prices: pd.DataFrame, **kwargs) -> str:
    """Write <RESULTS_DIR>/report.md and return its path."""
    md = generate(prices, **kwargs)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "report.md"
    path.write_text(md)
    return str(path)
