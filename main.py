"""Command line entry point for the ATR/news alert project.

Commands
--------
  python main.py start       First run: fetch data, verify, show current actions
  python main.py scan        Fetch prices + news, fire alerts, archive news
  python main.py backtest    Forward returns after ATR breaches
  python main.py portfolio   Portfolio simulation of the breach signal
  python main.py sweep       Hold-period x transaction-cost grid
  python main.py event-study Average path after a breach
  python main.py walk-forward Learn on past years, test on the next year
  python main.py ml          Walk-forward ML benchmark
  python main.py signal      Today's breaches + regime-conditioned expected drift
  python main.py verify      Data and artifact consistency checks
  python main.py report      Rebuild results/report.md from cached data
  python main.py report --refresh  Scan first, then rebuild results/report.md
  python main.py notify      Scan + today's signals, then send a digest
  python main.py schedule    Install a macOS LaunchAgent to run `notify` daily
  python main.py outcomes    Realized outcomes of past decisions (--backfill)
  python main.py compare     Signal numbers across portfolio baskets
  python main.py news-value  A/B test: breach-only vs breach+news (needs news archive)
  python main.py run         scan + backtest

Data output goes to ./data/. Reports and run artifacts go to ./results/.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

from atr_news_alert import config, prices as price_mod, news as news_mod
from atr_news_alert import (engine, backtest as bt, portfolio as pf, news_store,
                            results as results_mod, signal_analyzer as sa,
                            verify as verify_mod, report as report_mod,
                            notify as notify_mod, scheduler as scheduler_mod,
                            decision as decision_mod, pipeline as pipeline_mod,
                            outcomes as outcomes_mod, portfolios as portfolios_mod,
                            walkforward as wf_mod, ml as ml_mod)


def load_watchlist() -> pd.DataFrame:
    wl = pd.read_csv(config.WATCHLIST_CSV)
    wl["ticker"] = wl["ticker"].str.strip().str.upper()
    return wl


def cmd_scan(wl: pd.DataFrame) -> pd.DataFrame | None:
    tickers = wl["ticker"].tolist()
    print(f"[1/4] Fetching prices + ATR for {len(tickers)} tickers ...")
    prices = price_mod.fetch_prices(tickers)
    if prices.empty:
        print("No price data retrieved; aborting scan.")
        return None
    price_mod.save_prices(prices)
    latest = price_mod.latest_bar(prices)
    n_breach = int(latest["breach"].sum())
    print(f"      {len(prices):,} bars saved. {n_breach} ticker(s) breached "
          f"{config.ATR_BREACH_MULT} ATR on the latest bar.")

    print("[2/4] Ingesting RSS news ...")
    news = news_mod.fetch_news(wl)
    news_mod.save_news(news)
    n_archived = news_store.append_news(news)
    summary = news_mod.summarize_by_ticker(news)
    print(f"      {len(news)} unique headlines; {len(summary)} ticker(s) with "
          f"news in the last {config.NEWS_LOOKBACK_HOURS}h. "
          f"(+{n_archived} new to history archive)")

    print("[3/4] Scoring news-triggered ATR alerts ...")
    alerts = engine.build_alerts(latest, summary)
    engine.save_alerts(alerts)

    print(f"[4/4] {len(alerts)} alert(s) >= severity {config.MIN_ALERT_SEVERITY}\n")
    if alerts.empty:
        print("No alerts right now (need recent news AND an ATR breach on the same ticker).")
    else:
        print("=" * 78)
        for _, row in alerts.iterrows():
            print(engine.format_alert(row))
            print("-" * 78)
        print(f"\nSaved -> {config.ALERTS_CSV}")
    return alerts


def cmd_backtest(wl: pd.DataFrame) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)

    per_ticker, agg = bt.backtest(prices)
    if not agg or agg.get("events", 0) == 0:
        print("No ATR breach events found in history.")
        return
    bt.save_backtest(per_ticker)

    print("\n=== ATR Breach Backtest (returns measured in the breach direction) ===")
    print(f"Events: {agg['events']}  across {agg['tickers']} tickers"
          f"  (breach = |move| >= {config.ATR_BREACH_MULT} ATR)\n")
    for h in config.FORWARD_HORIZONS:
        print(f"  {h:>2}-day forward:  avg {agg[f'avg_fwd_{h}d_pct']:+.3f}%   "
              f"hit-rate {agg[f'hit_rate_{h}d_pct']:.1f}%")

    base = bt.random_baseline(prices, agg["events"])
    if base:
        print(f"\n--- Random-date baseline (null hypothesis, mean of {base['trials']} draws) ---")
        for h in config.FORWARD_HORIZONS:
            edge = agg[f"hit_rate_{h}d_pct"] - base[f"hit_rate_{h}d_pct"]
            print(f"  {h:>2}-day forward:  avg {base[f'avg_fwd_{h}d_pct']:+.3f}%   "
                  f"hit-rate {base[f'hit_rate_{h}d_pct']:.1f}%   "
                  f"(breach edge {edge:+.1f} pp)")

    # Where does the signal work? (sector breakdown)
    if "sector" in wl.columns:
        sectors = dict(zip(wl["ticker"], wl["sector"]))
        sec = bt.by_sector(prices, sectors)
        if not sec.empty:
            h = config.FORWARD_HORIZONS[-1]
            print(f"\n--- By sector ({h}-day forward, sorted by hit-rate) ---")
            view = sec[["sector", "events", f"hit_rate_{h}d", f"avg_fwd_{h}d"]]
            print(view.sort_values(f"hit_rate_{h}d", ascending=False).to_string(index=False))

    # Is the edge stable over time? (poor-man's walk-forward)
    yr = bt.by_year(prices)
    if not yr.empty:
        h = config.FORWARD_HORIZONS[-1]
        print(f"\n--- By year ({h}-day forward; edge should not live in one regime) ---")
        view = yr[["year", "events", f"hit_rate_{h}d", f"avg_fwd_{h}d"]]
        print(view.to_string(index=False))

    print(f"\nTop tickers by event count saved -> {config.BACKTEST_CSV}")
    print(per_ticker.head(10).to_string(index=False))


def _load_prices(wl: pd.DataFrame) -> pd.DataFrame:
    """Prices from the cache if present, else fetched fresh and saved."""
    if config.PRICES_CSV.exists():
        return pd.read_csv(config.PRICES_CSV, parse_dates=["date"])
    prices = price_mod.fetch_prices(wl["ticker"].tolist())
    price_mod.save_prices(prices)
    return prices


def cmd_portfolio(wl: pd.DataFrame, long_only: bool = False,
                  charts: bool = False) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    m = pf.simulate(prices, long_only=long_only)
    if not m:
        print("No price history to simulate.")
        return
    legs = "bullish breaches only" if long_only else "every ATR breach in its direction"
    print(f"\n=== Portfolio simulation: trade {legs} ===")
    print("(event-driven, equal-weight, one unit per name, net of costs)\n")
    print(pf.format_summary(m))
    edge = m["total_return_pct"] - m["benchmark_return_pct"]
    verdict = ("beats" if edge > 0 else "trails")
    print(f"\nNet of {m['cost_bps']:.0f} bps costs the strategy {verdict} buy-and-hold "
          f"by {edge:+.1f} pp total, while invested only "
          f"{m['pct_days_invested']:.0f}% of the time (the rest in cash).")
    note = ("Note: illustrative; no slippage modeling. Treat as signal evidence, "
            "not a live trading P&L." if long_only else
            "Note: illustrative; no slippage/borrow modeling; short legs assume "
            "shortability. Treat as signal evidence, not a live trading P&L.")
    print(note)

    written = results_mod.snapshot_portfolio(m, prices, command="portfolio")
    print("\nReproducible artifacts written to ./results/:")
    for name, path in written.items():
        print(f"  - {name:11} {path.name}")

    if charts:
        from atr_news_alert import charts as charts_mod
        imgs = charts_mod.render_portfolio(m)
        print("Charts written to ./results/:")
        for name, p in imgs.items():
            print(f"  - {name:11} {p.split('/')[-1]}")


def cmd_sweep(wl: pd.DataFrame, holds, costs, long_only: bool = False,
              charts: bool = False) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    print(f"Sweeping hold={holds} x cost_bps={costs} "
          f"({'long-only' if long_only else 'long/short'}) ...")
    grid, summary = pf.sweep(prices, holds, costs, long_only=long_only)
    if grid.empty:
        print("No results (empty price history).")
        return

    print("\n=== Parameter sweep (robustness across the grid) ===")
    print(grid.to_string(index=False))
    stab = summary["stability_score"]
    print(f"\nSharpe across {summary['cells']} cells: "
          f"mean {summary['sharpe_mean']} +/- {summary['sharpe_std']}  "
          f"(positive in {summary['sharpe_positive_pct']:.0f}% of cells)")
    print(f"Stability score (mean/std of Sharpe): {stab}  "
          f"(higher = more robust to parameter choice)")
    best = summary["best"]
    print(f"Best cell: hold {int(best['hold_days'])}d / {best['cost_bps']:.0f} bps "
          f"-> Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%")

    written = results_mod.snapshot_sweep(grid, summary, prices)
    print("\nReproducible artifacts written to ./results/:")
    for name, path in written.items():
        print(f"  - {name:11} {path.name}")

    if charts:
        from atr_news_alert import charts as charts_mod
        imgs = charts_mod.render_sweep(grid)
        print("Charts written to ./results/:")
        for name, p in imgs.items():
            print(f"  - {name:11} {p.split('/')[-1]}")


def cmd_event_study(wl: pd.DataFrame, charts: bool = False) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    study = sa.event_study(prices)
    if not study or study.get("n_events", 0) == 0:
        print("No breach events to study.")
        return
    print("\n=== Event study: what price does after a breach (signal-level) ===")
    print(sa.format_summary(study))

    written = results_mod.snapshot_event_study(study, prices)
    print("\nReproducible artifacts written to ./results/:")
    for name, path in written.items():
        print(f"  - {name:11} {path.name}")

    if charts:
        from atr_news_alert import charts as charts_mod
        imgs = charts_mod.render_event_study(study)
        print("Charts written to ./results/:")
        for name, p in imgs.items():
            print(f"  - {name:11} {p.split('/')[-1]}")


def _load_news_summary() -> pd.DataFrame | None:
    """Recent-news-per-ticker summary from the last scan, if available."""
    if not config.NEWS_CSV.exists():
        return None
    try:
        news = pd.read_csv(config.NEWS_CSV, parse_dates=["published"])
        return news_mod.summarize_by_ticker(news)
    except Exception:
        return None


def cmd_signal(wl: pd.DataFrame) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    res = sa.forward_signals(prices, news_summary=_load_news_summary())
    print("\n=== Today's ATR breach signals (forward-looking decision aid) ===")
    print(sa.format_signals(res))
    print("\n" + decision_mod.format_decisions(decision_mod.build_decisions(res)))
    print("\n(Expected drift is the regime-conditioned post-breach average from the "
          "event study; guidance, not a guarantee.)")


def cmd_verify(wl: pd.DataFrame) -> int:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    print("\n=== Verification checks ===")
    checks = verify_mod.run_checks(prices)
    print(verify_mod.format_report(checks))
    return 0 if verify_mod.all_passed(checks) else 1


def cmd_report(wl: pd.DataFrame, with_charts: bool = True,
               month: str | None = None, refresh: bool = False) -> bool:
    if refresh:
        print("Refreshing prices + news before regenerating the report ...")
        if cmd_scan(wl) is None:
            print("\nRefresh failed. The cached report was left untouched. "
                  "Use `python main.py report` if you want to rebuild from cache.")
            return False

    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)

    if month is None:
        path = report_mod.write_report(prices, news_summary=_load_news_summary(),
                                       with_charts=with_charts)
        print(f"\nResearch memo written -> {path}")
        if with_charts:
            print("Charts + manifests alongside it in ./results/.")
        else:
            print("Manifests alongside it in ./results/.")
        return True

    try:
        cutoff = (pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(0))
    except Exception:
        print(f"Invalid --month '{month}'; expected YYYY-MM")
        return False
    as_of = prices[prices["date"] <= cutoff]
    if as_of.empty:
        print(f"No data on/before {month}.")
        return False
    outdir = config.ROOT / "reports" / month
    orig = config.RESULTS_DIR
    config.RESULTS_DIR = outdir
    try:
        path = report_mod.write_report(as_of, with_charts=with_charts,
                                       on=cutoff.date())
    finally:
        config.RESULTS_DIR = orig
    print(f"\nAs-of {month} memo written -> {path}")
    print(f"(Generated using only data through {cutoff.date()}; no look-ahead.)")
    return True


def cmd_notify(wl: pd.DataFrame, do_scan: bool = True) -> None:
    """Run the daily lightweight pipeline."""
    result = pipeline_mod.run_daily(
        prices_loader=lambda: _load_prices(wl),
        news_loader=_load_news_summary,
        scan_fn=lambda: cmd_scan(wl),
        mode=pipeline_mod.LIGHT, do_scan=do_scan)
    print("\n" + decision_mod.format_decisions(result["decisions"]))
    print("\nDelivery:")
    for channel, state in result["delivery"].items():
        print(f"  - {channel:8} {state}")
    print(f"\nDecision contract -> {result['decisions_path']}")


def cmd_outcomes(wl: pd.DataFrame, backfill: bool = False) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    if backfill:
        n = outcomes_mod.backfill(prices)
        print(f"Backfilled {n:,} historical decisions (in-sample calibration).")
    else:
        outcomes_mod.update(prices)
    print("\n=== Decision calibration (does the system get it right?) ===")
    print(outcomes_mod.format_calibration(outcomes_mod.calibration()))
    if backfill:
        print("\n(Backfill applies the live decision rule across all history; an "
              "in-sample sanity check, not out-of-sample proof.)")


def cmd_walk_forward(wl: pd.DataFrame) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    print("\n=== Walk-forward (out-of-sample) evaluation ===")
    print("Learns the regime->behaviour rule on training years only, then tests "
          "it on the next unseen year.\n")
    res = wf_mod.walk_forward(prices)
    print(wf_mod.format_summary(res))
    print("\n(Every metric here is on data the rule was never fit on.)")


def cmd_ml(wl: pd.DataFrame, charts: bool = False) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    print("Training walk-forward continuation classifier (gradient boosting + "
          "logistic), benchmarked vs the regime rule ...")
    res = ml_mod.run_ml(prices)
    print("\n=== ML continuation model (out-of-sample, walk-forward) ===")
    print(ml_mod.format_summary(res))
    if charts and res.get("ok"):
        from atr_news_alert import charts as charts_mod
        imgs = charts_mod.render_ml(res)
        print("\nCharts written to ./results/:")
        for name, p in imgs.items():
            print(f"  - {name:11} {p.split('/')[-1]}")
    print("\n(Same walk-forward discipline as the rest of the system: every metric "
          "is on data the model was never trained on.)")


def cmd_compare(wl: pd.DataFrame) -> None:
    print("Loading prices (using cache if present) ...")
    prices = _load_prices(wl)
    print("\n=== Signal numbers across different portfolios (5-day horizon) ===")
    print("(hit5d = breach hit-rate, edge = vs random baseline, LO = long-only "
          "strategy, B&H = buy & hold)\n")
    print(portfolios_mod.format_table(portfolios_mod.compare(prices)))


def cmd_start(wl: pd.DataFrame) -> None:
    """Friendly first-run: fetch data if needed, verify, show today's actions."""
    print("=" * 70)
    print(" ATR News-Triggered Stock Alert System - quick start")
    print("=" * 70)
    if not config.PRICES_CSV.exists():
        print("\nNo cached data yet; fetching ~5y of prices for your watchlist "
              "(one-time, ~30-60s) ...")
        prices = price_mod.fetch_prices(wl["ticker"].tolist())
        price_mod.save_prices(prices)
    else:
        print("\nUsing cached prices in ./data/ (run `scan` to refresh).")

    print("\n[1/3] Integrity check ...")
    cmd_verify(wl)
    print("\n[2/3] Today's decisions for your portfolio ...")
    prices = _load_prices(wl)
    res = sa.forward_signals(prices, news_summary=_load_news_summary())
    print(decision_mod.format_decisions(decision_mod.build_decisions(res)))
    print("\n[3/3] What you can do next:")
    print("  python main.py signal      # today's breaches + recommended actions")
    print("  python main.py compare     # signal strength across portfolio types")
    print("  python main.py report      # full research memo -> results/report.md")
    print("  python main.py notify      # push today's digest (desktop/email)")
    print("  python main.py schedule    # run it automatically every day")
    print("  python main.py outcomes --backfill   # calibration: is it actually right?")


def cmd_schedule(hour: int, minute: int) -> None:
    info = scheduler_mod.install(hour, minute)
    print(f"Installed LaunchAgent to run `notify` daily at {info['time']}:")
    print(f"  plist -> {info['plist']}")
    print("\nActivate it with:")
    print(f"  {info['load_cmd']}")
    print("\nTo stop it later:")
    print(f"  {info['unload_cmd']}")
    print("\n(Logs: data/notify.out.log and data/notify.err.log)")


def cmd_news_value(wl: pd.DataFrame) -> None:
    print("Loading prices + news archive ...")
    prices = _load_prices(wl)
    res = news_store.compare_news_vs_breach(prices)
    cov = res["coverage"]
    print(f"\nNews archive: {cov['rows']:,} headlines across {cov['days']} day(s)"
          + (f" ({cov['start']} -> {cov['end']})" if cov["start"] else ""))
    if not res.get("ready"):
        print("\n" + res["note"])
        print("\nThis is the honest answer: the historical-news A/B test is a data "
              "problem. Every `scan` grows the archive; re-run this once it spans "
              "enough breach events.")
        return
    h = res["horizon"]
    print(f"\n=== Does news add value? ({h}-day forward, breach direction) ===")
    print(f"{'Bucket':14}{'Events':>8}{'Hit-rate':>10}{'Avg return':>12}")
    for name, b in res["buckets"].items():
        print(f"{name:14}{b['events']:>8}{b['hit_rate_pct']:>9.1f}%{b['avg_fwd_pct']:>11.3f}%")


def cmd_watch(wl: pd.DataFrame, interval_min: int) -> None:
    """Run a scan every `interval_min` minutes until interrupted (Ctrl-C)."""
    print(f"Watch mode: scanning every {interval_min} min. Ctrl-C to stop.\n")
    while True:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n========== SCAN @ {stamp} ==========")
        try:
            cmd_scan(wl)
        except Exception as exc:  # never let one bad cycle kill the loop
            print(f"  ! scan cycle failed: {exc}")
        nxt = datetime.now().strftime("%H:%M:%S")
        print(f"\nSleeping {interval_min} min (last cycle ended {nxt}) ...")
        time.sleep(interval_min * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["start", "scan", "backtest", "portfolio", "sweep",
                                 "event-study", "walk-forward", "ml", "signal",
                                 "verify", "report", "notify", "schedule",
                                 "outcomes", "compare", "news-value", "run", "watch"],
                        nargs="?", default="run")
    parser.add_argument("--interval", type=int, default=config.SCAN_INTERVAL_MINUTES,
                        help="watch mode: minutes between scans")
    parser.add_argument("--long-only", action="store_true",
                        help="portfolio/sweep mode: trade only bullish breaches (no shorts)")
    parser.add_argument("--hold-period", type=int, nargs="+",
                        default=config.SWEEP_HOLD_DAYS,
                        help="sweep mode: hold-period grid (trading days)")
    parser.add_argument("--costs", type=float, nargs="+",
                        default=config.SWEEP_COST_BPS,
                        help="sweep mode: cost grid in bps per turnover unit")
    parser.add_argument("--charts", action="store_true",
                        help="portfolio/sweep mode: also render PNG charts to ./results/")
    parser.add_argument("--no-charts", action="store_true",
                        help="report mode: skip PNG rendering (markdown + CSVs only)")
    parser.add_argument("--no-scan", action="store_true",
                        help="notify mode: use cached prices/news instead of a fresh scan")
    parser.add_argument("--at", default="16:45",
                        help="schedule mode: daily run time HH:MM (local), e.g. 16:45")
    parser.add_argument("--backfill", action="store_true",
                        help="outcomes mode: build calibration from all history (in-sample)")
    parser.add_argument("--month", default=None,
                        help="report mode: as-of historical memo for YYYY-MM (no look-ahead)")
    parser.add_argument("--refresh", action="store_true",
                        help="report mode: run scan before regenerating the memo")
    args = parser.parse_args(argv)

    wl = load_watchlist()
    if args.command == "start":
        cmd_start(wl)
        return 0
    if args.command == "outcomes":
        cmd_outcomes(wl, backfill=args.backfill)
        return 0
    if args.command == "compare":
        cmd_compare(wl)
        return 0
    if args.command == "watch":
        cmd_watch(wl, args.interval)
        return 0
    if args.command == "portfolio":
        cmd_portfolio(wl, long_only=args.long_only, charts=args.charts)
        return 0
    if args.command == "sweep":
        cmd_sweep(wl, args.hold_period, args.costs,
                  long_only=args.long_only, charts=args.charts)
        return 0
    if args.command == "event-study":
        cmd_event_study(wl, charts=args.charts)
        return 0
    if args.command == "signal":
        cmd_signal(wl)
        return 0
    if args.command == "verify":
        return cmd_verify(wl)
    if args.command == "walk-forward":
        cmd_walk_forward(wl)
        return 0
    if args.command == "ml":
        cmd_ml(wl, charts=args.charts)
        return 0
    if args.command == "report":
        if args.refresh and args.month is not None:
            print("`report --refresh` cannot be combined with `--month`; historical "
                  "reports are point-in-time snapshots from cached data.")
            return 2
        ok = cmd_report(wl, with_charts=not args.no_charts, month=args.month,
                        refresh=args.refresh)
        return 0 if ok else 1
    if args.command == "notify":
        cmd_notify(wl, do_scan=not args.no_scan)
        return 0
    if args.command == "schedule":
        try:
            hh, mm = (int(x) for x in args.at.split(":"))
        except ValueError:
            print(f"Invalid --at time '{args.at}'; expected HH:MM")
            return 2
        cmd_schedule(hh, mm)
        return 0
    if args.command == "news-value":
        cmd_news_value(wl)
        return 0
    if args.command in ("scan", "run"):
        cmd_scan(wl)
    if args.command in ("backtest", "run"):
        print()
        cmd_backtest(wl)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
