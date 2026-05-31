"""Continuation classifier benchmarked against the regime rule.

Predicts whether a breach will continue in its direction over the horizon, using
features known at the breach close. It is evaluated the same way as the rest of
the project.

Walk-forward CV trains on past years and predicts the next unseen year, pooled
out-of-sample. Every feature is causal (trailing only) so the model never sees
the test fold during training. Output includes probability calibration (a
reliability curve, not only accuracy) and permutation feature importance for
interpretability. Results are reported head-to-head against the regime rule
("continue in bull, revert in bear") to show whether the model adds anything.

Two models are used so the comparison is informative on its own: a gradient
boosted tree and logistic regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, signal_analyzer as sa

FEATURES = ["abs_move_in_atr", "breach_up", "atr_pct", "regime_bull",
            "mom_5d", "mom_20d", "dist_ma20", "vol_20d", "rel_volume"]


def build_dataset(prices: pd.DataFrame, *, horizon: int | None = None) -> pd.DataFrame:
    """One row per breach event with causal features + continuation label."""
    horizon = horizon or config.PORTFOLIO_HOLD_DAYS
    if prices.empty:
        return pd.DataFrame()
    regime = sa.market_regime(prices)
    df = prices.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker")
    ret = g["close"].pct_change()
    df["mom_5d"] = g["close"].shift(0) / g["close"].shift(5) - 1.0
    df["mom_20d"] = g["close"].shift(0) / g["close"].shift(20) - 1.0
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["dist_ma20"] = df["close"] / df["ma20"] - 1.0
    df["vol_20d"] = ret.groupby(df["ticker"]).transform(lambda s: s.rolling(20).std())
    if "volume" in df.columns:
        avg_vol = g["volume"].transform(lambda s: s.rolling(20).mean())
        df["rel_volume"] = df["volume"] / avg_vol
    else:
        df["rel_volume"] = 1.0   # neutral when volume isn't available
    df["fwd"] = g["close"].shift(-horizon) / df["close"] - 1.0

    df["abs_move_in_atr"] = df["move_in_atr"].abs()
    df["breach_up"] = (df["move_in_atr"] > 0).astype(float)
    df["regime_bull"] = (df["date"].map(regime) == "bull").astype(float)
    if "atr_pct" not in df.columns:
        df["atr_pct"] = 0.0   # volatility level unavailable -> neutral constant

    breaches = df[df["move_in_atr"].abs() >= config.ATR_BREACH_MULT].copy()
    breaches["continued"] = ((np.sign(breaches["move_in_atr"]) * breaches["fwd"]) > 0).astype(int)
    breaches["year"] = pd.to_datetime(breaches["date"]).dt.year
    # regime_bull is already in FEATURES; keep extra (non-feature) columns only.
    cols = FEATURES + ["continued", "year", "date", "ticker"]
    return breaches[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["continued"])


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05, random_state=0),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def walk_forward_ml(data: pd.DataFrame, *, min_train_years: int = 1) -> dict:
    """Expanding-window OOS evaluation of each model + the regime-rule baseline."""
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

    years = sorted(data["year"].unique())
    if len(years) <= min_train_years:
        return {}
    models = _models()
    oos = {name: {"proba": [], "y": []} for name in models}
    rule_pred, rule_y = [], []

    for i in range(min_train_years, len(years)):
        tr = data[data["year"].isin(years[:i])]
        te = data[data["year"] == years[i]]
        if tr["continued"].nunique() < 2 or te.empty:
            continue
        Xtr, ytr = tr[FEATURES].to_numpy(), tr["continued"].to_numpy()
        Xte, yte = te[FEATURES].to_numpy(), te["continued"].to_numpy()
        for name, model in models.items():
            model.fit(Xtr, ytr)
            oos[name]["proba"].extend(model.predict_proba(Xte)[:, 1].tolist())
            oos[name]["y"].extend(yte.tolist())
        # Regime-rule baseline: predict continuation iff bull regime.
        rule_pred.extend(te["regime_bull"].astype(int).tolist())
        rule_y.extend(yte.tolist())

    out = {"models": {}, "n_oos": len(rule_y)}
    for name, d in oos.items():
        y = np.array(d["y"]); p = np.array(d["proba"])
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        out["models"][name] = {
            "auc": round(float(roc_auc_score(y, p)), 4),
            "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "_proba": p, "_y": y,
        }
    if rule_y:
        ry, rp = np.array(rule_y), np.array(rule_pred)
        out["rule_baseline"] = {"accuracy": round(float(accuracy_score(ry, rp)), 4),
                                "n": int(len(ry))}
    out["base_rate"] = round(float(np.mean(rule_y)), 4) if rule_y else None
    return out


def feature_importance(data: pd.DataFrame, *, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance on a held-out last-year split (interpretability)."""
    from sklearn.inspection import permutation_importance

    years = sorted(data["year"].unique())
    if len(years) < 2:
        return pd.DataFrame()
    tr = data[data["year"].isin(years[:-1])]
    te = data[data["year"] == years[-1]]
    if tr["continued"].nunique() < 2 or te.empty:
        return pd.DataFrame()
    model = _models()["gradient_boosting"]
    model.fit(tr[FEATURES].to_numpy(), tr["continued"].to_numpy())
    r = permutation_importance(model, te[FEATURES].to_numpy(), te["continued"].to_numpy(),
                               n_repeats=n_repeats, random_state=0,
                               scoring="roc_auc")
    return (pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def run_ml(prices: pd.DataFrame, *, horizon: int | None = None) -> dict:
    data = build_dataset(prices, horizon=horizon)
    if data.empty or data["year"].nunique() < 2:
        return {"ok": False, "reason": "not enough data / years for walk-forward ML"}
    res = walk_forward_ml(data)
    res["ok"] = bool(res.get("models"))
    res["n_events"] = int(len(data))
    res["importance"] = feature_importance(data)
    res["horizon"] = horizon or config.PORTFOLIO_HOLD_DAYS
    return res


def format_summary(res: dict) -> str:
    if not res or not res.get("ok"):
        return f"ML not run: {res.get('reason', 'unknown')}" if res else "ML not run."
    L = [f"ML continuation classifier, walk-forward OOS "
         f"({res['n_oos']:,} predictions, base rate {res['base_rate']:.1%}):",
         "",
         f"  {'model':20}{'OOS AUC':>9}{'accuracy':>10}{'Brier':>8}"]
    for name, m in res["models"].items():
        L.append(f"  {name:20}{m['auc']:>9.3f}{m['accuracy']:>10.3f}{m['brier']:>8.3f}")
    rb = res.get("rule_baseline")
    if rb:
        L.append(f"  {'regime rule (base)':20}{'n/a':>9}{rb['accuracy']:>10.3f}{'n/a':>8}")
    imp = res.get("importance")
    if imp is not None and not imp.empty:
        top = ", ".join(f"{r.feature} ({r.importance:+.3f})"
                        for r in imp.head(4).itertuples())
        L += ["", f"  Top features (permutation AUC importance): {top}"]
    if res.get("models"):
        best = max(res["models"].values(), key=lambda m: m["auc"])
        edge = (best["accuracy"] - rb["accuracy"]) * 100 if rb else None
        if edge is not None:
            verdict = ("ML modestly beats the simple rule" if edge > 0.5 else
                       "the simple regime rule is competitive, so the added ML "
                       "complexity does not clearly pay")
            L += ["", f"  Verdict: {verdict} "
                  f"(best ML accuracy {best['accuracy']:.3f} vs rule {rb['accuracy']:.3f})."]
    return "\n".join(L)
