"""Train, evaluate and save the hybrid fraud detector.

Run from the repository root:

    python -m src.train --data data/creditcard.csv
    python -m src.train --data data/creditcard.csv --n-iter 20 --repeats 5
    python -m src.train --synthetic --n-iter 2 --repeats 1      # smoke test, no download

Models compared on the same held-out test set
  A. Isolation Forest only
  B. XGBoost + SMOTE (no anomaly feature)
  C. Isolation Forest score + XGBoost, no SMOTE   (ablation: does SMOTE help?)
  D. Hybrid: Isolation Forest score + XGBoost + SMOTE   <- saved model

Leakage controls: stratified split first; SMOTE only inside training folds; Isolation Forest
scores of training rows are out-of-fold; the decision threshold is chosen on out-of-fold
TRAIN predictions to meet a false-positive-rate budget, then applied once to the test set.

Outputs: models/fraud_model.joblib, models/metrics.json, models/figures/*.png and
dashboard/powerbi_data/scored_test.csv (git-ignored).
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import ISO_COLS, SEED, build_features, load_data
from .evaluate import (
    FPR_BUDGETS,
    bootstrap_pr_auc,
    measure_latency,
    operating_point,
    plot_cm,
    plot_curves,
    plot_eda,
    plot_importance,
    plot_model_comparison,
    plot_operating_points,
    plot_score_dist,
    recall_at_fpr,
    scored_frame,
)
from .models import SmoteXGB, iforest_scores, threshold_for_fpr, tune_hybrid

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FINAL_NAME = "D. Hybrid IF + XGBoost + SMOTE"


def main():
    ap = argparse.ArgumentParser(description="Train the hybrid fraud detector")
    ap.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--scores-out", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    ap.add_argument("--n-iter", type=int, default=12, help="RandomizedSearchCV iterations")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--max-fpr", type=float, default=0.001,
                    help="false-positive-rate budget for the headline operating point")
    ap.add_argument("--repeats", type=int, default=5,
                    help="extra repeated stratified hold-out runs for a mean +/- std (0 = skip)")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    models_dir = Path(args.models_dir)
    fig_dir = models_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.synthetic, args.keep_duplicates)
    plot_eda(df, fig_dir)
    X, y, iso_cols = build_features(df)
    print(f"[features] {X.shape[1]} base features (+1 Isolation Forest score for the hybrid)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"[split] train = {len(X_tr):,} ({int(y_tr.sum())} fraud) | test = {len(X_te):,} ({int(y_te.sum())} fraud)")

    print("\n[iforest] Computing out-of-fold anomaly scores ...")
    A_tr, A_te = X_tr[iso_cols].values, X_te[iso_cols].values
    iso_oof, iso_te, iforest = iforest_scores(A_tr, A_te)
    Xh_tr = X_tr.copy()
    Xh_tr["iforest_score"] = iso_oof
    Xh_te = X_te.copy()
    Xh_te["iforest_score"] = iso_te
    names_h = list(Xh_tr.columns)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)

    print("\n[xgboost] Randomized search (SMOTE inside each training fold, scoring = PR-AUC) ...")
    search = tune_hybrid(Xh_tr, y_tr, cv, n_iter=args.n_iter)
    best = search.best_params_
    print(f"  best CV PR-AUC = {search.best_score_:.4f}")
    print(f"  best params    = {best}")
    hybrid = search.best_estimator_

    print("\n[models] Fitting comparison models (hybrid's tuned parameters, not tuned separately) ...")
    base_params = dict(best)
    model_b = SmoteXGB(use_smote=True, **base_params).fit(X_tr, y_tr)
    model_c = SmoteXGB(use_smote=False, **base_params).fit(Xh_tr, y_tr)

    scores = {
        "A. Isolation Forest only": iso_te,
        "B. XGBoost + SMOTE (no IF)": model_b.predict_proba(X_te)[:, 1],
        "C. IF + XGBoost, no SMOTE": model_c.predict_proba(Xh_te)[:, 1],
        FINAL_NAME: hybrid.predict_proba(Xh_te)[:, 1],
    }
    p_final = scores[FINAL_NAME]

    comparison = {}
    print("\n" + "=" * 78)
    print("MODEL COMPARISON (held-out test set; recall@FPR uses the best possible threshold,")
    print("so it is for comparing models only. Deployable numbers follow below.)")
    print("=" * 78)
    print(f"{'Model':34s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'R@0.05%':>8s} {'R@0.1%':>8s} {'R@0.5%':>8s}")
    for name, p in scores.items():
        row = {"roc_auc": float(roc_auc_score(y_te, p)), "pr_auc": float(average_precision_score(y_te, p))}
        for b in FPR_BUDGETS:
            row[f"recall_at_fpr_{b}"] = recall_at_fpr(y_te, p, b)
        comparison[name] = row
        print(f"{name:34s} {row['roc_auc']:8.4f} {row['pr_auc']:8.4f} "
              f"{row['recall_at_fpr_0.0005']:8.3f} {row['recall_at_fpr_0.001']:8.3f} {row['recall_at_fpr_0.005']:8.3f}")

    print("\n[threshold] Out-of-fold train predictions for the final model ...")
    oof = cross_val_predict(hybrid, Xh_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    budgets = sorted(set(FPR_BUDGETS + [args.max_fpr]))
    ops = {}
    print("\n" + "=" * 78)
    print("DEPLOYABLE RESULTS: threshold fixed on TRAIN OOF predictions, applied once to TEST")
    print("=" * 78)
    for b in budgets:
        thr = threshold_for_fpr(y_tr, oof, b)
        op = operating_point(y_te, p_final, thr)
        ops[str(b)] = op
        print(f"FPR budget {b:.2%}: thr={thr:.4f} | recall {op['recall']:.1%} "
              f"(95% CI {op['recall_95ci'][0]:.1%}-{op['recall_95ci'][1]:.1%}) | "
              f"FPR {op['fpr']:.3%} | precision {op['precision']:.1%} | TP {op['tp']} FP {op['fp']} FN {op['fn']}")
    main_op = ops[str(args.max_fpr)]
    pr_ci = bootstrap_pr_auc(y_te.values, p_final)
    print(f"\nHybrid PR-AUC {comparison[FINAL_NAME]['pr_auc']:.4f} (95% CI {pr_ci[0]:.4f}-{pr_ci[1]:.4f}) | "
          f"fraud base rate {y_te.mean():.4f}")

    d_pr, c_pr = comparison[FINAL_NAME]["pr_auc"], comparison["C. IF + XGBoost, no SMOTE"]["pr_auc"]
    if d_pr <= c_pr:
        smote_note = (f"SMOTE did not improve PR-AUC over the no-SMOTE ablation on this split "
                      f"({d_pr:.4f} with SMOTE vs {c_pr:.4f} without).")
    else:
        smote_note = (f"SMOTE improved PR-AUC over the no-SMOTE ablation on this split "
                      f"({d_pr:.4f} with SMOTE vs {c_pr:.4f} without, a gain of {d_pr - c_pr:+.4f}).")
    print("\n[ablation] " + smote_note)

    print("\n[latency] Single-transaction scoring (Isolation Forest + XGBoost) ...")
    lat = measure_latency(iforest, hybrid, A_te, X_te.values)
    t = time.perf_counter()
    hybrid.predict_proba(Xh_te.values)
    batch_s = time.perf_counter() - t
    lat["batch_throughput_tx_per_s"] = float(len(Xh_te) / max(batch_s, 1e-9))
    print(f"  median {lat['median_ms']:.2f} ms | p95 {lat['p95_ms']:.2f} ms | "
          f"batch (XGBoost only) {lat['batch_throughput_tx_per_s']:,.0f} tx/s")

    rep = None
    if args.repeats > 0:
        print(f"\n[repeats] {args.repeats} repeated stratified hold-outs with the tuned params ...")
        rs = []
        for r in range(args.repeats):
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.20, stratify=y, random_state=100 + r)
            oof_r, te_r, _ = iforest_scores(Xa[iso_cols].values, Xb[iso_cols].values, seed=100 + r)
            Xa = Xa.copy()
            Xa["iforest_score"] = oof_r
            Xb = Xb.copy()
            Xb["iforest_score"] = te_r
            Xf, Xc, yf, yc = train_test_split(Xa, ya, test_size=0.20, stratify=ya, random_state=r)
            m = SmoteXGB(use_smote=True, **base_params).fit(Xf, yf)
            thr = threshold_for_fpr(yc, m.predict_proba(Xc)[:, 1], args.max_fpr)
            pb = m.predict_proba(Xb)[:, 1]
            o = operating_point(yb, pb, thr)
            rs.append({"recall": o["recall"], "fpr": o["fpr"], "precision": o["precision"],
                       "pr_auc": float(average_precision_score(yb, pb))})
            print(f"  run {r + 1}: recall {o['recall']:.1%} | FPR {o['fpr']:.3%} | "
                  f"precision {o['precision']:.1%} | PR-AUC {rs[-1]['pr_auc']:.3f}")
        rep = {k: [float(np.mean([x[k] for x in rs])), float(np.std([x[k] for x in rs]))] for k in rs[0]}
        print(f"  MEAN +/- STD: recall {rep['recall'][0]:.1%} +/- {rep['recall'][1]:.1%} | "
              f"FPR {rep['fpr'][0]:.3%} +/- {rep['fpr'][1]:.3%} | PR-AUC {rep['pr_auc'][0]:.3f} +/- {rep['pr_auc'][1]:.3f}")

    plot_curves(y_te.values, scores, fig_dir)
    plot_cm(main_op, fig_dir)
    plot_importance(names_h, np.asarray(hybrid.feature_importances_), fig_dir)
    plot_score_dist(y_te.values, p_final, main_op["threshold"], fig_dir)
    plot_model_comparison(comparison, fig_dir)
    plot_operating_points(ops, fig_dir)
    joblib.dump({"hybrid": hybrid, "iforest": iforest, "iso_cols": iso_cols,
                 "feature_names": names_h, "threshold": main_op["threshold"]},
                models_dir / "fraud_model.joblib")
    top = [names_h[i] for i in np.argsort(hybrid.feature_importances_)[::-1][:5]]
    metrics = {
        "n_transactions": int(len(df)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "n_test": int(len(y_te)), "n_test_fraud": int(y_te.sum()),
        "cv_pr_auc": float(search.best_score_),
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
        "comparison": comparison, "operating_points": ops, "main_fpr_budget": args.max_fpr,
        "hybrid_pr_auc_95ci": pr_ci, "latency": lat, "repeats": rep, "top_features": top,
        "smote_note": smote_note,
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    scores_path = Path(args.scores_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored_frame(X_te, y_te, p_final, main_op["threshold"]).to_csv(scores_path, index=False)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs saved to: {models_dir.resolve()}")


if __name__ == "__main__":
    main()
