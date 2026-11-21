"""Rebuild dashboard/powerbi_data/scored_test.csv from the SAVED model (no retraining).

The 80/20 split is deterministic (seed 42), so this scores the same held-out test rows that
``src.train`` evaluated. It also prints the confusion counts so you can compare them with
models/metrics.json.

    python -m src.export_scores --data data/creditcard.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from .data import SEED, build_features, load_data
from .evaluate import scored_frame
from . import predict

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Export the scored held-out test set")
    ap.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    args = ap.parse_args()

    predict.MODELS_DIR = Path(args.models_dir)
    predict._load.cache_clear()
    bundle = predict._load()
    thr = float(bundle["threshold"])

    X, y, _ = build_features(load_data(args.data))
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    Xh = X_te.copy()
    Xh["iforest_score"] = -bundle["iforest"].score_samples(X_te[bundle["iso_cols"]].values)
    prob = bundle["hybrid"].predict_proba(Xh[bundle["feature_names"]].values)[:, 1]

    scored = scored_frame(X_te, y_te, prob, thr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"Wrote {len(scored):,} scored rows to {out_path}")

    counts = scored["outcome"].value_counts()
    tp, fp, fn, tn = (int(counts.get(k, 0)) for k in ("True positive", "False positive", "False negative", "True negative"))
    print(f"At threshold {thr:.4f}: TP {tp} | FP {fp} | FN {fn} | TN {tn} | "
          f"recall {tp / max(tp + fn, 1):.1%} | precision {tp / max(tp + fp, 1):.1%} | "
          f"PR-AUC {average_precision_score(y_te, prob):.4f}")

    metrics_path = Path(args.models_dir) / "metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        ref = saved["operating_points"][str(saved["main_fpr_budget"])]
        same = (tp, fp, fn, tn) == (ref["tp"], ref["fp"], ref["fn"], ref["tn"])
        print("Matches models/metrics.json:", "yes" if same else
              f"NO (metrics.json has TP {ref['tp']} FP {ref['fp']} FN {ref['fn']} TN {ref['tn']})")


if __name__ == "__main__":
    main()
