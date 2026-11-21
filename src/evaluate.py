"""Metrics, confidence intervals, latency measurement and plots."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .data import SEED, TARGET

FPR_BUDGETS = [0.0005, 0.001, 0.005]


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    """Wilson 95% confidence interval for a proportion k / n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def recall_at_fpr(y, p, max_fpr):
    """Best achievable recall with FPR <= max_fpr (oracle threshold; for model comparison only)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= max_fpr].max())


def operating_point(y, p, thr):
    y = np.asarray(y)
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1), "recall_95ci": wilson(tp, tp + fn),
        "fpr": fp / max(fp + tn, 1), "fpr_95ci": wilson(fp, fp + tn),
        "precision": tp / max(tp + fp, 1),
    }


def bootstrap_pr_auc(y, p, n_boot=200, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def measure_latency(iforest, model, A_rows, X_rows, n=300):
    """Per-transaction latency: Isolation Forest score + XGBoost probability, one row at a time."""
    try:
        model.model_.set_params(n_jobs=1)
    except Exception:
        pass
    iforest.set_params(n_jobs=1)
    times = []
    n = min(n, len(A_rows))
    for i in range(n):
        t = time.perf_counter()
        s = -iforest.score_samples(A_rows[i:i + 1])
        feats = np.hstack([X_rows[i:i + 1], s.reshape(-1, 1)])
        model.predict_proba(feats)
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times[10:])  # drop warm-up
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "max_ms": float(times.max())}


def scored_frame(X_te, y_te, prob, thr):
    """Scored held-out test set in the layout used by sql/ and dashboard/ (see dashboard/README.md)."""
    flagged = (np.asarray(prob) >= thr).astype(int)
    actual = np.asarray(y_te)
    outcome = np.select(
        [(flagged == 1) & (actual == 1), (flagged == 1) & (actual == 0), (flagged == 0) & (actual == 1)],
        ["True positive", "False positive", "False negative"],
        default="True negative",
    )
    return pd.DataFrame({
        "row_id": X_te.index,
        "hour_of_day": np.floor(X_te["hour"].values).astype(int),
        "amount": np.expm1(X_te["log_amount"].values).round(2),
        "actual_fraud": actual,
        "fraud_probability": np.asarray(prob),
        "flagged": flagged,
        "outcome": outcome,
    })


# ---------------------------------------------------------------- plots
def plot_eda(df, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    c = df[TARGET].value_counts().sort_index()
    ax[0].bar(["Legit", "Fraud"], c.values, color=["#4C78A8", "#E45756"])
    ax[0].set_yscale("log")
    for i, v in enumerate(c.values):
        ax[0].text(i, v, f"{v:,}\n({v / c.sum():.2%})", ha="center", va="bottom")
    ax[0].set_title("Class balance (log scale)")
    for k, col, lab in [(0, "#4C78A8", "Legit"), (1, "#E45756", "Fraud")]:
        ax[1].hist(np.log1p(df.loc[df[TARGET] == k, "Amount"]), bins=40, alpha=0.6,
                   color=col, label=lab, density=True)
    ax[1].set_title("log(1 + Amount) by class")
    ax[1].legend()
    hrs = (df["Time"] % 86400) / 3600
    for k, col, lab in [(0, "#4C78A8", "Legit"), (1, "#E45756", "Fraud")]:
        ax[2].hist(hrs[df[TARGET] == k], bins=24, alpha=0.6, color=col, label=lab, density=True)
    ax[2].set_title("Hour of day by class")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(out / "eda_overview.png", dpi=150)
    plt.close()


def plot_curves(y, scores, out):
    plt.figure(figsize=(6.5, 5))
    for name, p in scores.items():
        pr, rc, _ = precision_recall_curve(y, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    plt.axhline(np.mean(y), color="k", ls="--", alpha=0.4)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (held-out test set)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    for b in FPR_BUDGETS:
        plt.axvline(b, color="grey", ls=":", alpha=0.5)
    plt.xscale("log")
    plt.xlim(1e-4, 1)
    plt.xlabel("False positive rate (log scale)")
    plt.ylabel("Recall (TPR)")
    plt.title("Recall vs false positive rate (dotted: 0.05%, 0.1%, 0.5%)")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150)
    plt.close()


def plot_cm(op, out):
    cm = np.array([[op["tn"], op["fp"]], [op["fn"], op["tp"]]])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > np.sqrt(cm.max()) else "black")
    plt.xticks([0, 1], ["Pred: Legit", "Pred: Fraud"])
    plt.yticks([0, 1], ["Actual: Legit", "Actual: Fraud"])
    plt.title(f"Hybrid model confusion matrix (thr={op['threshold']:.3f})")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_importance(names, imp, out):
    order = np.argsort(imp)[-15:]
    plt.figure(figsize=(7, 5.5))
    plt.barh(np.array(names)[order], np.array(imp)[order], color="#4C78A8")
    plt.title("Hybrid model feature importance (gain)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()


def plot_score_dist(y, p, thr, out):
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    plt.hist(p[y == 0], bins=bins, alpha=0.6, color="#4C78A8", label="Legit")
    plt.hist(p[y == 1], bins=bins, alpha=0.6, color="#E45756", label="Fraud")
    plt.axvline(thr, color="k", ls="--", label=f"Threshold {thr:.3f}")
    plt.yscale("log")
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Transactions (log scale)")
    plt.title("Score distribution (held-out test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "score_distribution.png", dpi=150)
    plt.close()


def plot_model_comparison(comparison, out):
    """PR-AUC and recall at 0.1% FPR for the four compared models (from metrics['comparison'])."""
    names = list(comparison)
    short = [n.split(".")[0] for n in names]
    colors = ["#9AA0A6", "#4C78A8", "#54A24B", "#E45756"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, key, title in [(ax[0], "pr_auc", "PR-AUC (higher is better)"),
                          (ax[1], "recall_at_fpr_0.001", "Recall at 0.1% false positive rate")]:
        vals = [comparison[n][key] for n in names]
        bars = a.bar(short, vals, color=colors[:len(names)])
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        a.set_ylim(0, 1.0)
        a.set_title(title, fontsize=11)
        a.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors[:len(names)]]
    fig.legend(handles, names, loc="lower center", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(out / "model_comparison.png", dpi=150)
    plt.close()


def plot_operating_points(ops, out):
    """Recall (with 95% CI) at each false-positive budget; annotated with precision and FP/FN counts."""
    keys = sorted(ops, key=float)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(keys))
    rec = np.array([ops[k]["recall"] for k in keys])
    lo = np.array([ops[k]["recall_95ci"][0] for k in keys])
    hi = np.array([ops[k]["recall_95ci"][1] for k in keys])
    ax.bar(xs, rec, color="#4C78A8", yerr=[rec - lo, hi - rec], capsize=6, ecolor="#333")
    for x, k in zip(xs, keys):
        o = ops[k]
        ax.text(x, 0.05, f"precision {o['precision']:.0%}\nFP {o['fp']}  FN {o['fn']}", ha="center",
                color="white", fontsize=9)
        ax.text(x, o["recall_95ci"][1] + 0.015, f"{o['recall']:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{float(k):.2%} FPR budget" for k in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall (share of fraud caught)")
    ax.set_title("Deployable operating points on the held-out test set (threshold set on training data)")
    ax.title.set_fontsize(10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "operating_points.png", dpi=150)
    plt.close()
