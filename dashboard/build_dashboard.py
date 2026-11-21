"""Render the two dashboard pages described in dashboard/README.md as PNG images.

Reads the scored held-out test set and applies the same logic as dashboard/measures.dax
(amount bands, outcome, recall, precision, false positive rate, money caught / missed / blocked).

    python dashboard/build_dashboard.py
    python dashboard/build_dashboard.py --scores path/to/scored_test.csv --out dashboard/screenshots
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]

BLUE, RED, ORANGE, GREEN, GREY = "#4C78A8", "#E45756", "#F58518", "#54A24B", "#9AA0A6"
AMOUNT_ORDER = ["Under 1", "1-10", "10-100", "100-500", "500+"]
OUTCOME_ORDER = ["True positive", "False negative", "False positive", "True negative"]
OUTCOME_COLORS = {"True positive": GREEN, "False negative": RED, "False positive": ORANGE, "True negative": GREY}


def add_columns(df):
    df = df.copy()
    df["amount_band"] = pd.cut(df["amount"], [-np.inf, 1, 10, 100, 500, np.inf], right=False, labels=AMOUNT_ORDER)
    if "outcome" not in df.columns:
        flag, act = df["flagged"] == 1, df["actual_fraud"] == 1
        df["outcome"] = np.select([flag & act, flag & ~act, ~flag & act],
                                  ["True positive", "False positive", "False negative"], default="True negative")
    return df


def card(ax, title, value, sub=""):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#F4F6F8", ec="#D9DEE3"))
    ax.text(0.5, 0.74, title, ha="center", va="center", fontsize=11, color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.42, value, ha="center", va="center", fontsize=26, fontweight="bold", color="#222", transform=ax.transAxes)
    ax.text(0.5, 0.14, sub, ha="center", va="center", fontsize=9, color="#777", transform=ax.transAxes)


def layout(title):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.suptitle(title, x=0.04, ha="left", fontsize=17, fontweight="bold")
    gs = fig.add_gridspec(2, 6, height_ratios=[0.5, 1.5], hspace=0.38, wspace=0.9,
                          left=0.10, right=0.97, top=0.88, bottom=0.12)
    return fig, gs


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)


def page1(df, out):
    total, fraud = len(df), int(df["actual_fraud"].sum())
    flagged = int(df["flagged"].sum())
    fig, gs = layout("Fraud Dashboard  |  Page 1: Fraud overview (held-out test set)")
    card(fig.add_subplot(gs[0, 0:3]), "Total transactions", f"{total:,}")
    card(fig.add_subplot(gs[0, 3:6]), "Actual fraud cases", f"{fraud:,}", f"{fraud / total:.3%} of transactions")

    ax = fig.add_subplot(gs[1, 0:3])
    by_hour = df[df["actual_fraud"] == 1].groupby("hour_of_day").size().reindex(range(24), fill_value=0)
    ax.bar(by_hour.index, by_hour.values, color=RED)
    ax.set_xticks(range(0, 24, 3))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Hour of day (relative to the first transaction)")
    ax.set_ylabel("Fraud cases")
    ax.set_title("Fraud cases by hour", loc="left", fontsize=12, fontweight="bold")
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, 3:6])
    g = df.groupby("amount_band", observed=False)["actual_fraud"].agg(["mean", "sum", "size"]).reindex(AMOUNT_ORDER)
    bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=BLUE)
    for b, rate, n in zip(bars, g["mean"] * 100, g["sum"]):
        ax.text(b.get_x() + b.get_width() / 2, rate, f"{rate:.2f}%\n({int(n)} fraud)", ha="center", va="bottom", fontsize=9)
    ax.axhline(100 * fraud / total, color="k", ls="--", lw=1, alpha=0.6, label=f"Overall ({fraud / total:.2%})")
    ax.set_ylim(0, max(g["mean"].max() * 100 * 1.3, 100 * fraud / total * 1.5))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xlabel("Transaction amount")
    ax.set_title("Fraud rate by amount band", loc="left", fontsize=12, fontweight="bold")
    clean_axes(ax)
    fig.text(0.10, 0.025, f"Flagged transactions at the decision threshold: {flagged:,}. "
             "Hourly and per-band fraud counts are small, so treat them as indicative.", fontsize=9, color="#666")
    fig.savefig(out / "page1_fraud_overview.png", dpi=110)
    plt.close(fig)


def page2(df, out):
    n = df["outcome"].value_counts()
    tp, fp, fn = (int(n.get(k, 0)) for k in ("True positive", "False positive", "False negative"))
    legit = int((df["actual_fraud"] == 0).sum())
    recall, precision, fpr = tp / max(tp + fn, 1), tp / max(tp + fp, 1), fp / max(legit, 1)

    fig, gs = layout("Fraud Dashboard  |  Page 2: Detection performance (held-out test set)")
    card(fig.add_subplot(gs[0, 0:2]), "Recall", f"{recall:.1%}", "fraud cases caught")
    card(fig.add_subplot(gs[0, 2:4]), "Precision", f"{precision:.1%}", "flagged transactions that are fraud")
    card(fig.add_subplot(gs[0, 4:6]), "False positive rate", f"{fpr:.3%}", "legitimate transactions flagged")

    ax = fig.add_subplot(gs[1, 0:3])
    vals = [int(n.get(k, 0)) for k in OUTCOME_ORDER]
    bars = ax.barh(OUTCOME_ORDER[::-1], vals[::-1], color=[OUTCOME_COLORS[k] for k in OUTCOME_ORDER[::-1]])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,}", va="center", fontsize=10)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(vals) * 4)
    ax.set_title("Prediction outcomes (log scale)", loc="left", fontsize=12, fontweight="bold")
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, 3:6])
    money = {
        "Fraud amount caught": df.loc[df["outcome"] == "True positive", "amount"].sum(),
        "Fraud amount missed": df.loc[df["outcome"] == "False negative", "amount"].sum(),
        "Legit amount blocked": df.loc[df["outcome"] == "False positive", "amount"].sum(),
    }
    bars = ax.bar(list(money), list(money.values()), color=[GREEN, RED, ORANGE])
    for b, v in zip(bars, money.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=10)
    caught, missed = money["Fraud amount caught"], money["Fraud amount missed"]
    share = caught / (caught + missed) if caught + missed else float("nan")
    ax.set_ylim(0, max(money.values()) * 1.2)
    ax.set_ylabel("Amount")
    ax.set_title(f"Money caught vs missed vs blocked ({share:.0%} of fraud amount caught)", loc="left",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    clean_axes(ax)
    fig.savefig(out / "page2_detection_performance.png", dpi=110)
    plt.close(fig)
    return recall, precision, fpr


def main():
    ap = argparse.ArgumentParser(description="Render the fraud dashboard pages as PNG images")
    ap.add_argument("--scores", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "screenshots"))
    args = ap.parse_args()

    scores = Path(args.scores)
    if not scores.exists():
        raise SystemExit(f"{scores} not found. Run `python -m src.export_scores --data data/creditcard.csv` first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_columns(pd.read_csv(scores))
    page1(df, out)
    r, p, f = page2(df, out)
    print(f"Wrote 2 dashboard images to {out}  (recall {r:.1%}, precision {p:.1%}, FPR {f:.3%})")


if __name__ == "__main__":
    main()
