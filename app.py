import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "models" / "figures"

V_COLS = [f"V{i}" for i in range(1, 29)]
REQUIRED = ["Time"] + V_COLS + ["Amount"]
TARGET = "Class"

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")
st.title("Credit Card Fraud Detection")
st.caption("Portfolio project, not a production fraud system.")

try:
    from src.predict import score
except ImportError:
    st.error(
        "`src/predict.py` with a `score(df)` function was not found. See `src/README.md` for the interface this app expects."
    )
    st.stop()

tab_csv, tab_single, tab_perf = st.tabs(["Score a CSV", "Single transaction", "Model performance"])

with tab_csv:
    st.write("Upload a CSV in the same format as `creditcard.csv`. Required columns:")
    st.code(", ".join(REQUIRED))
    st.caption("A `Class` column is optional. If present, the app also shows how well the model did.")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        data = pd.read_csv(upload)
        missing = [c for c in REQUIRED if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            try:
                result = score(data)
            except FileNotFoundError as err:
                st.error(str(err))
                st.stop()
            flagged = result[result["flagged"] == 1].sort_values("fraud_probability", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions scored", f"{len(result):,}")
            c2.metric("Flagged as fraud", f"{len(flagged):,}")
            c3.metric("Flag rate", f"{len(flagged) / len(result):.3%}")
            if TARGET in result.columns:
                actual = int((result[TARGET] == 1).sum())
                caught = int(((result["flagged"] == 1) & (result[TARGET] == 1)).sum())
                false_pos = int(((result["flagged"] == 1) & (result[TARGET] == 0)).sum())
                d1, d2, d3 = st.columns(3)
                d1.metric("Actual fraud cases", f"{actual:,}")
                d2.metric("Recall", f"{caught / actual:.1%}" if actual else "n/a")
                d3.metric("False positives", f"{false_pos:,}")
                st.caption("These numbers are only meaningful on data the model was not trained on.")
            st.dataframe(flagged.head(500))
            st.download_button(
                "Download flagged transactions",
                data=flagged.to_csv(index=False).encode("utf-8"),
                file_name="flagged_transactions.csv",
                mime="text/csv",
            )

with tab_single:
    st.write(
        "V1-V28 are anonymised PCA components, so they cannot be filled in by hand in a meaningful way. "
        "All zeros is roughly an 'average' transaction. For realistic tests, use the CSV tab with rows from the dataset."
    )
    a, b = st.columns(2)
    amount = a.number_input("Amount", min_value=0.0, max_value=30000.0, value=50.0, step=10.0)
    time_s = b.number_input("Time (seconds since first transaction)", min_value=0.0, max_value=172800.0, value=40000.0, step=100.0)
    values = {}
    with st.expander("PCA components V1-V28 (optional)"):
        cols = st.columns(4)
        for i, name in enumerate(V_COLS):
            values[name] = cols[i % 4].number_input(name, min_value=-60.0, max_value=60.0, value=0.0, step=0.1)
    row = pd.DataFrame([{"Time": time_s, **values, "Amount": amount}])
    try:
        scored = score(row)
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    prob = float(scored["fraud_probability"].iloc[0])
    st.metric("Fraud probability", f"{prob:.2%}")
    if int(scored["flagged"].iloc[0]) == 1:
        st.warning("Flagged as potential fraud.")
    else:
        st.success("Not flagged.")

with tab_perf:
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        main = m["operating_points"][str(m["main_fpr_budget"])]
        hybrid = m["comparison"]["D. Hybrid IF + XGBoost + SMOTE"]

        st.subheader("Held-out test set results")
        st.caption(
            f"{m['n_test']:,} transactions never used for training or tuning, of which only "
            f"{m['n_test_fraud']} are fraud ({m['fraud_rate']:.2%} fraud rate). "
            "With so few fraud cases, differences of a few points are within sampling noise."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hybrid PR-AUC", f"{hybrid['pr_auc']:.3f}")
        c1.caption(f"95% CI {m['hybrid_pr_auc_95ci'][0]:.3f} to {m['hybrid_pr_auc_95ci'][1]:.3f}")
        c2.metric("Hybrid ROC-AUC", f"{hybrid['roc_auc']:.3f}")
        c3.metric(f"Recall at {m['main_fpr_budget']:.1%} FPR budget", f"{main['recall']:.1%}")
        c3.caption(f"95% CI {main['recall_95ci'][0]:.1%} to {main['recall_95ci'][1]:.1%}")
        c4.metric("Precision at that threshold", f"{main['precision']:.1%}")
        c4.caption(f"{main['tp']} caught, {main['fp']} false alarms, {main['fn']} missed")

        st.subheader("Models compared")
        comp = pd.DataFrame(m["comparison"]).T
        comp = comp.rename(columns={
            "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
            "recall_at_fpr_0.0005": "Recall @0.05% FPR",
            "recall_at_fpr_0.001": "Recall @0.1% FPR",
            "recall_at_fpr_0.005": "Recall @0.5% FPR",
        })
        st.dataframe(comp.style.format("{:.3f}"))
        st.caption(
            "Recall at a given FPR here uses the best possible threshold on the test set, so it is for "
            "comparing models only. " + m.get("smote_note", "")
        )

        st.subheader("Deployable operating points")
        st.caption("Threshold chosen on out-of-fold training predictions, then applied once to the test set.")
        rows = []
        for budget, o in sorted(m["operating_points"].items(), key=lambda kv: float(kv[0])):
            rows.append({
                "FPR budget": f"{float(budget):.2%}",
                "Threshold": round(o["threshold"], 4),
                "Recall": f"{o['recall']:.1%} ({o['recall_95ci'][0]:.1%}-{o['recall_95ci'][1]:.1%})",
                "Actual FPR": f"{o['fpr']:.3%}",
                "Precision": f"{o['precision']:.1%}",
                "Caught": o["tp"], "False alarms": o["fp"], "Missed": o["fn"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        lat = m.get("latency")
        if lat:
            st.caption(
                f"Scoring latency per transaction (Isolation Forest + XGBoost, one row at a time): "
                f"median {lat['median_ms']:.1f} ms, 95th percentile {lat['p95_ms']:.1f} ms."
            )
        with st.expander("Full metrics (JSON)"):
            st.json(m)
    else:
        st.info("Add your evaluation metrics as `models/metrics.json` to show them here.")

    figures = sorted(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    if figures:
        st.subheader("Figures")
        cols = st.columns(2)
        for i, path in enumerate(figures):
            cols[i % 2].image(str(path), caption=path.stem.replace("_", " ").title())
    elif not METRICS_PATH.exists():
        st.info("Add plots as PNG files in `models/figures/` to show them here.")
