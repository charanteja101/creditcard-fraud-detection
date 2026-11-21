"""Data loading and feature engineering (shared by training and scoring)."""

import numpy as np
import pandas as pd

TARGET = "Class"
SEED = 42
V_COLS = [f"V{i}" for i in range(1, 29)]
REQUIRED = ["Time"] + V_COLS + ["Amount"]
# Columns fed to the Isolation Forest (the anomaly-score feature of the hybrid model).
ISO_COLS = V_COLS + ["log_amount"]


def make_synthetic(n=40000, fraud_rate=0.004, seed=SEED):
    """Synthetic data with the creditcard.csv schema (smoke tests only)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    V = rng.normal(0, 1, (n, 28))
    shift = rng.normal(0, 2.5, 28) * (rng.random(28) < 0.4)
    V[y == 1] += shift
    amount = rng.lognormal(3.5, 1.3, n)
    df = pd.DataFrame(V, columns=V_COLS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = amount
    df[TARGET] = y
    return df


def load_data(path, synthetic=False, keep_duplicates=False):
    if synthetic:
        print("[data] Using SYNTHETIC data (smoke test only, results are meaningless)")
        df = make_synthetic()
    else:
        df = pd.read_csv(path)
        print(f"[data] Loaded {path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    n0 = len(df)
    if not keep_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[clean] Dropped {n0 - len(df):,} exact duplicate rows -> {len(df):,} unique transactions")
    print(f"[data] Fraud cases: {int(df[TARGET].sum()):,} ({df[TARGET].mean():.3%})")
    return df


def make_features(df):
    """The 30 base features: V1-V28, log(1 + Amount) and hour of the (relative) day."""
    X = df[V_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    X["hour"] = (df["Time"] % 86400) / 3600.0
    return X


def build_features(df):
    """Training-time helper: features, target and the Isolation Forest input columns."""
    return make_features(df), df[TARGET].astype(int), list(ISO_COLS)


def check_columns(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
