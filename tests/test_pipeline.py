"""Tests for features, metrics and the Isolation Forest scoring (no xgboost / imbalanced-learn needed)."""

import numpy as np
import pandas as pd
import pytest

from src.data import ISO_COLS, REQUIRED, V_COLS, build_features, check_columns, load_data, make_features, make_synthetic
from src.evaluate import operating_point, recall_at_fpr, scored_frame, wilson
from src.models import iforest_scores, threshold_for_fpr


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(n=4000, fraud_rate=0.02)


def test_features_have_expected_columns_and_order(raw):
    X = make_features(raw)
    assert list(X.columns) == V_COLS + ["log_amount", "hour"]
    assert len(X) == len(raw)
    assert X["hour"].between(0, 24, inclusive="left").all()
    assert np.allclose(np.expm1(X["log_amount"]), raw["Amount"])


def test_build_features_returns_target_and_isolation_columns(raw):
    X, y, iso = build_features(raw)
    assert iso == ISO_COLS and all(c in X.columns for c in iso)
    assert set(y.unique()) <= {0, 1}
    assert "Class" not in X.columns


def test_check_columns_flags_missing(raw):
    check_columns(raw[REQUIRED])
    with pytest.raises(ValueError):
        check_columns(raw[REQUIRED].drop(columns=["V5"]))


def test_load_data_drops_exact_duplicates(tmp_path):
    df = make_synthetic(n=500)
    pd.concat([df, df.iloc[:20]]).to_csv(tmp_path / "d.csv", index=False)
    out = load_data(tmp_path / "d.csv")
    assert len(out) == 500
    assert len(load_data(tmp_path / "d.csv", keep_duplicates=True)) == 520


def test_wilson_interval_contains_estimate_and_is_ordered():
    lo, hi = wilson(78, 95)
    assert lo < 78 / 95 < hi
    assert 0 <= lo and hi <= 1
    assert all(np.isnan(wilson(0, 0)))


def test_operating_point_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1, 0.05, 0.01])
    op = operating_point(y, p, thr=0.5)
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (2, 1, 1, 4)
    assert op["recall"] == pytest.approx(2 / 3)
    assert op["fpr"] == pytest.approx(1 / 5)
    assert op["precision"] == pytest.approx(2 / 3)


def test_threshold_for_fpr_hits_the_budget():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(10000), np.ones(100)]).astype(int)
    p = np.concatenate([rng.random(10000) * 0.5, 0.5 + rng.random(100) * 0.5])
    thr = threshold_for_fpr(y, p, 0.01)
    assert ((p[y == 0] >= thr).mean()) == pytest.approx(0.01, abs=0.002)
    assert recall_at_fpr(y, p, 0.01) >= 0.99


def test_iforest_scores_shapes_and_no_leakage_of_training_rows(raw):
    X, y, iso = build_features(raw)
    A = X[iso].values
    oof, te, forest = iforest_scores(A[:3000], A[3000:], n_folds=3)
    assert oof.shape == (3000,) and te.shape == (1000,)
    # out-of-fold scores differ from scoring the same rows with the full forest
    assert not np.allclose(oof, -forest.score_samples(A[:3000]))


def test_scored_frame_matches_dashboard_contract(raw):
    X, y, _ = build_features(raw)
    prob = np.random.default_rng(1).random(len(X))
    out = scored_frame(X, y, prob, thr=0.9)
    assert list(out.columns) == ["row_id", "hour_of_day", "amount", "actual_fraud",
                                 "fraud_probability", "flagged", "outcome"]
    assert out["hour_of_day"].between(0, 23).all()
    assert np.allclose(out["amount"], raw["Amount"].round(2))
    assert set(out["outcome"]) <= {"True positive", "False positive", "False negative", "True negative"}
    assert (out["flagged"] == (prob >= 0.9)).all()
