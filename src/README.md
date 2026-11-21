# src/

Reusable code for the whole pipeline. Run everything from the repository root so `import src...` works.

| File | Purpose |
|---|---|
| `data.py` | Load data and drop duplicates, build the 30 base features, synthetic data for smoke tests |
| `models.py` | Isolation Forest out-of-fold scoring, the `SmoteXGB` classifier (SMOTE inside `fit`), hyper-parameter search, false-positive-rate threshold |
| `evaluate.py` | Wilson intervals, recall at a fixed FPR, operating points, bootstrap PR-AUC, latency, scored test set, plots |
| `train.py` | Training entry point: `python -m src.train --data data/creditcard.csv` |
| `export_scores.py` | Rebuilds `dashboard/powerbi_data/scored_test.csv` from the saved model and checks it against `metrics.json` |
| `predict.py` | `score(df)` used by `app.py` |

## Interface used by `app.py`

```python
def score(df: pd.DataFrame) -> pd.DataFrame:
    """df has the columns Time, V1..V28 and Amount (a Class column may be present and is ignored).
    Returns df with two added columns:
        fraud_probability   float between 0 and 1
        flagged             1 if the probability is at or above the decision threshold, else 0
    """
```

`score` applies the same feature engineering as training, adds the Isolation Forest score and then runs the XGBoost model. The decision threshold is stored inside `models/fraud_model.joblib`.
