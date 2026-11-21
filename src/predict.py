"""Scoring used by ``app.py``.

    from src.predict import score
    scored = score(df)   # df has Time, V1..V28, Amount (a Class column is ignored)

Run commands from the repository root so ``import src...`` works.
"""

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import check_columns, make_features

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def _load():
    path = MODELS_DIR / "fraud_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found in models/. Run `python -m src.train --data data/creditcard.csv` first."
        )
    # A model saved by the original single-file script refers to the class as `__main__.SmoteXGB`.
    # Registering the class there lets such a file load; models saved by `src.train` do not need it.
    from .models import SmoteXGB

    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "SmoteXGB"):
        main_module.SmoteXGB = SmoteXGB
    return joblib.load(path)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with fraud_probability and flagged (1 = at or above the decision threshold)."""
    bundle = _load()
    check_columns(df)
    X = make_features(df)
    A = X[bundle["iso_cols"]].values
    iforest = bundle["iforest"]
    iforest.set_params(n_jobs=-1 if len(df) > 1000 else 1)  # parallel only pays off for large batches
    Xh = X.copy()
    Xh["iforest_score"] = -iforest.score_samples(A)
    prob = bundle["hybrid"].predict_proba(Xh[bundle["feature_names"]].values)[:, 1]

    out = df.copy()
    out["fraud_probability"] = prob
    out["flagged"] = (prob >= bundle["threshold"]).astype(int)
    return out


def decision_threshold() -> float:
    return float(_load()["threshold"])
