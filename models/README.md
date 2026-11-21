# models/

Saved model and evaluation outputs from the committed run.

| Item | Content |
|---|---|
| `fraud_model.joblib` | Bundle with the hybrid XGBoost model (`hybrid`), the fitted Isolation Forest (`iforest`), the Isolation Forest input columns (`iso_cols`), the 31 model feature names in order (`feature_names`) and the decision threshold (`threshold`, 0.112, the 0.1% false-positive-rate budget). Loaded by `src/predict.py` |
| `metrics.json` | Test-set metrics: model comparison, operating points with confidence intervals, repeated hold-out results, latency, best hyper-parameters. The app's *Model performance* tab shows this file |
| `figures/` | EDA overview, precision-recall, recall versus false-positive rate, confusion matrix, feature importance, score distribution, model comparison and operating-point plots. The app shows every PNG here |

## Reproduce

```bash
pip install -r requirements.txt
python -m src.train --data data/creditcard.csv
```

- Random seed: 42 (splits, cross-validation, search, Isolation Forest and SMOTE). The repeated hold-outs use seeds 100 and up.
- Defaults: 12 randomised-search iterations x 5-fold CV (60 fits with SMOTE inside every fold), 5 repeated hold-outs. Use `--n-iter 20` for a wider search and `--repeats 0` to skip the repeats.
- Run time depends on your CPU; the script prints the total when it finishes. `--synthetic --n-iter 2 --repeats 1` is a smoke test that finishes in seconds.
- Also writes `dashboard/powerbi_data/scored_test.csv` (git-ignored).
- To check that the committed model still reproduces the reported numbers without retraining: `python -m src.export_scores --data data/creditcard.csv`.

## Compatibility notes

- `.joblib` files are pickles, so load them with scikit-learn, XGBoost and imbalanced-learn versions close to the ones used to create them. If loading fails after upgrading libraries, re-run the training command above.
- The committed bundle was created by the original single-file script, so it refers to the classifier as `__main__.SmoteXGB`. `src/predict.py` registers `src.models.SmoteXGB` under that name before loading. A bundle written by `python -m src.train` refers to `src.models.SmoteXGB` directly and needs no workaround. Load either from the repository root.
- `metrics.json` was edited once by hand: the `smote_note` text was reworded from an instruction to the author into a plain statement. All numbers are unchanged.
