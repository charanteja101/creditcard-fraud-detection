# notebooks/

| Notebook | Content |
|---|---|
| `01_eda.ipynb` | Class balance, duplicates, amount and time patterns, which PCA components separate fraud on their own |
| `02_experiments.ipynb` | Isolation Forest vs XGBoost, SMOTE vs no SMOTE, recall at a fixed false-positive rate, a random vs time-based split check, feature importance of the final model |

The notebooks are committed without outputs. Run them from this folder after downloading the data (`data/README.md`) and training the model (`models/README.md`):

```bash
cd notebooks
jupyter lab
```

They add the repository root to `sys.path` so they can import from `src/`. Model choices are made on a validation split carved out of the training data, so the final test set stays untouched. The final pipeline is `python -m src.train`.
