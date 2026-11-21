# Data

The raw data is not stored in this repository (the file is about 150 MB).

## Download

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (a free Kaggle account is needed).
2. Download `creditcard.csv`.
3. Place it here as `data/creditcard.csv`.

## About the dataset

Credit card transactions made by European cardholders over two days in September 2013. The features were anonymised with PCA before release, so the original transaction details are not available.

| Column | Meaning |
|---|---|
| `Time` | Seconds since the first transaction in the dataset (not a clock time) |
| `V1` - `V28` | Anonymised PCA components |
| `Amount` | Transaction amount |
| `Class` | Target: 1 = fraud, 0 = legitimate |

- 284,807 transactions, of which 492 (0.17%) are fraud.
- The file contains 1,081 exact duplicate rows (19 of them fraud), which are removed before splitting. This leaves 283,726 unique transactions, 473 of them fraud (0.167%).
- There are no missing values.
- `Time` is relative to the first transaction, so any hour-of-day feature is relative to the start of the two-day window rather than a clock time.
- The PCA transformation was applied by the dataset's authors before release, so it was not fitted on training rows only.

Please follow the dataset's licence and citation requirements on Kaggle.
