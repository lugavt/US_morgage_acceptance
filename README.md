# US_morgage_acceptance
FInal project for Interpretability, stability and algorithmic fairness. Predicting mortgage and assessing the fairness of the model

## Data

We use the [2025 HMDA Public LAR](https://ffiec.cfpb.gov/data-publication/dynamic-national-loan-level-dataset/2025) (Loan/Application Register), ~5.1 GB / 13.5M rows.

Not committed to git (too large — see `.gitignore`). Download it and place it at:

```
data/2025_public_lar_csv.csv
```

## Notebooks

- `notebooks/data_cleaning.ipynb` — loads the raw LAR data, derives the binary
  target (`action_taken`: 1 = originated, 3 = denied), checks missingness, and
  produces the train/val/test split.
