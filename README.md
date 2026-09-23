# US_morgage_acceptance
FInal project for Interpretability, stability and algorithmic fairness. Predicting mortgage and assessing the fairness of the model

## Data

We use the [2025 HMDA Public LAR](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2025) (Loan/Application Register), ~5.1 GB / 13.5M rows.

Not committed to git (too large — see `.gitignore`). Download it and place it at:

```
data/2025_public_lar_csv.csv
```

## Setup

```
pip install -r requirements.txt
```

## Notebooks

- `notebooks/data_cleaning.ipynb` — loads the raw LAR data with polars (kept out
  of pandas to stay within RAM on the full 13.5M-row file), runs EDA (action_taken
  distribution, missingness, protected-attribute distributions), derives the binary
  target (`action_taken`: 1, 2 = originated, 3 = denied, else dropped), flags
  target-leaking columns, and produces a 70/15/15 train/val/test split stratified
  on target × race × ethnicity × sex.

  **To get your own clean data:**
  1. Download the raw file (link above) and either place it at `data/2025_public_lar_csv.csv`,
     or edit `DATA_PATH` in the notebook's second cell to point at wherever you saved it.
  2. Run the notebook top to bottom (Kernel → Restart & Run All).
  3. `data/processed/{train,val,test}.parquet` are your clean, split datasets — load with
     `pd.read_parquet(...)` or `pl.read_parquet(...)`.

  These outputs are gitignored (too large for git, same as the raw file) — everyone regenerates
  them locally by running the notebook. `RANDOM_STATE` is fixed, so the same raw file always
  produces the same split, whoever runs it.

- `notebooks/target_relationships.ipynb` — approval rate vs. each protected attribute and core
  loan/financial field, using `data/processed/train.parquet` only (not val/test). Run
  `data_cleaning.ipynb` first to produce that file.
