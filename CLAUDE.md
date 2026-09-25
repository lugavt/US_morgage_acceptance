# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Final project for the HEC course *Interpretability, Stability and Algorithmic Fairness*:
modelling **US mortgage application acceptance** on the 2025 HMDA public Loan/Application
Register (LAR). The analysis dimension is as much fairness and explanation stability as raw
predictive accuracy, so protected-attribute columns (`derived_race`, `derived_ethnicity`,
`derived_sex`, `applicant_age`) matter both as potential features and as audit groups.

The repo currently contains **only the dataset** — no code has been committed yet. There is no
build, test, or lint setup to follow; establish one as code is added.

## Data

`data/2025_public_lar_csv.csv` — **4.8 GB, 13,543,606 rows, 99 columns**, one header row,
comma-delimited, all rows `activity_year == 2025`. Git-ignored: it is never committed, and any
machine that needs it must obtain it separately from the CFPB HMDA data portal.

Key columns:

- `action_taken` — the target. HMDA codes: `1` originated, `2` approved-not-accepted,
  `3` denied, `4` withdrawn by applicant, `5` file closed for incompleteness,
  `6` purchased loan, `7`/`8` preapproval denied/approved-not-accepted.
  A binary acceptance target normally keeps `1,2` vs `3` and **drops `4,5,6`** — `6` in
  particular is a loan bought on the secondary market, not an application decision, and is ~11%
  of rows.
- `derived_race`, `derived_ethnicity`, `derived_sex` — pre-aggregated protected attributes.
  In a 500k-row sample, ~16% of rows are `Race Not Available` and ~8% `Sex Not Available`;
  missingness here is itself non-random and should not be silently dropped.
- Applicant/loan features: `loan_amount`, `income`, `debt_to_income_ratio`,
  `combined_loan_to_value_ratio`, `property_value`, `loan_term`, `interest_rate`, `loan_purpose`,
  `lien_status`, `occupancy_type`.
- Tract context: `tract_minority_population_percent`, `tract_to_msa_income_percentage`,
  `ffiec_msa_md_median_family_income`, `tract_population` — proxies for race, so their inclusion
  is a fairness decision, not just a modelling one.

### HMDA sentinel values

Missing and coded values do **not** arrive as `NaN`. Treat these as missing before modelling:

- `NA` and empty strings throughout.
- `applicant_age` / `co_applicant_age`: `8888` = not applicable, `9999` = no co-applicant.
- `derived_msa_md`: `99999` = not in an MSA/MD.
- `debt_to_income_ratio` is **a string** with buckets like `"50%-60%"`, `">60%"`, `"<20%"` mixed
  with bare numbers — it needs explicit parsing, never a plain `astype(float)`.
- Many rate/fee columns are `"Exempt"` for exempt filers.

Columns like `county_code` and `census_tract` are zero-padded identifiers; read them as strings
or the leading zeros are lost (`04019` → `4019`).

## Working with a 4.8 GB CSV

Never `pd.read_csv` the whole file — it will exhaust memory. Instead:

- Pass `usecols=[...]` with only the needed columns, plus explicit `dtype=` to avoid the
  `low_memory` type-inference warnings this file triggers.
- Use `chunksize=` for full passes, or sample with `head -n N` into the scratchpad first.
- Convert once to Parquet (`pyarrow` is installed) and work from that for every later step.
- For quick exploration, `head`/`awk`/`wc -l` on the CSV beats loading it.

## Environment

Use the Anaconda interpreter — the system and Homebrew Pythons have **no** scientific stack:

```
/opt/anaconda3/bin/python
```

Available: Python 3.13.5, pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1, xgboost 3.1.2,
lightgbm 4.6.0, shap 0.51.0, pyarrow 19.0.0, matplotlib 3.10.0, seaborn 0.13.2.
Not installed (`conda install` / `pip install` if needed): `fairlearn`, `aif360`, `lime`,
`polars`, `duckdb`. `jupyter` is on PATH via Homebrew.

## Conventions

- `.gitignore` excludes `data/` and `*.csv`/`*.parquet` broadly. To commit a small results table,
  add an explicit negation (e.g. `!results/metrics.csv`) rather than loosening the rule.
- Keep derived datasets and model artifacts out of git; regenerate them from scripts instead.
- Write throwaway samples and intermediate files to the session scratchpad, not the project root.
