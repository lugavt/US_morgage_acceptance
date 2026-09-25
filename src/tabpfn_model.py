"""TabPFN mortgage-approval model exposed through the team's shared interface.

    fit(X_train, y_train) -> model
    predict_proba(model, X) -> np.ndarray of shape (n, 2)

`X` is a DataFrame of the cleaned HMDA columns (data/processed/*.parquet); feature selection,
context sampling and encoding all happen inside, so the stability/fairness harness can pass raw
cleaned rows straight in.

TabPFN has no training of its own: `fit` draws a stratified sample of the training rows (the
*context*), and every prediction reads that context together with the rows being scored. So a
bootstrap "refit" is simply a new context drawn from the resampled rows.

The saved model holds the context and the category codes, not the fitted network (which is ~900 MB
for TabPFN-3.5). The network is rebuilt from them on first use, on whatever device the loading
machine has; the same context and seed give the same model. Loading needs the same `tabpfn`
version and, for TabPFN-3.5, a Prior Labs key in `~/.tabpfn/token` or `TABPFN_TOKEN`.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# TabPFN refuses large inputs on CPU by default purely for speed reasons. This lifts only that
# guard; the model's own pre-training limits stay enforced.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

RANDOM_STATE = 42
MODEL_VERSION = "v3.5"   # TabPFN-3.5, the latest release (tabpfn 9.0.0)
CONTEXT_ROWS = 10_000    # training rows the model conditions on
N_ESTIMATORS = 4         # same validation AUC as 8 in the speed check, at half the scoring cost
BATCH_ROWS = 20_000      # rows per predict call; scores depend slightly on the batch, see notebook

# Same feature set as src/xgboost_model.py, so the two models differ only in the learner.
NUMERIC_FEATURES = [
    "loan_amount", "combined_loan_to_value_ratio", "loan_term", "intro_rate_period",
    "property_value", "total_units", "income", "debt_to_income_ratio",
]
ENGINEERED_FEATURES = ["loan_to_income"]
CATEGORICAL_FEATURES = [
    "conforming_loan_limit", "derived_loan_product_type", "derived_dwelling_category",
    "loan_type", "loan_purpose", "lien_status", "reverse_mortgage", "open_end_line_of_credit",
    "business_or_commercial_purpose", "negative_amortization", "interest_only_payment",
    "balloon_payment", "other_nonamortizing_features", "construction_method", "occupancy_type",
    "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
    "applicant_credit_score_type", "co_applicant_credit_score_type",
    "submission_of_application", "aus_1", "has_co_applicant",
]
FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
RAW_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns present in the cleaned dataset that the model never uses as inputs.
# "protected" columns stay in the data for the fairness evaluation only.
EXCLUDED = {
    "protected": [
        "derived_race", "derived_ethnicity", "derived_sex", "applicant_age", "co_applicant_age",
        "census_tract", "county_code", "state_code", "derived_msa_md",
        "tract_minority_population_percent", "tract_to_msa_income_percentage",
        "ffiec_msa_md_median_family_income", "tract_population", "tract_owner_occupied_units",
        "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
    ],
    "leakage": ["initially_payable_to_institution"],  # already deleted upstream; listed as in xgboost_model.py
    "race_proxy": ["lei"],
}
assert not set(FEATURES) & {c for cols in EXCLUDED.values() for c in cols}

# Context rows are drawn per stratum, with the same strata as the train/val/test split, so small
# demographic groups stay represented. Strata columns missing from X are simply skipped.
STRATA_COLS = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]


@dataclass
class TabPFNScorer:
    X_context: pd.DataFrame          # prepared features of the context rows (index = row labels in X_train)
    y_context: np.ndarray
    category_codes: dict
    model_version: str
    n_estimators: int
    random_state: int
    batch_rows: int
    classifier: object = field(default=None, repr=False)  # rebuilt from the context when needed

    def __getstate__(self):
        # Don't pickle the fitted network: it's rebuilt from the context on first use.
        return {**self.__dict__, "classifier": None}


def stratified_sample(X: pd.DataFrame, y, n: int | None, random_state: int = RANDOM_STATE) -> pd.Index:
    """Index labels of about `n` rows of X, the same fraction from every stratum (seeded)."""
    if n is None or n >= len(X):
        return X.index
    keys = pd.DataFrame({"target": np.asarray(y)}, index=X.index)
    for c in STRATA_COLS:
        if c in X.columns:
            keys[c] = X[c].astype("string").to_numpy()
    order = np.random.default_rng(random_state).permutation(len(X))
    keys = keys.iloc[order]
    groups = keys.groupby(list(keys.columns), dropna=False, sort=False)
    rank = groups.cumcount()
    size = groups["target"].transform("size")
    keep = (rank < (size * n / len(X)).round()).to_numpy()
    return keys.index[keep].sort_values()


def _as_str(col: pd.Series) -> pd.Series:
    # float-typed codes (1.0) must map to the same category as int codes (1)
    if pd.api.types.is_float_dtype(col):
        col = col.astype("Int64")
    return col.astype(str).where(col.notna())


def learn_category_codes(X: pd.DataFrame) -> dict:
    return {c: sorted(_as_str(X[c]).dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}


def build_features(X: pd.DataFrame, category_codes: dict) -> pd.DataFrame:
    """Raw cleaned rows -> TabPFN inputs: floats, categories as integer codes, NaN left as NaN.

    No scaling or one-hot encoding: TabPFN does its own preprocessing. A category not seen in the
    context becomes NaN.
    """
    out = X[RAW_COLUMNS].copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype(float)
    income_dollars = out["income"].where(out["income"] > 0) * 1000  # HMDA income is in $000s
    out["loan_to_income"] = out["loan_amount"] / income_dollars
    for c in CATEGORICAL_FEATURES:
        codes = {label: i for i, label in enumerate(category_codes[c])}
        out[c] = _as_str(out[c]).map(codes).astype(float)
    return out[FEATURES]


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_classifier(n_estimators: int = N_ESTIMATORS, model_version: str = MODEL_VERSION,
                    random_state: int = RANDOM_STATE):
    """TabPFN classifier with the model version set explicitly, whatever the library's default is."""
    import tabpfn
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    if model_version not in {v.value for v in ModelVersion}:
        raise RuntimeError(f"tabpfn {tabpfn.__version__} has no model {model_version!r}; "
                           "TabPFN-3.5 needs tabpfn==9.0.0.")
    return TabPFNClassifier.create_default_for_version(
        ModelVersion(model_version), n_estimators=n_estimators,
        categorical_features_indices=[FEATURES.index(c) for c in CATEGORICAL_FEATURES],
        device=device(), random_state=random_state)


def fit(X_train: pd.DataFrame, y_train, context_rows: int = CONTEXT_ROWS,
        n_estimators: int = N_ESTIMATORS, model_version: str = MODEL_VERSION,
        random_state: int = RANDOM_STATE, batch_rows: int = BATCH_ROWS) -> TabPFNScorer:
    """Draw a stratified context of `context_rows` from X_train and fit TabPFN on it.

    Pass the whole training split (or a bootstrap resample of it): the context is sampled here.
    """
    y_train = pd.Series(np.asarray(y_train), index=X_train.index)
    idx = stratified_sample(X_train, y_train, context_rows, random_state)
    context = X_train.loc[idx]
    codes = learn_category_codes(context)
    model = TabPFNScorer(
        X_context=build_features(context, codes), y_context=y_train.loc[idx].to_numpy(),
        category_codes=codes, model_version=model_version, n_estimators=n_estimators,
        random_state=random_state, batch_rows=batch_rows,
    )
    _ensure_classifier(model)
    return model


def _ensure_classifier(model: TabPFNScorer) -> None:
    if model.classifier is None:
        model.classifier = make_classifier(model.n_estimators, model.model_version, model.random_state)
        model.classifier.fit(model.X_context, model.y_context)


def predict_proba(model: TabPFNScorer, X: pd.DataFrame) -> np.ndarray:
    """[P(denied), P(approved)] for every row, scored in batches of `model.batch_rows`.

    Each batch re-reads the whole context, so pass many rows per call rather than one at a time.
    """
    _ensure_classifier(model)
    X_feat = build_features(X, model.category_codes)
    b = model.batch_rows
    p = np.concatenate([model.classifier.predict_proba(X_feat.iloc[i:i + b])[:, 1]
                        for i in range(0, len(X_feat), b)])
    return np.column_stack([1 - p, p])
