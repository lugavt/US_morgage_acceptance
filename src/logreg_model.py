"""Logistic regression mortgage-approval model exposed through the team's shared interface.

    fit(X_train, y_train) -> model
    predict_proba(model, X) -> np.ndarray of shape (n, 2)

`X` is a DataFrame of the cleaned HMDA columns (data/processed/*.parquet); feature selection
and preprocessing (imputation, scaling, one-hot encoding) all happen inside, so the
stability/fairness harness can pass raw cleaned rows straight in.

Same feature-trimming philosophy as src/xgboost_model.py and src/tabpfn_model.py (protected
attributes excluded, lei dropped as a race proxy), but a smaller final feature set — the
white-box model trims further for a simpler, more directly interpretable coefficient set.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PARAMS_PATH = Path(__file__).resolve().parent.parent / "models" / "logreg_params.json"
RANDOM_STATE = 42
DEFAULT_C = 1.0
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

NUMERIC_FEATURES = [
    "combined_loan_to_value_ratio", "debt_to_income_ratio", "income", "loan_amount",
    "loan_term", "property_value",
]
CATEGORICAL_FEATURES = [
    "derived_dwelling_category", "has_co_applicant", "lien_status",
    "loan_purpose", "loan_type", "occupancy_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns present in the cleaned dataset that this model never uses as inputs.
# "protected" columns stay in the data for the fairness evaluation only.
EXCLUDED = {
    "protected": [
        "derived_race", "derived_ethnicity", "derived_sex", "applicant_age", "co_applicant_age",
        "census_tract", "county_code", "state_code", "derived_msa_md",
        "tract_minority_population_percent", "tract_to_msa_income_percentage",
        "ffiec_msa_md_median_family_income", "tract_population", "tract_owner_occupied_units",
        "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
    ],
    "race_proxy": ["lei"],
}
assert not set(FEATURES) & {c for cols in EXCLUDED.values() for c in cols}


@dataclass
class LogRegScorer:
    pipeline: Pipeline  # ColumnTransformer + LogisticRegression, fit end to end
    C: float


def build_pipeline(C: float = DEFAULT_C) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def load_params() -> dict:
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text())
    return {"C": DEFAULT_C}


def fit(X_train: pd.DataFrame, y_train, C: float | None = None) -> LogRegScorer:
    """Fits on the given rows using the tuned C from models/logreg_params.json if it
    exists, else DEFAULT_C. Bootstrap refits call this exactly as-is — C is fixed once
    the real tuning run (in the notebook) has picked it, not re-searched per resample.
    """
    C = load_params()["C"] if C is None else C
    pipeline = build_pipeline(C)
    pipeline.fit(X_train[FEATURES], np.asarray(y_train))
    return LogRegScorer(pipeline, C)


def predict_proba(model: LogRegScorer, X: pd.DataFrame) -> np.ndarray:
    return model.pipeline.predict_proba(X[FEATURES])


def tune(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val, c_grid=C_GRID) -> LogRegScorer:
    """Grid-searches C on (X_val, y_val). Run once, in the notebook, to pick C — not
    part of the shared fit()/predict_proba() contract the evaluation notebooks call.
    """
    best_auc, best_model = -1.0, None
    for c in c_grid:
        model = fit(X_train, y_train, C=c)
        auc = roc_auc_score(y_val, predict_proba(model, X_val)[:, 1])
        print(f"  C={c}: validation ROC-AUC={auc:.5f}")
        if auc > best_auc:
            best_auc, best_model = auc, model
    print(f"Best C: {best_model.C} (validation ROC-AUC={best_auc:.5f})")
    return best_model
