"""XGBoost mortgage-approval model exposed through the team's shared interface.

    fit(X_train, y_train) -> model
    predict_proba(model, X) -> np.ndarray of shape (n, 2)

`X` is a DataFrame of the cleaned HMDA columns (data/processed/*.parquet); feature selection,
encoding and calibration all happen inside, so the stability/fairness harness can pass raw
cleaned rows straight in.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PARAMS_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_params.json"
RANDOM_STATE = 42

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
    "leakage": ["initially_payable_to_institution"],
    "race_proxy": ["lei"],
}
assert not set(FEATURES) & {c for cols in EXCLUDED.values() for c in cols}

DEFAULT_PARAMS = dict(
    n_estimators=500, learning_rate=0.1, max_depth=8, min_child_weight=1, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, scale_pos_weight=1.0,
)


@dataclass
class XGBScorer:
    booster: xgb.XGBClassifier
    cat_dtypes: dict
    calibrator: object
    calibration: str


def build_features(df: pd.DataFrame, cat_dtypes: dict) -> pd.DataFrame:
    X = df[RAW_COLUMNS].copy()
    income_dollars = X["income"].where(X["income"] > 0) * 1000  # HMDA income is in $000s
    X["loan_to_income"] = X["loan_amount"] / income_dollars
    for c in CATEGORICAL_FEATURES:
        X[c] = _as_str(X[c]).astype(cat_dtypes[c])
    return X[FEATURES]


def _as_str(col: pd.Series) -> pd.Series:
    # float-typed codes (1.0) must map to the same category as int codes (1)
    if pd.api.types.is_float_dtype(col):
        col = col.astype("Int64")
    return col.astype(str).where(col.notna())


def learn_cat_dtypes(df: pd.DataFrame) -> dict:
    return {
        c: pd.CategoricalDtype(sorted(_as_str(df[c]).dropna().unique()))
        for c in CATEGORICAL_FEATURES
    }


def load_params() -> dict:
    if PARAMS_PATH.exists():
        return {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text())}
    return dict(DEFAULT_PARAMS)


def make_classifier(params: dict, early_stopping_rounds=None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _fit_calibrator(p_raw: np.ndarray, y: np.ndarray, method: str):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
    if method == "platt":
        return LogisticRegression().fit(_logit(p_raw).reshape(-1, 1), y)
    return None


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(X_train: pd.DataFrame, y_train, params: dict | None = None, calibration: str = "isotonic",
        calib_frac: float = 0.1, eval_set=None, early_stopping_rounds: int = 50) -> XGBScorer:
    """Train booster on (1 - calib_frac) of X_train and calibrate on the held-out rest.

    `eval_set=(X_val, y_val)` turns on early stopping (used once, in the notebook, to fix
    n_estimators); without it the booster trains for params["n_estimators"] rounds, which is
    what bootstrap refits should use.
    """
    params = load_params() if params is None else params
    y_train = np.asarray(y_train)
    cat_dtypes = learn_cat_dtypes(X_train)

    if calibration == "none":
        X_fit, y_fit, X_cal, y_cal = X_train, y_train, None, None
    else:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train, y_train, test_size=calib_frac, stratify=y_train, random_state=RANDOM_STATE
        )

    clf = make_classifier(params, early_stopping_rounds if eval_set is not None else None)
    fit_kwargs = {"verbose": 100}
    if eval_set is not None:
        X_val, y_val = eval_set
        fit_kwargs["eval_set"] = [(build_features(X_val, cat_dtypes), np.asarray(y_val))]
    clf.fit(build_features(X_fit, cat_dtypes), y_fit, **fit_kwargs)

    calibrator = None
    if X_cal is not None:
        p_cal = clf.predict_proba(build_features(X_cal, cat_dtypes))[:, 1]
        calibrator = _fit_calibrator(p_cal, y_cal, calibration)
    return XGBScorer(clf, cat_dtypes, calibrator, calibration)


def predict_raw(model: XGBScorer, X: pd.DataFrame) -> np.ndarray:
    return model.booster.predict_proba(build_features(X, model.cat_dtypes))[:, 1]


def predict_proba(model: XGBScorer, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    p = predict_raw(model, X)
    if calibrated and model.calibrator is not None:
        if model.calibration == "isotonic":
            p = model.calibrator.predict(p)
        else:
            p = model.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    return np.column_stack([1 - p, p])
