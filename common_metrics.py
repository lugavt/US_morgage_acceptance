"""
Shared scaffolding for the interpretability, stability, and fairness evaluation
notebooks under notebooks/evaluation/.

Every model owner drops a `<name>_model.py` file into `src/` (copy
`src/TEMPLATE_model.py`, matching one of MODEL_NAMES below) exposing `fit()` and
`predict_proba()`. Once that file exists, `load_model_module()` picks it up
automatically and every notebook here lights up for that model with no other code
changes needed.
"""

import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "processed"
SRC_DIR = REPO_ROOT / "src"
ARTIFACTS_DIR = REPO_ROOT / "models"

# joblib.load() on a saved model unpickles a reference to its own module (e.g.
# `xgboost_model.XGBScorer`) — this makes that resolvable via a plain import.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Locked feature / protected-set decision ---

FEATURES = [
    "applicant_credit_score_type",
    "aus_1",
    "balloon_payment",
    "business_or_commercial_purpose",
    "co_applicant_credit_score_type",
    "combined_loan_to_value_ratio",
    "conforming_loan_limit",
    "construction_method",
    "debt_to_income_ratio",
    "derived_dwelling_category",
    "derived_loan_product_type",
    "has_co_applicant",
    "income",
    "interest_only_payment",
    "intro_rate_period",
    "lei",
    "lien_status",
    "loan_amount",
    "loan_purpose",
    "loan_term",
    "loan_type",
    "manufactured_home_land_property_interest",
    "manufactured_home_secured_property_type",
    "negative_amortization",
    "occupancy_type",
    "open_end_line_of_credit",
    "other_nonamortizing_features",
    "property_value",
    "reverse_mortgage",
    "submission_of_application",
    "total_units",
]

# Excluded from every model's X
PROTECTED_COLS = [
    "applicant_age",
    "census_tract",
    "co_applicant_age",
    "county_code",
    "derived_ethnicity",
    "derived_msa_md",
    "derived_race",
    "derived_sex",
    "state_code",
    "tract_median_age_of_housing_units",
    "tract_minority_population_percent",
    "tract_one_to_four_family_homes",
    "tract_owner_occupied_units",
    "tract_population",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
]

# Subset of PROTECTED_COLS actually tested for fairness (people-level ECOA-protected
# classes) — the other 11 are geographic/tract-level training exclusions, not group
# variables to audit against directly.
AUDIT_COLS = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age", "co_applicant_age"]

assert not (set(FEATURES) & set(PROTECTED_COLS)), "a protected column ended up in FEATURES"

MODEL_NAMES = ["xgboost", "logreg", "tabpfn"]

# p > LGD / (MARGIN + LGD) with MARGIN=3%, LGD=35% (see notebooks/xgboost_model.ipynb §7) —
# the loan amount cancels out of the per-loan P&L comparison, so this same probability cutoff
# is intended to apply to every model and every applicant, not just XGBoost's own tuning.
# TODO(team): MARGIN/LGD themselves still need final team sign-off.
TEAM_THRESHOLD = 0.9210526315789473


def team_pnl_fn(y_true, y_pred, profit: float = 1.0, loss: float = 5.0) -> float:
    """
    Placeholder economic P&L metric — one unit of profit per correctly approved
    loan, `loss` units lost per approved loan that actually defaults/was denied.
    TODO(team): replace profit/loss with the agreed cost matrix from Task 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct_approvals = int(((y_pred == 1) & (y_true == 1)).sum())
    bad_approvals = int(((y_pred == 1) & (y_true == 0)).sum())
    return correct_approvals * profit - bad_approvals * loss


def load_split(name: str) -> pd.DataFrame:
    """name is one of 'train', 'val', 'test'."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run notebooks/data_cleaning.ipynb first.")
    return pd.read_parquet(path)


def get_X_y(df: pd.DataFrame):
    """Model input matrix (locked FEATURES only) and target."""
    return df[FEATURES].copy(), df["target"].copy()


def get_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Protected-attribute columns for the fairness audit (never joined into X)."""
    return df[AUDIT_COLS].copy()


def load_model_module(name: str):
    """
    Returns the src/<name>_model module if the model owner has dropped it in,
    else None. Notebooks check for None and skip that model gracefully — that stays
    true right up until each model is actually ready.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name {name!r}, expected one of {MODEL_NAMES}")
    module_path = SRC_DIR / f"{name}_model.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"{name}_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def available_models() -> dict:
    """{name: module} for every model owner who has dropped their file in so far."""
    return {name: mod for name in MODEL_NAMES if (mod := load_model_module(name)) is not None}


# Each model owner's own training notebook already scores the test set once and exports
# it here — reusing that means fairness.ipynb never needs to load or run a model for these.
PRECOMPUTED_PREDICTIONS = {
    "logreg": REPO_ROOT / "data" / "log-reg-results" / "test_predictions_with_sensitive.parquet",
    "tabpfn": REPO_ROOT / "data" / "tabpfn-results" / "test_predictions_with_sensitive.parquet",
}


def get_precomputed_predictions(name: str):
    """
    Returns the model owner's own exported test-set predictions (index = original position
    in test.parquet), or None if none exists yet for this model (xgboost has none, so
    fairness.ipynb falls back to live predict_proba for it). Only y_true/y_pred_proba and the
    protected columns come from the file — y_pred is intentionally NOT read from it: the
    export was thresholded at 0.5, not TEAM_THRESHOLD, so callers must rebuild y_pred
    themselves to stay consistent with any model that does need live inference.
    """
    path = PRECOMPUTED_PREDICTIONS.get(name)
    if path is None or not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.rename(columns={"true_target": "y_true", "predicted_probability": "y_pred_proba"}).drop(columns=["predicted_label"])


def run_step(results: dict, key: str, fn, *args, **kwargs) -> bool:
    """
    Runs fn(*args, **kwargs) and stores it at results[key] if it succeeds; prints a short
    message and leaves results[key] unset if it raises. Every metric in every notebook goes
    through this, so one metric failing (a missing token, a slow timeout, a data quirk) never
    takes any other metric down with it, for that model or any other.
    """
    try:
        results[key] = fn(*args, **kwargs)
        return True
    except Exception as e:
        print(f"    {key} failed ({type(e).__name__}: {e}) — skipping")
        return False


def report_status() -> None:
    ready = list(available_models().keys())
    waiting = [n for n in MODEL_NAMES if n not in ready]
    print(f"Ready:   {ready or '(none yet)'}")
    print(f"Waiting: {waiting or '(none — all models in)'}")


def get_or_fit_model(name: str, module, X_train, y_train):
    """
    Loads models/<name>_model.joblib if a model owner has saved one (the real,
    already-tuned model), refitting from scratch only as a fallback when no saved
    artifact exists, or when the saved one doesn't actually work against the
    current code (e.g. it was trained on a feature set that's since changed).
    Retraining a multi-thousand-tree model on the full training set just to
    explain it is pure waste when the real one is on disk — only the stability
    notebook has a genuine reason to call fit() repeatedly, since resampling and
    refitting IS the measurement there.

    A fallback refit gets written back to artifact_path, so whichever notebook
    hits the stale/missing case first pays for it once and every later call —
    same notebook, another notebook, a different run entirely — loads the fresh
    one instead of refitting again. Only call this with the FULL X_train/y_train:
    passing a subsample would get cached as if it were the real model.
    """
    import joblib

    artifact_path = ARTIFACTS_DIR / f"{name}_model.joblib"
    if artifact_path.exists():
        model = joblib.load(artifact_path)
        try:
            module.predict_proba(model, X_train.iloc[:2])
            return model
        except Exception as e:
            print(f"  {artifact_path.name} exists but doesn't match the current code ({e}) — refitting instead")

    model = module.fit(X_train, y_train)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    print(f"  saved freshly-fit model to {artifact_path.name} — later calls will load it instead of refitting")
    return model


def manual_permutation_importance(model, module, X, y, n_repeats: int = 10, random_state: int = 42) -> dict:
    """
    Model-agnostic permutation importance against AUC, working purely off the
    module.predict_proba(model, X) contract — doesn't assume a sklearn estimator,
    so it works identically for every model regardless of what fit() returned.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(random_state)
    baseline_auc = roc_auc_score(y, module.predict_proba(model, X)[:, 1])
    importances = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            shuffled_auc = roc_auc_score(y, module.predict_proba(model, X_shuffled)[:, 1])
            drops.append(baseline_auc - shuffled_auc)
        importances[col] = float(np.mean(drops))
    return importances


def importance_vector(model_type: str, model, module, X_ref, y_ref) -> np.ndarray:
    """
    A feature-importance vector over FEATURES, ordered consistently, regardless of
    model type — this is what makes d(f1, f2) = ||φ1 - φ2|| comparable across
    xgboost/logreg/tabpfn.
    """
    if model_type == "xgboost":
        for candidate in (model, getattr(model, "booster", None)):
            if candidate is None:
                continue
            try:
                gain = candidate.get_booster().get_score(importance_type="gain")
                return np.array([gain.get(f, 0.0) for f in FEATURES])
            except AttributeError:
                continue
    if model_type == "logreg":
        try:
            return np.asarray(model.coef_[0])
        except AttributeError:
            pass  # fall through if fit() didn't return a raw sklearn estimator
        try:
            # a wrapper (e.g. LogRegScorer) around a ColumnTransformer + LogisticRegression
            # pipeline — sum |coef| across each column's one-hot-expanded coefficients to get
            # one number per original feature, using the transformer's own structure rather
            # than parsing encoded feature-name strings (fragile with underscored column names)
            pipeline = model.pipeline
            coefs = pipeline.named_steps["classifier"].coef_[0]
            preprocessor = pipeline.named_steps["preprocessor"]
            agg, idx = {}, 0
            for name, trans, cols in preprocessor.transformers_:
                if name == "cat":
                    sizes = [len(c) for c in trans.named_steps["onehot"].categories_]
                else:
                    sizes = [1] * len(cols)
                for col, size in zip(cols, sizes):
                    agg[col] = float(np.sum(np.abs(coefs[idx:idx + size])))
                    idx += size
            return np.array([agg.get(f, 0.0) for f in FEATURES])
        except (AttributeError, KeyError):
            pass
    # tabpfn, and the fallback for xgboost/logreg if the native attribute isn't there
    perm = manual_permutation_importance(model, module, X_ref, y_ref, n_repeats=5)
    return np.array([perm[f] for f in FEATURES])
