"""
Template for a model owner's module.

Copy this file to `<name>_model.py` in this same folder, where <name> is one of
common_metrics.MODEL_NAMES ("xgboost", "logreg", "tabpfn") — e.g. `xgboost_model.py`.
As soon as it's here, every notebook under notebooks/ (interpretability, stability,
fairness) picks it up automatically. No other code changes needed.

Both functions receive/return plain pandas DataFrames built from exactly
common_metrics.FEATURES — do your own feature engineering (encoding, scaling,
whatever your model needs) inside fit(), not before calling it, so the stability
notebook's bootstrap can call fit() fresh on resampled raw rows.
"""


def fit(X_train, y_train):
    """
    X_train: DataFrame with exactly common_metrics.FEATURES columns (raw, uncleaned
        beyond what data_cleaning.ipynb already did — do your own encoding/scaling
        here, not upstream).
    y_train: Series, the binary target.

    Returns: whatever object predict_proba() below knows how to call. Doesn't have
    to be a bare sklearn/xgboost model — return a small pipeline object, a tuple,
    a dict, anything — predict_proba() just needs to know what to do with it.
    """
    raise NotImplementedError("Replace this with your model's actual fit logic.")


def predict_proba(model, X):
    """
    model: whatever fit() returned.
    X: DataFrame with exactly common_metrics.FEATURES columns.

    Returns: np.ndarray of shape (n, 2) — column 1 is P(target=1), matching
    sklearn's predict_proba convention. Every metrics notebook calls this and
    indexes [:, 1] itself.
    """
    raise NotImplementedError("Replace this with your model's actual predict_proba logic.")
