"""
src/model_trainer.py
====================
Step 8: Train and cross-validate all models.

Regression  (Y_reg  → next-week return):
  LinearRegression | RandomForestRegressor | DecisionTreeRegressor

Classification  (Y_clf3 → tăng / đi ngang / giảm,   Y_clf → binary):
  LogisticRegression | RandomForestClassifier | DecisionTreeClassifier

Cross-validation uses TimeSeriesSplit (never random K-Fold).
"""
import logging
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tscv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=config.TSCV_N_SPLITS)


def _cv_reg(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    scores = cross_val_score(
        model, X, y, cv=_tscv(),
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    rmse = np.sqrt(-scores)
    return {"cv_rmse_mean": round(float(rmse.mean()), 6),
            "cv_rmse_std":  round(float(rmse.std()),  6)}


def _cv_clf(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    scores = cross_val_score(
        model, X, y, cv=_tscv(),
        scoring="f1_weighted", n_jobs=-1,
    )
    return {"cv_f1_mean": round(float(scores.mean()), 4),
            "cv_f1_std":  round(float(scores.std()),  4)}


# ─────────────────────────────────────────────────────────────────────────────
# Step 8a – Regression models
# ─────────────────────────────────────────────────────────────────────────────

def train_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train LinearRegression, RandomForestRegressor, DecisionTreeRegressor.
    Each is evaluated with TimeSeriesSplit CV before the final fit.

    Returns
    -------
    models    : dict  name → fitted model
    cv_scores : dict  name → {cv_rmse_mean, cv_rmse_std}
    """
    candidates = {
        "LinearRegression":      LinearRegression(**config.LR_PARAMS),
        "RandomForestRegressor": RandomForestRegressor(**config.RF_REG_PARAMS),
        "DecisionTreeRegressor": DecisionTreeRegressor(**config.DT_REG_PARAMS),
    }

    models, cv_scores = {}, {}
    for name, mdl in candidates.items():
        cv = _cv_reg(mdl, X_train, y_train)
        cv_scores[name] = cv
        mdl.fit(X_train, y_train)
        models[name] = mdl
        log.info(
            f"  [REG] {name:<30} "
            f"CV RMSE = {cv['cv_rmse_mean']:.6f} ± {cv['cv_rmse_std']:.6f}"
        )

    return models, cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# Step 8b/c – Classification models
# ─────────────────────────────────────────────────────────────────────────────

def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    suffix: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train LogisticRegression, RandomForestClassifier, DecisionTreeClassifier.

    Parameters
    ----------
    suffix : str
        Appended to each model name for disambiguation, e.g. "_3cls" or "_bin".

    Returns
    -------
    models, cv_scores  (same structure as train_regressors)
    """
    n_classes = y_train.dropna().nunique()
    if n_classes < 2:
        log.warning(
            f"  Only {n_classes} unique class in y_train{suffix}. "
            "Skipping classifiers."
        )
        return {}, {}

    candidates = {
        f"LogisticRegression{suffix}":      LogisticRegression(**config.LOGIT_PARAMS),
        f"RandomForestClassifier{suffix}":  RandomForestClassifier(**config.RF_CLF_PARAMS),
        f"DecisionTreeClassifier{suffix}":  DecisionTreeClassifier(**config.DT_CLF_PARAMS),
    }

    models, cv_scores = {}, {}
    for name, mdl in candidates.items():
        try:
            cv = _cv_clf(mdl, X_train, y_train)
            cv_scores[name] = cv
            mdl.fit(X_train, y_train)
            models[name] = mdl
            log.info(
                f"  [CLF] {name:<40} "
                f"CV F1 = {cv['cv_f1_mean']:.4f} ± {cv['cv_f1_std']:.4f}"
            )
        except Exception as exc:
            log.warning(f"  [CLF] {name} failed: {exc}")

    return models, cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_models(models: Dict[str, Any], directory: str = None) -> None:
    """Serialise all models to disk as .pkl files."""
    directory = directory or config.MODELS_DIR
    os.makedirs(directory, exist_ok=True)
    for name, mdl in models.items():
        path = os.path.join(directory, f"{name}.pkl")
        joblib.dump(mdl, path)
    log.info(f"  Saved {len(models)} model(s) → {directory}")


def load_model(name: str, directory: str = None) -> Any:
    """Load a single model from disk."""
    directory = directory or config.MODELS_DIR
    return joblib.load(os.path.join(directory, f"{name}.pkl"))
