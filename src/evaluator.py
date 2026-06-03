"""
src/evaluator.py
================
Step 9: Evaluate all trained models and select the best regressor.

Regression metrics  : MAE, RMSE, R², Directional Accuracy
Classification metrics : Accuracy, F1-weighted, ROC-AUC
Overfitting check   : train vs test performance gap
"""
import logging
import os
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regression evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_regression(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate regression models on the hold-out test set.

    Metrics
    -------
    MAE  – Mean Absolute Error
    RMSE – Root Mean Squared Error
    R²   – Coefficient of determination
    DirAcc – Fraction of weeks where predicted sign == actual sign
             (most relevant metric for trading)

    Returns
    -------
    pd.DataFrame sorted by RMSE (ascending = better).
    """
    rows = []
    for name, mdl in models.items():
        yp = mdl.predict(X_test)
        mae    = mean_absolute_error(y_test, yp)
        rmse   = np.sqrt(mean_squared_error(y_test, yp))
        r2     = r2_score(y_test, yp)
        diracc = float(np.mean(np.sign(yp) == np.sign(y_test.values)))

        rows.append({
            "Model":   name,
            "MAE":     round(mae,    6),
            "RMSE":    round(rmse,   6),
            "R2":      round(r2,     4),
            "DirAcc":  round(diracc, 4),
        })
        log.info(
            f"  [REG] {name:<30}  "
            f"MAE={mae:.6f}  RMSE={rmse:.6f}  "
            f"R²={r2:+.4f}  DirAcc={diracc:.4f}"
        )

    df = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Classification evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_classification(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot_dir: str = None,
) -> pd.DataFrame:
    """
    Evaluate classification models on the hold-out test set.

    Returns
    -------
    pd.DataFrame sorted by F1-weighted (descending = better).
    """
    rows = []
    for name, mdl in models.items():
        yp  = mdl.predict(X_test)
        acc = accuracy_score(y_test, yp)
        f1  = f1_score(y_test, yp, average="weighted", zero_division=0)

        try:
            proba = mdl.predict_proba(X_test)
            nc = proba.shape[1]
            auc = (
                roc_auc_score(y_test, proba[:, 1])
                if nc == 2
                else roc_auc_score(
                    y_test, proba,
                    multi_class="ovr", average="weighted"
                )
            )
        except Exception:
            auc = float("nan")

        rows.append({
            "Model":       name,
            "Accuracy":    round(acc, 4),
            "F1_weighted": round(f1,  4),
            "ROC_AUC":     round(auc, 4),
        })
        log.info(
            f"  [CLF] {name:<40}  "
            f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}"
        )
        log.info(
            "\n" + classification_report(y_test, yp, zero_division=0)
        )

        if plot_dir:
            _save_confusion_matrix(y_test, yp, name, plot_dir)

    df = pd.DataFrame(rows).set_index("Model").sort_values(
        "F1_weighted", ascending=False
    )
    return df


def _save_confusion_matrix(y_true, y_pred, name: str, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {name}", fontsize=9)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"cm_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Overfitting / underfitting detection
# ─────────────────────────────────────────────────────────────────────────────

def check_overfit(
    models: Dict[str, Any],
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_te: pd.DataFrame, y_te: pd.Series,
    task: str = "reg",
) -> pd.DataFrame:
    """
    Compare train vs test performance to flag overfitting/underfitting.

    Rules
    -----
    Regression  : overfit  if test_RMSE > train_RMSE × 1.5
                  underfit if train_RMSE itself is high (> 3 %)
    Classification : overfit  if train_Acc − test_Acc > 0.15
                     underfit if train_Acc < 0.55

    Returns
    -------
    pd.DataFrame with columns [Train_RMSE/Acc, Test_RMSE/Acc, Status].
    """
    rows = []
    for name, mdl in models.items():
        tr_pred = mdl.predict(X_tr)
        te_pred = mdl.predict(X_te)

        if task == "reg":
            tr_s = float(np.sqrt(mean_squared_error(y_tr, tr_pred)))
            te_s = float(np.sqrt(mean_squared_error(y_te, te_pred)))
            metric = "RMSE"
            overfit   = te_s > tr_s * 1.5
            underfit  = tr_s > 0.03
        else:
            tr_s = float(accuracy_score(y_tr, tr_pred))
            te_s = float(accuracy_score(y_te, te_pred))
            metric = "Acc"
            overfit  = (tr_s - te_s) > 0.15
            underfit = tr_s < 0.55

        if overfit:
            status = "⚠ Overfit"
        elif underfit:
            status = "⚠ Underfit"
        else:
            status = "✓ OK"

        rows.append({
            "Model": name,
            f"Train_{metric}": round(tr_s, 4),
            f"Test_{metric}":  round(te_s, 4),
            "Status": status,
        })
        log.info(
            f"  [OverfitCheck] {name:<40} "
            f"train={tr_s:.4f}  test={te_s:.4f}  {status}"
        )

    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────────────────────────────────────
# Best model selection
# ─────────────────────────────────────────────────────────────────────────────

def select_best_regressor(
    reg_results: pd.DataFrame,
    reg_models: Dict[str, Any],
    overfit_df: pd.DataFrame,
) -> Tuple[str, Any]:
    """
    Pick the best regression model by RMSE, skipping overfit models.
    Falls back to the best RMSE model if all are flagged.

    Returns (model_name, fitted_model).
    """
    for name in reg_results.index:   # sorted by RMSE ascending
        status = overfit_df.loc[name, "Status"] if name in overfit_df.index else ""
        if "Overfit" not in str(status):
            rmse = reg_results.loc[name, "RMSE"]
            log.info(f"  ✅ Best regressor selected: {name}  (RMSE={rmse:.6f})")
            return name, reg_models[name]

    # fallback
    best = reg_results.index[0]
    log.warning(
        f"  ⚠ All regressors flagged as overfit. "
        f"Using '{best}' as fallback – consider tuning hyperparameters."
    )
    return best, reg_models[best]


# ─────────────────────────────────────────────────────────────────────────────
# Prediction plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    models: Dict[str, Any],
    X_te: pd.DataFrame,
    y_te: pd.Series,
    plot_dir: str = None,
) -> None:
    """Plot actual vs predicted weekly returns for all regression models."""
    plot_dir = plot_dir or config.PLOTS_DIR
    os.makedirs(plot_dir, exist_ok=True)

    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, mdl) in zip(axes, models.items()):
        yp = pd.Series(mdl.predict(X_te), index=y_te.index)
        ax.plot(y_te.index,  y_te.values, label="Actual",    color="black",     lw=1.5)
        ax.plot(yp.index,    yp.values,   label="Predicted", color="steelblue", lw=1.0, ls="--")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{name} – Actual vs Predicted Returns (Test set)", fontsize=9)
        ax.set_ylabel("Weekly Return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(plot_dir, "regression_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Prediction plot saved: {path}")
