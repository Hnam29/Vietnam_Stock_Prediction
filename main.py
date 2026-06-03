#!/usr/bin/env python3
"""
main.py
=======
Stock Prediction Pipeline – entry point.

Corrected step order (no data leakage):
  0   Load + resample daily → weekly          [data_loader]
  1   Forward-fill missing prices             [preprocessor]
  2.x Feature engineering on returns          [preprocessor]
  4   Drop NaN warm-up rows                   [preprocessor]
      ↑ Save X_predict_raw before dropna
  5   Chronological train / test split        [preprocessor]
  3   Winsorize  (fit on Train only)          [preprocessor]
  6   Scale      (fit on Train only)          [preprocessor]
  7   Feature selection (Train only)          [feature_selector]
  8   Train all models with TSCV              [model_trainer]
      Evaluate + overfit check                [evaluator]
  9   Select best regressor                   [evaluator]
  10  Predict next week                       [predictor]

Usage
-----
  python main.py
"""

import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd

import config
from src.data_loader import load_and_prepare
from src.evaluator import (
    check_overfit,
    eval_classification,
    eval_regression,
    plot_predictions,
    select_best_regressor,
)
from src.feature_selector import run_feature_selection
from src.model_trainer import save_models, train_classifiers, train_regressors
from src.predictor import predict, print_report
from src.preprocessor import (
    TARGET_COLS,
    build_feature_matrix,
    forward_fill,
    get_feature_cols,
    scale,
    time_split,
    winsorize,
)


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def _setup_dirs() -> None:
    for d in [
        config.DATA_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
        config.PLOTS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def _setup_logging() -> logging.Logger:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.REPORTS_DIR, f"run_{ts}.log")
    fmt      = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger("main")


def _save_reports(
    ts: str,
    reg_results: pd.DataFrame,
    clf3_results: pd.DataFrame,
    overfit_reg: pd.DataFrame,
    overfit_clf: pd.DataFrame,
    prediction: dict,
) -> None:
    rd = config.REPORTS_DIR
    reg_results.to_csv(   os.path.join(rd, f"regression_results_{ts}.csv"))
    clf3_results.to_csv(  os.path.join(rd, f"clf3_results_{ts}.csv"))
    overfit_reg.to_csv(   os.path.join(rd, f"overfit_regression_{ts}.csv"))
    overfit_clf.to_csv(   os.path.join(rd, f"overfit_clf3_{ts}.csv"))
    pred_path = os.path.join(rd, f"prediction_{ts}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    _setup_dirs()
    log = _setup_logging()

    SEP = "=" * 70
    log.info(SEP)
    log.info("  STOCK PREDICTION PIPELINE  –  START")
    log.info(SEP)

    # ─── STEP 0: Load + resample ──────────────────────────────────────────
    log.info("── STEP 0: Load + resample daily → weekly ──────────────────────")
    stock_w, market_w = load_and_prepare()

    # ─── STEP 1: Forward-fill ─────────────────────────────────────────────
    log.info("── STEP 1: Forward-fill ────────────────────────────────────────")
    stock_w  = forward_fill(stock_w)
    market_w = forward_fill(market_w)

    # ─── STEPS 2.x + 4: Feature engineering + drop NaNs ──────────────────
    log.info("── STEPS 2.x + 4: Feature engineering ─────────────────────────")
    feature_df, X_pred_raw = build_feature_matrix(stock_w, market_w)
    feat_cols = get_feature_cols(feature_df)

    # ─── STEP 5: Time-series split ────────────────────────────────────────
    log.info("── STEP 5: Chronological split ─────────────────────────────────")
    train_df, test_df = time_split(feature_df)

    X_tr_raw  = train_df[feat_cols]
    X_te_raw  = test_df[feat_cols]

    y_tr_reg  = train_df["Y_reg"];   y_te_reg  = test_df["Y_reg"]
    y_tr_clf  = train_df["Y_clf"];   y_te_clf  = test_df["Y_clf"]
    y_tr_c3   = train_df["Y_clf3"];  y_te_c3   = test_df["Y_clf3"]

    # ─── STEP 3: Winsorize (AFTER split — fit on Train only) ──────────────
    log.info("── STEP 3: Winsorize (fit on Train only) ───────────────────────")
    X_tr_w, X_te_w, X_pred_w, _bounds = winsorize(
        X_tr_raw, X_te_raw, X_pred_raw, feat_cols
    )

    # ─── STEP 7: Feature selection (Train only!) ──────────────────────────
    log.info("── STEP 7: Feature selection ───────────────────────────────────")
    selected = run_feature_selection(
        X_tr_w, y_tr_reg, plot_dir=config.PLOTS_DIR
    )

    X_tr_sel   = X_tr_w[selected]
    X_te_sel   = X_te_w[selected]
    X_pred_sel = X_pred_w[selected]

    # ─── STEP 6: Scale (fit on Train only, after feature selection) ───────
    log.info("── STEP 6: Scale (fit on Train only) ───────────────────────────")
    X_tr_s, X_te_s, X_pred_s, _scaler = scale(
        X_tr_sel, X_te_sel, X_pred_sel
    )

    # ─── STEP 8a: Regression models ───────────────────────────────────────
    log.info("── STEP 8a: Train regression models ────────────────────────────")
    reg_models, _reg_cv = train_regressors(X_tr_s, y_tr_reg)

    # ─── STEP 8b: 3-class classifiers (tăng / đi ngang / giảm) ──────────
    log.info("── STEP 8b: Train 3-class classifiers ──────────────────────────")
    clf3_models, _clf3_cv = train_classifiers(X_tr_s, y_tr_c3, suffix="_3cls")

    # ─── STEP 8c: Binary classifiers (tăng / giảm) ───────────────────────
    log.info("── STEP 8c: Train binary classifiers ───────────────────────────")
    clf2_models, _clf2_cv = train_classifiers(X_tr_s, y_tr_clf, suffix="_bin")

    # Persist all models
    save_models({**reg_models, **clf3_models, **clf2_models})

    # ─── Evaluate regression ──────────────────────────────────────────────
    log.info("── EVALUATE: Regression ────────────────────────────────────────")
    reg_results  = eval_regression(reg_models, X_te_s, y_te_reg)
    overfit_reg  = check_overfit(
        reg_models, X_tr_s, y_tr_reg, X_te_s, y_te_reg, task="reg"
    )
    plot_predictions(reg_models, X_te_s, y_te_reg, config.PLOTS_DIR)

    # ─── Evaluate classification ──────────────────────────────────────────
    log.info("── EVALUATE: 3-class classifiers ───────────────────────────────")
    clf3_results = eval_classification(
        clf3_models, X_te_s, y_te_c3, config.PLOTS_DIR
    )
    overfit_clf3 = check_overfit(
        clf3_models, X_tr_s, y_tr_c3, X_te_s, y_te_c3, task="clf"
    )

    log.info("── EVALUATE: Binary classifiers ────────────────────────────────")
    clf2_results = eval_classification(
        clf2_models, X_te_s, y_te_clf, config.PLOTS_DIR
    )

    # ─── Select best regressor ────────────────────────────────────────────
    log.info("── Selecting best regression model ─────────────────────────────")
    best_reg_name, best_reg_model = select_best_regressor(
        reg_results, reg_models, overfit_reg
    )

    # ─── STEP 10: Predict next week ───────────────────────────────────────
    log.info("── STEP 10: Predict next week ───────────────────────────────────")
    current_price = float(stock_w["close"].iloc[-1])

    result = predict(
        best_reg_name  = best_reg_name,
        best_reg_model = best_reg_model,
        clf3_models    = clf3_models,
        X_latest       = X_pred_s,
        current_price  = current_price,
    )
    print_report(result)

    # ─── Save all reports ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_reports(
        ts, reg_results, clf3_results,
        overfit_reg, overfit_clf3, result,
    )
    log.info(f"  Reports saved to: {config.REPORTS_DIR}")

    log.info(SEP)
    log.info("  PIPELINE COMPLETE  ✓")
    log.info(SEP)
    return result


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
