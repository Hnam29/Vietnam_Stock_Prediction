"""
src/predictor.py
================
Step 10: Generate the final prediction for next week.

Output:
  - Best regression model → predicted return → predicted price
  - Logistic (3-class)    → trend direction  + class probabilities
  - Consensus check       → regression trend == Logistic trend?

"Confirmed" signal: both models agree on direction.
"Divergent" signal: models disagree – treat with caution.
"""
import logging
from typing import Any, Dict

import config

log = logging.getLogger(__name__)

# ─── Direction helpers ────────────────────────────────────────────────────────

_THR = config.SIDEWAYS_THRESHOLD

LABEL_VI  = {1: "TĂNG ▲",     0: "ĐI NGANG ↔",   -1: "GIẢM ▼"}
LABEL_DIR = {1: "UP",         0: "SIDEWAYS",       -1: "DOWN"}

REG_LABEL = {
    "UP":       "TĂNG ▲",
    "SIDEWAYS": "ĐI NGANG ↔",
    "DOWN":     "GIẢM ▼",
}


def _reg_direction(ret: float) -> str:
    if ret > _THR:  return "UP"
    if ret < -_THR: return "DOWN"
    return "SIDEWAYS"


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    best_reg_name:  str,
    best_reg_model: Any,
    clf3_models:    Dict[str, Any],
    X_latest:       Any,           # scaled single-row DataFrame
    current_price:  float,
) -> Dict[str, Any]:
    """
    Build the full prediction dictionary for next week.

    Regression path
    ---------------
    predicted_return = best_reg_model.predict(X_latest)
    predicted_price  = current_price × (1 + predicted_return)

    Classification path (Logistic, 3-class)
    ----------------------------------------
    Classes: -1 = giảm | 0 = đi ngang | 1 = tăng
    Probabilities are surfaced for transparency.

    Consensus
    ---------
    "Confirmed"  → regression direction == Logistic direction
    "Divergent"  → they disagree; add caution note

    Returns
    -------
    Dict with all prediction fields (ready for JSON serialisation).
    """
    # ── Regression ────────────────────────────────────────────────────────
    pred_ret   = float(best_reg_model.predict(X_latest)[0])
    pred_price = current_price * (1 + pred_ret)
    reg_dir    = _reg_direction(pred_ret)

    # ── Logistic (3-class) ────────────────────────────────────────────────
    logit_key = next(
        (k for k in clf3_models if "Logistic" in k), None
    )

    clf_out: Dict[str, Any] = {}
    logit_dir: str = None

    if logit_key:
        logit   = clf3_models[logit_key]
        classes = logit.classes_
        proba   = logit.predict_proba(X_latest)[0]
        pred_cls = int(logit.predict(X_latest)[0])

        # Sort probabilities descending for display
        proba_dict = dict(
            sorted(
                {LABEL_VI.get(int(c), str(c)): round(float(p) * 100, 2)
                 for c, p in zip(classes, proba)}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        logit_dir = LABEL_DIR.get(pred_cls, "SIDEWAYS")
        clf_out = {
            "model":               logit_key,
            "prediction_label":    LABEL_VI.get(pred_cls, "?"),
            "prediction_direction": logit_dir,
            "probabilities":       proba_dict,
        }
        log.info(
            f"  Logistic prediction: {clf_out['prediction_label']}  "
            + "  ".join(f"{k}={v:.1f}%" for k, v in proba_dict.items())
        )

    # ── Consensus ─────────────────────────────────────────────────────────
    if logit_dir is None:
        consensus_label = "⚠ Không có mô hình phân loại"
        confirmed = False
    elif reg_dir == logit_dir:
        consensus_label = "✅ XÁC NHẬN  –  Hồi quy & Logistic đồng thuận"
        confirmed = True
    else:
        consensus_label = (
            "⚠ PHÂN KỲ  –  Tín hiệu mâu thuẫn "
            f"(Reg={REG_LABEL[reg_dir]}, Logistic={clf_out.get('prediction_label','')})"
        )
        confirmed = False

    log.info(f"  Consensus: {consensus_label}")

    return {
        "best_reg_model":        best_reg_name,
        "current_price":         round(current_price, 2),
        "predicted_return_pct":  round(pred_ret * 100, 4),
        "predicted_price":       round(pred_price, 2),
        "regression_trend":      REG_LABEL[reg_dir],
        "regression_direction":  reg_dir,
        "classification":        clf_out,
        "consensus":             consensus_label,
        "is_confirmed":          confirmed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: Dict[str, Any]) -> None:
    """Print a formatted prediction report to stdout."""
    W    = 66
    line = "═" * W
    thin = "─" * W

    print(f"\n{line}")
    print(f"{'📊  KẾT QUẢ DỰ ĐOÁN TUẦN TỚI':^{W}}")
    print(line)
    print(f"  Mô hình hồi quy tốt nhất : {result['best_reg_model']}")
    print(thin)
    print(f"  Giá hiện tại             : {result['current_price']:>14,.2f}")
    print(f"  Tỷ suất sinh lời dự đoán : {result['predicted_return_pct']:>+13.4f} %")
    print(f"  Giá dự đoán tuần tới     : {result['predicted_price']:>14,.2f}")
    print(f"  Xu hướng (hồi quy)       : {result['regression_trend']}")
    print(thin)

    clf = result.get("classification", {})
    if clf:
        print(f"  Dự đoán Logistic (3 lớp) : {clf.get('prediction_label', 'N/A')}")
        print("  Xác suất từng xu hướng   :")
        for label, prob in clf.get("probabilities", {}).items():
            bar   = "█" * max(1, int(prob / 3.5))
            arrow = " ◀" if label == clf.get("prediction_label") else ""
            print(f"    {label:<20}  {prob:6.2f}%  {bar}{arrow}")
    else:
        print("  (Không có mô hình phân loại)")

    print(thin)
    print(f"  Đồng thuận mô hình       : {result['consensus']}")
    print(line)

    # Interpretation hint
    if result["is_confirmed"]:
        print("  💡 Tín hiệu được xác nhận bởi cả hai mô hình.")
    else:
        print("  💡 Tín hiệu phân kỳ – khuyến nghị chờ thêm xác nhận.")
    print()
