"""
src/feature_selector.py
========================
Steps 7.1 – 7.3: Feature selection.

CRITICAL: ALL steps are fitted on X_train ONLY to prevent data leakage.

7.1  Spearman correlation filter:
       For each pair with |r| > threshold, keep whichever has higher
       correlation with Y_reg.  Drop the weaker one.

7.2  Iterative VIF filter (Linear / Logistic Regression guard):
       Drop the highest-VIF feature until all remaining VIF ≤ threshold.
       Stops early to guarantee ≥ 3 features remain.

7.3  Random Forest feature importance filter:
       Drop features whose RF importance < threshold (noise / irrelevant).
"""
import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / IDE environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config

log = logging.getLogger(__name__)

# Optional dependency – graceful fallback if not installed
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif_fn
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ─────────────────────────────────────────────────────────────────────────────
# Step 7.1 – Spearman Correlation Filter
# ─────────────────────────────────────────────────────────────────────────────

def correlation_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> List[str]:
    """
    Remove redundant features using pairwise Spearman correlation.

    For every pair (A, B) with |corr| > CORRELATION_THRESHOLD:
      - Compute each feature's Spearman |r| with the regression target.
      - Drop the one with the LOWER target correlation.

    Parameters
    ----------
    X_train : feature matrix (training data only)
    y_train : regression target (Y_reg)

    Returns
    -------
    List of feature names to KEEP.
    """
    thr = config.CORRELATION_THRESHOLD
    log.info(f"[7.1] Correlation filter  (Spearman |r| > {thr})")

    ff_corr = X_train.corr(method="spearman").abs()
    tgt_corr = X_train.corrwith(y_train, method="spearman").abs()

    # Upper triangle only (avoid double-counting pairs)
    upper = ff_corr.where(
        np.triu(np.ones(ff_corr.shape, dtype=bool), k=1)
    )

    to_drop: set = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > thr].tolist()
        for partner in partners:
            if col in to_drop or partner in to_drop:
                continue
            # Drop the one with lower correlation to the target
            loser = (
                col
                if tgt_corr.get(col, 0) < tgt_corr.get(partner, 0)
                else partner
            )
            winner = partner if loser == col else col
            to_drop.add(loser)
            log.debug(
                f"  Pair ({col}, {partner})  "
                f"ff_corr={ff_corr.loc[col, partner]:.3f}  "
                f"→ keep '{winner}', drop '{loser}'"
            )

    kept = [c for c in X_train.columns if c not in to_drop]
    log.info(
        f"  Dropped {len(to_drop)}: {sorted(to_drop)}\n"
        f"  Remaining: {len(kept)} features"
    )
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Step 7.2 – VIF Filter
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vif(X: pd.DataFrame) -> pd.Series:
    """Return a Series of VIF values keyed by column name."""
    arr = X.values.astype(float)
    vif_vals = {}
    for i, col in enumerate(X.columns):
        try:
            v = _vif_fn(arr, i)
        except Exception:
            v = float("inf")
        vif_vals[col] = v
    return pd.Series(vif_vals)


def vif_filter(X_train: pd.DataFrame) -> List[str]:
    """
    Iteratively drop the highest-VIF feature until all remaining
    features have VIF ≤ VIF_THRESHOLD.

    Always keeps at least 3 features (safety guard).

    Returns
    -------
    List of feature names to KEEP.
    """
    if not _HAS_STATSMODELS:
        log.warning(
            "[7.2] statsmodels not installed → VIF filter skipped.\n"
            "      Install with: pip install statsmodels"
        )
        return list(X_train.columns)

    thr = config.VIF_THRESHOLD
    log.info(f"[7.2] VIF filter  (threshold = {thr})")
    features = list(X_train.columns)

    while len(features) > 3:
        vif = _compute_vif(X_train[features])
        worst_feat = vif.idxmax()
        worst_val  = vif[worst_feat]
        if worst_val <= thr:
            break
        log.debug(f"  Drop '{worst_feat}' (VIF = {worst_val:.2f})")
        features.remove(worst_feat)

    # Log final VIF table
    final_vif = _compute_vif(X_train[features]).sort_values(ascending=False)
    log.info("  Final VIF:")
    for feat, val in final_vif.items():
        flag = "  ✓" if val <= thr else "  ⚠"
        log.info(f"    {feat:<25} VIF = {val:6.2f}{flag}")
    log.info(f"  Remaining: {len(features)} features")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Step 7.3 – RF Feature Importance Filter
# ─────────────────────────────────────────────────────────────────────────────

def importance_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    plot_path: str = None,
) -> List[str]:
    """
    Train a base Random Forest on the current feature set and drop
    any feature whose importance < IMPORTANCE_THRESHOLD.

    Parameters
    ----------
    X_train    : feature matrix (training data only)
    y_train    : regression target (Y_reg)
    plot_path  : if provided, save a horizontal bar-chart to this path

    Returns
    -------
    List of feature names to KEEP.
    """
    thr = config.IMPORTANCE_THRESHOLD
    log.info(f"[7.3] RF Importance filter  (threshold = {thr:.0%})")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    imp = pd.Series(
        rf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)

    # ── Plot ──────────────────────────────────────────────────────────────
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        colors = ["#d62728" if v < thr else "#1f77b4" for v in imp]
        fig, ax = plt.subplots(figsize=(9, max(3, len(imp) * 0.40)))
        imp.plot.barh(ax=ax, color=colors, edgecolor="none")
        ax.axvline(thr, color="red", ls="--", lw=1.2,
                   label=f"Threshold {thr:.0%}")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances (Step 7.3)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log.info(f"  Plot saved: {plot_path}")

    to_drop = imp[imp < thr].index.tolist()
    kept    = imp[imp >= thr].sort_values(ascending=False).index.tolist()

    log.info(
        f"  Dropped {len(to_drop)} low-importance: {to_drop}\n"
        f"  Remaining: {len(kept)} features"
    )
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_selection(
    X_train: pd.DataFrame,
    y_train_reg: pd.Series,
    plot_dir: str = None,
) -> List[str]:
    """
    Full pipeline: 7.1 → 7.2 → 7.3.

    MUST be called with training data ONLY.
    Returns the final list of selected feature column names.
    """
    log.info("═══ FEATURE SELECTION (Train only) ════════════════════════════")

    # 7.1
    kept = correlation_filter(X_train, y_train_reg)
    # 7.2
    kept = vif_filter(X_train[kept])
    # 7.3
    plot_path = (
        os.path.join(plot_dir, "feature_importance.png")
        if plot_dir else None
    )
    kept = importance_filter(X_train[kept], y_train_reg, plot_path)

    log.info(
        f"═══ Final feature set ({len(kept)}): "
        f"{sorted(kept)} ════════════════"
    )
    return kept
