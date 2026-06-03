"""
src/preprocessor.py
====================
Implements Steps 1, 2.x, 3, 4, 5, 6 of the pipeline.

IMPORTANT ordering (no data leakage):
  Step 1  – Forward-fill prices
  Step 2.x – Feature engineering (on returns, not raw prices)
  Step 4  – Drop NaN rows from warm-up
       ↑ Save X_predict_raw (latest row) BEFORE dropna
  Step 5  – Chronological train/test split
  Step 3  – Winsorize   ← fit on TRAIN only
  Step 6  – Scale       ← fit on TRAIN only
"""
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config

log = logging.getLogger(__name__)

TARGET_COLS = ["Y_reg", "Y_clf", "Y_clf3"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Forward-fill
# ─────────────────────────────────────────────────────────────────────────────

def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing weekly bars by carrying the previous week's value forward.
    No mean/median/backward fill – forward-fill only.
    """
    n_missing = df.isnull().sum().sum()
    df = df.ffill()
    if n_missing:
        log.info(f"  Forward-fill: {n_missing} missing values filled.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2.1 – Returns & Target Variables
# ─────────────────────────────────────────────────────────────────────────────

def _returns(close: pd.Series, name: str = "ret") -> pd.Series:
    r = close.pct_change()
    r.name = name
    return r


def _create_targets(stock_ret: pd.Series) -> pd.DataFrame:
    """
    Y_reg   = next-week return  (shift -1)           [regression target]
    Y_clf   = 1 if Y_reg > 0 else 0                  [binary classification]
    Y_clf3  = 1 (tăng) / 0 (đi ngang) / -1 (giảm)   [3-class classification]

    The last row will have NaN targets (future unknown) and is removed
    by dropna().  We save it separately as X_predict_raw before dropna.
    """
    thr = config.SIDEWAYS_THRESHOLD
    t = pd.DataFrame(index=stock_ret.index)
    t["Y_reg"]  = stock_ret.shift(-1)
    t["Y_clf"]  = (t["Y_reg"] > 0).astype(float)
    t["Y_clf3"] = np.select(
        [t["Y_reg"] > thr, t["Y_reg"] < -thr], [1.0, -1.0], default=0.0
    )
    # Propagate NaN from Y_reg to clf targets
    t.loc[t["Y_reg"].isna(), ["Y_clf", "Y_clf3"]] = np.nan
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Step 2.2 – Momentum Features  (all computed on RETURNS, not prices)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _macd(ret: pd.Series) -> pd.DataFrame:
    """MACD on returns series: line, signal, histogram."""
    f, sl, sg = config.MACD_FAST_PERIOD, config.MACD_SLOW_PERIOD, config.MACD_SIGNAL_PERIOD
    line   = _ema(ret, f) - _ema(ret, sl)
    signal = _ema(line, sg)
    return pd.DataFrame(
        {"MACD_line": line, "MACD_signal": signal, "MACD_hist": line - signal},
        index=ret.index,
    )


def _rsi(ret: pd.Series) -> pd.Series:
    """Wilder RSI on returns series."""
    p = config.RSI_PERIOD
    d = ret.diff()
    gain = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss = (-d.clip(upper=0)).ewm(com=p - 1, min_periods=p).mean()
    rs   = gain / (loss + 1e-10)
    rsi  = 100 - 100 / (1 + rs)
    rsi.name = "RSI"
    return rsi


def _bollinger(ret: pd.Series) -> pd.DataFrame:
    """Bollinger Bands: SMA±2σ on returns. Also exports BB_width and BB_pct."""
    p, nσ = config.BB_PERIOD, config.BB_STD_DEV
    mid   = ret.rolling(p).mean()
    std   = ret.rolling(p).std()
    upper = mid + nσ * std
    lower = mid - nσ * std
    width = upper - lower
    pct   = (ret - lower) / (width + 1e-10)   # 0 = at lower, 1 = at upper
    return pd.DataFrame(
        {"BB_upper": upper, "BB_lower": lower,
         "BB_mid": mid, "BB_width": width, "BB_pct": pct},
        index=ret.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2.3 – Risk & Autoregressive Features
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_vol(ret: pd.Series) -> pd.Series:
    vol = ret.rolling(config.ROLLING_VOL_WINDOW).std()
    vol.name = "Rolling_Vol"
    return vol


def _stock_lags(ret: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {f"Lag_{k}": ret.shift(k) for k in config.LAG_PERIODS},
        index=ret.index,
    )


def _market_lag(mkt_ret: pd.Series) -> pd.Series:
    s = mkt_ret.shift(1)
    s.name = "Market_Lag1"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def get_close(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in df.columns if c.lower() == "close"), None)
    if col is None:
        raise ValueError(f"No 'close' column in DataFrame. Columns: {df.columns.tolist()}")
    return df[col]


def build_feature_matrix(
    stock_w: pd.DataFrame,
    market_w: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble complete feature + target DataFrame, then:
      - Extract the latest row as X_predict_raw (BEFORE dropna)
      - Drop NaN rows (warm-up rows + last prediction row)

    Returns
    -------
    feature_df : pd.DataFrame
        Clean rows with valid features AND valid targets.
    X_predict_raw : pd.DataFrame
        Single-row DataFrame with feature values for the live prediction.
        All TARGET_COLS have been removed.
    """
    s_close = get_close(stock_w)
    m_close = get_close(market_w)

    # Align to shared weekly index
    idx = s_close.index.intersection(m_close.index)
    if len(idx) == 0:
        raise ValueError("Stock and market data have no overlapping weekly dates.")
    s_close = s_close.loc[idx]
    m_close = m_close.loc[idx]

    s_ret = _returns(s_close, "stock_ret")
    m_ret = _returns(m_close, "market_ret")

    full_df = pd.concat([
        s_ret.to_frame(),
        m_ret.to_frame(),
        _macd(s_ret),
        _rsi(s_ret).to_frame(),
        _bollinger(s_ret),
        _rolling_vol(s_ret).to_frame(),
        _stock_lags(s_ret),
        _market_lag(m_ret).to_frame(),
        _create_targets(s_ret),   # Y_reg, Y_clf, Y_clf3
    ], axis=1)

    # ── Save prediction row (has valid features, NaN targets) ──────────────
    X_predict_raw = full_df.iloc[[-1]].drop(columns=TARGET_COLS, errors="ignore").copy()
    if X_predict_raw.isnull().any().any():
        n_nan = X_predict_raw.isnull().sum().sum()
        log.warning(
            f"  X_predict_raw has {n_nan} NaN feature(s). "
            "Series may be too short for the longest rolling window."
        )

    # ── Step 4: Drop NaN rows ──────────────────────────────────────────────
    n_before = len(full_df)
    full_df = full_df.dropna()
    n_after = len(full_df)
    log.info(f"  dropna: removed {n_before - n_after} warm-up rows → {n_after} clean rows")

    if n_after < 60:
        log.warning(
            f"  Only {n_after} rows remaining after dropna. "
            "Consider using a longer history for reliable models."
        )

    return full_df, X_predict_raw


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return column names that are features (excludes TARGET_COLS)."""
    return [c for c in df.columns if c not in TARGET_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Time-Series Split
# ─────────────────────────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    ratio: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split.  NO shuffling. NO random_state.
    The oldest (ratio * 100)% of rows become Train.
    """
    ratio = ratio or config.TRAIN_RATIO
    n     = int(len(df) * ratio)
    train, test = df.iloc[:n].copy(), df.iloc[n:].copy()

    log.info(
        f"  Train: {train.index[0].date()} → {train.index[-1].date()} "
        f"({len(train)} rows)"
    )
    log.info(
        f"  Test : {test.index[0].date()} → {test.index[-1].date()} "
        f"({len(test)} rows)"
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Winsorization  (AFTER split, fit on Train only)
# ─────────────────────────────────────────────────────────────────────────────

def winsorize(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_predict: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Tuple]]:
    """
    Compute 1st/99th percentile bounds on X_train only.
    Apply the SAME bounds to clip X_test and X_predict.

    Returns
    -------
    X_tr_w, X_te_w, X_pr_w : winsorized DataFrames
    bounds : dict mapping column → (lower_bound, upper_bound)
    """
    lo_p = config.WINSOR_LOWER_PCTILE
    hi_p = config.WINSOR_UPPER_PCTILE

    bounds: Dict[str, Tuple] = {}
    Xtr = X_train.copy()
    Xte = X_test.copy()
    Xpr = X_predict.copy()

    for col in cols:
        if col not in X_train.columns:
            continue
        lo = float(X_train[col].quantile(lo_p))
        hi = float(X_train[col].quantile(hi_p))
        bounds[col] = (lo, hi)
        for frame in (Xtr, Xte, Xpr):
            if col in frame.columns:
                frame[col] = frame[col].clip(lo, hi)

    log.info(f"  Winsorized {len(bounds)} features (percentile bounds from Train only).")
    return Xtr, Xte, Xpr, bounds


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Scaling  (AFTER split, fit on Train only)
# ─────────────────────────────────────────────────────────────────────────────

def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_predict: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """
    Fit scaler on X_train.  Transform all three sets with the same params.

    Returns
    -------
    X_tr_s, X_te_s, X_pr_s : scaled DataFrames (same column names)
    scaler : fitted scaler object (for external use / serialisation)
    """
    stype = config.SCALER_TYPE
    scaler = StandardScaler() if stype == "standard" else MinMaxScaler()
    cols  = X_train.columns

    # Fill any residual NaN in X_predict before transform
    if X_predict.isnull().any().any():
        train_means = X_train.mean()
        X_predict = X_predict.fillna(train_means)

    Xtr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    Xte_s = pd.DataFrame(scaler.transform(X_test),      columns=cols, index=X_test.index)
    Xpr_s = pd.DataFrame(scaler.transform(X_predict),   columns=cols, index=X_predict.index)

    log.info(
        f"  Scaling ({stype}): fit on {len(X_train)} train rows, "
        f"transformed train + test + predict."
    )
    return Xtr_s, Xte_s, Xpr_s, scaler
