"""
config.py
=========
Central configuration for the Stock Prediction Pipeline.
Change parameters here only – never hard-code in src/*.py.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")

# ─── Input files ──────────────────────────────────────────────────────────────
# Place your CSV files here. Required columns: time, open, high, low, close, volume
STOCK_FILE  = os.path.join(DATA_DIR, "stock_data.csv")
MARKET_FILE = os.path.join(DATA_DIR, "vnindex_data.csv")
DATE_COLUMN = "time"

# ─── Feature engineering ──────────────────────────────────────────────────────
# All indicators are computed on the WEEKLY RETURNS series (not on raw price).

# MACD: EMA-fast, EMA-slow, Signal periods (in weeks)
MACD_FAST_PERIOD   = 12
MACD_SLOW_PERIOD   = 26
MACD_SIGNAL_PERIOD = 9

# RSI period (weeks)
RSI_PERIOD = 14

# Bollinger Bands: rolling window (weeks), number of std-dev
BB_PERIOD  = 20
BB_STD_DEV = 2

# Rolling volatility window (weeks)
ROLLING_VOL_WINDOW = 12

# Lagged return lags (list of integers)
LAG_PERIODS = [1, 2, 3]

# ─── Outlier handling ─────────────────────────────────────────────────────────
# Winsorization percentiles. Computed on Train only (no leakage).
WINSOR_LOWER_PCTILE = 0.01   # 1st  percentile
WINSOR_UPPER_PCTILE = 0.99   # 99th percentile

# ─── Train / test split ───────────────────────────────────────────────────────
# Chronological split – the OLDEST (train_ratio)% becomes Train.
TRAIN_RATIO = 0.80

# ─── Scaling ──────────────────────────────────────────────────────────────────
# "standard" = StandardScaler (z-score)  |  "minmax" = MinMaxScaler
SCALER_TYPE = "standard"

# ─── Feature selection thresholds ────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.80   # |Spearman r| > this → drop weaker of the pair
VIF_THRESHOLD         = 5.0    # Iteratively drop highest-VIF until all ≤ this
IMPORTANCE_THRESHOLD  = 0.01   # RF importance < 1% → drop

# ─── Target variable ──────────────────────────────────────────────────────────
# Weekly returns within (-threshold, +threshold) are labelled "sideways" (Y_clf3 = 0)
SIDEWAYS_THRESHOLD = 0.005   # 0.5 %

# ─── Model hyperparameters ────────────────────────────────────────────────────
LR_PARAMS = {"fit_intercept": True}

RF_REG_PARAMS = {
    "n_estimators": 200, "max_depth": 5,
    "min_samples_split": 10, "min_samples_leaf": 5,
    "random_state": 42, "n_jobs": -1,
}

RF_CLF_PARAMS = {
    "n_estimators": 200, "max_depth": 5,
    "min_samples_split": 10, "min_samples_leaf": 5,
    "random_state": 42, "n_jobs": -1,
    "class_weight": "balanced",
}

DT_REG_PARAMS = {
    "max_depth": 4, "min_samples_split": 10,
    "min_samples_leaf": 5, "random_state": 42,
}

DT_CLF_PARAMS = {
    "max_depth": 4, "min_samples_split": 10,
    "min_samples_leaf": 5, "random_state": 42,
    "class_weight": "balanced",
}

LOGIT_PARAMS = {
    "C": 1.0, "max_iter": 1000,
    "solver": "lbfgs", "random_state": 42,
    "class_weight": "balanced",
}

# ─── Cross-validation ─────────────────────────────────────────────────────────
# Number of folds for TimeSeriesSplit (used during CV scoring, not hyperparameter tuning)
TSCV_N_SPLITS = 5

# ─── Misc ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
