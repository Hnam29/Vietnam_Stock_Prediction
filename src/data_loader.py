"""
src/data_loader.py
==================
Step 0 – Load raw daily OHLCV CSVs and resample to weekly frequency.

Resampling rules (week ending Friday):
  open   → first of week
  high   → max of week
  low    → min of week
  close  → last of week   (adjusted close)
  volume → sum of week
"""
import logging
from typing import Tuple

import pandas as pd

import config

log = logging.getLogger(__name__)


# ─── CSV loader ───────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load an OHLCV CSV file.
    - Parses the date column (config.DATE_COLUMN)
    - Sorts ascending
    - Sets date as the index
    """
    log.info(f"Loading: {filepath}")
    try:
        # utf-8-sig strips the UTF-8 BOM (﻿) that Excel/some crawlers add.
        # sep=None with python engine auto-detects comma vs semicolon.
        df = pd.read_csv(filepath, sep=None, engine="python", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\nData file not found: {filepath}\n"
            "Place your CSV files in the data/ folder.\n"
            "Required columns: time, open, high, low, close, volume"
        )

    # Strip residual BOM / leading whitespace from column names (defensive)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

    dc = config.DATE_COLUMN
    if dc not in df.columns:
        raise ValueError(
            f"Date column '{dc}' not found in {filepath}.\n"
            f"Columns present: {df.columns.tolist()}"
        )

    df[dc] = pd.to_datetime(df[dc])
    df = df.sort_values(dc).drop_duplicates(subset=dc, keep="last")
    df = df.set_index(dc)

    log.info(
        f"  {len(df):,} rows  |  "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    return df


# ─── Resampler ────────────────────────────────────────────────────────────────

def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a daily OHLCV DataFrame to weekly (week ending Friday).
    Drops weeks where ALL values are NaN (e.g. holiday-only weeks).
    """
    col_map = {c.lower(): c for c in df.columns}

    agg: dict = {}
    for canon, rule in [
        ("open", "first"), ("high", "max"),
        ("low", "min"),   ("close", "last"),
        ("volume", "sum"),
    ]:
        if canon in col_map:
            agg[col_map[canon]] = rule

    if not agg:
        raise ValueError("No standard OHLCV columns found.")

    weekly = df[list(agg.keys())].resample("W-FRI").agg(agg)
    weekly = weekly.dropna(how="all")
    log.info(f"  → {len(weekly)} weekly bars after resampling.")
    return weekly


# ─── Validation ───────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label}: DataFrame is empty.")

    close_candidates = [c for c in df.columns if c.lower() == "close"]
    if not close_candidates:
        raise ValueError(
            f"{label}: 'close' column not found. "
            f"Columns: {df.columns.tolist()}"
        )

    close_col = close_candidates[0]
    n_nonpos = (df[close_col] <= 0).sum()
    if n_nonpos:
        log.warning(f"{label}: {n_nonpos} non-positive close values detected.")

    n_dup = df.index.duplicated().sum()
    if n_dup:
        log.warning(f"{label}: {n_dup} duplicate dates were removed (kept last).")


# ─── Public entry point ───────────────────────────────────────────────────────

def load_and_prepare(
    stock_file: str = None,
    market_file: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full load pipeline:
      1. Read daily CSVs
      2. Validate
      3. Resample to weekly

    Returns
    -------
    stock_weekly, market_weekly : pd.DataFrame
        Weekly OHLCV DataFrames with DatetimeIndex.
    """
    stock_file  = stock_file  or config.STOCK_FILE
    market_file = market_file or config.MARKET_FILE

    log.info("── Loading stock data ──────────────────────────────────────────")
    stock_daily  = load_csv(stock_file)
    _validate(stock_daily, "Stock")
    stock_weekly = resample_weekly(stock_daily)

    log.info("── Loading market (VNINDEX) data ────────────────────────────────")
    market_daily  = load_csv(market_file)
    _validate(market_daily, "VNINDEX")
    market_weekly = resample_weekly(market_daily)

    return stock_weekly, market_weekly
