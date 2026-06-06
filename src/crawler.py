"""
src/crawler.py
==============
Crawl dữ liệu OHLCV từ vnstock cho 1 mã cổ phiếu bất kỳ + VNINDEX.

Sử dụng cùng API như notebook crawl_single_stock.ipynb.
Output DataFrame đồng dạng với CSV hiện tại (index='time', columns: open/high/low/close/volume).
"""
import logging
from typing import Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

VNINDEX_SYMBOL = "VNINDEX"
DEFAULT_SOURCE  = "KBS"
DEFAULT_INTERVAL = "1D"


# ─── API Key registration ─────────────────────────────────────────────────────

def register_api_key(api_key: str) -> str:
    """
    Đăng ký API key với vnstock.
    Returns: 'community' nếu thành công, 'guest' nếu thất bại.
    """
    if not api_key or not api_key.strip():
        log.info("Không có API key → chế độ Guest.")
        return "guest"
    try:
        from vnstock import register_user
        register_user(api_key.strip())
        log.info("API key đã đăng ký thành công → chế độ Community.")
        return "community"
    except Exception as e:
        log.warning(f"Đăng ký API key thất bại: {e} → chế độ Guest.")
        return "guest"


# ─── Core fetch function ──────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol:   str,
    start:    str,
    end:      str,
    interval: str = DEFAULT_INTERVAL,
    source:   str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV từ vnstock cho 1 mã cổ phiếu.

    Parameters
    ----------
    symbol   : Mã cổ phiếu (VD: 'VCB', 'HPG', 'VNINDEX')
    start    : Ngày bắt đầu 'YYYY-MM-DD'
    end      : Ngày kết thúc 'YYYY-MM-DD'
    interval : '1D' (mặc định – theo ngày)
    source   : Nguồn dữ liệu (mặc định 'KBS')

    Returns
    -------
    pd.DataFrame với DatetimeIndex ('time') và các cột: open, high, low, close, volume
    """
    from vnstock.api.quote import Quote

    log.info(f"Crawling {symbol} [{start} → {end}] from {source}")

    stock = Quote(symbol=symbol, source=source)
    df = stock.history(start=start, end=end, interval=interval)

    if df is None or df.empty:
        raise ValueError(
            f"Không có dữ liệu cho mã '{symbol}' "
            f"trong khoảng {start} – {end}.\n"
            "Kiểm tra lại mã cổ phiếu và khoảng thời gian."
        )

    # Chuẩn hoá tên cột time/date
    time_col = next(
        (c for c in df.columns if "time" in c.lower() or "date" in c.lower()),
        None,
    )
    if time_col and time_col != "time":
        df = df.rename(columns={time_col: "time"})
    elif time_col is None:
        # Nếu thời gian là index
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "time"})

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last")
    df = df.set_index("time")

    # Đảm bảo chỉ giữ các cột OHLCV
    ohlcv_cols = [c for c in df.columns if c.lower() in {"open", "high", "low", "close", "volume"}]
    df = df[ohlcv_cols]

    # Chuẩn hóa tên cột về chữ thường
    df.columns = [c.lower() for c in df.columns]

    log.info(
        f"  {symbol}: {len(df):,} rows | "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    return df


# ─── Public entry point ───────────────────────────────────────────────────────

def fetch_stock_and_market(
    ticker:   str,
    start:    str,
    end:      str,
    source:   str = DEFAULT_SOURCE,
    api_key:  Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crawl đồng thời dữ liệu cổ phiếu và VNINDEX.

    Returns
    -------
    stock_daily, market_daily : pd.DataFrame
        Daily OHLCV DataFrames với DatetimeIndex.
    """
    if api_key:
        register_api_key(api_key)

    ticker = ticker.strip().upper()

    stock_daily  = fetch_ohlcv(ticker,         start, end, source=source)
    market_daily = fetch_ohlcv(VNINDEX_SYMBOL, start, end, source=source)

    return stock_daily, market_daily
