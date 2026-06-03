# Data Directory

Place your CSV files here before running the pipeline.

## Required files

| File | Description |
|---|---|
| `stock_data.csv` | Daily OHLCV of the target stock |
| `vnindex_data.csv` | Daily OHLCV of VN-Index (market benchmark) |

## Required columns

```
time, open, high, low, close, volume
```

- `time`  – date string in any standard format (`YYYY-MM-DD`, `DD/MM/YYYY`, etc.)
- `close` – **adjusted close price** (split- and dividend-adjusted)

## Notes

- Data should be **daily frequency**. The pipeline resamples it to weekly automatically.
- The longer your history, the more reliable the models. A minimum of **3 years** (≈ 156 weekly bars after feature warm-up) is recommended.
- Both files must cover an overlapping date range. The pipeline will automatically align them.
- Duplicate dates are dropped (keeping the last occurrence).
