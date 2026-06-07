# Stock Prediction Pipeline - [Link Website](https://vietnamstockprediction.streamlit.app/)

Dự báo giá cổ phiếu tuần tới bằng ML, áp dụng đúng nguyên tắc **no data leakage**.

---

## Cấu trúc project

```
stock_prediction/
├── config.py               ← Tất cả tham số cấu hình
├── main.py                 ← Entry point – chạy toàn bộ pipeline
├── requirements.txt
├── README.md
│
├── data/
│   ├── stock_data.csv      ← Dữ liệu daily OHLCV cổ phiếu (bạn cung cấp)
│   ├── vnindex_data.csv    ← Dữ liệu daily OHLCV VN-Index (bạn cung cấp)
│   └── README.md
│
├── src/
│   ├── data_loader.py      ← Bước 0: Load + resample daily → weekly
│   ├── preprocessor.py     ← Bước 1, 2.x, 3, 4, 5, 6
│   ├── feature_selector.py ← Bước 7.1 (Corr), 7.2 (VIF), 7.3 (RF Importance)
│   ├── model_trainer.py    ← Bước 8: Train + TimeSeriesSplit CV
│   ├── evaluator.py        ← Bước 9: Metrics + overfit check + plots
│   └── predictor.py        ← Bước 10: Dự đoán + consensus check
│
└── outputs/
    ├── models/             ← Models đã lưu (.pkl)
    ├── reports/            ← CSV results + JSON prediction + log file
    └── plots/              ← Feature importance chart + prediction plots + confusion matrices
```

---

## Quy trình pipeline (đúng thứ tự, không data leakage)

```
Bước 0   Load daily CSV + Resample → weekly (W-FRI)
Bước 1   Forward-fill giá thiếu
Bước 2.x Feature engineering trên RETURNS (không phải giá):
           MACD · RSI · Bollinger Bands · Rolling Vol · Lags
           Y_reg (return T+1) · Y_clf (binary) · Y_clf3 (3-class)
Bước 4   Drop NaN warm-up rows  |  Lưu X_predict (hàng cuối)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bước 5   Chronological Split 80/20 (Train / Test)
         ━━━━━━━━━━ Ranh giới chống leakage ━━
Bước 3   Winsorize  — fit trên Train ONLY
Bước 6   Scale      — fit trên Train ONLY
Bước 7   Feature Selection — chỉ dùng Train:
           7.1 Spearman Correlation (|r| > 0.8 → loại)
           7.2 VIF (> 5 → loại dần)
           7.3 RF Importance (< 1% → loại)
Bước 8   Train Models với TimeSeriesSplit CV:
           Regression:      LinearRegression | RF | DecisionTree
           Classification:  Logistic | RF | DT  (3 lớp + binary)
Bước 9   Evaluate + Overfit check + Chọn best regressor
Bước 10  Predict tuần tới:
           → Giá dự đoán (Regression)
           → Xu hướng + Xác suất (Logistic 3-class)
           → Consensus check
```

---

## Cài đặt & Chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Đặt file CSV vào thư mục data/
#    data/stock_data.csv
#    data/vnindex_data.csv

# 3. (Tùy chọn) Chỉnh tham số trong config.py

# 4. Chạy pipeline
python main.py
```

---

## Output mẫu

```
══════════════════════════════════════════════════════════════════════
 📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (CROSS-VALIDATION)
══════════════════════════════════════════════════════════════════════
 [1] MÔ HÌNH REGRESSION:
 ────────────────────────────────────────────────────────────────────
 • Linear : MAE = 0.0559 | RMSE = 0.0745 | R2 = -1.0137 | Dir.Acc = 57.1%
 • RF_Reg : MAE = 0.0449 | RMSE = 0.0583 | R2 = -0.1838 | Dir.Acc = 54.4%
 • DT_Reg : MAE = 0.0518 | RMSE = 0.0678 | R2 = -0.6168 | Dir.Acc = 58.3%

 [2] MÔ HÌNH CLASSIFICATION (BINARY):
 ────────────────────────────────────────────────────────────────────
 • Logistic : Accuracy = 57.1% | F1-Score = 0.5317 | ROC-AUC = 0.5100
 • RF_Clf   : Accuracy = 53.2% | F1-Score = 0.5267 | ROC-AUC = 0.5220
 • DT_Clf   : Accuracy = 51.2% | F1-Score = 0.4779 | ROC-AUC = 0.4983

══════════════════════════════════════════════════════════════════════
 📈 KẾT QUẢ DỰ BÁO TUẦN TỚI (CONSENSUS CHECK)
══════════════════════════════════════════════════════════════════════
 [Regression] Dự báo Return : +0.58% (TĂNG ▲)  ->  Giá: 42,243
 [Classifier] Dự báo Xu hướng: TĂNG ▲ (Xác suất: 50.9%)

 ✅ ĐỒNG THUẬN: Giá cổ phiếu trong 1 tuần tới sẽ TĂNG
══════════════════════════════════════════════════════════════════════
```

---

## Các tham số quan trọng trong `config.py`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `TRAIN_RATIO` | `0.80` | 80% dữ liệu cũ nhất làm Train |
| `SIDEWAYS_THRESHOLD` | `0.005` | Return trong ±0.5% = đi ngang |
| `CORRELATION_THRESHOLD` | `0.80` | Ngưỡng Spearman loại feature trùng |
| `VIF_THRESHOLD` | `5.0` | Ngưỡng VIF loại đa cộng tuyến |
| `IMPORTANCE_THRESHOLD` | `0.01` | RF importance < 1% → loại |
| `WINSOR_LOWER_PCTILE` | `0.01` | Winsorize tại percentile 1% |
| `WINSOR_UPPER_PCTILE` | `0.99` | Winsorize tại percentile 99% |
| `SCALER_TYPE` | `"standard"` | `"standard"` hoặc `"minmax"` |

---

## Lưu ý quan trọng

- **Adjusted close**: cột `close` là giá đã điều chỉnh (split + dividend).
- **Dữ liệu tối thiểu**: nên có ≥ 3 năm để có đủ hàng sau khi warm-up (MACD cần 26 tuần, BB cần 20 tuần).
- **Không dự đoán mù**: tín hiệu "PHÂN KỲ" (⚠) có nghĩa hai mô hình không đồng ý — hãy chờ thêm tín hiệu xác nhận.
- **Đây là công cụ hỗ trợ phân tích**, không phải lời khuyên đầu tư (disclaimer).
