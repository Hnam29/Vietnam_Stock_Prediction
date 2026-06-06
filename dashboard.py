"""
dashboard.py
============
Streamlit dashboard cho Vietnam Stock Prediction Pipeline.

Chạy: streamlit run dashboard.py
"""

import sys
import os
import logging
import warnings
warnings.filterwarnings("ignore")

# ── Ensure project root is on path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="🇻🇳 Vietnam Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
/* Style texts and labels inside sidebar specifically */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {
    color: #f1f5f9 !important;
}

/* Ensure inputs, selects, date pickers are readable */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] div[role="listbox"],
[data-testid="stSidebar"] div[data-baseweb="select"] {
    color: #f1f5f9 !important;
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}

/* Fix dropdown menu listbox option styling */
div[role="listbox"] div[role="option"] {
    color: #f1f5f9 !important;
    background-color: #1e1b4b !important;
}
div[role="listbox"] div[role="option"]:hover,
div[role="listbox"] div[role="option"][aria-selected="true"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] input::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.main-header p {
    color: rgba(255,255,255,0.5);
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* Scorecard */
.scorecard {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(20px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.scorecard:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.scorecard .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 0.5rem;
}
.scorecard .value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.scorecard .sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
    margin-top: 0.4rem;
}

/* Trend colors */
.up    { color: #34d399; }
.down  { color: #f87171; }
.side  { color: #fbbf24; }
.blue  { color: #38bdf8; }
.purple { color: #a78bfa; }

/* Section title */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* Confirmed / Divergent badge */
.badge-confirmed {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-divergent {
    display: inline-block;
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Info box */
.info-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
}
.info-box .row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
}
.info-box .row:last-child { border-bottom: none; }
.info-box .row .key   { color: rgba(255,255,255,0.4); }
.info-box .row .val   { font-weight: 600; }

/* Plotly chart container */
.chart-container {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    overflow: hidden;
}

/* Streamlit native tweaks */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 4px 20px rgba(99,102,241,0.5);
    transform: translateY(-1px);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #38bdf8); }

/* Spinner */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* Alerts */
.stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _trend_color(direction: str) -> str:
    return {"UP": "#34d399", "DOWN": "#f87171", "SIDEWAYS": "#fbbf24"}.get(direction, "#e2e8f0")

def _trend_icon(direction: str) -> str:
    return {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "↔"}.get(direction, "—")

def _trend_vi(direction: str) -> str:
    return {"UP": "TĂNG", "DOWN": "GIẢM", "SIDEWAYS": "ĐI NGANG"}.get(direction, direction)

def _fmt_price(p: float) -> str:
    return f"{p:,.0f}"

def _fmt_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#f8fafc"),
    margin=dict(l=10, r=60, t=40, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)", tickfont=dict(color="#cbd5e1")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)", tickfont=dict(color="#cbd5e1")),
    legend=dict(
        font=dict(color="#f8fafc", size=11),
        bgcolor="rgba(15, 23, 42, 0.75)",
        bordercolor="rgba(255, 255, 255, 0.15)",
        borderwidth=1
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline_from_df(
    stock_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
) -> dict:
    """Chạy toàn bộ ML pipeline với DataFrames đã có sẵn (không cần đọc CSV)."""
    import config
    from src.data_loader import resample_weekly
    from src.preprocessor import (
        TARGET_COLS, build_feature_matrix,
        forward_fill, get_feature_cols,
        scale, time_split, winsorize,
    )
    from src.feature_selector import run_feature_selection
    from src.model_trainer import train_classifiers, train_regressors
    from src.evaluator import (
        check_overfit, eval_classification,
        eval_regression, select_best_regressor,
    )
    from src.predictor import predict

    # Step 0: Resample daily → weekly
    stock_w  = resample_weekly(stock_daily)
    market_w = resample_weekly(market_daily)

    # Step 1: Forward-fill
    stock_w  = forward_fill(stock_w)
    market_w = forward_fill(market_w)

    # Steps 2.x + 4: Feature engineering
    feature_df, X_pred_raw = build_feature_matrix(stock_w, market_w)
    feat_cols = get_feature_cols(feature_df)

    # Step 5: Split
    train_df, test_df = time_split(feature_df)

    X_tr_raw = train_df[feat_cols]
    X_te_raw = test_df[feat_cols]
    y_tr_reg = train_df["Y_reg"];  y_te_reg = test_df["Y_reg"]
    y_tr_c3  = train_df["Y_clf3"]; y_te_c3  = test_df["Y_clf3"]

    # Step 3: Winsorize
    X_tr_w, X_te_w, X_pred_w, _bounds = winsorize(X_tr_raw, X_te_raw, X_pred_raw, feat_cols)

    # Step 7: Feature selection
    selected = run_feature_selection(X_tr_w, y_tr_reg, plot_dir=None)
    X_tr_sel   = X_tr_w[selected]
    X_te_sel   = X_te_w[selected]
    X_pred_sel = X_pred_w[selected]

    # Step 6: Scale
    X_tr_s, X_te_s, X_pred_s, _scaler = scale(X_tr_sel, X_te_sel, X_pred_sel)

    # Step 8: Train models
    reg_models,  _ = train_regressors(X_tr_s, y_tr_reg)
    clf3_models, _ = train_classifiers(X_tr_s, y_tr_c3, suffix="_3cls")

    # Evaluate
    reg_results  = eval_regression(reg_models, X_te_s, y_te_reg)
    overfit_reg  = check_overfit(reg_models, X_tr_s, y_tr_reg, X_te_s, y_te_reg, task="reg")
    clf3_results = eval_classification(clf3_models, X_te_s, y_te_c3)

    # Best regressor
    best_reg_name, best_reg_model = select_best_regressor(reg_results, reg_models, overfit_reg)

    # Predict
    current_price = float(stock_w["close"].iloc[-1])
    result = predict(
        best_reg_name=best_reg_name,
        best_reg_model=best_reg_model,
        clf3_models=clf3_models,
        X_latest=X_pred_s,
        current_price=current_price,
    )

    return {
        "prediction":    result,
        "reg_results":   reg_results,
        "clf3_results":  clf3_results,
        "overfit_reg":   overfit_reg,
        "stock_weekly":  stock_w,
        "stock_daily":   stock_daily,
        "selected_features": selected,
        "best_reg_name": best_reg_name,
        "test_actual":   y_te_reg,
        "test_pred":     pd.Series(best_reg_model.predict(X_te_s), index=y_te_reg.index),
        "feat_cols":     feat_cols,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def chart_price_forecast(stock_weekly: pd.DataFrame, result: dict) -> go.Figure:
    """Line chart: Lịch sử giá weekly + forecast điểm tuần tới."""
    prices = stock_weekly["close"].copy()
    last_date  = prices.index[-1]
    last_price = float(prices.iloc[-1])
    pred_price = result["predicted_price"]
    direction  = result["regression_direction"]

    # Ước tính ngày tuần tới (next Friday)
    days_ahead = (4 - last_date.weekday()) % 7 + 7
    next_date  = last_date + pd.Timedelta(days=days_ahead if days_ahead > 0 else 7)

    color_fc   = _trend_color(direction)

    # RMSE từ best model (lấy từ reg_results nếu có, không thì 2%)
    # Đây là giá trị mang tính minh hoạ – confidence band ±RMSE*current_price
    est_err_pct = abs(result["predicted_return_pct"]) * 0.5 + 1.0
    upper = pred_price * (1 + est_err_pct / 100)
    lower = pred_price * (1 - est_err_pct / 100)

    # Chỉ hiển thị 52 tuần gần nhất cho rõ
    prices_disp = prices.iloc[-52:] if len(prices) > 52 else prices

    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=prices_disp.index, y=prices_disp.values,
        mode="lines", name="Giá đóng cửa (tuần)",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
        hovertemplate="%{x|%Y-%m-%d}<br>Giá: %{y:,.0f}<extra></extra>",
    ))

    # Connector line from last point to forecast
    fig.add_trace(go.Scatter(
        x=[last_date, next_date],
        y=[last_price, pred_price],
        mode="lines",
        name="Dự báo xu hướng",
        line=dict(color=color_fc, width=2.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Giá: %{y:,.0f}<extra></extra>",
    ))

    # Forecast point
    fig.add_trace(go.Scatter(
        x=[next_date], y=[pred_price],
        mode="markers+text",
        name=f"Dự báo: {_fmt_price(pred_price)}",
        marker=dict(symbol="diamond", size=14, color=color_fc,
                    line=dict(color="white", width=2)),
        text=[f"  {_fmt_price(pred_price)}"],
        textposition="middle right",
        textfont=dict(color=color_fc, size=13, family="Inter"),
        hovertemplate=f"Tuần tới<br>Giá dự báo: {_fmt_price(pred_price)}<extra></extra>",
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=[next_date, next_date],
        y=[lower, upper],
        mode="lines+markers",
        name="Biên sai số (±ước tính)",
        line=dict(color=color_fc, width=0),
        marker=dict(symbol="line-ew", size=12, color=color_fc, line=dict(color=color_fc, width=2)),
        showlegend=True,
        hovertemplate=f"Lower: {_fmt_price(lower)}<br>Upper: {_fmt_price(upper)}<extra></extra>",
    ))
    # Shade confidence
    fig.add_vrect(
        x0=last_date, x1=next_date,
        fillcolor="rgba(255,255,255,0.025)", layer="below", line_width=0,
    )

    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="📈 Lịch sử Giá Tuần & Dự Báo Tuần Tới", font=dict(size=15, color="#f8fafc")),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], tickformat=",.0f"),
        height=400,
        hovermode="x unified",
    )
    fig.update_layout(**layout)
    return fig


def chart_probability(clf_out: dict) -> go.Figure:
    """Horizontal bar chart xác suất 3 xu hướng từ Logistic."""
    if not clf_out:
        fig = go.Figure()
        fig.add_annotation(text="Không có mô hình phân loại", showarrow=False,
                           font=dict(color="#e2e8f0", size=14))
        fig.update_layout(**PLOT_LAYOUT, height=280)
        return fig

    proba = clf_out.get("probabilities", {})
    pred_label = clf_out.get("prediction_label", "")

    labels, values, colors = [], [], []
    color_map = {"TĂNG ▲": "#34d399", "ĐI NGANG ↔": "#fbbf24", "GIẢM ▼": "#f87171"}
    for label, val in proba.items():
        labels.append(label)
        values.append(val)
        colors.append(color_map.get(label, "#a78bfa"))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=colors,
            opacity=[1.0 if l == pred_label else 0.45 for l in labels],
            line=dict(color="rgba(255,255,255,0.15)", width=1),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=13, color="#e2e8f0"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="🎯 Xác Suất Xu Hướng (Logistic 3-class)", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(**PLOT_LAYOUT["xaxis"], range=[0, max(values) * 1.25],
                   ticksuffix="%", showgrid=True),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], showgrid=False),
        height=280,
        showlegend=False,
    )
    fig.update_layout(**layout)
    return fig


def chart_candlestick(stock_daily: pd.DataFrame, n_weeks: int = 16) -> go.Figure:
    """Candlestick chart N tuần gần nhất (daily candles) + volume."""
    cutoff = stock_daily.index[-1] - pd.Timedelta(weeks=n_weeks)
    df = stock_daily[stock_daily.index >= cutoff].copy()

    colors_vol = ["rgba(52,211,153,0.5)" if c >= o else "rgba(248,113,113,0.5)"
                  for c, o in zip(df["close"], df["open"])]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#34d399", decreasing_line_color="#f87171",
        increasing_fillcolor="rgba(52,211,153,0.7)",
        decreasing_fillcolor="rgba(248,113,113,0.7)",
        name="OHLC",
    ), row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume", marker_color=colors_vol,
        hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text=f"🕯 Candlestick {n_weeks} Tuần Gần Nhất (Daily)", font=dict(size=14, color="#f8fafc")),
        height=440,
        xaxis=dict(**PLOT_LAYOUT["xaxis"], rangeslider=dict(visible=False)),
        xaxis2=dict(**PLOT_LAYOUT["xaxis"]),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], tickformat=",.0f"),
        yaxis2=dict(**PLOT_LAYOUT["yaxis"], tickformat=".2s"),
        showlegend=False,
    )
    fig.update_layout(**layout)
    return fig


def chart_rsi_macd(stock_weekly: pd.DataFrame) -> go.Figure:
    """RSI + MACD technical indicators panel (weekly)."""
    import config

    close = stock_weekly["close"].copy()
    # --- RSI (mirrors preprocessor._rsi logic: computed on returns series) ---
    p   = config.RSI_PERIOD
    ret = close.pct_change()          # weekly return series
    d   = ret.diff()                  # same as preprocessor._rsi
    gain  = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss  = (-d.clip(upper=0)).ewm(com=p - 1, min_periods=p).mean()
    rs    = gain / (loss + 1e-10)
    rsi   = (100 - 100 / (1 + rs)).iloc[-60:]

    # --- MACD ---
    ret = close.pct_change()
    f, sl, sg = config.MACD_FAST_PERIOD, config.MACD_SLOW_PERIOD, config.MACD_SIGNAL_PERIOD
    ema_fast = ret.ewm(span=f, adjust=False).mean()
    ema_slow = ret.ewm(span=sl, adjust=False).mean()
    macd_line   = (ema_fast - ema_slow).iloc[-60:]
    macd_signal = macd_line.ewm(span=sg, adjust=False).mean()
    macd_hist   = macd_line - macd_signal

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.06,
        subplot_titles=["RSI (14 tuần)", "MACD (12/26/9)"],
    )

    # RSI
    rsi_color = ["#f87171" if v > 70 else "#34d399" if v < 30 else "#38bdf8" for v in rsi]
    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi.values,
        mode="lines", name="RSI",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
        hovertemplate="%{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.6)", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(52,211,153,0.6)", row=1, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.05)", layer="below",
                  row=1, col=1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(52,211,153,0.05)", layer="below",
                  row=1, col=1, line_width=0)

    # MACD Line + Signal
    fig.add_trace(go.Scatter(
        x=macd_line.index, y=macd_line.values,
        mode="lines", name="MACD",
        line=dict(color="#a78bfa", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>MACD: %{y:.5f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=macd_signal.index, y=macd_signal.values,
        mode="lines", name="Signal",
        line=dict(color="#f97316", width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{y:.5f}<extra></extra>",
    ), row=2, col=1)
    # MACD Histogram
    hist_colors = ["rgba(52,211,153,0.5)" if v >= 0 else "rgba(248,113,113,0.5)" for v in macd_hist]
    fig.add_trace(go.Bar(
        x=macd_hist.index, y=macd_hist.values,
        name="Histogram", marker_color=hist_colors,
        hovertemplate="%{x|%Y-%m-%d}<br>Hist: %{y:.5f}<extra></extra>",
    ), row=2, col=1)

    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="📊 Chỉ Số Kỹ Thuật – RSI & MACD (Weekly)", font=dict(size=14, color="#f8fafc")),
        height=440,
        hovermode="x unified",
        showlegend=True,
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#f8fafc", size=13)
    fig.update_layout(**layout)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=1, col=1, range=[0, 100])
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=2, col=1)
    return fig


def chart_model_comparison(test_actual: pd.Series, test_pred: pd.Series) -> go.Figure:
    """Actual vs Predicted return scatter + time series."""
    fig = make_subplots(
        rows=1, cols=2,
        column_titles=["Actual vs Predicted Return (Test)", "Return Timeline"],
        horizontal_spacing=0.08,
    )

    # Scatter
    colors_scatter = ["#34d399" if (a > 0) == (p > 0) else "#f87171"
                      for a, p in zip(test_actual, test_pred)]
    fig.add_trace(go.Scatter(
        x=test_actual.values, y=test_pred.values,
        mode="markers", name="Predictions",
        marker=dict(color=colors_scatter, size=7, opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
        hovertemplate="Actual: %{x:.4f}<br>Pred: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    # Perfect line
    rng = [min(test_actual.min(), test_pred.min()), max(test_actual.max(), test_pred.max())]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines", name="Perfect",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1),
        showlegend=False,
    ), row=1, col=1)

    # Timeline
    fig.add_trace(go.Scatter(
        x=test_actual.index, y=test_actual.values,
        mode="lines", name="Actual",
        line=dict(color="#38bdf8", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=test_pred.index, y=test_pred.values,
        mode="lines", name="Predicted",
        line=dict(color="#a78bfa", width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Pred: %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot",
                  row=1, col=2)

    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="🔬 Best Model – Actual vs Predicted (Test Set)", font=dict(size=14, color="#f8fafc")),
        height=360,
        hovermode="x unified",
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#f8fafc", size=13)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_scorecards(result: dict, ticker: str):
    direction = result["regression_direction"]
    color     = _trend_color(direction)
    ret_pct   = result["predicted_return_pct"]
    pred_price = result["predicted_price"]
    curr_price = result["current_price"]
    confirmed  = result["is_confirmed"]
    clf        = result.get("classification", {})
    prob_top   = max(clf.get("probabilities", {}).values(), default=None) if clf else None

    # Calculate Consensus Confidence Index (CCI)
    consensus_score = 100.0 if confirmed else 30.0
    if prob_top:
        # Scale margin over random baseline (33.33%) up to target threshold (50.0%)
        clf_score = min(100.0, max(0.0, (prob_top - 33.33) / 16.67 * 100.0))
    else:
        clf_score = 0.0
    
    cci = round(0.5 * consensus_score + 0.5 * clf_score, 1)
    
    if cci >= 70:
        cci_level = "CAO"
        cci_color = "#34d399"
    elif cci >= 50:
        cci_level = "VỪA"
        cci_color = "#fbbf24"
    else:
        cci_level = "THẤP"
        cci_color = "#f87171"

    badge_confirmed = (
        '<span class="badge-confirmed">✅ Đồng thuận</span>'
        if confirmed else
        '<span class="badge-divergent">⚠ Phân kỳ</span>'
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        icon = _trend_icon(direction)
        vi   = _trend_vi(direction)
        st.markdown(f"""
        <div class="scorecard">
            <div class="label">🎯 Xu Hướng Tuần Tới</div>
            <div class="value" style="color:{color}">{icon} {vi}</div>
            <div class="sub">{badge_confirmed}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        sign_color = "#34d399" if ret_pct >= 0 else "#f87171"
        st.markdown(f"""
        <div class="scorecard">
            <div class="label">📈 Tỷ Suất Dự Báo</div>
            <div class="value" style="color:{sign_color}">{_fmt_pct(ret_pct)}</div>
            <div class="sub">Tuần tới so với hiện tại</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="scorecard">
            <div class="label">💰 Giá Dự Báo</div>
            <div class="value" style="color:#38bdf8">{_fmt_price(pred_price)}</div>
            <div class="sub">Hiện tại: {_fmt_price(curr_price)}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="scorecard">
            <div class="label">🤖 Độ Tin Cậy</div>
            <div class="value" style="color:{cci_color}">{cci}% ({cci_level})</div>
            <div class="sub">Chỉ số đồng thuận & xác suất</div>
        </div>""", unsafe_allow_html=True)


def render_model_table(reg_results: pd.DataFrame, overfit_reg: pd.DataFrame, best_name: str):
    """Bảng so sánh performance các regression models."""
    merged = reg_results.copy()
    if "Status" in overfit_reg.columns:
        merged = merged.join(overfit_reg[["Status"]], how="left")

    # Highlight best
    def _style(row):
        base = [""] * len(row)
        if row.name == best_name:
            return ["background-color: rgba(99,102,241,0.2); font-weight:600;"] * len(row)
        return base

    merged_display = merged.copy()
    merged_display.index.name = "Model"

    # Format numbers
    for col in ["MAE", "RMSE"]:
        if col in merged_display.columns:
            merged_display[col] = merged_display[col].apply(lambda x: f"{x:.6f}")
    if "R2" in merged_display.columns:
        merged_display["R2"] = merged_display["R2"].apply(lambda x: f"{x:+.4f}")
    if "DirAcc" in merged_display.columns:
        merged_display["DirAcc"] = merged_display["DirAcc"].apply(lambda x: f"{x:.1%}")

    st.dataframe(
        merged_display.style.apply(_style, axis=1),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>Vietnam Stock Predictor</h1>
        <p>Dự báo xu hướng cổ phiếu sử dụng Machine Learning · Dữ liệu từ vnstock</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cấu Hình Phân Tích")
        st.markdown("---")

        ticker = st.text_input(
            "📌 Mã cổ phiếu",
            value="VCB",
            placeholder="VCB, HPG, FPT, BID...",
            help="Nhập mã cổ phiếu niêm yết trên HOSE/HNX/UPCOM",
        ).strip().upper()

        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input(
                "📅 Từ ngày",
                value=pd.Timestamp("2018-01-01"),
                min_value=pd.Timestamp("2010-01-01"),
            )
        with col_e:
            end_date = st.date_input(
                "📅 Đến ngày",
                value=pd.Timestamp.today(),
            )

        st.markdown("---")
        st.markdown("### 🔑 vnstock API")
        api_key = st.text_input(
            "API Key (Community)",
            value="",
            type="password",
            placeholder="vnstock_xxxxxx... (tuỳ chọn)",
            help="Lấy tại https://vnstock.site/ để tăng giới hạn request",
        )

        source = st.selectbox(
            "Nguồn dữ liệu",
            options=["KBS", "VCI", "TCBS"],
            index=0,
            help="KBS thường ổn định nhất với community API",
        )

        st.markdown("---")
        run_btn = st.button("▶ Phân Tích Ngay", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div style="color:rgba(255,255,255,0.3); font-size:0.75rem; line-height:1.6">
        <b>Pipeline:</b><br>
        0. Crawl → Resample<br>
        1. Forward-fill<br>
        2. Feature engineering<br>
        3. Winsorize (Train only)<br>
        4. Feature selection<br>
        5. Scale (Train only)<br>
        6. Train: LR · RF · DT<br>
        7. Evaluate & select best<br>
        8. Predict next week ✓
        </div>
        """, unsafe_allow_html=True)

    # ── Main content ──────────────────────────────────────────────────────────
    if not run_btn:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding: 4rem 0; color: rgba(255,255,255,0.3)">
            <div style="font-size: 5rem">📊</div>
            <p style="font-size: 1.2rem; margin-top: 1rem">
                Nhập mã cổ phiếu và nhấn <b style="color:rgba(255,255,255,0.5)">▶ Phân Tích Ngay</b> để bắt đầu
            </p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem">
                VD: VCB · HPG · FPT · BID · MBB · TCB · ACB · SSI
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Validation ────────────────────────────────────────────────────────────
    if not ticker:
        st.error("❌ Vui lòng nhập mã cổ phiếu.")
        return
    if start_date >= end_date:
        st.error("❌ Ngày bắt đầu phải trước ngày kết thúc.")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # ── Pipeline execution ────────────────────────────────────────────────────
    progress = st.progress(0, text="⏳ Đang khởi động...")

    try:
        # Step 1: Crawl
        progress.progress(10, text=f"🌐 Đang crawl dữ liệu {ticker} từ vnstock...")
        from src.crawler import fetch_stock_and_market

        stock_daily, market_daily = fetch_stock_and_market(
            ticker=ticker,
            start=start_str,
            end=end_str,
            source=source,
            api_key=api_key if api_key else None,
        )
        progress.progress(30, text="🔧 Đang chạy pipeline ML...")

        # Step 2: Run ML pipeline
        # Suppress all logging noise inside Streamlit
        try:
            logging.disable(logging.CRITICAL)
            pipeline_out = run_pipeline_from_df(stock_daily, market_daily)
        finally:
            logging.disable(logging.NOTSET)

        progress.progress(100, text="✅ Hoàn tất!")
        import time; time.sleep(0.4)
        progress.empty()

    except ValueError as e:
        progress.empty()
        st.error(f"❌ Lỗi dữ liệu: {e}")
        return
    except Exception as e:
        progress.empty()
        st.error(f"❌ Đã xảy ra lỗi: {e}")
        st.exception(e)
        return

    result      = pipeline_out["prediction"]
    stock_w     = pipeline_out["stock_weekly"]
    stock_d     = pipeline_out["stock_daily"]
    reg_results = pipeline_out["reg_results"]
    clf3_res    = pipeline_out["clf3_results"]
    overfit_reg = pipeline_out["overfit_reg"]
    best_name   = pipeline_out["best_reg_name"]
    selected    = pipeline_out["selected_features"]
    test_actual = pipeline_out["test_actual"]
    test_pred   = pipeline_out["test_pred"]
    clf_out     = result.get("classification", {})

    # ── Scorecards ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Kết Quả Dự Báo</div>', unsafe_allow_html=True)
    render_scorecards(result, ticker)

    # ── Row 2: Main charts ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Biểu Đồ Chính</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        fig_price = chart_price_forecast(stock_w, result)
        st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        fig_prob = chart_probability(clf_out)
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

        # Consensus info box
        direction = result["regression_direction"]
        color     = _trend_color(direction)
        st.markdown(f"""
        <div class="info-box">
            <div class="row">
                <span class="key">Mô hình Hồi quy</span>
                <span class="val" style="color:{color}">{result['regression_trend']}</span>
            </div>
            <div class="row">
                <span class="key">Mô hình Phân loại</span>
                <span class="val">{clf_out.get('prediction_label', 'N/A')}</span>
            </div>
            <div class="row">
                <span class="key">Tín hiệu</span>
                <span class="val">{'✅ Đồng thuận' if result['is_confirmed'] else '⚠ Phân kỳ'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 3: Creative charts ────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Phân Tích Kỹ Thuật</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig_candle = chart_candlestick(stock_d)
        st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        fig_rsi_macd = chart_rsi_macd(stock_w)
        st.plotly_chart(fig_rsi_macd, use_container_width=True, config={"displayModeBar": False})

    # ── Row 3b: Actual vs Predicted ───────────────────────────────────────────
    fig_comp = chart_model_comparison(test_actual, test_pred)
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    # ── Row 4: Model Performance ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 So Sánh Hiệu Suất Các Mô Hình</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Regression Models", "📋 3-Class Classifiers", "🔍 Overfit Check"])

    with tab1:
        render_model_table(reg_results, overfit_reg, best_name)

    with tab2:
        st.dataframe(clf3_res, use_container_width=True)

    with tab3:
        st.dataframe(overfit_reg, use_container_width=True)

    # ── Footer: Model Info ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">ℹ️ Thông Tin Mô Hình & Pipeline</div>', unsafe_allow_html=True)

    with st.expander("📦 Chi tiết cấu hình & features", expanded=False):
        import config
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏅 Mô hình tốt nhất (Regression)**")
            best_rmse = reg_results.loc[best_name, "RMSE"] if best_name in reg_results.index else "N/A"
            best_dacc = reg_results.loc[best_name, "DirAcc"] if best_name in reg_results.index else "N/A"
            st.markdown(f"""
            <div class="info-box">
                <div class="row"><span class="key">Tên model</span><span class="val">{best_name}</span></div>
                <div class="row"><span class="key">RMSE (test)</span><span class="val">{best_rmse}</span></div>
                <div class="row"><span class="key">DirAcc (test)</span><span class="val">{best_dacc}</span></div>
                <div class="row"><span class="key">Ngưỡng sideways</span><span class="val">±{config.SIDEWAYS_THRESHOLD*100:.1f}%</span></div>
                <div class="row"><span class="key">Train/Test split</span><span class="val">{int(config.TRAIN_RATIO*100)}/{int((1-config.TRAIN_RATIO)*100)}</span></div>
                <div class="row"><span class="key">Scaler</span><span class="val">{config.SCALER_TYPE.title()}</span></div>
                <div class="row"><span class="key">TSCV folds</span><span class="val">{config.TSCV_N_SPLITS}</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**⚙️ Hyperparameters RF Regressor**")
            st.json(config.RF_REG_PARAMS)

        with col2:
            st.markdown(f"**✅ Selected Features ({len(selected)} features)**")
            for i, f in enumerate(sorted(selected), 1):
                st.markdown(f"&nbsp;&nbsp;`{i:02d}.` `{f}`", unsafe_allow_html=True)

            st.markdown("**📐 Technical Indicator Config**")
            st.markdown(f"""
            <div class="info-box">
                <div class="row"><span class="key">MACD</span><span class="val">{config.MACD_FAST_PERIOD}/{config.MACD_SLOW_PERIOD}/{config.MACD_SIGNAL_PERIOD}</span></div>
                <div class="row"><span class="key">RSI period</span><span class="val">{config.RSI_PERIOD} tuần</span></div>
                <div class="row"><span class="key">Bollinger Band</span><span class="val">{config.BB_PERIOD}w ±{config.BB_STD_DEV}σ</span></div>
                <div class="row"><span class="key">Rolling Vol</span><span class="val">{config.ROLLING_VOL_WINDOW} tuần</span></div>
                <div class="row"><span class="key">Lag periods</span><span class="val">{config.LAG_PERIODS}</span></div>
                <div class="row"><span class="key">Winsorize</span><span class="val">{int(config.WINSOR_LOWER_PCTILE*100)}–{int(config.WINSOR_UPPER_PCTILE*100)} percentile</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; padding: 2rem 0 1rem">
        ⚠️ Thông tin chỉ mang tính tham khảo học thuật, không phải khuyến nghị đầu tư.<br>
        Dữ liệu từ <b>vnstock</b> · Mô hình: Linear Regression, Random Forest, Decision Tree
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
