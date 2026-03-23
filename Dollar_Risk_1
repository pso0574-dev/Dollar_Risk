# streamlit_app.py
# ============================================================
# FRED + yfinance Entry Timing Dashboard
# ------------------------------------------------------------
# Purpose:
#   - Use FRED as a macro filter
#   - Use yfinance as an execution trigger
#   - Summarize current market regime
#   - Help decide: Wait / Starter Buy / Add / Aggressive Buy
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests python-dateutil
#
# Environment:
#   Set FRED_API_KEY in environment variables
#   or create .streamlit/secrets.toml with:
#       FRED_API_KEY="YOUR_FRED_API_KEY"
# ============================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Macro Entry Timing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Constants
# -----------------------------
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Unemployment Rate": "UNRATE",
    "Initial Claims": "ICSA",
    "Industrial Production": "INDPRO",
    "M2": "M2SL",
    "10Y Yield": "DGS10",
    "2Y Yield": "DGS2",
    "3M Yield": "TB3MS",
    "Fed Funds": "FEDFUNDS",
}

YF_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "HYG": "HYG",
    "TLT": "TLT",
    "IEF": "IEF",
    "GLD": "GLD",
    "XLE": "XLE",
    "UUP": "UUP",       # Dollar ETF proxy
    "WTI": "CL=F",      # Crude Oil futures
    "Brent": "BZ=F",
    "BTC": "BTC-USD",
    "SOXX": "SOXX",
}

DEFAULT_LOOKBACK_YEARS = 10

# -----------------------------
# Helpers
# -----------------------------
def get_fred_api_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        pass
    if not key:
        key = os.getenv("FRED_API_KEY")
    return key


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fred_series(
    series_id: str,
    api_key: str,
    start_date: str = "2000-01-01",
) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    r = requests.get(FRED_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)

    df = pd.DataFrame(obs)
    if df.empty:
        return pd.Series(dtype=float)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    s = df["value"].astype(float)
    s.name = series_id
    return s


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_close(
    tickers: List[str],
    period: str = "15y",
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if df.empty:
        return pd.DataFrame()

    # Case 1: multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        close_map = {}
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t]
                if "Close" in sub.columns:
                    close_map[t] = sub["Close"]
                elif len(sub.columns) > 0:
                    close_map[t] = sub.iloc[:, 0]
        out = pd.DataFrame(close_map)
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    # Case 2: single ticker flat columns
    if "Close" in df.columns:
        t = tickers[0] if len(tickers) == 1 else "Close"
        out = pd.DataFrame({t: df["Close"]})
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    return pd.DataFrame()


def resample_monthly_last(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    return s.resample("M").last().dropna()


def yoy_change(s: pd.Series) -> pd.Series:
    return s.pct_change(12) * 100.0


def mom_change(s: pd.Series) -> pd.Series:
    return s.pct_change(1) * 100.0


def safe_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def safe_prev(s: pd.Series, periods: int = 1) -> float:
    s = s.dropna()
    return float(s.iloc[-1 - periods]) if len(s) > periods else np.nan


def pct_from_n_days(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return (s.iloc[-1] / s.iloc[-1 - n] - 1.0) * 100.0


def pct_from_n_periods(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return (s.iloc[-1] / s.iloc[-1 - n] - 1.0) * 100.0


def rolling_max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    if p.empty:
        return np.nan
    running_max = p.cummax()
    dd = p / running_max - 1.0
    return dd.iloc[-1] * 100.0


def compute_dma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def higher_low_signal(prices: pd.Series, lookback: int = 120) -> bool:
    p = prices.dropna()
    if len(p) < lookback:
        return False
    recent = p.iloc[-lookback:]
    half = lookback // 2
    low1 = recent.iloc[:half].min()
    low2 = recent.iloc[half:].min()
    return low2 > low1


def slope_positive(s: pd.Series, window: int = 20) -> bool:
    x = s.dropna()
    if len(x) < window:
        return False
    y = x.iloc[-window:].values
    coeff = np.polyfit(np.arange(len(y)), y, 1)[0]
    return coeff > 0


def slope_negative(s: pd.Series, window: int = 20) -> bool:
    x = s.dropna()
    if len(x) < window:
        return False
    y = x.iloc[-window:].values
    coeff = np.polyfit(np.arange(len(y)), y, 1)[0]
    return coeff < 0


def fmt_num(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{nd}f}"


def fmt_pct(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{nd}f}%"


def make_line_chart(
    df: pd.DataFrame,
    title: str,
    ytitle: str = "",
    normalize: bool = False,
    height: int = 450,
) -> go.Figure:
    plot_df = df.copy()
    if normalize and not plot_df.empty:
        plot_df = plot_df / plot_df.iloc[0] * 100.0

    fig = go.Figure()
    for c in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[c],
                mode="lines",
                name=c,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title=ytitle,
        hovermode="x unified",
        legend_title="Series",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_bar_score_chart(scores: Dict[str, int], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            text=list(scores.values()),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Score",
    )
    return fig


@dataclass
class SignalResult:
    label: str
    value: str
    status: str
    score: int
    comment: str


def status_emoji(status: str) -> str:
    status = status.lower()
    if status == "positive":
        return "🟢"
    if status == "neutral":
        return "🟡"
    if status == "negative":
        return "🔴"
    return "⚪"


# -----------------------------
# Sidebar
# -----------------------------
st.title("FRED + yfinance Entry Timing Dashboard")

with st.sidebar:
    st.header("Settings")

    lookback_label = st.selectbox(
        "Chart lookback",
        ["1y", "2y", "3y", "5y", "10y", "max"],
        index=4,
    )

    price_ma_fast = st.number_input("Fast MA", min_value=10, max_value=200, value=50, step=5)
    price_ma_slow = st.number_input("Slow MA", min_value=50, max_value=400, value=200, step=10)

    starter_allocation = st.slider("Starter buy %", 0, 100, 30, 5)
    add_allocation = st.slider("Add %", 0, 100, 40, 5)
    final_allocation = st.slider("Final %", 0, 100, 30, 5)

    target_equity = st.selectbox("Primary equity trigger", ["QQQ", "SPY"], index=0)
    oil_proxy = st.selectbox("Oil proxy", ["WTI", "Brent"], index=0)
    bond_proxy = st.selectbox("Bond proxy", ["TLT", "IEF"], index=0)

    st.markdown("---")
    st.caption("Tip: set FRED_API_KEY in environment variables or Streamlit secrets.")

fred_api_key = get_fred_api_key()

if not fred_api_key:
    st.error("FRED_API_KEY not found. Please set it in environment variables or .streamlit/secrets.toml")
    st.stop()

# -----------------------------
# Data loading
# -----------------------------
start_date = (pd.Timestamp.today() - relativedelta(years=DEFAULT_LOOKBACK_YEARS + 5)).strftime("%Y-%m-%d")

with st.spinner("Loading FRED data..."):
    fred_data = {}
    for name, sid in FRED_SERIES.items():
        try:
            fred_data[name] = fetch_fred_series(sid, fred_api_key, start_date=start_date)
        except Exception as e:
            fred_data[name] = pd.Series(dtype=float)
            st.warning(f"Failed to load FRED series {name} ({sid}): {e}")

yf_names = list(YF_TICKERS.keys())
yf_symbols = list(YF_TICKERS.values())

with st.spinner("Loading Yahoo Finance data..."):
    try:
        yf_close = fetch_yf_close(yf_symbols, period="15y", interval="1d")
    except Exception as e:
        st.error(f"Failed to load Yahoo Finance data: {e}")
        st.stop()

# Rename Yahoo columns from symbol to friendly names
rename_map = {v: k for k, v in YF_TICKERS.items()}
yf_close = yf_close.rename(columns=rename_map)

if yf_close.empty:
    st.error("No Yahoo Finance data loaded.")
    st.stop()

# -----------------------------
# Derived series
# -----------------------------
# FRED monthly transforms
cpi = resample_monthly_last(fred_data["CPI"])
core_cpi = resample_monthly_last(fred_data["Core CPI"])
unrate = resample_monthly_last(fred_data["Unemployment Rate"])
claims = fred_data["Initial Claims"].resample("W").last().dropna()
indpro = resample_monthly_last(fred_data["Industrial Production"])
m2 = resample_monthly_last(fred_data["M2"])
y10 = fred_data["10Y Yield"].dropna()
y2 = fred_data["2Y Yield"].dropna()
y3m = fred_data["3M Yield"].dropna()
fedfunds = resample_monthly_last(fred_data["Fed Funds"])

cpi_yoy = yoy_change(cpi)
core_cpi_yoy = yoy_change(core_cpi)
indpro_yoy = yoy_change(indpro)
m2_yoy = yoy_change(m2)

yc_10y_2y = (y10 - y2).dropna()
yc_10y_3m = (y10.resample("M").last() - y3m.resample("M").last()).dropna()

# Price series
spy = yf_close["SPY"].dropna() if "SPY" in yf_close.columns else pd.Series(dtype=float)
qqq = yf_close["QQQ"].dropna() if "QQQ" in yf_close.columns else pd.Series(dtype=float)
iwm = yf_close["IWM"].dropna() if "IWM" in yf_close.columns else pd.Series(dtype=float)
hyg = yf_close["HYG"].dropna() if "HYG" in yf_close.columns else pd.Series(dtype=float)
tlt = yf_close["TLT"].dropna() if "TLT" in yf_close.columns else pd.Series(dtype=float)
ief = yf_close["IEF"].dropna() if "IEF" in yf_close.columns else pd.Series(dtype=float)
gld = yf_close["GLD"].dropna() if "GLD" in yf_close.columns else pd.Series(dtype=float)
xle = yf_close["XLE"].dropna() if "XLE" in yf_close.columns else pd.Series(dtype=float)
uup = yf_close["UUP"].dropna() if "UUP" in yf_close.columns else pd.Series(dtype=float)
wti = yf_close["WTI"].dropna() if "WTI" in yf_close.columns else pd.Series(dtype=float)
brent = yf_close["Brent"].dropna() if "Brent" in yf_close.columns else pd.Series(dtype=float)
soxx = yf_close["SOXX"].dropna() if "SOXX" in yf_close.columns else pd.Series(dtype=float)

oil_series = wti if oil_proxy == "WTI" else brent
bond_series = tlt if bond_proxy == "TLT" else ief
equity_series = qqq if target_equity == "QQQ" else spy

# Moving averages
equity_ma_fast = compute_dma(equity_series, price_ma_fast)
equity_ma_slow = compute_dma(equity_series, price_ma_slow)

spy_ma_fast = compute_dma(spy, price_ma_fast)
spy_ma_slow = compute_dma(spy, price_ma_slow)
qqq_ma_fast = compute_dma(qqq, price_ma_fast)
qqq_ma_slow = compute_dma(qqq, price_ma_slow)

# -----------------------------
# Scores / Signals
# -----------------------------
macro_signals: List[SignalResult] = []
market_signals: List[SignalResult] = []

# Macro 1: Oil pressure cooling?
oil_3m = pct_from_n_days(oil_series, 63)
oil_1m = pct_from_n_days(oil_series, 21)
if pd.notna(oil_1m) and pd.notna(oil_3m):
    if oil_1m < 0 and oil_3m < 10:
        macro_signals.append(SignalResult("Oil Pressure", fmt_pct(oil_1m), "positive", 1, "Oil pressure cooling"))
    elif oil_1m < oil_3m:
        macro_signals.append(SignalResult("Oil Pressure", fmt_pct(oil_1m), "neutral", 0, "Oil still elevated but momentum cooling"))
    else:
        macro_signals.append(SignalResult("Oil Pressure", fmt_pct(oil_1m), "negative", 0, "Oil still pressuring inflation"))
else:
    macro_signals.append(SignalResult("Oil Pressure", "N/A", "neutral", 0, "Insufficient data"))

# Macro 2: 10Y yield falling?
y10_1m = pct_from_n_periods(y10, 21)
y10_last = safe_last(y10)
y10_prev_month_level = safe_prev(y10, 21)
if pd.notna(y10_last) and pd.notna(y10_prev_month_level):
    if y10_last < y10_prev_month_level:
        macro_signals.append(SignalResult("10Y Yield Trend", f"{fmt_num(y10_last)}%", "positive", 1, "Discount rate easing"))
    else:
        macro_signals.append(SignalResult("10Y Yield Trend", f"{fmt_num(y10_last)}%", "negative", 0, "Discount rate still elevated"))
else:
    macro_signals.append(SignalResult("10Y Yield Trend", "N/A", "neutral", 0, "Insufficient data"))

# Macro 3: Core CPI cooling?
core_last = safe_last(core_cpi_yoy)
core_prev = safe_prev(core_cpi_yoy, 1)
if pd.notna(core_last) and pd.notna(core_prev):
    if core_last < core_prev:
        macro_signals.append(SignalResult("Core CPI YoY", fmt_pct(core_last), "positive", 1, "Core inflation cooling"))
    elif abs(core_last - core_prev) < 0.15:
        macro_signals.append(SignalResult("Core CPI YoY", fmt_pct(core_last), "neutral", 0, "Core inflation sticky"))
    else:
        macro_signals.append(SignalResult("Core CPI YoY", fmt_pct(core_last), "negative", 0, "Core inflation re-accelerating"))
else:
    macro_signals.append(SignalResult("Core CPI YoY", "N/A", "neutral", 0, "Insufficient data"))

# Macro 4: Unemployment not spiking too hard?
un_last = safe_last(unrate)
un_3m_ago = safe_prev(unrate, 3)
if pd.notna(un_last) and pd.notna(un_3m_ago):
    delta = un_last - un_3m_ago
    if 0 <= delta <= 0.3:
        macro_signals.append(SignalResult("Unemployment", fmt_num(un_last), "positive", 1, "Soft deterioration, not severe"))
    elif delta < 0:
        macro_signals.append(SignalResult("Unemployment", fmt_num(un_last), "neutral", 0, "Labor market still firm"))
    else:
        macro_signals.append(SignalResult("Unemployment", fmt_num(un_last), "negative", 0, "Labor market weakening fast"))
else:
    macro_signals.append(SignalResult("Unemployment", "N/A", "neutral", 0, "Insufficient data"))

# Macro 5: Yield curve steepening?
yc_last = safe_last(yc_10y_2y)
yc_1m_ago = safe_prev(yc_10y_2y, 21)
if pd.notna(yc_last) and pd.notna(yc_1m_ago):
    if yc_last > yc_1m_ago:
        macro_signals.append(SignalResult("10Y-2Y Curve", fmt_num(yc_last), "positive", 1, "Curve steepening"))
    else:
        macro_signals.append(SignalResult("10Y-2Y Curve", fmt_num(yc_last), "neutral", 0, "No meaningful steepening"))
else:
    macro_signals.append(SignalResult("10Y-2Y Curve", "N/A", "neutral", 0, "Insufficient data"))

macro_score = sum(x.score for x in macro_signals)

# Market 1: Primary equity above fast MA
eq_last = safe_last(equity_series)
eq_ma_fast_last = safe_last(equity_ma_fast)
if pd.notna(eq_last) and pd.notna(eq_ma_fast_last):
    if eq_last > eq_ma_fast_last:
        market_signals.append(SignalResult(f"{target_equity} > {price_ma_fast}DMA", fmt_num(eq_last), "positive", 1, "Short-to-medium trend recovered"))
    else:
        market_signals.append(SignalResult(f"{target_equity} > {price_ma_fast}DMA", fmt_num(eq_last), "negative", 0, "Still below fast trend"))
else:
    market_signals.append(SignalResult(f"{target_equity} > {price_ma_fast}DMA", "N/A", "neutral", 0, "Insufficient data"))

# Market 2: Primary equity above slow MA
eq_ma_slow_last = safe_last(equity_ma_slow)
if pd.notna(eq_last) and pd.notna(eq_ma_slow_last):
    if eq_last > eq_ma_slow_last:
        market_signals.append(SignalResult(f"{target_equity} > {price_ma_slow}DMA", fmt_num(eq_last), "positive", 1, "Long trend confirmed"))
    else:
        market_signals.append(SignalResult(f"{target_equity} > {price_ma_slow}DMA", fmt_num(eq_last), "negative", 0, "Long trend not confirmed"))
else:
    market_signals.append(SignalResult(f"{target_equity} > {price_ma_slow}DMA", "N/A", "neutral", 0, "Insufficient data"))

# Market 3: Higher low structure
if higher_low_signal(equity_series, lookback=120):
    market_signals.append(SignalResult("Higher Low", "True", "positive", 1, "Base-building structure"))
else:
    market_signals.append(SignalResult("Higher Low", "False", "negative", 0, "No base structure yet"))

# Market 4: HYG improving?
if slope_positive(hyg, window=20):
    market_signals.append(SignalResult("HYG Trend", fmt_num(safe_last(hyg)), "positive", 1, "Credit market supportive"))
else:
    market_signals.append(SignalResult("HYG Trend", fmt_num(safe_last(hyg)), "negative", 0, "Credit market not supportive"))

# Market 5: Bond proxy improving?
if slope_positive(bond_series, window=20):
    market_signals.append(SignalResult(f"{bond_proxy} Trend", fmt_num(safe_last(bond_series)), "positive", 1, "Bond market aligns with easing"))
else:
    market_signals.append(SignalResult(f"{bond_proxy} Trend", fmt_num(safe_last(bond_series)), "negative", 0, "Bond market not confirming easing"))

market_score = sum(x.score for x in market_signals)
total_score = macro_score + market_score

# -----------------------------
# Regime / Entry mode
# -----------------------------
if total_score <= 3:
    risk_environment = "Risk-Off"
    trend_status = "Downtrend / Fragile"
    entry_mode = "WAIT"
    entry_comment = "Macro and/or market confirmation is insufficient."
elif 4 <= total_score <= 6:
    risk_environment = "Neutral"
    trend_status = "Base Building"
    entry_mode = "STARTER BUY"
    entry_comment = f"Consider initial allocation around {starter_allocation}%."
elif 7 <= total_score <= 8:
    risk_environment = "Improving"
    trend_status = "Trend Confirmation"
    entry_mode = "ADD"
    entry_comment = f"Add on confirmation. Suggested cumulative exposure: {starter_allocation + add_allocation}%."
else:
    risk_environment = "Risk-On"
    trend_status = "Uptrend"
    entry_mode = "AGGRESSIVE BUY"
    entry_comment = f"Trend and macro aligned. Full staged allocation possible: {starter_allocation + add_allocation + final_allocation}%."

# -----------------------------
# Lookback slicing
# -----------------------------
def slice_lookback(df_or_s: pd.DataFrame | pd.Series, label: str):
    if label == "max":
        return df_or_s
    years = int(label.replace("y", ""))
    cutoff = pd.Timestamp.today() - relativedelta(years=years)
    return df_or_s[df_or_s.index >= cutoff]

# -----------------------------
# Header metrics
# -----------------------------
m1, m2c, m3, m4, m5 = st.columns(5)
m1.metric("Macro Score", f"{macro_score}/5")
m2c.metric("Market Score", f"{market_score}/5")
m3.metric("Total Score", f"{total_score}/10")
m4.metric("Risk Environment", risk_environment)
m5.metric("Entry Mode", entry_mode)

st.caption(entry_comment)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Snapshot",
    "Macro Scorecard",
    "Signal Charts",
    "Trend Confirmation",
    "Regime & Entry Plan"
])

# ============================================================
# TAB 1: Market Snapshot
# ============================================================
with tab1:
    st.subheader("Market Snapshot")

    snapshot_assets = {
        "SPY": spy,
        "QQQ": qqq,
        "IWM": iwm,
        "HYG": hyg,
        "TLT": tlt,
        "IEF": ief,
        "GLD": gld,
        "XLE": xle,
        "UUP": uup,
        oil_proxy: oil_series,
        "SOXX": soxx,
    }

    rows = []
    for name, s in snapshot_assets.items():
        if s.empty:
            continue
        rows.append({
            "Asset": name,
            "Last": safe_last(s),
            "1W %": pct_from_n_days(s, 5),
            "1M %": pct_from_n_days(s, 21),
            "3M %": pct_from_n_days(s, 63),
            "6M %": pct_from_n_days(s, 126),
            "1Y %": pct_from_n_days(s, 252),
            "Drawdown %": rolling_max_drawdown(s),
        })

    snap_df = pd.DataFrame(rows)
    st.dataframe(
        snap_df.style.format({
            "Last": "{:,.2f}",
            "1W %": "{:,.2f}",
            "1M %": "{:,.2f}",
            "3M %": "{:,.2f}",
            "6M %": "{:,.2f}",
            "1Y %": "{:,.2f}",
            "Drawdown %": "{:,.2f}",
        }),
        use_container_width=True
    )

    chart_df = pd.DataFrame({
        "SPY": slice_lookback(spy, lookback_label),
        "QQQ": slice_lookback(qqq, lookback_label),
        "IWM": slice_lookback(iwm, lookback_label),
        "HYG": slice_lookback(hyg, lookback_label),
        bond_proxy: slice_lookback(bond_series, lookback_label),
        oil_proxy: slice_lookback(oil_series, lookback_label),
    }).dropna(how="all")

    fig = make_line_chart(
        chart_df,
        title=f"Normalized Multi-Asset Snapshot ({lookback_label})",
        ytitle="Indexed to 100",
        normalize=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: Macro Scorecard
# ============================================================
with tab2:
    st.subheader("Macro Scorecard")

    macro_table = pd.DataFrame([{
        "Signal": s.label,
        "Value": s.value,
        "Status": status_emoji(s.status),
        "Score": s.score,
        "Comment": s.comment
    } for s in macro_signals])

    market_table = pd.DataFrame([{
        "Signal": s.label,
        "Value": s.value,
        "Status": status_emoji(s.status),
        "Score": s.score,
        "Comment": s.comment
    } for s in market_signals])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Macro Signals")
        st.dataframe(macro_table, use_container_width=True, hide_index=True)
        macro_bar = make_bar_score_chart({s.label: s.score for s in macro_signals}, "Macro Signal Scores")
        st.plotly_chart(macro_bar, use_container_width=True)

    with c2:
        st.markdown("#### Market Signals")
        st.dataframe(market_table, use_container_width=True, hide_index=True)
        market_bar = make_bar_score_chart({s.label: s.score for s in market_signals}, "Market Signal Scores")
        st.plotly_chart(market_bar, use_container_width=True)

    st.markdown("#### Macro Data Snapshot")
    macro_snapshot = pd.DataFrame([
        {"Series": "CPI YoY", "Last": safe_last(cpi_yoy), "Prev": safe_prev(cpi_yoy, 1)},
        {"Series": "Core CPI YoY", "Last": safe_last(core_cpi_yoy), "Prev": safe_prev(core_cpi_yoy, 1)},
        {"Series": "Unemployment Rate", "Last": safe_last(unrate), "Prev": safe_prev(unrate, 1)},
        {"Series": "Initial Claims", "Last": safe_last(claims), "Prev": safe_prev(claims, 1)},
        {"Series": "Industrial Production YoY", "Last": safe_last(indpro_yoy), "Prev": safe_prev(indpro_yoy, 1)},
        {"Series": "M2 YoY", "Last": safe_last(m2_yoy), "Prev": safe_prev(m2_yoy, 1)},
        {"Series": "10Y Yield", "Last": safe_last(y10), "Prev": safe_prev(y10, 21)},
        {"Series": "2Y Yield", "Last": safe_last(y2), "Prev": safe_prev(y2, 21)},
        {"Series": "10Y-2Y", "Last": safe_last(yc_10y_2y), "Prev": safe_prev(yc_10y_2y, 21)},
    ])
    st.dataframe(
        macro_snapshot.style.format({"Last": "{:,.2f}", "Prev": "{:,.2f}"}),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TAB 3: Signal Charts
# ============================================================
with tab3:
    st.subheader("Signal Charts")

    c1, c2 = st.columns(2)

    with c1:
        infl_df = pd.concat([
            slice_lookback(cpi_yoy.rename("CPI YoY"), lookback_label),
            slice_lookback(core_cpi_yoy.rename("Core CPI YoY"), lookback_label),
        ], axis=1).dropna(how="all")
        fig = make_line_chart(infl_df, f"Inflation Trend ({lookback_label})", ytitle="%")
        st.plotly_chart(fig, use_container_width=True)

        yc_df = pd.concat([
            slice_lookback(yc_10y_2y.rename("10Y-2Y"), lookback_label),
            slice_lookback(yc_10y_3m.rename("10Y-3M"), lookback_label),
        ], axis=1).dropna(how="all")
        fig = make_line_chart(yc_df, f"Yield Curve ({lookback_label})", ytitle="%p")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rates_df = pd.concat([
            slice_lookback(y10.rename("10Y"), lookback_label),
            slice_lookback(y2.rename("2Y"), lookback_label),
            slice_lookback(fedfunds.rename("Fed Funds"), lookback_label),
        ], axis=1).dropna(how="all")
        fig = make_line_chart(rates_df, f"Rates ({lookback_label})", ytitle="%")
        st.plotly_chart(fig, use_container_width=True)

        growth_df = pd.concat([
            slice_lookback(unrate.rename("Unemployment"), lookback_label),
            slice_lookback(indpro_yoy.rename("Industrial Production YoY"), lookback_label),
            slice_lookback(m2_yoy.rename("M2 YoY"), lookback_label),
        ], axis=1).dropna(how="all")
        fig = make_line_chart(growth_df, f"Growth / Liquidity ({lookback_label})", ytitle="%")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: Trend Confirmation
# ============================================================
with tab4:
    st.subheader("Trend Confirmation")

    trend_price = slice_lookback(equity_series, lookback_label)
    trend_fast = slice_lookback(equity_ma_fast, lookback_label)
    trend_slow = slice_lookback(equity_ma_slow, lookback_label)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_price.index, y=trend_price, mode="lines", name=target_equity))
    fig.add_trace(go.Scatter(x=trend_fast.index, y=trend_fast, mode="lines", name=f"{price_ma_fast}DMA"))
    fig.add_trace(go.Scatter(x=trend_slow.index, y=trend_slow, mode="lines", name=f"{price_ma_slow}DMA"))
    fig.update_layout(
        title=f"{target_equity} Trend Confirmation ({lookback_label})",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{target_equity} Last", fmt_num(safe_last(equity_series)))
    c2.metric(f"{price_ma_fast}DMA", fmt_num(safe_last(equity_ma_fast)))
    c3.metric(f"{price_ma_slow}DMA", fmt_num(safe_last(equity_ma_slow)))
    c4.metric("Higher Low", "Yes" if higher_low_signal(equity_series) else "No")

    st.markdown("#### Risk-On Confirmation Basket")
    basket = pd.concat([
        slice_lookback(spy.rename("SPY"), lookback_label),
        slice_lookback(qqq.rename("QQQ"), lookback_label),
        slice_lookback(iwm.rename("IWM"), lookback_label),
        slice_lookback(hyg.rename("HYG"), lookback_label),
        slice_lookback(bond_series.rename(bond_proxy), lookback_label),
        slice_lookback(soxx.rename("SOXX"), lookback_label),
    ], axis=1).dropna(how="all")

    fig = make_line_chart(
        basket,
        title=f"Risk-On Basket ({lookback_label})",
        ytitle="Indexed to 100",
        normalize=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5: Regime & Entry Plan
# ============================================================
with tab5:
    st.subheader("Regime & Entry Plan")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Current Regime")
        regime_df = pd.DataFrame([
            {"Item": "Risk Environment", "Value": risk_environment},
            {"Item": "Trend Status", "Value": trend_status},
            {"Item": "Entry Mode", "Value": entry_mode},
            {"Item": "Macro Score", "Value": f"{macro_score}/5"},
            {"Item": "Market Score", "Value": f"{market_score}/5"},
            {"Item": "Total Score", "Value": f"{total_score}/10"},
        ])
        st.dataframe(regime_df, use_container_width=True, hide_index=True)

        st.markdown("#### Suggested Allocation Plan")
        alloc_df = pd.DataFrame([
            {"Step": "Starter Buy", "Allocation %": starter_allocation, "Condition": "Macro stabilizing, early confirmation"},
            {"Step": "Add", "Allocation %": add_allocation, "Condition": "Trend confirmed, price > fast MA, supportive credit/bonds"},
            {"Step": "Final Add", "Allocation %": final_allocation, "Condition": "Price > slow MA, pullback holds, regime improving"},
        ])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Decision Rules")
        rules = pd.DataFrame([
            {"Score Range": "0-3", "Mode": "WAIT", "Interpretation": "Risk-off. Preserve cash and observe."},
            {"Score Range": "4-6", "Mode": "STARTER BUY", "Interpretation": "Base building. Small initial entry only."},
            {"Score Range": "7-8", "Mode": "ADD", "Interpretation": "Trend confirmation improving. Add in stages."},
            {"Score Range": "9-10", "Mode": "AGGRESSIVE BUY", "Interpretation": "Macro + market aligned. Full staged plan allowed."},
        ])
        st.dataframe(rules, use_container_width=True, hide_index=True)

        st.markdown("#### Live Interpretation")
        st.info(entry_comment)

    st.markdown("#### What this dashboard is checking")
    st.write(
        """
        1. **Macro filter:** Is inflation pressure cooling and are rates becoming less restrictive?  
        2. **Stress check:** Is the market stopping its deterioration?  
        3. **Trend confirmation:** Has price actually recovered enough to justify staged buying?
        """
    )

    st.markdown("#### Raw Signal Summary")
    raw_all = pd.DataFrame([{
        "Category": "Macro",
        "Signal": s.label,
        "Value": s.value,
        "Status": s.status,
        "Score": s.score,
        "Comment": s.comment
    } for s in macro_signals] + [{
        "Category": "Market",
        "Signal": s.label,
        "Value": s.value,
        "Status": s.status,
        "Score": s.score,
        "Comment": s.comment
    } for s in market_signals])

    st.dataframe(raw_all, use_container_width=True, hide_index=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Framework: FRED for macro regime filtering + yfinance for execution timing. "
    "Use as a decision-support tool, not as financial advice."
)
