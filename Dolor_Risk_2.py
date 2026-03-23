# streamlit_app.py
# ============================================================
# QQQ / SPY Phase Buying Strategy Dashboard
# ------------------------------------------------------------
# Fixes in this version:
#   - Fixed empty explanation boxes
#   - Replaced fragile HTML wrapper + Streamlit markdown mixing
#   - "Why the system is cautious" and "What to watch next"
#     now always render visibly inside proper containers
#   - Kept safe dataframe rendering without style.format()
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests python-dateutil
#
# FRED API:
#   Set FRED_API_KEY in environment variables
#   or .streamlit/secrets.toml:
#       FRED_API_KEY="YOUR_KEY"
# ============================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="QQQ / SPY Phase Buying Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
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
    "UUP": "UUP",
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "SOXX": "SOXX",
}

DEFAULT_LOOKBACK_YEARS = 10

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.summary-box {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 16px;
    background: rgba(255,255,255,0.02);
    min-height: 110px;
    margin-bottom: 10px;
}

.summary-label {
    font-size: 0.95rem;
    opacity: 0.8;
    margin-bottom: 8px;
}

.summary-value {
    font-size: 1.4rem;
    font-weight: 800;
    line-height: 1.2;
}

.summary-sub {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-top: 8px;
}

.card-box {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 14px;
    background: rgba(255,255,255,0.02);
    margin-bottom: 10px;
}

.card-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.card-status {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.card-value {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.card-sub {
    font-size: 0.88rem;
    opacity: 0.84;
    margin-bottom: 4px;
}

.card-comment {
    font-size: 0.88rem;
    opacity: 0.92;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 800;
    margin-top: 0.3rem;
    margin-bottom: 0.8rem;
}

.info-box {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.02);
    min-height: 280px;
    margin-bottom: 10px;
}

.info-box h4 {
    margin-top: 0.1rem;
    margin-bottom: 0.8rem;
    font-size: 1rem;
}

.info-box ul {
    margin-top: 0.2rem;
    padding-left: 1.2rem;
}

.info-box li {
    margin-bottom: 0.45rem;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------
@dataclass
class SignalResult:
    label: str
    value: str
    status: str
    score: float
    comment: str
    detail: str = ""
    importance: str = "Medium"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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
def fetch_fred_series(series_id: str, api_key: str, start_date: str = "2000-01-01") -> pd.Series:
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
    return df["value"].astype(float)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_close(tickers: List[str], period: str = "15y", interval: str = "1d") -> pd.DataFrame:
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

    if "Close" in df.columns:
        t = tickers[0] if len(tickers) == 1 else "Close"
        out = pd.DataFrame({t: df["Close"]})
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    return pd.DataFrame()


def resample_monthly_last(s: pd.Series) -> pd.Series:
    return s.resample("ME").last().dropna() if not s.empty else s


def yoy_change(s: pd.Series) -> pd.Series:
    return s.pct_change(12) * 100.0


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


def compute_dma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def rolling_max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    if p.empty:
        return np.nan
    running_max = p.cummax()
    dd = p / running_max - 1.0
    return dd.iloc[-1] * 100.0


def higher_low_signal(prices: pd.Series, lookback: int = 120) -> int:
    p = prices.dropna()
    if len(p) < lookback:
        return 0
    recent = p.iloc[-lookback:]
    half = lookback // 2
    low1 = recent.iloc[:half].min()
    low2 = recent.iloc[half:].min()
    ratio = low2 / low1 if low1 != 0 else 1.0

    if ratio > 1.03:
        return 2
    if ratio > 1.00:
        return 1
    return 0


def slope_value(s: pd.Series, window: int = 20) -> float:
    x = s.dropna()
    if len(x) < window:
        return np.nan
    y = x.iloc[-window:].values
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def score_slope(series: pd.Series, window: int = 20, flat_threshold_ratio: float = 0.0005) -> float:
    s = series.dropna()
    if len(s) < window:
        return 0.0
    slope = slope_value(s, window=window)
    level = abs(s.iloc[-1])
    if pd.isna(slope) or level == 0:
        return 0.0
    ratio = slope / level
    if ratio > flat_threshold_ratio:
        return 1.0
    if ratio > -flat_threshold_ratio:
        return 0.5
    return 0.0


def score_price_vs_ma(price_last: float, ma_last: float, neutral_band: float = 0.01) -> float:
    if pd.isna(price_last) or pd.isna(ma_last) or ma_last == 0:
        return 0.0
    rel = price_last / ma_last - 1.0
    if rel > neutral_band:
        return 1.0
    if rel >= -neutral_band:
        return 0.5
    return 0.0


def score_to_status(score: float) -> str:
    if score >= 1.0:
        return "positive"
    if score >= 0.5:
        return "neutral"
    return "negative"


def status_emoji(status: str) -> str:
    if status == "positive":
        return "🟢"
    if status == "neutral":
        return "🟡"
    if status == "negative":
        return "🔴"
    return "⚪"


def fmt_num(x: float, nd: int = 2) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.{nd}f}"


def fmt_pct(x: float, nd: int = 2) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.{nd}f}%"


def readiness_percent(total_score: float, max_score: float = 10.0) -> int:
    if max_score <= 0:
        return 0
    pct = int(round((total_score / max_score) * 100))
    return max(0, min(100, pct))


def slice_lookback(df_or_s: pd.DataFrame | pd.Series, label: str):
    if label == "max":
        return df_or_s
    years = int(label.replace("y", ""))
    cutoff = pd.Timestamp.today() - relativedelta(years=years)
    return df_or_s[df_or_s.index >= cutoff]


def make_line_chart(df: pd.DataFrame, title: str, ytitle: str = "", normalize: bool = False, height: int = 430) -> go.Figure:
    plot_df = df.copy()
    if normalize and not plot_df.empty:
        plot_df = plot_df / plot_df.iloc[0] * 100.0

    fig = go.Figure()
    for c in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[c], mode="lines", name=c))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title=ytitle,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="Series",
    )
    return fig


def render_summary_box(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-label">{label}</div>
            <div class="summary-value">{value}</div>
            <div class="summary-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_signal_card(title: str, status: str, value: str, sub_value: str, comment: str) -> None:
    emoji = status_emoji(status)
    st.markdown(
        f"""
        <div class="card-box">
            <div class="card-title">{title}</div>
            <div class="card-status">{emoji} {status.capitalize()}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub">{sub_value}</div>
            <div class="card-comment">{comment}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_info_box(title: str, items: List[str], empty_text: str) -> None:
    html = [f'<div class="info-box"><h4>{title}</h4>']
    if items:
        html.append("<ul>")
        for item in items:
            html.append(f"<li>{item}</li>")
        html.append("</ul>")
    else:
        html.append(f"<div>{empty_text}</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def strategy_profile_map(profile: str) -> Dict[str, Dict[str, int]]:
    if profile == "Conservative":
        return {
            "Risk-Off": {"QQQ": 20, "SPY": 80},
            "Base Building": {"QQQ": 30, "SPY": 70},
            "Trend Confirmation": {"QQQ": 40, "SPY": 60},
            "Risk-On": {"QQQ": 50, "SPY": 50},
        }
    if profile == "Aggressive":
        return {
            "Risk-Off": {"QQQ": 40, "SPY": 60},
            "Base Building": {"QQQ": 45, "SPY": 55},
            "Trend Confirmation": {"QQQ": 60, "SPY": 40},
            "Risk-On": {"QQQ": 70, "SPY": 30},
        }
    return {
        "Risk-Off": {"QQQ": 30, "SPY": 70},
        "Base Building": {"QQQ": 40, "SPY": 60},
        "Trend Confirmation": {"QQQ": 50, "SPY": 50},
        "Risk-On": {"QQQ": 60, "SPY": 40},
    }


def current_phase_from_entry_mode(entry_mode: str) -> str:
    mapping = {
        "WAIT": "Risk-Off",
        "STARTER BUY": "Base Building",
        "ADD": "Trend Confirmation",
        "AGGRESSIVE BUY": "Risk-On",
    }
    return mapping.get(entry_mode, "Risk-Off")


def build_phase_plan(
    total_capital: float,
    phase_allocations: Dict[str, float],
    qqq_spy_weights: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    rows = []
    for phase in ["Risk-Off", "Base Building", "Trend Confirmation", "Risk-On"]:
        phase_pct = phase_allocations[phase]
        phase_amount = total_capital * phase_pct / 100.0
        qqq_w = qqq_spy_weights[phase]["QQQ"]
        spy_w = qqq_spy_weights[phase]["SPY"]
        qqq_amount = phase_amount * qqq_w / 100.0
        spy_amount = phase_amount * spy_w / 100.0

        rows.append({
            "Phase": phase,
            "Phase Allocation %": phase_pct,
            "Phase Amount": phase_amount,
            "QQQ Weight %": qqq_w,
            "SPY Weight %": spy_w,
            "QQQ Buy Amount": qqq_amount,
            "SPY Buy Amount": spy_amount,
        })
    return pd.DataFrame(rows)


def build_action_items(signals: List[SignalResult], fast_ma: int, slow_ma: int, bond_proxy: str) -> List[str]:
    mapping = {
        "Oil Pressure": "Oil momentum cooling further",
        "10Y Yield": "10Y yield trending lower",
        "Core CPI YoY": "Core CPI continuing to cool",
        "Unemployment": "Labor softening without sharp spike",
        "10Y-2Y Curve": "Yield curve steepening meaningfully",
        f"QQQ > {fast_ma}DMA": f"QQQ reclaiming {fast_ma}DMA",
        f"QQQ > {slow_ma}DMA": f"QQQ reclaiming {slow_ma}DMA",
        "Higher Low": "Higher low structure forming",
        "HYG Trend": "Credit market stabilizing",
        f"{bond_proxy} Trend": f"{bond_proxy} confirming easing",
    }
    out = []
    for s in signals:
        if s.score < 1.0:
            out.append(f"<b>{s.label}</b> — {mapping.get(s.label, 'Needs improvement')}")
    return out[:6]


def build_weak_items(signals: List[SignalResult]) -> List[str]:
    out = []
    for s in signals:
        if s.score == 0.0:
            out.append(f"<b>{s.label}</b> — {s.comment}")
    return out[:6]


def format_mixed_value(x: Any) -> str:
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return f"{x:,.2f}"
    return str(x)


def display_dataframe(df: pd.DataFrame, numeric_decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].apply(
                lambda x: f"{x:,.{numeric_decimals}f}" if pd.notna(x) else "N/A"
            )
        else:
            out[col] = out[col].astype(str)
    return out

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.title("QQQ / SPY Phase Buying Strategy")

with st.sidebar:
    st.header("Market Settings")

    lookback_label = st.selectbox("Chart lookback", ["1y", "2y", "3y", "5y", "10y", "max"], index=4)
    price_ma_fast = st.number_input("Fast MA", min_value=10, max_value=200, value=50, step=5)
    price_ma_slow = st.number_input("Slow MA", min_value=50, max_value=400, value=200, step=10)
    oil_proxy = st.selectbox("Oil proxy", ["WTI", "Brent"], index=0)
    bond_proxy = st.selectbox("Bond proxy", ["TLT", "IEF"], index=0)

    st.markdown("---")
    st.header("Strategy Options")

    strategy_profile = st.selectbox(
        "Risk Profile",
        ["Conservative", "Balanced", "Aggressive"],
        index=1
    )

    total_capital = st.number_input(
        "Total planned capital",
        min_value=1000.0,
        value=100000.0,
        step=1000.0
    )

    phase_mode = st.radio(
        "Phase Allocation Mode",
        ["Default", "Custom"],
        index=0
    )

    default_phase_alloc = {
        "Risk-Off": 20.0,
        "Base Building": 30.0,
        "Trend Confirmation": 30.0,
        "Risk-On": 20.0,
    }

    if phase_mode == "Custom":
        risk_off_alloc = st.slider("Risk-Off allocation %", 0, 100, 20, 5)
        base_alloc = st.slider("Base Building allocation %", 0, 100, 30, 5)
        confirm_alloc = st.slider("Trend Confirmation allocation %", 0, 100, 30, 5)
        risk_on_alloc = st.slider("Risk-On allocation %", 0, 100, 20, 5)

        phase_total = risk_off_alloc + base_alloc + confirm_alloc + risk_on_alloc
        if phase_total != 100:
            st.warning(f"Current phase allocation sum is {phase_total}%. It should equal 100%.")
        phase_allocations = {
            "Risk-Off": float(risk_off_alloc),
            "Base Building": float(base_alloc),
            "Trend Confirmation": float(confirm_alloc),
            "Risk-On": float(risk_on_alloc),
        }
    else:
        phase_allocations = default_phase_alloc

    show_debug = st.checkbox("Show debug", value=False)

fred_api_key = get_fred_api_key()
if not fred_api_key:
    st.error("FRED_API_KEY not found. Please set it in environment variables or .streamlit/secrets.toml")
    st.stop()

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
start_date = (pd.Timestamp.today() - relativedelta(years=DEFAULT_LOOKBACK_YEARS + 5)).strftime("%Y-%m-%d")

with st.spinner("Loading FRED data..."):
    fred_data = {}
    for name, sid in FRED_SERIES.items():
        try:
            fred_data[name] = fetch_fred_series(sid, fred_api_key, start_date=start_date)
        except Exception as e:
            fred_data[name] = pd.Series(dtype=float)
            st.warning(f"Failed to load FRED series {name} ({sid}): {e}")

with st.spinner("Loading Yahoo Finance data..."):
    try:
        yf_close = fetch_yf_close(list(YF_TICKERS.values()), period="15y", interval="1d")
    except Exception as e:
        st.error(f"Failed to load Yahoo Finance data: {e}")
        st.stop()

rename_map = {v: k for k, v in YF_TICKERS.items()}
yf_close = yf_close.rename(columns=rename_map)

if yf_close.empty:
    st.error("No Yahoo Finance data loaded.")
    st.stop()

# ------------------------------------------------------------
# Derived data
# ------------------------------------------------------------
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
yc_10y_3m = (y10.resample("ME").last() - y3m.resample("ME").last()).dropna()

spy = yf_close["SPY"].dropna() if "SPY" in yf_close.columns else pd.Series(dtype=float)
qqq = yf_close["QQQ"].dropna() if "QQQ" in yf_close.columns else pd.Series(dtype=float)
iwm = yf_close["IWM"].dropna() if "IWM" in yf_close.columns else pd.Series(dtype=float)
hyg = yf_close["HYG"].dropna() if "HYG" in yf_close.columns else pd.Series(dtype=float)
tlt = yf_close["TLT"].dropna() if "TLT" in yf_close.columns else pd.Series(dtype=float)
ief = yf_close["IEF"].dropna() if "IEF" in yf_close.columns else pd.Series(dtype=float)
uup = yf_close["UUP"].dropna() if "UUP" in yf_close.columns else pd.Series(dtype=float)
wti = yf_close["WTI"].dropna() if "WTI" in yf_close.columns else pd.Series(dtype=float)
brent = yf_close["Brent"].dropna() if "Brent" in yf_close.columns else pd.Series(dtype=float)
soxx = yf_close["SOXX"].dropna() if "SOXX" in yf_close.columns else pd.Series(dtype=float)

oil_series = wti if oil_proxy == "WTI" else brent
bond_series = tlt if bond_proxy == "TLT" else ief

qqq_ma_fast = compute_dma(qqq, price_ma_fast)
qqq_ma_slow = compute_dma(qqq, price_ma_slow)

# ------------------------------------------------------------
# Signals
# ------------------------------------------------------------
macro_signals: List[SignalResult] = []
market_signals: List[SignalResult] = []

oil_len = len(oil_series.dropna())
oil_last = safe_last(oil_series)
oil_1m = pct_from_n_days(oil_series, 21)
oil_3m = pct_from_n_days(oil_series, 63)

if oil_len < 70 or pd.isna(oil_1m) or pd.isna(oil_3m):
    score = 0.5
    value = "N/A"
    detail = f"{oil_proxy} data unstable"
    comment = "Insufficient or unstable data; treated as neutral"
else:
    value = f"{fmt_pct(oil_1m)}"
    detail = f"Last {fmt_num(oil_last)} | 3M {fmt_pct(oil_3m)}"
    if oil_1m < -3 and oil_3m < 5:
        score = 1.0
        comment = "Oil pressure clearly cooling"
    elif oil_1m < oil_3m:
        score = 0.5
        comment = "Oil still elevated but momentum cooling"
    else:
        score = 0.0
        comment = "Oil still pressuring inflation"

macro_signals.append(SignalResult("Oil Pressure", value, score_to_status(score), score, comment, detail, "High"))

y10_last = safe_last(y10)
y10_1m_ago = safe_prev(y10, 21)
if pd.notna(y10_last) and pd.notna(y10_1m_ago):
    delta = y10_last - y10_1m_ago
    if delta < -0.15:
        score = 1.0
        comment = "Discount rate easing meaningfully"
    elif delta <= 0.05:
        score = 0.5
        comment = "Rates stabilizing"
    else:
        score = 0.0
        comment = "Discount rate still elevated"
    detail = f"1M delta {fmt_num(delta)}%p"
else:
    score = 0.5
    comment = "Insufficient data"
    detail = "No stable comparison"
macro_signals.append(SignalResult("10Y Yield", f"{fmt_num(y10_last)}%", score_to_status(score), score, comment, detail, "High"))

core_last = safe_last(core_cpi_yoy)
core_prev = safe_prev(core_cpi_yoy, 1)
if pd.notna(core_last) and pd.notna(core_prev):
    diff = core_last - core_prev
    if diff < -0.10:
        score = 1.0
        comment = "Core inflation cooling"
    elif diff <= 0.10:
        score = 0.5
        comment = "Core inflation sticky but not worsening much"
    else:
        score = 0.0
        comment = "Core inflation re-accelerating"
    detail = f"Prev {fmt_pct(core_prev)} | Delta {fmt_num(diff)}"
else:
    score = 0.5
    comment = "Insufficient data"
    detail = "No stable comparison"
macro_signals.append(SignalResult("Core CPI YoY", fmt_pct(core_last), score_to_status(score), score, comment, detail, "Medium"))

un_last = safe_last(unrate)
un_3m_ago = safe_prev(unrate, 3)
if pd.notna(un_last) and pd.notna(un_3m_ago):
    delta = un_last - un_3m_ago
    if 0 <= delta <= 0.30:
        score = 1.0
        comment = "Soft deterioration, not severe"
    elif -0.10 <= delta < 0:
        score = 0.5
        comment = "Labor market still firm"
    else:
        score = 0.0
        comment = "Labor market weakening fast"
    detail = f"3M delta {fmt_num(delta)}"
else:
    score = 0.5
    comment = "Insufficient data"
    detail = "No stable comparison"
macro_signals.append(SignalResult("Unemployment", fmt_num(un_last), score_to_status(score), score, comment, detail, "Medium"))

yc_last = safe_last(yc_10y_2y)
yc_1m_ago = safe_prev(yc_10y_2y, 21)
if pd.notna(yc_last) and pd.notna(yc_1m_ago):
    delta = yc_last - yc_1m_ago
    if delta > 0.15:
        score = 1.0
        comment = "Curve steepening"
    elif delta >= -0.05:
        score = 0.5
        comment = "No major deterioration"
    else:
        score = 0.0
        comment = "Curve not improving"
    detail = f"1M delta {fmt_num(delta)}"
else:
    score = 0.5
    comment = "Insufficient data"
    detail = "No stable comparison"
macro_signals.append(SignalResult("10Y-2Y Curve", fmt_num(yc_last), score_to_status(score), score, comment, detail, "Medium"))

qqq_last = safe_last(qqq)
qqq_fast_last = safe_last(qqq_ma_fast)
qqq_gap_fast = (qqq_last / qqq_fast_last - 1.0) * 100.0 if pd.notna(qqq_last) and pd.notna(qqq_fast_last) and qqq_fast_last != 0 else np.nan
score = score_price_vs_ma(qqq_last, qqq_fast_last, neutral_band=0.01)
comment = "Above fast trend" if score == 1.0 else ("Near fast trend" if score == 0.5 else "Still below fast trend")
detail = f"Current {fmt_num(qqq_last)} vs {price_ma_fast}DMA {fmt_num(qqq_fast_last)} | Gap {fmt_pct(qqq_gap_fast)}"
market_signals.append(SignalResult(f"QQQ > {price_ma_fast}DMA", fmt_num(qqq_last), score_to_status(score), score, comment, detail, "High"))

qqq_slow_last = safe_last(qqq_ma_slow)
qqq_gap_slow = (qqq_last / qqq_slow_last - 1.0) * 100.0 if pd.notna(qqq_last) and pd.notna(qqq_slow_last) and qqq_slow_last != 0 else np.nan
score = score_price_vs_ma(qqq_last, qqq_slow_last, neutral_band=0.02)
comment = "Long trend confirmed" if score == 1.0 else ("Near long trend" if score == 0.5 else "Long trend not confirmed")
detail = f"Current {fmt_num(qqq_last)} vs {price_ma_slow}DMA {fmt_num(qqq_slow_last)} | Gap {fmt_pct(qqq_gap_slow)}"
market_signals.append(SignalResult(f"QQQ > {price_ma_slow}DMA", fmt_num(qqq_last), score_to_status(score), score, comment, detail, "High"))

hl = higher_low_signal(qqq, 120)
score = 1.0 if hl == 2 else (0.5 if hl == 1 else 0.0)
comment = "Base-building structure" if score == 1.0 else ("Weak higher-low attempt" if score == 0.5 else "No base structure yet")
market_signals.append(SignalResult("Higher Low", "True" if hl > 0 else "False", score_to_status(score), score, comment, "Structure over 120 trading days", "High"))

hyg_last = safe_last(hyg)
hyg_slope = slope_value(hyg, 20)
score = score_slope(hyg, 20, 0.0005)
comment = "Credit market supportive" if score == 1.0 else ("Credit market stabilizing" if score == 0.5 else "Credit market not supportive")
market_signals.append(SignalResult("HYG Trend", fmt_num(hyg_last), score_to_status(score), score, comment, f"Last {fmt_num(hyg_last)} | 20d slope {fmt_num(hyg_slope, 6)}", "Medium"))

bond_last = safe_last(bond_series)
bond_slope = slope_value(bond_series, 20)
score = score_slope(bond_series, 20, 0.0005)
comment = "Bond market confirms easing" if score == 1.0 else ("Bond market stabilizing" if score == 0.5 else "Bond market not confirming easing")
market_signals.append(SignalResult(f"{bond_proxy} Trend", fmt_num(bond_last), score_to_status(score), score, comment, f"Last {fmt_num(bond_last)} | 20d slope {fmt_num(bond_slope, 6)}", "Medium"))

macro_score = float(sum(s.score for s in macro_signals))
market_score = float(sum(s.score for s in market_signals))
total_score = macro_score + market_score
entry_readiness = readiness_percent(total_score)

# ------------------------------------------------------------
# Regime classification
# ------------------------------------------------------------
if total_score <= 3.0:
    risk_environment = "Risk-Off"
    trend_status = "Downtrend / Fragile"
    entry_mode = "WAIT"
    entry_comment = "Macro and/or market confirmation is insufficient."
elif total_score <= 5.0:
    risk_environment = "Neutral"
    trend_status = "Base Building"
    entry_mode = "STARTER BUY"
    entry_comment = "Initial staged buying is possible."
elif total_score <= 7.5:
    risk_environment = "Improving"
    trend_status = "Trend Confirmation"
    entry_mode = "ADD"
    entry_comment = "Trend is improving. Add in stages."
else:
    risk_environment = "Risk-On"
    trend_status = "Uptrend"
    entry_mode = "AGGRESSIVE BUY"
    entry_comment = "Macro and market are aligned."

current_phase = current_phase_from_entry_mode(entry_mode)

# ------------------------------------------------------------
# Strategy plan
# ------------------------------------------------------------
qqq_spy_weights = strategy_profile_map(strategy_profile)
phase_plan_df = build_phase_plan(total_capital, phase_allocations, qqq_spy_weights)

current_phase_row = phase_plan_df.loc[phase_plan_df["Phase"] == current_phase].iloc[0]
current_phase_amount = float(current_phase_row["Phase Amount"])
current_qqq_amount = float(current_phase_row["QQQ Buy Amount"])
current_spy_amount = float(current_phase_row["SPY Buy Amount"])
current_qqq_weight = int(current_phase_row["QQQ Weight %"])
current_spy_weight = int(current_phase_row["SPY Weight %"])

watch_items = build_action_items(macro_signals + market_signals, price_ma_fast, price_ma_slow, bond_proxy)
weak_items = build_weak_items(macro_signals + market_signals)

# ------------------------------------------------------------
# Top summary
# ------------------------------------------------------------
st.markdown('<div class="section-title">Top Summary</div>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
with s1:
    render_summary_box("Current Market Phase", current_phase, f"Risk environment: {risk_environment}")
with s2:
    render_summary_box("Entry Mode", entry_mode, entry_comment)
with s3:
    render_summary_box("Entry Readiness", f"{entry_readiness}%", f"Total score {total_score:.1f}/10")
with s4:
    render_summary_box("Selected Strategy", strategy_profile, f"Current phase mix: QQQ {current_qqq_weight}% / SPY {current_spy_weight}%")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Strategy Overview",
    "Phase Buy Plan",
    "Market Dashboard",
    "Detailed Data"
])

# ------------------------------------------------------------
# Tab 1: Strategy Overview
# ------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Current Phase Action</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_summary_box("Current Phase Budget", f"{current_phase_amount:,.0f}", f"Total capital {total_capital:,.0f}")
    with c2:
        render_summary_box("QQQ Buy Amount", f"{current_qqq_amount:,.0f}", f"Weight {current_qqq_weight}% in {current_phase}")
    with c3:
        render_summary_box("SPY Buy Amount", f"{current_spy_amount:,.0f}", f"Weight {current_spy_weight}% in {current_phase}")

    st.markdown('<div class="section-title">Market Signals</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, sig in zip(cols, market_signals):
        with col:
            render_signal_card(sig.label, sig.status, sig.value, sig.detail, sig.comment)

    st.markdown('<div class="section-title">Macro Signals</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, sig in zip(cols, macro_signals):
        with col:
            render_signal_card(sig.label, sig.status, sig.value, sig.detail, sig.comment)

    st.markdown('<div class="section-title">Why the system is cautious / What to watch next</div>', unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        render_info_box(
            "Why the system is cautious",
            weak_items,
            "No major weak signal detected."
        )

    with right:
        render_info_box(
            "What to watch next",
            watch_items,
            "No major improvement required."
        )

# ------------------------------------------------------------
# Tab 2: Phase Buy Plan
# ------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Phase-Based QQQ / SPY Buying Plan</div>', unsafe_allow_html=True)

    display_df = phase_plan_df.copy()
    display_df["Active"] = display_df["Phase"].apply(lambda x: "✅ Current" if x == current_phase else "")
    display_df = display_df[[
        "Phase", "Active", "Phase Allocation %", "Phase Amount",
        "QQQ Weight %", "SPY Weight %", "QQQ Buy Amount", "SPY Buy Amount"
    ]]

    st.dataframe(display_dataframe(display_df), use_container_width=True, hide_index=True)

    chart_df = phase_plan_df.set_index("Phase")[["QQQ Buy Amount", "SPY Buy Amount"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["QQQ Buy Amount"], name="QQQ Buy Amount"))
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["SPY Buy Amount"], name="SPY Buy Amount"))
    fig.update_layout(
        barmode="stack",
        title="Phase Buy Amount by ETF",
        height=420,
        yaxis_title="Amount",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Strategy logic")
    logic_df = pd.DataFrame([
        {"Phase": "Risk-Off", "Meaning": "Environment still fragile", "Action": "Small and defensive, SPY-heavy"},
        {"Phase": "Base Building", "Meaning": "Deterioration slows", "Action": "Starter buy, gradual QQQ increase"},
        {"Phase": "Trend Confirmation", "Meaning": "Trend recovery appears", "Action": "Balanced or QQQ-leaning add"},
        {"Phase": "Risk-On", "Meaning": "Trend and macro aligned", "Action": "More aggressive QQQ allocation"},
    ])
    st.dataframe(logic_df.astype(str), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Tab 3: Market Dashboard
# ------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)

    snapshot_assets = {
        "SPY": spy,
        "QQQ": qqq,
        "IWM": iwm,
        "HYG": hyg,
        bond_proxy: bond_series,
        oil_proxy: oil_series,
        "SOXX": soxx,
        "UUP": uup,
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
    snapshot_df = pd.DataFrame(rows)
    st.dataframe(display_dataframe(snapshot_df), use_container_width=True, hide_index=True)

    multi_asset_df = pd.DataFrame({
        "SPY": slice_lookback(spy, lookback_label),
        "QQQ": slice_lookback(qqq, lookback_label),
        "HYG": slice_lookback(hyg, lookback_label),
        bond_proxy: slice_lookback(bond_series, lookback_label),
        oil_proxy: slice_lookback(oil_series, lookback_label),
        "SOXX": slice_lookback(soxx, lookback_label),
    }).dropna(how="all")
    st.plotly_chart(
        make_line_chart(multi_asset_df, f"Normalized Multi-Asset Snapshot ({lookback_label})", "Indexed to 100", normalize=True),
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        inflation_df = pd.concat([
            slice_lookback(cpi_yoy.rename("CPI YoY"), lookback_label),
            slice_lookback(core_cpi_yoy.rename("Core CPI YoY"), lookback_label),
        ], axis=1).dropna(how="all")
        st.plotly_chart(
            make_line_chart(inflation_df, f"Inflation Trend ({lookback_label})", "%"),
            use_container_width=True
        )

        curve_df = pd.concat([
            slice_lookback(yc_10y_2y.rename("10Y-2Y"), lookback_label),
            slice_lookback(yc_10y_3m.rename("10Y-3M"), lookback_label),
        ], axis=1).dropna(how="all")
        st.plotly_chart(
            make_line_chart(curve_df, f"Yield Curve ({lookback_label})", "%p"),
            use_container_width=True
        )

    with right:
        rates_df = pd.concat([
            slice_lookback(y10.rename("10Y"), lookback_label),
            slice_lookback(y2.rename("2Y"), lookback_label),
            slice_lookback(fedfunds.rename("Fed Funds"), lookback_label),
        ], axis=1).dropna(how="all")
        st.plotly_chart(
            make_line_chart(rates_df, f"Rates ({lookback_label})", "%"),
            use_container_width=True
        )

        trend_df = pd.DataFrame({
            "QQQ": slice_lookback(qqq, lookback_label),
            f"{price_ma_fast}DMA": slice_lookback(qqq_ma_fast, lookback_label),
            f"{price_ma_slow}DMA": slice_lookback(qqq_ma_slow, lookback_label),
        }).dropna(how="all")
        st.plotly_chart(
            make_line_chart(trend_df, f"QQQ Trend Confirmation ({lookback_label})", "Price"),
            use_container_width=True
        )

# ------------------------------------------------------------
# Tab 4: Detailed Data
# ------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">Detailed Signal Table</div>', unsafe_allow_html=True)

    raw_df = pd.DataFrame([{
        "Category": "Macro",
        "Signal": s.label,
        "Value": s.value,
        "Status": s.status,
        "Score": s.score,
        "Importance": s.importance,
        "Detail": s.detail,
        "Comment": s.comment,
    } for s in macro_signals] + [{
        "Category": "Market",
        "Signal": s.label,
        "Value": s.value,
        "Status": s.status,
        "Score": s.score,
        "Importance": s.importance,
        "Detail": s.detail,
        "Comment": s.comment,
    } for s in market_signals])

    st.dataframe(display_dataframe(raw_df), use_container_width=True, hide_index=True)

    with st.expander("Macro snapshot", expanded=False):
        macro_snapshot = pd.DataFrame([
            {"Series": "CPI YoY", "Last": safe_last(cpi_yoy), "Prev": safe_prev(cpi_yoy, 1)},
            {"Series": "Core CPI YoY", "Last": safe_last(core_cpi_yoy), "Prev": safe_prev(core_cpi_yoy, 1)},
            {"Series": "Unemployment", "Last": safe_last(unrate), "Prev": safe_prev(unrate, 1)},
            {"Series": "Initial Claims", "Last": safe_last(claims), "Prev": safe_prev(claims, 1)},
            {"Series": "Industrial Production YoY", "Last": safe_last(indpro_yoy), "Prev": safe_prev(indpro_yoy, 1)},
            {"Series": "M2 YoY", "Last": safe_last(m2_yoy), "Prev": safe_prev(m2_yoy, 1)},
            {"Series": "10Y Yield", "Last": safe_last(y10), "Prev": safe_prev(y10, 21)},
            {"Series": "2Y Yield", "Last": safe_last(y2), "Prev": safe_prev(y2, 21)},
            {"Series": "10Y-2Y", "Last": safe_last(yc_10y_2y), "Prev": safe_prev(yc_10y_2y, 21)},
        ])
        st.dataframe(display_dataframe(macro_snapshot), use_container_width=True, hide_index=True)

    with st.expander("Current phase execution summary", expanded=False):
        current_exec_df = pd.DataFrame([
            {"Item": "Current Phase", "Value": current_phase},
            {"Item": "Risk Profile", "Value": strategy_profile},
            {"Item": "Phase Allocation %", "Value": float(current_phase_row["Phase Allocation %"])},
            {"Item": "Phase Budget", "Value": current_phase_amount},
            {"Item": "QQQ Weight %", "Value": current_qqq_weight},
            {"Item": "SPY Weight %", "Value": current_spy_weight},
            {"Item": "QQQ Buy Amount", "Value": current_qqq_amount},
            {"Item": "SPY Buy Amount", "Value": current_spy_amount},
        ])
        current_exec_df["Value"] = current_exec_df["Value"].apply(format_mixed_value)
        st.dataframe(current_exec_df.astype(str), use_container_width=True, hide_index=True)

    if show_debug:
        with st.expander("Debug", expanded=False):
            debug_df = pd.DataFrame([
                {"Metric": "Oil Length", "Value": oil_len},
                {"Metric": "Oil 1M %", "Value": oil_1m},
                {"Metric": "Oil 3M %", "Value": oil_3m},
                {"Metric": "QQQ Last", "Value": qqq_last},
                {"Metric": "QQQ 50DMA", "Value": qqq_fast_last},
                {"Metric": "QQQ 200DMA", "Value": qqq_slow_last},
                {"Metric": "HYG Slope 20d", "Value": hyg_slope},
                {"Metric": f"{bond_proxy} Slope 20d", "Value": bond_slope},
                {"Metric": "Total Score", "Value": total_score},
                {"Metric": "Entry Readiness %", "Value": entry_readiness},
            ])
            st.dataframe(display_dataframe(debug_df, numeric_decimals=6), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("This dashboard combines market regime detection with phase-based QQQ/SPY staged buying execution.")
