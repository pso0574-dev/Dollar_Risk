"""
유가 예측 대시보드 - Streamlit App
실행: streamlit run oil_price_predictor.py
필요 패키지: pip install streamlit yfinance pandas numpy plotly scikit-learn ta requests
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="유가 예측 대시보드",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2236;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-yellow: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: rgba(59,130,246,0.2);
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

.stApp { background-color: var(--bg-dark); }

/* 헤더 */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2044 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle {
    color: var(--text-secondary);
    margin-top: 0.4rem;
    font-size: 0.95rem;
}

/* 메트릭 카드 */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.2s;
}
.metric-card:hover { border-color: var(--accent-blue); }
.metric-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-delta-up { color: var(--accent-green); font-size: 0.85rem; }
.metric-delta-down { color: var(--accent-red); font-size: 0.85rem; }

/* 섹션 헤더 */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 1rem 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* 신호 배지 */
.signal-buy {
    background: rgba(16,185,129,0.15);
    color: var(--accent-green);
    border: 1px solid rgba(16,185,129,0.3);
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.signal-sell {
    background: rgba(239,68,68,0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239,68,68,0.3);
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.signal-neutral {
    background: rgba(245,158,11,0.15);
    color: var(--accent-yellow);
    border: 1px solid rgba(245,158,11,0.3);
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Streamlit 요소 오버라이드 */
div[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border);
}
.stButton button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    width: 100%;
}
.stButton button:hover { opacity: 0.9; }
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

.info-box {
    background: var(--bg-card2);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: var(--text-secondary);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 데이터 로드 함수
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_oil_data(ticker: str, period: str) -> pd.DataFrame:
    """yfinance에서 원유 가격 데이터 로드"""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            st.error(f"데이터를 가져올 수 없습니다: {ticker}")
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_related_data() -> dict:
    """연관 지표 데이터 로드 (달러인덱스, 금, 천연가스)"""
    tickers = {
        "달러인덱스 (DXY)": "DX-Y.NYB",
        "금 (Gold)": "GC=F",
        "천연가스": "NG=F",
        "S&P 500": "^GSPC",
    }
    result = {}
    for name, tkr in tickers.items():
        try:
            d = yf.download(tkr, period="3mo", auto_adjust=True, progress=False)
            if not d.empty:
                d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
                result[name] = d["Close"].iloc[-1], d["Close"].pct_change().iloc[-1] * 100
        except:
            pass
    return result


# ─────────────────────────────────────────────
# 기술적 지표 계산
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    # 이동평균
    df["MA20"]  = close.rolling(20).mean()
    df["MA50"]  = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # 볼린저 밴드
    df["BB_mid"]   = close.rolling(20).mean()
    df["BB_std"]   = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # OBV
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].squeeze().iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].squeeze().iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    # 수익률
    df["Returns"]    = close.pct_change()
    df["Returns_5d"] = close.pct_change(5)

    return df


# ─────────────────────────────────────────────
# 신호 판단
# ─────────────────────────────────────────────
def get_signals(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = df["Close"].squeeze().iloc[-1]
    signals = {}

    # RSI
    rsi = last["RSI"]
    if rsi < 30:
        signals["RSI"] = ("매수", "과매도 구간 (RSI < 30)", "buy")
    elif rsi > 70:
        signals["RSI"] = ("매도", "과매수 구간 (RSI > 70)", "sell")
    else:
        signals["RSI"] = ("중립", f"RSI = {rsi:.1f}", "neutral")

    # MACD
    if last["MACD"] > last["MACD_signal"]:
        signals["MACD"] = ("매수", "MACD > Signal (골든 크로스)", "buy")
    else:
        signals["MACD"] = ("매도", "MACD < Signal (데드 크로스)", "sell")

    # MA
    if close > last["MA20"] and last["MA20"] > last["MA50"]:
        signals["이동평균"] = ("매수", "가격 > MA20 > MA50 (상승추세)", "buy")
    elif close < last["MA20"] and last["MA20"] < last["MA50"]:
        signals["이동평균"] = ("매도", "가격 < MA20 < MA50 (하락추세)", "sell")
    else:
        signals["이동평균"] = ("중립", "혼조세", "neutral")

    # 볼린저 밴드
    if close < last["BB_lower"]:
        signals["볼린저밴드"] = ("매수", "하단 밴드 이탈 (반등 가능)", "buy")
    elif close > last["BB_upper"]:
        signals["볼린저밴드"] = ("매도", "상단 밴드 이탈 (조정 가능)", "sell")
    else:
        signals["볼린저밴드"] = ("중립", "밴드 내 위치", "neutral")

    return signals


# ─────────────────────────────────────────────
# ML 예측 모델
# ─────────────────────────────────────────────
def build_ml_model(df: pd.DataFrame, forecast_days: int = 5):
    features = ["MA20", "MA50", "RSI", "MACD", "MACD_signal",
                "BB_upper", "BB_lower", "ATR", "Returns", "Returns_5d", "OBV"]
    close = df["Close"].squeeze()

    data = df[features].copy()
    data["target"] = close.shift(-forecast_days)
    data = data.dropna()

    X = data[features]
    y = data["target"]

    if len(X) < 60:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    best_model, best_mae = None, float("inf")
    metrics = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        preds = m.predict(X_test_s)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        metrics[name] = {"MAE": mae, "R²": r2}
        if mae < best_mae:
            best_mae   = mae
            best_model = m

    # 미래 예측
    last_features = scaler.transform(X.iloc[[-1]])
    predicted_price = best_model.predict(last_features)[0]

    return best_model, scaler, predicted_price, metrics


# ─────────────────────────────────────────────
# 차트 그리기
# ─────────────────────────────────────────────
def make_main_chart(df: pd.DataFrame, show_bb: bool, show_ma: bool):
    close = df["Close"].squeeze()
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=["가격 차트", "RSI (14)", "MACD"]
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(),
        high=df["High"].squeeze(), low=df["Low"].squeeze(),
        close=close,
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        name="가격"
    ), row=1, col=1)

    # MA
    if show_ma:
        for col, color, name in [
            ("MA20", "#60a5fa", "MA20"),
            ("MA50", "#f59e0b", "MA50"),
            ("MA200", "#a78bfa", "MA200"),
        ]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], line=dict(color=color, width=1.5),
                name=name, opacity=0.9
            ), row=1, col=1)

    # 볼린저 밴드
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            line=dict(color="rgba(139,92,246,0.4)", width=1, dash="dot"),
            name="BB Upper", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            fill="tonexty", fillcolor="rgba(139,92,246,0.05)",
            line=dict(color="rgba(139,92,246,0.4)", width=1, dash="dot"),
            name="BB Band"
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        line=dict(color="#f59e0b", width=1.5), name="RSI"
    ), row=2, col=1)
    for lvl, color in [(70, "rgba(239,68,68,0.3)"), (30, "rgba(16,185,129,0.3)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=color, row=2, col=1)

    # MACD
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_hist"],
        marker_color=colors, name="MACD Hist", opacity=0.7
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        line=dict(color="#60a5fa", width=1.5), name="MACD"
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_signal"],
        line=dict(color="#f59e0b", width=1.5), name="Signal"
    ), row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        font=dict(color="#94a3b8", family="Noto Sans KR"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(17,24,39,0.9)", bordercolor="rgba(59,130,246,0.3)",
            borderwidth=1, font=dict(size=11)
        ),
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis3=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis3=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def make_returns_dist(df: pd.DataFrame):
    returns = df["Returns"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=60,
        marker_color="#3b82f6", opacity=0.7,
        name="수익률 분포"
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.update_layout(
        title="일간 수익률 분포",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        font=dict(color="#94a3b8"),
        height=300,
        margin=dict(t=40, b=20, l=10, r=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def make_volatility_chart(df: pd.DataFrame):
    vol = df["Returns"].rolling(20).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=vol,
        fill="tozeroy", fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="#8b5cf6", width=1.5),
        name="연환산 변동성 (20일)"
    ))
    fig.update_layout(
        title="역사적 변동성",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        font=dict(color="#94a3b8"),
        height=300,
        margin=dict(t=40, b=20, l=10, r=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", ticksuffix="%"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    st.markdown("---")

    oil_type = st.selectbox(
        "원유 종류",
        ["WTI 원유 (CL=F)", "브렌트 원유 (BZ=F)"],
        help="WTI: 미국 기준 원유 / 브렌트: 국제 기준 원유"
    )
    ticker_map = {"WTI 원유 (CL=F)": "CL=F", "브렌트 원유 (BZ=F)": "BZ=F"}
    selected_ticker = ticker_map[oil_type]

    period = st.selectbox(
        "데이터 기간",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
        format_func=lambda x: {"3mo":"3개월","6mo":"6개월","1y":"1년","2y":"2년","5y":"5년"}[x]
    )

    forecast_days = st.slider("예측 기간 (일)", 1, 30, 5)

    st.markdown("---")
    st.markdown("**차트 옵션**")
    show_ma = st.toggle("이동평균선 (MA)", value=True)
    show_bb = st.toggle("볼린저 밴드", value=True)

    st.markdown("---")
    refresh = st.button("🔄 데이터 새로고침")
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.6'>
    ⚠️ <b>면책 고지</b><br>
    본 앱은 교육·참고 목적이며,<br>
    실제 투자 권유가 아닙니다.<br>
    투자 손실에 대한 책임은<br>
    사용자 본인에게 있습니다.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 메인 콘텐츠
# ─────────────────────────────────────────────

# 헤더
st.markdown(f"""
<div class="hero-header">
    <div class="hero-title">🛢️ 유가 예측 대시보드</div>
    <div class="hero-subtitle">
        기술적 분석 · 머신러닝 예측 · 실시간 시장 데이터 &nbsp;|&nbsp; {oil_type}
        &nbsp;|&nbsp; 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

# 데이터 로드
with st.spinner("📡 시장 데이터 불러오는 중..."):
    df_raw = load_oil_data(selected_ticker, period)

if df_raw.empty:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도하세요.")
    st.stop()

df = compute_indicators(df_raw.copy())
close_series = df["Close"].squeeze()

current_price = close_series.iloc[-1]
prev_price    = close_series.iloc[-2]
price_change  = current_price - prev_price
pct_change    = (price_change / prev_price) * 100

# ─── 주요 지표 카드 ───
col1, col2, col3, col4, col5 = st.columns(5)

delta_color = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
delta_icon  = "▲" if price_change >= 0 else "▼"

with col1:
    st.metric("현재가", f"${current_price:.2f}", f"{delta_icon} {abs(price_change):.2f} ({abs(pct_change):.2f}%)")
with col2:
    st.metric("52주 최고", f"${close_series.rolling(252).max().iloc[-1]:.2f}")
with col3:
    st.metric("52주 최저", f"${close_series.rolling(252).min().iloc[-1]:.2f}")
with col4:
    st.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
with col5:
    vol_20 = df["Returns"].rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    st.metric("변동성 (20일)", f"{vol_20:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─── 탭 레이아웃 ───
tab1, tab2, tab3, tab4 = st.tabs(["📈 차트 분석", "🔔 매매 신호", "🤖 ML 예측", "🌐 연관 시장"])

# ──────────────── TAB 1: 차트 ────────────────
with tab1:
    fig_main = make_main_chart(df, show_bb, show_ma)
    st.plotly_chart(fig_main, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(make_returns_dist(df), use_container_width=True)
    with col_b:
        st.plotly_chart(make_volatility_chart(df), use_container_width=True)

    # 최근 데이터 테이블
    st.markdown('<div class="section-header">📋 최근 가격 데이터</div>', unsafe_allow_html=True)
    display_df = df[["Open","High","Low","Close","Volume","RSI","MACD"]].tail(15).copy()
    display_df = display_df.round(2)
    display_df.columns = ["시가","고가","저가","종가","거래량","RSI","MACD"]
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    st.dataframe(display_df[::-1], use_container_width=True, height=300)


# ──────────────── TAB 2: 매매 신호 ────────────────
with tab2:
    signals = get_signals(df)

    st.markdown('<div class="section-header">🔔 기술적 분석 신호 종합</div>', unsafe_allow_html=True)

    buy_count     = sum(1 for v in signals.values() if v[2] == "buy")
    sell_count    = sum(1 for v in signals.values() if v[2] == "sell")
    neutral_count = sum(1 for v in signals.values() if v[2] == "neutral")

    overall = "매수 우세" if buy_count > sell_count else ("매도 우세" if sell_count > buy_count else "중립")
    overall_class = "signal-buy" if buy_count > sell_count else ("signal-sell" if sell_count > buy_count else "signal-neutral")

    st.markdown(f"""
    <div style='background: var(--bg-card2); border-radius:12px; padding:1.5rem; margin-bottom:1rem; text-align:center;'>
        <div style='font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.5rem'>종합 신호</div>
        <span class="{overall_class}" style="font-size:1.2rem; padding: 0.4rem 1.5rem">{overall}</span>
        <div style='margin-top:1rem; display:flex; justify-content:center; gap:2rem;'>
            <div><span style='color:#10b981; font-size:1.4rem; font-weight:700'>{buy_count}</span>
                 <div style='color:var(--text-secondary);font-size:0.8rem'>매수</div></div>
            <div><span style='color:#94a3b8; font-size:1.4rem; font-weight:700'>{neutral_count}</span>
                 <div style='color:var(--text-secondary);font-size:0.8rem'>중립</div></div>
            <div><span style='color:#ef4444; font-size:1.4rem; font-weight:700'>{sell_count}</span>
                 <div style='color:var(--text-secondary);font-size:0.8rem'>매도</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for indicator, (signal, desc, stype) in signals.items():
        badge_class = {"buy": "signal-buy", "sell": "signal-sell", "neutral": "signal-neutral"}[stype]
        st.markdown(f"""
        <div style='background:var(--bg-card); border:1px solid var(--border); border-radius:10px;
                    padding:1rem 1.2rem; margin-bottom:0.6rem; display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <div style='font-weight:600; font-size:0.95rem'>{indicator}</div>
                <div style='color:var(--text-secondary); font-size:0.82rem; margin-top:0.2rem'>{desc}</div>
            </div>
            <span class="{badge_class}">{signal}</span>
        </div>
        """, unsafe_allow_html=True)

    # 지지/저항선
    st.markdown('<div class="section-header">📊 지지선 / 저항선 (최근 20일)</div>', unsafe_allow_html=True)
    recent = close_series.tail(20)
    support    = recent.min()
    resistance = recent.max()
    pivot      = (df["High"].squeeze().tail(1).values[0] +
                  df["Low"].squeeze().tail(1).values[0] +
                  current_price) / 3

    c1, c2, c3 = st.columns(3)
    c1.metric("지지선 (Support)", f"${support:.2f}", f"{((current_price-support)/support*100):+.1f}%")
    c2.metric("피벗 포인트", f"${pivot:.2f}", f"{((current_price-pivot)/pivot*100):+.1f}%")
    c3.metric("저항선 (Resistance)", f"${resistance:.2f}", f"{((current_price-resistance)/resistance*100):+.1f}%")


# ──────────────── TAB 3: ML 예측 ────────────────
with tab3:
    st.markdown('<div class="section-header">🤖 머신러닝 가격 예측</div>', unsafe_allow_html=True)

    with st.spinner(f"모델 학습 중 ({forecast_days}일 후 예측)..."):
        model, scaler, predicted_price, metrics = build_ml_model(df, forecast_days)

    if model is None:
        st.warning("데이터가 부족합니다. 더 긴 기간을 선택해 주세요.")
    else:
        price_diff = predicted_price - current_price
        pct_diff   = (price_diff / current_price) * 100

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("현재가", f"${current_price:.2f}")
        with col_p2:
            st.metric(f"{forecast_days}일 후 예측가",
                      f"${predicted_price:.2f}",
                      f"{'+' if price_diff>=0 else ''}{price_diff:.2f} ({pct_diff:+.1f}%)")
        with col_p3:
            direction = "📈 상승" if price_diff >= 0 else "📉 하락"
            st.metric("예측 방향", direction)

        # 신뢰 구간 (ATR 기반)
        atr = df["ATR"].iloc[-1]
        upper_ci = predicted_price + 1.5 * atr
        lower_ci = predicted_price - 1.5 * atr

        st.markdown(f"""
        <div class="info-box">
            📐 <b>예측 신뢰 구간</b> (±1.5×ATR):
            하단 <b>${lower_ci:.2f}</b> ~ 상단 <b>${upper_ci:.2f}</b>
            &nbsp;|&nbsp; ATR = {atr:.2f}
        </div>
        """, unsafe_allow_html=True)

        # 모델 성능
        st.markdown('<div class="section-header">📊 모델 성능 비교</div>', unsafe_allow_html=True)
        if metrics:
            perf_df = pd.DataFrame(metrics).T.round(4)
            perf_df.columns = ["평균절대오차 (MAE)", "결정계수 (R²)"]
            st.dataframe(perf_df, use_container_width=True)

        # 특성 중요도
        if hasattr(model, "feature_importances_"):
            features = ["MA20","MA50","RSI","MACD","MACD_signal","BB_upper","BB_lower","ATR","Returns","Returns_5d","OBV"]
            imp_df = pd.DataFrame({
                "특성": features,
                "중요도": model.feature_importances_
            }).sort_values("중요도", ascending=True)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["중요도"], y=imp_df["특성"],
                orientation="h",
                marker=dict(
                    color=imp_df["중요도"],
                    colorscale=[[0, "#1e3a5f"], [1, "#3b82f6"]]
                )
            ))
            fig_imp.update_layout(
                title="특성 중요도",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.8)",
                font=dict(color="#94a3b8"),
                height=350,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            ⚠️ ML 예측은 과거 패턴 기반이며, 뉴스·지정학적 이벤트·OPEC 결정 등 
            돌발 변수는 반영되지 않습니다. 참고 지표로만 활용하세요.
        </div>
        """, unsafe_allow_html=True)


# ──────────────── TAB 4: 연관 시장 ────────────────
with tab4:
    st.markdown('<div class="section-header">🌐 연관 지표 현황</div>', unsafe_allow_html=True)

    with st.spinner("연관 데이터 불러오는 중..."):
        related = load_related_data()

    if related:
        cols = st.columns(len(related))
        for i, (name, (price, chg)) in enumerate(related.items()):
            with cols[i]:
                st.metric(name, f"{price:.2f}", f"{'+' if chg>=0 else ''}{chg:.2f}%")

    # 상관관계 분석
    st.markdown('<div class="section-header">📊 상관관계 분석</div>', unsafe_allow_html=True)

    with st.spinner("상관관계 계산 중..."):
        corr_tickers = {"WTI (CL=F)": "CL=F", "브렌트 (BZ=F)": "BZ=F",
                        "달러인덱스": "DX-Y.NYB", "금": "GC=F", "천연가스": "NG=F"}
        price_data = {}
        for name, tkr in corr_tickers.items():
            try:
                d = yf.download(tkr, period="1y", auto_adjust=True, progress=False)
                if not d.empty:
                    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
                    price_data[name] = d["Close"].squeeze()
            except:
                pass

    if len(price_data) >= 2:
        corr_df = pd.DataFrame(price_data).pct_change().dropna().corr().round(3)
        fig_corr = go.Figure(go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale=[[0, "#ef4444"], [0.5, "#0f172a"], [1, "#10b981"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_df.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=13),
        ))
        fig_corr.update_layout(
            title="수익률 상관계수 행렬 (최근 1년)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.8)",
            font=dict(color="#94a3b8"),
            height=400,
            margin=dict(t=50, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            💡 <b>해석 팁</b>: 달러인덱스와 원유는 일반적으로 <b>음의 상관관계</b>(-0.3 ~ -0.7).
            달러 강세 시 원유 약세 경향. 금과는 <b>양의 상관관계</b>를 보이는 경우가 많습니다.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("상관관계 분석을 위한 데이터가 부족합니다.")


# ─── 푸터 ───
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem; padding:1rem 0'>
    🛢️ 유가 예측 대시보드 &nbsp;|&nbsp; 데이터: Yahoo Finance &nbsp;|&nbsp;
    Built with Streamlit · Plotly · scikit-learn
</div>
""", unsafe_allow_html=True)
