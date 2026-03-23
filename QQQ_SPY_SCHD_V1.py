# streamlit_app.py
# ============================================================
# QQQ / SPY / SCHD
# 50MA / 200MA Trend Analysis + Buy Timing Dashboard
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
# ============================================================

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="QQQ / SPY / SCHD MA Analysis",
    layout="wide"
)

st.title("QQQ / SPY / SCHD - 50 / 200 MA Analysis")
st.caption("Trend analysis, cross events, buy timing scoring, and simple backtest")


# ============================================================
# User inputs
# ============================================================
with st.sidebar:
    st.header("Settings")

    default_tickers = ["QQQ", "SPY", "SCHD"]
    tickers_input = st.text_input(
        "Tickers (comma separated)",
        value="QQQ,SPY,SCHD"
    )
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

    period = st.selectbox(
        "Download period",
        ["3y", "5y", "10y", "max"],
        index=1
    )

    short_window = st.number_input("Short MA", min_value=5, max_value=300, value=50, step=1)
    long_window = st.number_input("Long MA", min_value=20, max_value=500, value=200, step=1)

    if short_window >= long_window:
        st.warning("Short MA should be smaller than Long MA.")

    lookback_signal = st.selectbox(
        "Signal chart lookback",
        ["6M", "1Y", "3Y", "5Y", "ALL"],
        index=2
    )

    st.markdown("---")
    st.subheader("Buy scoring logic")

    weight_price_above_ma200 = st.number_input("Price > MA200 score", value=2, step=1)
    weight_ma50_above_ma200 = st.number_input("MA50 > MA200 score", value=2, step=1)
    weight_price_above_ma50 = st.number_input("Price > MA50 score", value=1, step=1)
    weight_ma50_slope_up = st.number_input("MA50 slope up score", value=1, step=1)
    weight_not_overextended = st.number_input("Not too extended score", value=1, step=1)
    penalty_too_extended = st.number_input("Too extended penalty", value=1, step=1)

    overextended_threshold = st.slider(
        "Overextended threshold vs MA50 (%)",
        min_value=1.0,
        max_value=20.0,
        value=8.0,
        step=0.5
    )

    st.markdown("---")
    show_markers = st.checkbox("Show cross markers", value=True)
    show_buy_points = st.checkbox("Show buy candidate markers", value=True)

    st.markdown("---")
    st.subheader("Simple backtest")
    initial_capital = st.number_input("Initial capital", min_value=1000, value=10000, step=1000)
    backtest_ticker = st.selectbox("Backtest ticker", tickers if tickers else default_tickers, index=0)


# ============================================================
# Helpers
# ============================================================
@st.cache_data(show_spinner=False)
def download_price_data(tickers: List[str], period: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )

    if df.empty:
        return df

    # Normalize output shape
    # For single ticker, yfinance sometimes returns single-level columns
    if len(tickers) == 1 and not isinstance(df.columns, pd.MultiIndex):
        t = tickers[0]
        df.columns = pd.MultiIndex.from_product([[t], df.columns])

    return df


def get_single_ticker_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float)

    try:
        s = raw[ticker]["Close"].copy()
    except Exception:
        return pd.Series(dtype=float)

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = ticker
    return s


def prepare_ticker_df(close: pd.Series, short_w: int, long_w: int) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df["Close"] = close
    df["MA_Short"] = close.rolling(short_w).mean()
    df["MA_Long"] = close.rolling(long_w).mean()

    df["Price_vs_MA_Short_%"] = (df["Close"] / df["MA_Short"] - 1.0) * 100.0
    df["Price_vs_MA_Long_%"] = (df["Close"] / df["MA_Long"] - 1.0) * 100.0
    df["MA_Short_vs_MA_Long_%"] = (df["MA_Short"] / df["MA_Long"] - 1.0) * 100.0

    df["MA_Short_Slope"] = df["MA_Short"].diff(5)
    df["MA_Long_Slope"] = df["MA_Long"].diff(5)

    df["GoldenCross"] = (
        (df["MA_Short"] > df["MA_Long"]) &
        (df["MA_Short"].shift(1) <= df["MA_Long"].shift(1))
    )

    df["DeadCross"] = (
        (df["MA_Short"] < df["MA_Long"]) &
        (df["MA_Short"].shift(1) >= df["MA_Long"].shift(1))
    )

    df["Above_MA_Short"] = df["Close"] > df["MA_Short"]
    df["Above_MA_Long"] = df["Close"] > df["MA_Long"]
    df["MA_Short_Above_MA_Long"] = df["MA_Short"] > df["MA_Long"]
    df["MA_Short_Slope_Up"] = df["MA_Short_Slope"] > 0

    return df


def score_latest_row(df: pd.DataFrame, overextended_threshold_pct: float) -> Tuple[int, Dict[str, int]]:
    latest = df.iloc[-1]
    score_parts = {
        "price_above_ma200": 0,
        "ma50_above_ma200": 0,
        "price_above_ma50": 0,
        "ma50_slope_up": 0,
        "not_overextended": 0,
        "overextended_penalty": 0,
    }

    if pd.notna(latest["MA_Long"]) and latest["Close"] > latest["MA_Long"]:
        score_parts["price_above_ma200"] = weight_price_above_ma200

    if pd.notna(latest["MA_Short"]) and pd.notna(latest["MA_Long"]) and latest["MA_Short"] > latest["MA_Long"]:
        score_parts["ma50_above_ma200"] = weight_ma50_above_ma200

    if pd.notna(latest["MA_Short"]) and latest["Close"] > latest["MA_Short"]:
        score_parts["price_above_ma50"] = weight_price_above_ma50

    if pd.notna(latest["MA_Short_Slope"]) and latest["MA_Short_Slope"] > 0:
        score_parts["ma50_slope_up"] = weight_ma50_slope_up

    if pd.notna(latest["Price_vs_MA_Short_%"]):
        if latest["Price_vs_MA_Short_%"] <= overextended_threshold_pct:
            score_parts["not_overextended"] = weight_not_overextended
        else:
            score_parts["overextended_penalty"] = -penalty_too_extended

    total = sum(score_parts.values())
    return total, score_parts


def classify_state(df: pd.DataFrame, score: int) -> str:
    latest = df.iloc[-1]

    if pd.isna(latest["MA_Long"]) or pd.isna(latest["MA_Short"]):
        return "Not enough data"

    if (
        latest["Close"] > latest["MA_Long"]
        and latest["MA_Short"] > latest["MA_Long"]
        and latest["Close"] > latest["MA_Short"]
        and latest["MA_Short_Slope"] > 0
    ):
        if latest["Price_vs_MA_Short_%"] > overextended_threshold:
            return "Strong Uptrend (Extended)"
        return "Strong Uptrend"

    if (
        latest["Close"] > latest["MA_Long"]
        and latest["MA_Short"] > latest["MA_Long"]
        and latest["Close"] <= latest["MA_Short"]
    ):
        return "Pullback in Uptrend"

    if (
        latest["Close"] < latest["MA_Long"]
        and latest["MA_Short"] < latest["MA_Long"]
    ):
        return "Weak / Below 200MA"

    if (
        latest["Close"] > latest["MA_Long"]
        and latest["MA_Short"] <= latest["MA_Long"]
    ) or (
        latest["Close"] < latest["MA_Long"]
        and latest["MA_Short"] > latest["MA_Long"]
    ):
        return "Potential Reversal"

    if score >= 5:
        return "Buy-Friendly"
    if score >= 3:
        return "Watch / Partial Buy"
    return "Wait"


def latest_cross_dates(df: pd.DataFrame) -> Tuple[str, str]:
    golden_dates = df.index[df["GoldenCross"]].tolist()
    dead_dates = df.index[df["DeadCross"]].tolist()

    last_golden = golden_dates[-1].strftime("%Y-%m-%d") if golden_dates else "-"
    last_dead = dead_dates[-1].strftime("%Y-%m-%d") if dead_dates else "-"
    return last_golden, last_dead


def buy_candidate_signal(df: pd.DataFrame) -> pd.Series:
    """
    A practical buy candidate:
    1) Price > MA_Long
    2) MA_Short > MA_Long
    3) Yesterday price <= MA_Short and today price > MA_Short
    4) MA_Short slope positive
    """
    signal = (
        (df["Close"] > df["MA_Long"]) &
        (df["MA_Short"] > df["MA_Long"]) &
        (df["Close"].shift(1) <= df["MA_Short"].shift(1)) &
        (df["Close"] > df["MA_Short"]) &
        (df["MA_Short_Slope"] > 0)
    )
    return signal.fillna(False)


def apply_lookback(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return df

    end_date = df.index.max()

    if mode == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif mode == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif mode == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif mode == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        return df

    return df[df.index >= start_date]


def make_price_chart(df: pd.DataFrame, ticker: str, state: str) -> go.Figure:
    plot_df = apply_lookback(df, lookback_signal)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Close"],
        mode="lines",
        name=f"{ticker} Close",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["MA_Short"],
        mode="lines",
        name=f"MA{short_window}",
        line=dict(width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["MA_Long"],
        mode="lines",
        name=f"MA{long_window}",
        line=dict(width=2, dash="dash")
    ))

    if show_markers:
        golden = plot_df[plot_df["GoldenCross"]]
        dead = plot_df[plot_df["DeadCross"]]

        if not golden.empty:
            fig.add_trace(go.Scatter(
                x=golden.index,
                y=golden["Close"],
                mode="markers",
                name="Golden Cross",
                marker=dict(symbol="triangle-up", size=10)
            ))

        if not dead.empty:
            fig.add_trace(go.Scatter(
                x=dead.index,
                y=dead["Close"],
                mode="markers",
                name="Dead Cross",
                marker=dict(symbol="triangle-down", size=10)
            ))

    if show_buy_points:
        buys = plot_df[plot_df["BuyCandidate"]]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers",
                name="Buy Candidate",
                marker=dict(symbol="star", size=11)
            ))

    fig.update_layout(
        title=f"{ticker} Price / MA Analysis - {state}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig


def make_relative_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    plot_df = apply_lookback(df, lookback_signal)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Price_vs_MA_Short_%"],
        mode="lines",
        name="Price vs MA50 (%)"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Price_vs_MA_Long_%"],
        mode="lines",
        name="Price vs MA200 (%)"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["MA_Short_vs_MA_Long_%"],
        mode="lines",
        name="MA50 vs MA200 (%)"
    ))

    fig.add_hline(y=0, line_dash="dash")
    fig.add_hline(y=overextended_threshold, line_dash="dot")
    fig.add_hline(y=-overextended_threshold, line_dash="dot")

    fig.update_layout(
        title=f"{ticker} Relative Distance Analysis",
        xaxis_title="Date",
        yaxis_title="Percent (%)",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig


def simple_strategy_backtest(df: pd.DataFrame, initial_capital: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simple regime strategy:
    Invest when Close > MA_Long
    Stay in cash when Close <= MA_Long
    """
    bt = df.copy().dropna(subset=["Close", "MA_Long"]).copy()
    if bt.empty:
        return pd.DataFrame(), {}

    bt["DailyReturn"] = bt["Close"].pct_change().fillna(0.0)
    bt["Signal"] = (bt["Close"] > bt["MA_Long"]).astype(int)
    bt["StrategyReturn"] = bt["Signal"].shift(1).fillna(0) * bt["DailyReturn"]

    bt["BuyHoldEquity"] = initial_capital * (1 + bt["DailyReturn"]).cumprod()
    bt["StrategyEquity"] = initial_capital * (1 + bt["StrategyReturn"]).cumprod()

    def calc_mdd(equity: pd.Series) -> float:
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        return dd.min() * 100.0

    bh_total = (bt["BuyHoldEquity"].iloc[-1] / initial_capital - 1.0) * 100.0
    st_total = (bt["StrategyEquity"].iloc[-1] / initial_capital - 1.0) * 100.0

    metrics = {
        "Buy & Hold Total Return (%)": bh_total,
        "Strategy Total Return (%)": st_total,
        "Buy & Hold MDD (%)": calc_mdd(bt["BuyHoldEquity"]),
        "Strategy MDD (%)": calc_mdd(bt["StrategyEquity"]),
        "Days Invested (%)": bt["Signal"].mean() * 100.0,
    }
    return bt, metrics


def make_backtest_chart(bt: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bt.index,
        y=bt["BuyHoldEquity"],
        mode="lines",
        name="Buy & Hold"
    ))
    fig.add_trace(go.Scatter(
        x=bt.index,
        y=bt["StrategyEquity"],
        mode="lines",
        name=f"Regime: Price > MA{long_window}"
    ))

    fig.update_layout(
        title=f"{ticker} Simple Backtest",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig


# ============================================================
# Main data load
# ============================================================
with st.spinner("Downloading market data..."):
    raw = download_price_data(tickers, period)

if raw.empty:
    st.error("Failed to download data. Please check ticker names or network connection.")
    st.stop()


# ============================================================
# Prepare all ticker data
# ============================================================
all_data: Dict[str, pd.DataFrame] = {}
summary_rows = []

for ticker in tickers:
    close = get_single_ticker_close(raw, ticker)
    if close.empty:
        continue

    df = prepare_ticker_df(close, short_window, long_window)
    df["BuyCandidate"] = buy_candidate_signal(df)

    score, score_parts = score_latest_row(df, overextended_threshold)
    state = classify_state(df, score)
    last_golden, last_dead = latest_cross_dates(df)

    latest = df.iloc[-1]

    summary_rows.append({
        "Ticker": ticker,
        "Close": round(float(latest["Close"]), 2) if pd.notna(latest["Close"]) else np.nan,
        f"MA{short_window}": round(float(latest["MA_Short"]), 2) if pd.notna(latest["MA_Short"]) else np.nan,
        f"MA{long_window}": round(float(latest["MA_Long"]), 2) if pd.notna(latest["MA_Long"]) else np.nan,
        "Price vs MA50 (%)": round(float(latest["Price_vs_MA_Short_%"]), 2) if pd.notna(latest["Price_vs_MA_Short_%"]) else np.nan,
        "Price vs MA200 (%)": round(float(latest["Price_vs_MA_Long_%"]), 2) if pd.notna(latest["Price_vs_MA_Long_%"]) else np.nan,
        "MA50 vs MA200 (%)": round(float(latest["MA_Short_vs_MA_Long_%"]), 2) if pd.notna(latest["MA_Short_vs_MA_Long_%"]) else np.nan,
        "Score": score,
        "State": state,
        "Last Golden Cross": last_golden,
        "Last Dead Cross": last_dead,
        "Buy Candidate Today": bool(df["BuyCandidate"].iloc[-1]),
    })

    df.attrs["score"] = score
    df.attrs["score_parts"] = score_parts
    df.attrs["state"] = state
    all_data[ticker] = df


if not all_data:
    st.error("No valid ticker data available.")
    st.stop()


# ============================================================
# Summary table
# ============================================================
st.subheader("1) Current Summary")

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by=["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)
st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ============================================================
# Regime filter interpretation
# ============================================================
st.subheader("2) Interpretation")

for ticker in tickers:
    if ticker not in all_data:
        continue

    df = all_data[ticker]
    score = df.attrs["score"]
    state = df.attrs["state"]
    score_parts = df.attrs["score_parts"]
    latest = df.iloc[-1]

    with st.expander(f"{ticker} - detailed interpretation", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**State:** {state}")
            st.markdown(f"**Score:** {score}")
            st.write("Score breakdown:")
            st.json(score_parts)

        with col2:
            buy_text = "YES" if df["BuyCandidate"].iloc[-1] else "NO"
            st.markdown(f"**Buy candidate today:** {buy_text}")
            st.markdown(f"**Close:** {latest['Close']:.2f}")
            if pd.notna(latest["MA_Short"]):
                st.markdown(f"**MA{short_window}:** {latest['MA_Short']:.2f}")
            if pd.notna(latest["MA_Long"]):
                st.markdown(f"**MA{long_window}:** {latest['MA_Long']:.2f}")

        st.info(
            "Suggested reading:\n"
            "- Strong Uptrend: trend-following is favorable\n"
            "- Pullback in Uptrend: good area for staged buying\n"
            "- Potential Reversal: watch closely, confirmation needed\n"
            "- Weak / Below 200MA: defensive stance"
        )


# ============================================================
# Charts per ticker
# ============================================================
st.subheader("3) Price / Moving Average Charts")

for ticker in tickers:
    if ticker not in all_data:
        continue

    df = all_data[ticker]
    state = df.attrs["state"]

    fig = make_price_chart(df, ticker, state)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = make_relative_chart(df, ticker)
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# Buy timing table
# ============================================================
st.subheader("4) Recent Buy Candidate Signals")

signal_rows = []
for ticker in tickers:
    if ticker not in all_data:
        continue

    df = all_data[ticker]
    recent_signals = df[df["BuyCandidate"]].tail(10)

    for idx, row in recent_signals.iterrows():
        signal_rows.append({
            "Ticker": ticker,
            "Date": idx.strftime("%Y-%m-%d"),
            "Close": round(float(row["Close"]), 2),
            f"MA{short_window}": round(float(row["MA_Short"]), 2) if pd.notna(row["MA_Short"]) else np.nan,
            f"MA{long_window}": round(float(row["MA_Long"]), 2) if pd.notna(row["MA_Long"]) else np.nan,
            "Price vs MA50 (%)": round(float(row["Price_vs_MA_Short_%"]), 2) if pd.notna(row["Price_vs_MA_Short_%"]) else np.nan,
            "Price vs MA200 (%)": round(float(row["Price_vs_MA_Long_%"]), 2) if pd.notna(row["Price_vs_MA_Long_%"]) else np.nan,
        })

if signal_rows:
    signal_df = pd.DataFrame(signal_rows).sort_values(by=["Date", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    st.dataframe(signal_df, use_container_width=True, hide_index=True)
else:
    st.write("No recent buy candidate signals found.")


# ============================================================
# Compare normalized performance
# ============================================================
st.subheader("5) Normalized Comparison")

norm_fig = go.Figure()

for ticker in tickers:
    if ticker not in all_data:
        continue

    plot_df = apply_lookback(all_data[ticker], lookback_signal).copy()
    if plot_df.empty:
        continue

    norm = plot_df["Close"] / plot_df["Close"].iloc[0] * 100.0
    norm_fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=norm,
        mode="lines",
        name=ticker
    ))

norm_fig.update_layout(
    title="Normalized Price Comparison (Start = 100)",
    xaxis_title="Date",
    yaxis_title="Normalized Index",
    height=450,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
)
st.plotly_chart(norm_fig, use_container_width=True)


# ============================================================
# Backtest section
# ============================================================
st.subheader("6) Simple Backtest")

if backtest_ticker in all_data:
    bt_df, bt_metrics = simple_strategy_backtest(all_data[backtest_ticker], initial_capital)

    if bt_df.empty:
        st.warning("Backtest could not be computed due to insufficient data.")
    else:
        metric_cols = st.columns(len(bt_metrics))
        for i, (k, v) in enumerate(bt_metrics.items()):
            metric_cols[i].metric(k, f"{v:.2f}")

        bt_fig = make_backtest_chart(bt_df, backtest_ticker)
        st.plotly_chart(bt_fig, use_container_width=True)

        with st.expander("Backtest data"):
            st.dataframe(
                bt_df[["Close", "MA_Long", "Signal", "BuyHoldEquity", "StrategyEquity"]].tail(200),
                use_container_width=True
            )


# ============================================================
# Strategy notes
# ============================================================
st.subheader("7) Practical Strategy Notes")

st.markdown(f"""
**How this dashboard reads the market:**

1. **Price > MA{long_window}**
   - Long-term trend is constructive.

2. **MA{short_window} > MA{long_window}**
   - Medium-term trend is stronger than long-term trend.

3. **Price falls near MA{short_window} and then reclaims it**
   - Often a practical pullback entry in an uptrend.

4. **Too far above MA{short_window}**
   - Can mean trend is strong, but entry may be late and extended.

5. **Price < MA{long_window} and MA{short_window} < MA{long_window}**
   - Usually a weak trend structure. Better to be selective.

**Typical interpretation by ETF:**
- **QQQ**: aggressive trend-following
- **SPY**: market regime filter
- **SCHD**: steadier pullback accumulation
""")


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("Built for practical MA-based trend and buy timing analysis.")
