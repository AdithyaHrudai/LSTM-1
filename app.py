# =============================================================================
# StockSense AI  —  Indian NSE Stock Prediction & Analysis Platform
# =============================================================================
# All logic lives in modules; this file is layout + wiring.
# Interface modeled after the reference single-file app with tab-based layout.
# =============================================================================

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from config import (
    STOCK_UNIVERSE, TICKERS, CURRENCY_SYMBOL, SEQUENCE_LENGTH,
    get_display_name, get_sector, get_logo_url, get_search_term,
)
from technical import compute_all_indicators
from sentiment import (
    fetch_news, analyze_sentiment,
    compute_market_sentiment, generate_sentiment_summary,
)
from strategy import evaluate_strategy, compute_trigger_prices
from model import (
    load_lstm_model, get_scalers_for_ticker, prepare_data_for_prediction,
    make_prediction, make_7day_prediction,
)
from glossary import render_glossary

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="StockSense AI", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .block-container{padding-top:1rem;}
    .big-metric{font-size:2.2rem;font-weight:700;}

    /* Cards — always light background with dark text (works in both themes) */
    .card{background:#f0f2f6;padding:18px;border-radius:10px;border-left:5px solid #1f77b4;margin-bottom:10px;color:#0e1117 !important;}
    .card h2, .card h3, .card b, .card p, .card span, .card small, .card a, .card br{color:#0e1117 !important;}
    .card a{color:#1565c0 !important;text-decoration:underline;}
    .card small{color:#333 !important;}
    .card-green{background:#d4edda;border-left-color:#28a745;}
    .card-green b, .card-green p, .card-green span{color:#155724 !important;}
    .card-red{background:#f8d7da;border-left-color:#dc3545;}
    .card-red b, .card-red p, .card-red span{color:#721c24 !important;}
    .card-yellow{background:#fff3cd;border-left-color:#ffc107;}
    .card-yellow b, .card-yellow p, .card-yellow span{color:#856404 !important;}

    /* Sentiment labels — high contrast colors that work on any bg */
    .sentiment-pos{color:#1b8a2a !important;font-weight:700;}
    .sentiment-neg{color:#c0392b !important;font-weight:700;}
    .sentiment-neu{color:#7f8c8d !important;font-weight:700;}

    .tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.82em;font-weight:600;margin:2px;}
    .tag-buy{background:#d4edda;color:#155724;}
    .tag-sell{background:#f8d7da;color:#721c24;}
    .tag-hold{background:#fff3cd;color:#856404;}

    .footer-text{color:#9e9e9e !important;}
</style>
""", unsafe_allow_html=True)

S = CURRENCY_SYMBOL  # shorthand


# ─────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────
def fmt(num):
    if num is None or num == "N/A":
        return "N/A"
    num = float(num)
    if abs(num) >= 1e12: return f"{S}{num/1e12:.2f}T"
    if abs(num) >= 1e9:  return f"{S}{num/1e9:.2f}B"
    if abs(num) >= 1e7:  return f"{S}{num/1e7:.2f}Cr"
    if abs(num) >= 1e5:  return f"{S}{num/1e5:.2f}L"
    if abs(num) >= 1e3:  return f"{S}{num/1e3:.2f}K"
    return f"{S}{num:.2f}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None, "No data available for this ticker"
        return data, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600, show_spinner=False)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'name': info.get('longName', info.get('shortName', get_display_name(ticker))),
            'sector': info.get('sector', get_sector(ticker)),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'dividend_yield': info.get('dividendYield'),
            '52w_high': info.get('fiftyTwoWeekHigh'),
            '52w_low': info.get('fiftyTwoWeekLow'),
            'avg_volume': info.get('averageVolume'),
            'beta': info.get('beta'),
            'eps': info.get('trailingEps'),
            'currency': info.get('currency', 'INR'),
        }
    except Exception:
        return None


def compute_risk_metrics(df):
    returns = df["Close"].pct_change().dropna()
    ann_vol = float(returns.std() * np.sqrt(252) * 100)
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min() * 100)
    mean_ret = float(returns.mean() * 252)
    sharpe = float((mean_ret - 0.06) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0.0
    return dict(volatility=ann_vol, max_drawdown=max_dd, sharpe=sharpe)


def compute_fibonacci(df, window=120):
    recent = df.tail(window)
    hi = float(recent["High"].max())
    lo = float(recent["Low"].min())
    diff = hi - lo
    return {
        "0% (High)": hi, "23.6%": hi - 0.236*diff, "38.2%": hi - 0.382*diff,
        "50.0%": hi - 0.5*diff, "61.8%": hi - 0.618*diff,
        "78.6%": hi - 0.786*diff, "100% (Low)": lo,
    }


def sentiment_label(score):
    if score >= 0.05:  return "Positive", "sentiment-pos"
    if score <= -0.05: return "Negative", "sentiment-neg"
    return "Neutral", "sentiment-neu"


# ─────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────
def chart_candlestick_volume(df, n=120, title="Price & Volume"):
    d = df.tail(n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                  low=d["Low"], close=d["Close"], name="OHLC"), row=1, col=1)
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=colors,
                         name="Volume", opacity=0.5), row=2, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False,
                      height=520, template="plotly_white", showlegend=False)
    fig.update_yaxes(title_text=f"Price ({S})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def chart_rsi(df, n=120):
    d = df.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI_14"], name="RSI", line=dict(color="#7b1fa2", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.07)
    fig.update_layout(title="RSI (14)", height=280, template="plotly_white", yaxis_range=[0, 100])
    return fig


def chart_macd(df, n=120):
    d = df.tail(n)
    fig = go.Figure()
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in d["MACD_hist"]]
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"], name="Histogram", marker_color=colors, opacity=0.5))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_line"], name="MACD", line=dict(color="#1976d2", width=2)))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], name="Signal", line=dict(color="#ff9800", width=2)))
    fig.update_layout(title="MACD (12, 26, 9)", height=300, template="plotly_white")
    return fig


def chart_bollinger(df, n=120):
    d = df.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_upper"], name="Upper", line=dict(color="#aaa", dash="dash")))
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_lower"], name="Lower", line=dict(color="#aaa", dash="dash"),
                             fill="tonexty", fillcolor="rgba(173,216,230,0.15)"))
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_middle"], name="SMA20", line=dict(color="#ff9800", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close", line=dict(color="#1976d2", width=2)))
    fig.update_layout(title="Bollinger Bands (20, 2)", height=400, template="plotly_white")
    return fig


def chart_ema(df, n=250):
    d = df.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close", line=dict(color="#333", width=1.5)))
    for span, clr in [(12, "#42a5f5"), (26, "#66bb6a"), (50, "#ffa726")]:
        col = f"EMA_{span}"
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=f"EMA {span}", line=dict(color=clr, width=1.5)))
    if "SMA_200" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA_200"], name="SMA 200", line=dict(color="#ef5350", width=1.5)))
    fig.update_layout(title="EMA Crossovers", height=400, template="plotly_white")
    return fig


def chart_stochastic(df, n=120):
    d = df.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Stoch_K"], name="%K", line=dict(color="#1976d2", width=2)))
    fig.add_trace(go.Scatter(x=d.index, y=d["Stoch_D"], name="%D", line=dict(color="#ff9800", width=2)))
    fig.add_hline(y=80, line_dash="dash", line_color="red")
    fig.add_hline(y=20, line_dash="dash", line_color="green")
    fig.update_layout(title="Stochastic Oscillator (14, 3)", height=280, template="plotly_white", yaxis_range=[0, 100])
    return fig


def chart_adx(df, n=120):
    d = df.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["ADX_14"], name="ADX", line=dict(color="#333", width=2)))
    fig.add_hline(y=25, line_dash="dot", line_color="gray", annotation_text="Trend threshold")
    fig.update_layout(title="ADX - Trend Strength", height=300, template="plotly_white")
    return fig


def chart_prediction(hist_close, pred_prices, dates_future, title="LSTM Forecast"):
    fig = go.Figure()
    hist = hist_close.tail(90)
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color="#1976d2", width=2)))
    fig.add_trace(go.Scatter(x=dates_future, y=pred_prices, name="Predicted",
                             line=dict(color="#f57c00", width=2, dash="dash")))
    upper = pred_prices * np.array([1 + 0.005*(i+1) for i in range(len(pred_prices))])
    lower = pred_prices * np.array([1 - 0.005*(i+1) for i in range(len(pred_prices))])
    fig.add_trace(go.Scatter(x=list(dates_future)+list(dates_future)[::-1],
                             y=list(upper)+list(lower)[::-1],
                             fill="toself", fillcolor="rgba(255,152,0,0.12)",
                             line=dict(width=0), name="Confidence Band"))
    fig.add_vline(x=hist.index[-1], line_dash="dot", line_color="gray")
    fig.update_layout(title=title, height=420, template="plotly_white")
    return fig


def chart_sentiment_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain=dict(x=[0,1], y=[0,1]),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#1976d2"),
                   steps=[
                       dict(range=[0, 25], color="#ffcdd2"),
                       dict(range=[25, 45], color="#fff9c4"),
                       dict(range=[45, 55], color="#e0e0e0"),
                       dict(range=[55, 75], color="#c8e6c9"),
                       dict(range=[75, 100], color="#a5d6a7"),
                   ],
                   threshold=dict(line=dict(color="black", width=3), thickness=0.8, value=score)),
        title=dict(text="Overall Sentiment"),
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20))
    return fig


# ═════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════
def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 📈 StockSense AI")
        st.markdown("---")
        display_options = {f"{get_display_name(t)} ({t.replace('.NS','')})": t for t in TICKERS}
        company_label = st.selectbox("Select Company", list(display_options.keys()), index=0)
        ticker = display_options[company_label]
        company_name = get_display_name(ticker)
        st.caption(f"NSE Ticker: **{ticker}**")

        st.markdown("---")
        st.markdown("### 📚 Pages")
        page = st.radio("Go to:", ["📈 Stock Analysis", "📖 Glossary"],
                        index=0, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "**StockSense AI** combines LSTM deep learning, technical analysis "
            "(RSI, EMA, MACD, Bollinger, ADX, Stochastic), and NLP sentiment analysis "
            "for a comprehensive stock evaluation system."
        )
        st.warning(
            "**Disclaimer:** For educational & research purposes only. "
            "NOT financial advice. Always consult a certified advisor."
        )

    # ── Glossary page ──
    if page == "📖 Glossary":
        render_glossary()
        return

    # ── Fetch Data ──
    with st.spinner(f"Loading data for {company_name}..."):
        data_raw, err = fetch_stock_data(ticker)
    if err or data_raw is None:
        st.error(f"Could not fetch data for **{company_name}** ({ticker}). {err or ''}")
        return
    if len(data_raw) < SEQUENCE_LENGTH + 10:
        st.error(f"Insufficient data ({len(data_raw)} rows). Need >= {SEQUENCE_LENGTH + 10}.")
        return

    df = compute_all_indicators(data_raw)
    latest = df.iloc[-1]
    info = get_stock_info(ticker)

    current_price = float(latest["Close"])
    prev_close = float(df["Close"].iloc[-2])
    daily_chg = current_price - prev_close
    daily_pct = (daily_chg / prev_close) * 100

    # ── Header ──
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        name_disp = info["name"] if info else company_name
        st.markdown(f"# {name_disp}")
        sector = info["sector"] if info else get_sector(ticker)
        industry = info["industry"] if info else ""
        st.caption(f"{ticker}  •  {sector}  •  {industry}")
    with hc2:
        st.metric("Live Price", f"{S}{current_price:,.2f}", f"{daily_chg:+,.2f} ({daily_pct:+.2f}%)")

    # ── Load model & sentiment (needed across tabs) ──
    model, per_stock_scalers, model_info, model_error = load_lstm_model()

    with st.spinner("Analyzing market sentiment..."):
        news_articles, news_err = fetch_news(ticker, company_name)
        news_texts = [a['title'] for a in news_articles]
        news_sentiments = analyze_sentiment(news_texts)
        sentiment_score, sentiment_label_str, sentiment_emoji, news_agg = \
            compute_market_sentiment(news_sentiments)

    # Prepare predictions
    sentiment_compound = (sentiment_score - 50) / 50
    predicted_price = None
    predictions_7d = None
    X_test = None
    if model is not None and not model_error and per_stock_scalers is not None:
        feature_scaler, target_scaler = get_scalers_for_ticker(per_stock_scalers, ticker)
        X_test = prepare_data_for_prediction(df, feature_scaler, sentiment_compound)
        if X_test is not None:
            predicted_price = make_prediction(model, X_test, target_scaler, df)
            predictions_7d = make_7day_prediction(model, X_test, target_scaler, df, sentiment_score)

    # Compute strategy (EMA 20 + RSI 14)
    rsi_val = float(latest.get('RSI_14', 50))
    ema_20_val = float(latest.get('EMA_20', current_price))
    atr_val = float(latest.get('ATR_14', 0))
    prev_ema_20 = float(df['EMA_20'].iloc[-2]) if len(df) >= 2 and 'EMA_20' in df.columns else ema_20_val
    vol = float(latest.get('Volume', 0))
    avg_vol = float(latest.get('Vol_SMA_20', 0))

    strategy_result = evaluate_strategy(
        close=current_price, ema_20=ema_20_val, prev_ema_20=prev_ema_20,
        rsi=rsi_val, volume=vol, avg_volume=avg_vol,
    )
    triggers = compute_trigger_prices(current_price, atr_val, rsi_val)

    # ── Tabs ──
    tab_ov, tab_tech, tab_pred, tab_strat, tab_sent = st.tabs([
        "📊 Overview", "📈 Technical Analysis", "🤖 AI Prediction",
        "🎯 Trading Strategy", "📰 Sentiment Analysis",
    ])

    # ╔════════════════════════════════════╗
    # ║  TAB 1 – OVERVIEW                 ║
    # ╚════════════════════════════════════╝
    with tab_ov:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Market Cap", fmt(info["market_cap"]) if info and info["market_cap"] else "N/A")
        m2.metric("P/E Ratio", f"{info['pe_ratio']:.2f}" if info and info.get("pe_ratio") else "N/A")
        m3.metric("P/B Ratio", f"{info['pb_ratio']:.2f}" if info and info.get("pb_ratio") else "N/A")
        m4.metric("EPS", f"{S}{info['eps']:.2f}" if info and info.get("eps") else "N/A")
        m5.metric("Beta", f"{info['beta']:.2f}" if info and info.get("beta") else "N/A")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📋 Price Summary")
            ma7  = float(df["Close"].tail(7).mean())
            ma30 = float(df["Close"].tail(30).mean())
            ma90 = float(df["Close"].tail(90).mean())
            summary = pd.DataFrame({
                "Metric": ["Current Price", "Previous Close", "Day Change",
                           "7-Day MA", "30-Day MA", "90-Day MA",
                           "52-Week High", "52-Week Low", "Avg Volume"],
                "Value": [
                    f"{S}{current_price:,.2f}", f"{S}{prev_close:,.2f}",
                    f"{S}{daily_chg:+,.2f} ({daily_pct:+.2f}%)",
                    f"{S}{ma7:,.2f}", f"{S}{ma30:,.2f}", f"{S}{ma90:,.2f}",
                    f"{S}{info['52w_high']:,.2f}" if info and info.get("52w_high") else "N/A",
                    f"{S}{info['52w_low']:,.2f}" if info and info.get("52w_low") else "N/A",
                    f"{info['avg_volume']:,.0f}" if info and info.get("avg_volume") else "N/A",
                ]
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("#### 📅 Period Returns")
            def ret(n):
                if len(df) < n: return None
                old = float(df["Close"].iloc[-n])
                return ((current_price - old) / old) * 100
            periods = {"1 Week": 5, "1 Month": 22, "3 Months": 66,
                       "6 Months": 132, "1 Year": 252, "3 Years": 756, "5 Years": 1260}
            ret_rows = [{"Period": lbl, "Return": f"{ret(n):+.2f}%" if ret(n) is not None else "N/A"}
                        for lbl, n in periods.items()]
            st.dataframe(pd.DataFrame(ret_rows), hide_index=True, use_container_width=True)

            if info and info.get("52w_high") and info.get("52w_low"):
                rng = info["52w_high"] - info["52w_low"]
                pos = ((current_price - info["52w_low"]) / rng) if rng > 0 else 0.5
                st.markdown(f"**52-Week Range Position:** {pos*100:.1f}%")
                st.progress(min(max(pos, 0.0), 1.0))

        st.markdown("---")
        st.markdown("#### ⚡ Risk Metrics")
        rm = compute_risk_metrics(df)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Annualised Volatility", f"{rm['volatility']:.1f}%")
        rc2.metric("Max Drawdown", f"{rm['max_drawdown']:.1f}%")
        rc3.metric("Sharpe Ratio (rf=6%)", f"{rm['sharpe']:.2f}")

        st.markdown("---")
        st.plotly_chart(chart_candlestick_volume(df, n=180,
                        title=f"{company_name} - 6 Month Price & Volume"), use_container_width=True)

        with st.expander("📋 Recent Trading Data (last 30 days)"):
            recent = df[["Open","High","Low","Close","Volume"]].tail(30).sort_index(ascending=False).copy()
            recent.index = recent.index.strftime("%Y-%m-%d")
            for c in ["Open","High","Low","Close"]:
                recent[c] = recent[c].map(lambda x: f"{S}{x:,.2f}")
            recent["Volume"] = recent["Volume"].map(lambda x: f"{x:,.0f}")
            st.dataframe(recent, use_container_width=True)

    # ╔════════════════════════════════════╗
    # ║  TAB 2 – TECHNICAL ANALYSIS       ║
    # ╚════════════════════════════════════╝
    with tab_tech:
        st.markdown("### 📈 Technical Indicator Dashboard")
        st.markdown("#### Current Readings")

        ic1, ic2, ic3, ic4 = st.columns(4)
        rsi_val = float(latest.get("RSI_14", 0))
        ic1.metric("RSI (14)", f"{rsi_val:.1f}",
                    "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
        macd_val = float(latest.get('MACD_line', 0))
        macd_sig = float(latest.get('MACD_signal', 0))
        ic2.metric("MACD", f"{macd_val:.2f}",
                    "Bullish" if macd_val > macd_sig else "Bearish")
        adx_val = float(latest.get('ADX_14', 0))
        ic3.metric("ADX", f"{adx_val:.1f}",
                    "Strong Trend" if adx_val > 25 else "Weak/No Trend")
        atr_val = float(latest.get('ATR_14', 0))
        ic4.metric("ATR (14)", f"{S}{atr_val:.2f}")

        ic5, ic6, ic7, ic8 = st.columns(4)
        ema50 = float(latest.get('EMA_50', 0))
        ic5.metric("EMA 50", f"{S}{ema50:,.2f}",
                    "Above" if current_price > ema50 else "Below")
        sma200 = float(latest.get('SMA_200', 0))
        ic6.metric("SMA 200", f"{S}{sma200:,.2f}",
                    "Above" if current_price > sma200 else "Below")
        stoch_k = float(latest.get('Stoch_K', 0))
        ic7.metric("Stoch %K", f"{stoch_k:.1f}",
                    "Overbought" if stoch_k > 80 else ("Oversold" if stoch_k < 20 else "Neutral"))
        bb_width = float(latest.get('BB_width', 0))
        ic8.metric("BB Width", f"{bb_width:.3f}")

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(chart_rsi(df), use_container_width=True)
            st.plotly_chart(chart_macd(df), use_container_width=True)
            st.plotly_chart(chart_adx(df), use_container_width=True)
        with right:
            st.plotly_chart(chart_bollinger(df), use_container_width=True)
            st.plotly_chart(chart_ema(df), use_container_width=True)
            st.plotly_chart(chart_stochastic(df), use_container_width=True)

        with st.expander("📖 Indicator Interpretation Guide"):
            guide = pd.DataFrame({
                "Indicator": ["RSI","MACD","Bollinger Bands","EMA 50 / SMA 200","ADX","Stochastic","ATR","BB Width"],
                "What It Measures": [
                    "Momentum - overbought / oversold conditions",
                    "Trend direction & momentum via MA convergence",
                    "Volatility & mean-reversion zones",
                    "Trend direction; Golden Cross (EMA50>SMA200) = bullish",
                    "Trend strength (not direction); >25 = trending",
                    "Momentum oscillator; <20 oversold, >80 overbought",
                    "Average daily volatility in price terms",
                    "Bollinger Band width - narrow = squeeze (breakout imminent)",
                ],
                "Bullish Signal": [
                    "RSI < 30","MACD > Signal","Price at lower band","Price > both MAs",
                    "ADX rising > 25", "%K crosses %D below 20","Low ATR = breakout setup",
                    "Narrow width expanding",
                ],
                "Bearish Signal": [
                    "RSI > 70","MACD < Signal","Price at upper band","Price < both MAs",
                    "ADX falling < 20","%K crosses %D above 80","High ATR = volatile",
                    "Wide width contracting",
                ],
            })
            st.dataframe(guide, hide_index=True, use_container_width=True)

    # ╔════════════════════════════════════╗
    # ║  TAB 3 – AI PREDICTION            ║
    # ╚════════════════════════════════════╝
    with tab_pred:
        st.markdown("### 🤖 LSTM Deep-Learning Price Prediction")

        if model_error or model is None:
            st.warning(
                "No trained model found. Run `python train_model.py` to train the LSTM model."
            )
            st.info(
                f"The training script downloads 10 years of data for all 20 Indian NSE stocks, "
                f"computes technical indicators (RSI, MACD, Bollinger Bands), and trains a "
                f"universal LSTM model with {SEQUENCE_LENGTH}-day look-back and 9 features."
            )
        elif predicted_price is not None and predictions_7d is not None:
            chg1 = ((predicted_price - current_price) / current_price) * 100
            chg7 = ((predictions_7d[-1] - current_price) / current_price) * 100

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Predicted Tomorrow", f"{S}{predicted_price:,.2f}", f"{chg1:+.2f}%")
            pc2.metric("Predicted Day 3", f"{S}{predictions_7d[2]:,.2f}",
                        f"{((predictions_7d[2]-current_price)/current_price)*100:+.2f}%")
            pc3.metric("Predicted Day 7", f"{S}{predictions_7d[-1]:,.2f}", f"{chg7:+.2f}%")

            st.markdown("---")

            # How the prediction works (always visible, not in expander)
            st.markdown("""
<div class="card">
<b>How this prediction works:</b><br>
The LSTM neural network analyses the last 60 trading days of price data along with
technical indicators (RSI, MACD histogram, Bollinger Band width) and market sentiment.
The next-day prediction blends the LSTM output (70% weight) with short-term technical
momentum (30% weight), capped at a realistic ±3% daily move. The 7-day forecast extends
this using a multi-factor rolling approach that gradually shifts weight from the LSTM
signal toward momentum and mean-reversion as uncertainty grows further out.
</div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            last_date = df.index[-1]
            future_dates_7 = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
            st.plotly_chart(chart_prediction(
                df["Close"], np.array(predictions_7d), future_dates_7,
                title=f"{company_name} - 7-Day LSTM Forecast"
            ), use_container_width=True)

            st.markdown("#### 📅 7-Day Forecast Table")
            fdf = pd.DataFrame({
                "Date": future_dates_7.strftime("%Y-%m-%d (%a)"),
                f"Predicted ({S})": [f"{S}{p:,.2f}" for p in predictions_7d],
                "Change from Today": [f"{((p-current_price)/current_price)*100:+.2f}%" for p in predictions_7d],
            })
            st.dataframe(fdf, hide_index=True, use_container_width=True)

            n_features = model_info.get('expected_features', 9)
            metrics = model_info.get('metrics', {})
            overall = metrics.get('overall', {})

            with st.expander("🧠 Model Architecture & Training Details"):
                st.markdown(f"""
#### Universal Multi-Stock LSTM Model

This is **not** a per-stock model. A single LSTM network was trained on **all 20 Indian NSE stocks
simultaneously**, learning general price patterns across sectors (IT, Auto, FMCG, Pharma, Chemicals, etc.).
This gives it broader generalisation ability compared to per-stock models.

---

**Input Shape:** `(batch, {SEQUENCE_LENGTH}, {n_features})`

| Layer | Configuration |
|-------|--------------|
| LSTM 1 | 128 units, return_sequences=True |
| Dropout | 30% |
| LSTM 2 | 64 units |
| Dropout | 30% |
| Dense | 32 units, ReLU activation |
| Output | 1 unit (predicted close price) |

---

**9 Input Features (per timestep):**

| # | Feature | Source |
|---|---------|--------|
| 1-5 | Open, High, Low, Close, Volume | Raw OHLCV from yfinance |
| 6 | RSI (14) | `ta` library — momentum oscillator |
| 7 | MACD Histogram | `ta` library — trend momentum |
| 8 | Bollinger Band Width | `ta` library — volatility measure |
| 9 | Sentiment (10-day avg) | Synthetic price-action proxy during training; live VADER score at inference |

---

**Training Configuration:**
- **Data:** 2016-03-01 to 2026-03-01 (10 years, ~2000 trading days per stock)
- **Stocks:** 20 Indian NSE companies across diverse sectors
- **Sequence length:** {SEQUENCE_LENGTH} days look-back
- **Optimiser:** Adam (lr=0.0003) with ReduceLROnPlateau (factor=0.5, patience=8)
- **Loss:** Mean Squared Error (MSE)
- **Early stopping:** patience=20, restore best weights
- **Split:** 85% train / 15% validation (shuffled, seed=42)
- **Scaling:** MinMaxScaler(0,1) fitted on combined data from all stocks
""")
                if overall:
                    st.markdown(f"""
**Validation Performance:**
- RMSE: {S}{overall.get('rmse', 'N/A')} | MAE: {S}{overall.get('mae', 'N/A')}
- MAPE: {overall.get('mape', 'N/A')}% | R²: {overall.get('r2', 'N/A')}
""")
                st.markdown(f"""
---

**Prediction Pipeline:**

1. **Next-day:** Raw LSTM output is capped to ±3%/day, then blended — **70% LSTM + 30% technical momentum** when both agree on direction, **50/50** when they disagree
2. **7-day forecast:** Multi-signal rolling prediction:
   - Momentum (30%) + EMA trend (15%) + LSTM signal (15%, decaying) + Growth floor (15%) + Sentiment (10%) + Mean-reversion (5%) + Bounce detection (10%)
   - Daily moves capped at 1.5× historical volatility
   - Confidence band widens ±0.5%/day

**Model file:** `{model_info.get('model_file', 'N/A')}`
                """)
        else:
            st.warning("Could not prepare data for LSTM prediction. Need 60+ trading days with all indicators computed.")

    # ╔════════════════════════════════════╗
    # ║  TAB 4 – TRADING STRATEGY         ║
    # ╚════════════════════════════════════╝
    with tab_strat:
        st.markdown("### 🎯 EMA 20 + RSI 14 Strategy (Optimized)")

        sig = strategy_result['signal']
        strength = strategy_result['strength']
        confidence = strategy_result['confidence']
        reason = strategy_result['reason']

        # Color mapping
        if sig in ('STRONG BUY', 'BUY'):
            sig_css = 'card-green'
        elif sig in ('STRONG SELL', 'SELL'):
            sig_css = 'card-red'
        else:
            sig_css = 'card-yellow'

        # ── Signal Card ──
        st.markdown(f"""
<div class="card {sig_css}" style="text-align:center;">
    <h2 style="margin:0;">{sig}</h2>
    <p style="margin:4px 0 0;">Signal Strength: <b>{strength}</b>/100 &nbsp;|&nbsp;
    Confidence: <b>{confidence}</b></p>
</div>""", unsafe_allow_html=True)

        # Reason
        st.info(reason)

        # ── Indicator Cards ──
        st.markdown("---")
        st.markdown("#### 📊 Strategy Indicators")
        k1, k2, k3 = st.columns(3)

        # EMA 20 card
        ema_side = strategy_result['close_vs_ema']
        ema_css = 'card-green' if ema_side == 'Above' else 'card-red'
        k1.markdown(f"""
<div class="card {ema_css}">
    <b>EMA 20 (Trend Filter)</b><br>
    Price: <b>{ema_side}</b> EMA 20<br>
    EMA 20 = {S}{ema_20_val:,.2f}<br>
    Distance: <b>{strategy_result['ema_distance_pct']:+.2f}%</b><br>
    Direction: <b>{strategy_result['ema_trend']}</b>
</div>""", unsafe_allow_html=True)

        # RSI card
        rsi_display = strategy_result['rsi']
        if rsi_display > 70:
            rsi_zone, rsi_css = 'Overbought', 'card-red'
        elif rsi_display > 55:
            rsi_zone, rsi_css = 'Bullish ✅', 'card-green'
        elif rsi_display < 45:
            rsi_zone, rsi_css = 'Weak', 'card-red'
        else:
            rsi_zone, rsi_css = 'Neutral', 'card-yellow'
        k2.markdown(f"""
<div class="card {rsi_css}">
    <b>RSI 14 (Momentum)</b><br>
    RSI = <b>{rsi_display}</b><br>
    Zone: <b>{rsi_zone}</b><br>
    Buy zone: 55–70<br>
    Exit below: 45
</div>""", unsafe_allow_html=True)

        # Volume card
        vol_ratio = strategy_result.get('volume_ratio')
        vol_confirm = vol_ratio is not None and vol_ratio > 1.3
        if vol_ratio:
            vol_css = 'card-green' if vol_confirm else 'card-yellow'
            k3.markdown(f"""
<div class="card {vol_css}">
    <b>Volume Confirmation</b><br>
    Volume: <b>{vol/1e6:.1f}M</b><br>
    Ratio: <b>{vol_ratio:.2f}x avg</b><br>
    Confirms: <b>{'Yes ✅' if vol_confirm else 'No'}</b>
</div>""", unsafe_allow_html=True)
        else:
            k3.markdown("""
<div class="card card-yellow">
    <b>Volume</b><br>
    Data not available
</div>""", unsafe_allow_html=True)

        # ── Trigger Prices ──
        st.markdown("---")
        st.markdown("#### 💰 Suggested Trade Levels")
        entry = triggers['entry']
        sl = triggers['stop_loss']
        t1 = triggers['target_1']
        t2 = triggers['target_2']
        risk_per = triggers['risk_per_share']
        rr = triggers['risk_reward']

        tl1, tl2, tl3, tl4 = st.columns(4)
        tl1.markdown(f'<div class="card card-green" style="text-align:center;"><b>Entry</b><br><h3>{S}{entry:,.2f}</h3></div>', unsafe_allow_html=True)
        tl2.markdown(f'<div class="card card-red" style="text-align:center;"><b>Stop Loss</b><br><h3>{S}{sl:,.2f}</h3></div>', unsafe_allow_html=True)
        tl3.markdown(f'<div class="card card-green" style="text-align:center;"><b>Target 1</b><br><h3>{S}{t1:,.2f}</h3></div>', unsafe_allow_html=True)
        tl4.markdown(f'<div class="card card-green" style="text-align:center;"><b>Target 2</b><br><h3>{S}{t2:,.2f}</h3></div>', unsafe_allow_html=True)

        st.caption(f"Risk/share: {S}{risk_per:,.2f} | Risk:Reward = 1:{rr} | SL method: 1.5x ATR")

        # ── Position Sizing ──
        if risk_per > 0:
            st.markdown("---")
            with st.expander("📐 Position Sizing Calculator"):
                st.markdown(f"""
| Risk Budget | Max Shares | Position Size |
|-------------|-----------|---------------|
| {S}5,000 | {int(5000/risk_per)} shares | {S}{int(5000/risk_per)*current_price:,.0f} |
| {S}10,000 | {int(10000/risk_per)} shares | {S}{int(10000/risk_per)*current_price:,.0f} |
| {S}25,000 | {int(25000/risk_per)} shares | {S}{int(25000/risk_per)*current_price:,.0f} |
                """)

        # ── Strategy Rules ──
        st.markdown("---")
        with st.expander("📖 Strategy Rules — EMA 20 + RSI 14"):
            st.markdown("""
**🟢 ENTRY (Long Only):**
```
IF Close > EMA20
AND EMA20 is rising (today > yesterday)
AND RSI > 55
AND RSI < 70  (avoid overbought)
THEN → BUY at next candle open
```

**🔴 EXIT (any one triggers):**
```
IF Close < EMA20    → Trend broken
OR RSI < 45         → Momentum weakness
THEN → EXIT at next candle open
```

**📊 Indicator Formulas:**
- **EMA 20:** k = 2/(20+1) = 0.0952, EMAₜ = Closeₜ × k + EMAₜ₋₁ × (1-k)
- **RSI 14:** Standard Wilder RSI = 100 − (100 / (1 + RS)), where RS = Avg Gain / Avg Loss

**🔐 Why these thresholds?**
- RSI > 55 (not 50): Avoids weak fake moves, confirms real momentum
- RSI < 70: Prevents buying into overbought conditions
- RSI < 45 exit: Early warning of momentum loss before full breakdown
- 1.5x ATR stop-loss: Captures normal volatility without getting stopped out too early
            """)

    # ╔════════════════════════════════════╗
    # ║  TAB 5 – SENTIMENT ANALYSIS       ║
    # ╚════════════════════════════════════╝
    with tab_sent:
        st.markdown("### 📰 News Sentiment Analysis (FinBERT)")

        s_tab1, s_tab2 = st.tabs(["📰 Latest News", "📊 Sentiment Summary"])

        with s_tab1:
            st.markdown(f"#### Latest News — {company_name}")
            st.caption("Source: Google News RSS (Indian locale) | Sentiment: FinBERT")
            if news_err:
                st.warning(f"⚠️ {news_err}")
            elif not news_articles:
                st.info("No recent news articles found for this company. Try again later.")
            else:
                for i, article in enumerate(news_articles[:10], 1):
                    sent = news_sentiments[i-1] if i-1 < len(news_sentiments) else None
                    if sent:
                        lbl, cls = sentiment_label(sent['compound'])
                        conf = sent.get('finbert_confidence', 0)
                        score_text = f'<span class="{cls}">FinBERT: {lbl} ({sent["compound"]:+.3f}) | Confidence: {conf:.0%}</span>'
                    else:
                        score_text = ""
                    st.markdown(f"""
<div class="card" style="padding:12px 16px;">
    <b>{i}. {article['title']}</b><br>
    <small>📰 {article['source']} &nbsp;|&nbsp; 📅 {article['published']}</small><br>
    {score_text}<br>
    <a href="{article['link']}" target="_blank" rel="noopener noreferrer">Read full article →</a>
</div>""", unsafe_allow_html=True)

        with s_tab2:
            st.markdown("#### 📊 Sentiment Dashboard")

            all_compounds = [s['compound'] for s in news_sentiments]

            if all_compounds:
                avg_sent = float(np.mean(all_compounds))
                pos_count = sum(1 for s in all_compounds if s >= 0.05)
                neg_count = sum(1 for s in all_compounds if s <= -0.05)
                neu_count = len(all_compounds) - pos_count - neg_count

                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(chart_sentiment_gauge(sentiment_score), use_container_width=True)
                with g2:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=["Positive","Negative","Neutral"],
                        values=[pos_count, neg_count, neu_count],
                        marker_colors=["#66bb6a","#ef5350","#bdbdbd"], hole=0.45)])
                    fig_pie.update_layout(title="Sentiment Distribution", height=280, margin=dict(t=50,b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Articles", len(all_compounds))
                s2.metric("Positive", f"{pos_count} ({pos_count/len(all_compounds)*100:.0f}%)")
                s3.metric("Negative", f"{neg_count} ({neg_count/len(all_compounds)*100:.0f}%)")
                s4.metric("Avg Compound", f"{avg_sent:+.3f}")

                st.markdown("---")
                st.markdown("#### 💡 Market Sentiment Insight")
                summary = generate_sentiment_summary(
                    ticker, news_sentiments,
                    news_agg, sentiment_score, sentiment_label_str,
                )
                st.info(summary)
            else:
                st.warning("No sentiment data available to analyze.")

    # ── Footer ──
    st.markdown("---")
    st.markdown(f"""
<p class='footer-text' style='text-align:center;color:#9e9e9e;font-size:0.8em;'>
StockSense AI &nbsp;|&nbsp; Data as of {df.index[-1].strftime('%B %d, %Y')} &nbsp;|&nbsp;
Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
Model: {model_info.get('model_file', 'N/A')} &nbsp;|&nbsp;
Sentiment: FinBERT &nbsp;|&nbsp;
Strategy: EMA 20 + RSI 14 &nbsp;|&nbsp;
Educational purposes only
</p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
