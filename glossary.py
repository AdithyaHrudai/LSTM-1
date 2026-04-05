"""
Glossary Page for StockSense AI — Indian NSE Edition
Explains key stock market terms, concepts, and indicators with Indian examples.
"""

import streamlit as st
import pandas as pd


def render_glossary():
    """Render the full glossary page."""
    st.markdown("<h1 style='text-align: center;'>📖 Stock Market Glossary</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Key terms, indicators, and concepts used in StockSense AI — Indian NSE Edition</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Table of Contents
    st.markdown("### 📑 Table of Contents")
    st.markdown("""
    1. [Stock Market Basics](#stock-market-basics)
    2. [Price & Volume Concepts](#price-volume-concepts)
    3. [Moving Averages](#moving-averages)
    4. [Technical Indicators](#technical-indicators)
    5. [Support, Resistance & Trigger Prices](#support-resistance-trigger-prices)
    6. [Sentiment Analysis Terms](#sentiment-analysis-terms)
    7. [AI & Machine Learning Concepts](#ai-machine-learning-concepts)
    8. [Trading Terminology](#trading-terminology)
    """)

    st.markdown("---")

    # ================================================================
    # SECTION 1: STOCK MARKET BASICS
    # ================================================================
    st.markdown("## 📌 Stock Market Basics")

    with st.expander("🏷️ Ticker Symbol", expanded=False):
        st.markdown("""
        **Definition:** A unique abbreviation used to identify a publicly traded company on a stock exchange.

        **Example:** `TCS.NS` = Tata Consultancy Services (NSE), `INFY.NS` = Infosys (NSE)
        """)
        st.dataframe(pd.DataFrame({
            "Company": ["Tata Consultancy Services", "Bajaj Auto", "Pidilite Industries", "Hero MotoCorp", "Polycab India"],
            "Ticker": ["TCS.NS", "BAJAJ-AUTO.NS", "PIDILITIND.NS", "HEROMOTOCO.NS", "POLYCAB.NS"],
            "Exchange": ["NSE", "NSE", "NSE", "NSE", "NSE"]
        }), hide_index=True, use_container_width=True)
        st.info("💡 **Good to know:** `.NS` indicates the National Stock Exchange (NSE) and `.BO` indicates the Bombay Stock Exchange (BSE).")

    with st.expander("💰 Market Capitalization (Market Cap)", expanded=False):
        st.markdown("""
        **Definition:** The total market value of a company's outstanding shares.

        > **Market Cap = Current Share Price × Total Outstanding Shares**

        **Example:** If TCS trades at ₹4,000 and has 366 Cr shares → Market Cap ≈ ₹14.6 Lakh Crore
        """)
        st.dataframe(pd.DataFrame({
            "Category": ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"],
            "Market Cap Range": ["> ₹5 Lakh Cr", "₹50,000 Cr – ₹5 Lakh Cr", "₹10,000 – ₹50,000 Cr", "₹2,000 – ₹10,000 Cr", "< ₹2,000 Cr"],
            "Example": ["TCS, Reliance", "Bajaj Auto, Pidilite", "Polycab, Cyient", "Force Motors, UBL", "Smaller listed cos"],
            "Risk Level": ["Low", "Low-Medium", "Medium", "High", "Very High"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Optimal for beginners:** Large Cap and Mega Cap stocks — more stable, less volatile.")

    with st.expander("📊 P/E Ratio (Price-to-Earnings)", expanded=False):
        st.markdown("""
        **Definition:** Measures how much investors are willing to pay per rupee of earnings.

        > **P/E Ratio = Share Price ÷ Earnings Per Share (EPS)**

        **Example:** If Pidilite trades at ₹3,000 and EPS is ₹30 → P/E = 100
        """)
        st.dataframe(pd.DataFrame({
            "P/E Range": ["< 15", "15 – 25", "25 – 40", "> 40"],
            "Interpretation": ["Undervalued or slow growth", "Fairly valued", "High growth expected", "Potentially overvalued"],
            "Indian Sectors": ["PSU Banks, Mining", "FMCG, Auto", "IT Services", "Specialty Chemicals, Growth stocks"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Note:** Indian growth stocks often trade at higher P/E ratios. Compare within the same sector.")

    with st.expander("📈 52-Week High / Low", expanded=False):
        st.markdown("""
        **Definition:** The highest and lowest prices a stock has traded at during the past 52 weeks (1 year).

        **How to use it:**
        - **Near 52-week high:** Stock has momentum but may be overextended
        - **Near 52-week low:** Could be undervalued or in a downtrend
        - **Mid-range:** Neutral territory
        """)
        st.success("✅ **Optimal position:** Look for stocks in the 40-70% range of their 52-week band.")

    st.markdown("---")

    # ================================================================
    # SECTION 2: PRICE & VOLUME CONCEPTS
    # ================================================================
    st.markdown("## 📌 Price & Volume Concepts")

    with st.expander("📊 OHLCV (Open, High, Low, Close, Volume)", expanded=False):
        st.markdown("""
        **Definition:** The five fundamental data points for each trading day:

        | Term | Meaning |
        |------|---------|
        | **Open** | Price at which the stock starts trading (9:15 AM IST) |
        | **High** | Highest price during the trading day |
        | **Low** | Lowest price during the trading day |
        | **Close** | Final price at market close (3:30 PM IST) |
        | **Volume** | Total number of shares traded |
        """)
        st.info("💡 **In this project:** Our LSTM model uses OHLCV plus RSI, MACD histogram, Bollinger Band width, and sentiment — 9 features across 60 timesteps.")

    with st.expander("📶 Trading Volume", expanded=False):
        st.markdown("""
        **How to interpret:**
        - **Rising price + Rising volume** = Strong bullish signal ✅
        - **Rising price + Falling volume** = Weak rally, potential reversal ⚠️
        - **Falling price + Rising volume** = Strong selling pressure 🔴
        - **Falling price + Falling volume** = Selling exhaustion, potential bottom 🟡
        """)
        st.success("✅ **What to look for:** Always check if price moves are backed by volume. High volume = conviction.")

    st.markdown("---")

    # ================================================================
    # SECTION 3: MOVING AVERAGES
    # ================================================================
    st.markdown("## 📌 Moving Averages")

    with st.expander("📏 SMA (Simple Moving Average)", expanded=False):
        st.markdown("""
        **Definition:** The average closing price over a specific number of days.

        > **SMA(n) = Sum of last n closing prices ÷ n**
        """)
        st.dataframe(pd.DataFrame({
            "SMA Period": ["7-day", "20-day", "50-day", "200-day"],
            "Used For": ["Short-term trend", "Bollinger Band center", "Medium-term trend", "Major trend direction"],
            "Signal": ["Quick reactions", "Swing trading", "Golden/Death Cross", "Bull/Bear market line"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Key signal:** When the 50-day SMA crosses above the 200-day SMA → **Golden Cross** (bullish). Opposite → **Death Cross** (bearish).")

    with st.expander("📐 EMA (Exponential Moving Average)", expanded=False):
        st.markdown("""
        **Definition:** Like SMA but gives **more weight to recent prices**, making it more responsive.

        **Common EMA periods used in StockSense AI:**
        """)
        st.dataframe(pd.DataFrame({
            "EMA Period": ["5-day", "12-day", "26-day", "50-day"],
            "Used For": ["Ultra short-term", "MACD fast line", "MACD slow line", "Trend confirmation"]
        }), hide_index=True, use_container_width=True)

    st.markdown("---")

    # ================================================================
    # SECTION 4: TECHNICAL INDICATORS
    # ================================================================
    st.markdown("## 📌 Technical Indicators")

    with st.expander("📊 RSI (Relative Strength Index)", expanded=False):
        st.markdown("""
        **Definition:** A momentum oscillator (0-100) that measures the speed of price changes.

        > **RSI = 100 − (100 ÷ (1 + RS))**
        > where RS = Avg Gain / Avg Loss over 14 periods
        """)
        st.dataframe(pd.DataFrame({
            "RSI Range": ["0 – 30", "30 – 45", "45 – 55", "55 – 70", "70 – 100"],
            "Signal": ["Oversold 🟢", "Weakly bearish", "Neutral", "Weakly bullish", "Overbought 🔴"],
            "Action": ["Potential buy", "Watch", "No signal", "Trend continuation", "Potential sell"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **In StockSense AI:** RSI contributes 15 points to the strategy score. Buy below 30, sell above 70.")

    with st.expander("📈 MACD (Moving Average Convergence Divergence)", expanded=False):
        st.markdown("""
        **Components:**
        - **MACD Line** = 12-day EMA − 26-day EMA
        - **Signal Line** = 9-day EMA of MACD Line
        - **Histogram** = MACD Line − Signal Line
        """)
        st.dataframe(pd.DataFrame({
            "Signal": ["MACD crosses above Signal", "MACD crosses below Signal", "Histogram growing", "Histogram shrinking"],
            "Meaning": ["Bullish crossover 🟢", "Bearish crossover 🔴", "Momentum increasing", "Momentum weakening"],
            "Action": ["Consider buying", "Consider selling", "Hold long", "Prepare for reversal"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **In StockSense AI:** MACD contributes 15 points. Crossovers get bonus weighting.")

    with st.expander("📉 Bollinger Bands", expanded=False):
        st.markdown("""
        **Definition:** A volatility indicator with three lines:
        - **Upper Band** = 20-day SMA + 2 × Std Dev
        - **Middle Band** = 20-day SMA
        - **Lower Band** = 20-day SMA − 2 × Std Dev

        **Bollinger Band Width** is used as a feature in our LSTM model.
        """)
        st.dataframe(pd.DataFrame({
            "Price Position": ["Above Upper", "Near Upper", "Near Middle", "Near Lower", "Below Lower"],
            "Signal": ["Overbought", "Approaching resistance", "Fair value", "Approaching support", "Oversold"],
            "Bands Width": ["Wide = High vol", "", "Normal", "", "Narrow = Squeeze (breakout coming)"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Strategy:** Buy near lower band, sell near upper. When bands narrow ('squeeze'), expect a big move.")

    with st.expander("📊 ADX (Average Directional Index)", expanded=False):
        st.markdown("""
        **Definition:** Measures the **strength** of a trend (not direction). Range: 0-100.
        """)
        st.dataframe(pd.DataFrame({
            "ADX Range": ["0 – 25", "25 – 50", "50 – 75", "75 – 100"],
            "Interpretation": ["Weak/No trend (ranging)", "Trending", "Strong trend", "Extremely strong trend"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Tip:** Only use trend-following strategies (MACD, EMA cross) when ADX > 25. In low-ADX markets, use range strategies (Bollinger, RSI).")

    with st.expander("📊 ATR (Average True Range)", expanded=False):
        st.markdown("""
        **Definition:** Measures market **volatility** by averaging the true range over 14 periods.

        **In StockSense AI:** ATR is used to calculate:
        - **Stop Loss** = Buy Price − 1.5 × ATR
        - **Take Profit 1** = Buy Price + 1.5 × ATR
        - **Take Profit 2** = Buy Price + 3.0 × ATR
        """)

    st.markdown("---")

    # ================================================================
    # SECTION 5: SUPPORT, RESISTANCE & TRIGGER PRICES
    # ================================================================
    st.markdown("## 📌 Support, Resistance & Trigger Prices")

    with st.expander("🧱 Support & Resistance (Pivot Points)", expanded=False):
        st.markdown("""
        **Definition:**
        - **Support:** Price level where a stock tends to stop falling — like a "floor"
        - **Resistance:** Price level where a stock tends to stop rising — like a "ceiling"

        **StockSense AI uses Classic Pivot Points:**

        > Pivot = (High + Low + Close) / 3
        > R1 = 2 × Pivot − Low, R2 = Pivot + (High − Low)
        > S1 = 2 × Pivot − High, S2 = Pivot − (High − Low)

        **Example for Hero MotoCorp at ₹5,200:**
        """)
        st.dataframe(pd.DataFrame({
            "Level": ["R2", "R1", "Pivot", "S1", "S2"],
            "Price": ["₹5,350", "₹5,280", "₹5,220", "₹5,150", "₹5,080"],
            "Meaning": ["Strong resistance", "First resistance", "Pivot point", "First support", "Strong support"]
        }), hide_index=True, use_container_width=True)

    with st.expander("🎯 Trigger Prices (Buy, Sell, Stop Loss, Take Profit)", expanded=False):
        st.markdown("""
        **Definition:** Specific price levels at which to execute trades.

        **How StockSense AI calculates them:**
        - **Buy Price** = Near S1 support, adjusted by 0.5 × ATR
        - **Sell Price** = Near R1 resistance, adjusted by 0.5 × ATR
        - **Stop Loss** = Buy Price − 1.5 × ATR (limits downside risk)
        - **Take Profit 1** = Buy Price + 1.5 × ATR (1:1 risk-reward)
        - **Take Profit 2** = Buy Price + 3.0 × ATR (2:1 risk-reward)
        - **Risk-Reward Ratio** = (TP2 − Buy) / (Buy − SL)
        """)
        st.success("✅ **Rule:** Always maintain a risk-reward ratio of at least 1.5:1. Never risk more than 2% of your portfolio on a single trade.")

    st.markdown("---")

    # ================================================================
    # SECTION 6: SENTIMENT ANALYSIS
    # ================================================================
    st.markdown("## 📌 Sentiment Analysis Terms")

    with st.expander("🧠 VADER Sentiment Analysis", expanded=False):
        st.markdown("""
        **Definition:** A lexicon-based sentiment tool for social media text.

        **Output scores:**
        """)
        st.dataframe(pd.DataFrame({
            "Score": ["Compound", "Positive", "Negative", "Neutral"],
            "Range": ["-1.0 to +1.0", "0-1", "0-1", "0-1"],
            "Classification": ["≥ 0.05 = Positive, ≤ -0.05 = Negative", "", "", ""]
        }), hide_index=True, use_container_width=True)

    with st.expander("📊 Market Sentiment Score (0-100)", expanded=False):
        st.markdown("""
        **Calculation:** News (60%) + Reddit (40%) → Composite score 0-100.

        **Sources:** Google News (Indian locale) + Reddit (r/IndianStockMarket, r/IndianStreetBets, r/IndiaInvestments)
        """)
        st.dataframe(pd.DataFrame({
            "Score Range": ["0 – 25", "26 – 45", "46 – 55", "56 – 75", "76 – 100"],
            "Label": ["Very Bearish 🔴🔴", "Bearish 🔴", "Neutral 🟡", "Bullish 🟢", "Very Bullish 🟢🟢"],
        }), hide_index=True, use_container_width=True)
        st.success("✅ **In StockSense AI:** Sentiment contributes 20 points (highest weight) to the strategy score.")

    st.markdown("---")

    # ================================================================
    # SECTION 7: AI & ML CONCEPTS
    # ================================================================
    st.markdown("## 📌 AI & Machine Learning Concepts")

    with st.expander("🧠 LSTM (Long Short-Term Memory)", expanded=False):
        st.markdown("""
        **Definition:** A recurrent neural network designed to learn patterns in sequential data.

        **Our model architecture:**
        """)
        st.dataframe(pd.DataFrame({
            "Component": ["Input", "LSTM Layer 1", "LSTM Layer 2", "Dense Layer", "Output", "Lookback", "Features"],
            "Details": [
                "60 days of data", "128 units, return_sequences=True",
                "64 units", "32 units, ReLU", "1 predicted close price",
                "60 timesteps",
                "9 features (OHLCV + RSI + MACD_hist + BB_width + Sentiment)"
            ]
        }), hide_index=True, use_container_width=True)
        st.warning("⚠️ **Limitation:** LSTM predicts based on patterns only. It cannot account for breaking news, earnings, or black swan events.")

    with st.expander("📏 MinMaxScaler (Normalization)", expanded=False):
        st.markdown("""
        **Definition:** Scales all values to 0-1 range.

        > **Scaled Value = (Value − Min) ÷ (Max − Min)**

        **Why needed:** Stock prices range from ₹50 to ₹50,000+. Neural networks work best with normalized inputs.
        After prediction, we **inverse transform** to get the actual ₹ price.
        """)

    st.markdown("---")

    # ================================================================
    # SECTION 8: TRADING TERMINOLOGY
    # ================================================================
    st.markdown("## 📌 Trading Terminology")

    with st.expander("🐂 Bullish vs 🐻 Bearish", expanded=False):
        st.markdown("""
        | Term | Meaning |
        |------|---------|
        | **Bullish** 🐂 | Expecting prices to **rise** |
        | **Bearish** 🐻 | Expecting prices to **fall** |
        | **Bull Market** | Extended period of rising prices (> 20% gain) |
        | **Bear Market** | Extended period of falling prices (> 20% decline) |

        **Example:** "I'm bullish on TCS" = I think TCS price will go up.
        """)

    with st.expander("🥧 Portfolio Diversification", expanded=False):
        st.markdown("""
        **Definition:** Spreading investments across sectors to reduce risk.

        **Example diversified Indian portfolio:**
        """)
        st.dataframe(pd.DataFrame({
            "Allocation": ["25%", "20%", "15%", "15%", "15%", "10%"],
            "Sector": ["IT Services", "Automobiles", "FMCG", "Pharma", "Capital Goods", "Chemicals"],
            "Example": ["TCS, LTIMindtree", "Hero MotoCorp, Bajaj Auto", "HUL, Godfrey Phillips", "Divi's Labs, Torrent", "ABB India", "Pidilite"],
            "Risk": ["Medium", "Medium", "Low", "Medium", "Medium-High", "Medium"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Rule:** No single stock > 10-15% of your portfolio. Diversify across 3-4 sectors minimum.")

    with st.expander("💲 SIP (Systematic Investment Plan)", expanded=False):
        st.markdown("""
        **Definition:** Investing a fixed amount at regular intervals (similar to Dollar-Cost Averaging).

        **Example: Investing ₹10,000/month in Bajaj Auto:**
        """)
        st.dataframe(pd.DataFrame({
            "Month": ["January", "February", "March", "April"],
            "Price": ["₹9,000", "₹9,500", "₹8,800", "₹9,200"],
            "Shares": ["1.11", "1.05", "1.14", "1.09"],
            "Cost": ["₹10,000", "₹10,000", "₹10,000", "₹10,000"]
        }), hide_index=True, use_container_width=True)
        st.success("✅ **Why SIP works:** You buy more shares when prices are low and fewer when high. Removes emotion from investing.")

    st.markdown("---")

    # Quick Reference Card
    st.markdown("### 🎯 Quick Reference: Optimal Values")
    st.dataframe(pd.DataFrame({
        "Indicator": ["P/E Ratio", "RSI", "MACD", "Bollinger Position", "ADX", "Sentiment Score", "Risk-Reward"],
        "Optimal Range": ["15 – 25 (sector-specific)", "40 – 60", "Above signal line", "20% – 80% of bands", "> 25 (trending)", "56 – 75 (Bullish)", "> 1.5:1"],
        "Buy Signal": ["< 15 (value)", "< 30 (oversold)", "Bullish crossover", "Near lower band", "> 25 + bullish", "> 75 (very bullish)", "High ratio"],
        "Sell Signal": ["> 40 (expensive)", "> 70 (overbought)", "Bearish crossover", "Near upper band", "< 25 (no trend)", "< 25 (very bearish)", "Low ratio"]
    }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: gray; font-size: 12px;'>
    📖 This glossary is for educational purposes only. Always do your own research before making investment decisions.
    All examples use Indian NSE stocks. Prices are illustrative.
    </p>
    """, unsafe_allow_html=True)
