"""
Technical Analysis Module for StockSense AI — Indian NSE Edition
Uses the `ta` library for all indicator computation.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


# ============================================================
# CORE: Compute All Indicators
# ============================================================

def compute_all_indicators(df):
    """Add all technical indicator columns to a DataFrame with OHLCV data.
    Modifies df in-place and returns it."""
    close = df['Close']
    high = df['High']
    low = df['Low']

    # --- Simple Moving Averages ---
    for period in [7, 20, 50, 200]:
        sma = SMAIndicator(close=close, window=period)
        df[f'SMA_{period}'] = sma.sma_indicator()

    # --- Exponential Moving Averages ---
    for period in [5, 12, 20, 26, 50]:
        ema = EMAIndicator(close=close, window=period)
        df[f'EMA_{period}'] = ema.ema_indicator()

    # --- RSI (14) ---
    rsi = RSIIndicator(close=close, window=14)
    df['RSI_14'] = rsi.rsi()

    # --- MACD (12, 26, 9) ---
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    # --- Bollinger Bands (20, 2) ---
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = bb.bollinger_wband()

    # --- ATR (14) ---
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df['ATR_14'] = atr.average_true_range()

    # --- ADX (14) ---
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    df['ADX_14'] = adx.adx()

    # --- Stochastic Oscillator (14, 3) ---
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # --- Volume SMA (20) ---
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()

    return df


# ============================================================
# SIGNAL FUNCTIONS
# ============================================================

def get_rsi_signal(rsi):
    """Interpret RSI value into a signal dict."""
    if rsi is None or np.isnan(rsi):
        return {"signal": "Neutral", "color": "gray", "detail": "RSI unavailable"}
    if rsi < 30:
        return {"signal": "Oversold", "color": "green", "detail": f"RSI {rsi:.1f} — Potential buying opportunity"}
    if rsi > 70:
        return {"signal": "Overbought", "color": "red", "detail": f"RSI {rsi:.1f} — Potential selling pressure"}
    if rsi < 45:
        return {"signal": "Weakly Bearish", "color": "orange", "detail": f"RSI {rsi:.1f} — Below midpoint"}
    if rsi > 55:
        return {"signal": "Weakly Bullish", "color": "lightgreen", "detail": f"RSI {rsi:.1f} — Above midpoint"}
    return {"signal": "Neutral", "color": "gray", "detail": f"RSI {rsi:.1f} — No strong signal"}


def get_macd_signal(macd_line, signal_line, prev_macd=None, prev_signal=None):
    """Detect MACD crossover and momentum."""
    if macd_line is None or signal_line is None or np.isnan(macd_line) or np.isnan(signal_line):
        return {"signal": "Neutral", "color": "gray", "detail": "MACD unavailable"}

    histogram = macd_line - signal_line

    # Crossover detection
    if prev_macd is not None and prev_signal is not None:
        if not (np.isnan(prev_macd) or np.isnan(prev_signal)):
            prev_hist = prev_macd - prev_signal
            if prev_hist <= 0 < histogram:
                return {"signal": "Bullish Crossover", "color": "green",
                        "detail": f"MACD crossed above signal — Buy signal"}
            if prev_hist >= 0 > histogram:
                return {"signal": "Bearish Crossover", "color": "red",
                        "detail": f"MACD crossed below signal — Sell signal"}

    if histogram > 0:
        return {"signal": "Bullish", "color": "lightgreen",
                "detail": f"MACD above signal ({histogram:.2f}) — Upward momentum"}
    return {"signal": "Bearish", "color": "orange",
            "detail": f"MACD below signal ({histogram:.2f}) — Downward momentum"}


def get_ema_crossover_signal(ema_short, ema_long):
    """Detect EMA crossover (golden/death cross using EMA 12/26)."""
    if ema_short is None or ema_long is None or np.isnan(ema_short) or np.isnan(ema_long):
        return {"signal": "Neutral", "color": "gray", "detail": "EMA unavailable"}
    diff_pct = (ema_short - ema_long) / ema_long * 100
    if diff_pct > 1:
        return {"signal": "Strong Bullish", "color": "green",
                "detail": f"EMA 12 is {diff_pct:.2f}% above EMA 26 — Strong uptrend"}
    if diff_pct > 0:
        return {"signal": "Bullish", "color": "lightgreen",
                "detail": f"EMA 12 above EMA 26 by {diff_pct:.2f}% — Mild uptrend"}
    if diff_pct > -1:
        return {"signal": "Bearish", "color": "orange",
                "detail": f"EMA 12 below EMA 26 by {abs(diff_pct):.2f}% — Mild downtrend"}
    return {"signal": "Strong Bearish", "color": "red",
            "detail": f"EMA 12 is {abs(diff_pct):.2f}% below EMA 26 — Strong downtrend"}


def get_bollinger_signal(close, upper, lower, width):
    """Interpret Bollinger Band position."""
    if any(v is None or np.isnan(v) for v in [close, upper, lower, width]):
        return {"signal": "Neutral", "color": "gray", "detail": "Bollinger Bands unavailable"}
    if close > upper:
        return {"signal": "Overbought", "color": "red",
                "detail": f"Price above upper band — Potential pullback"}
    if close < lower:
        return {"signal": "Oversold", "color": "green",
                "detail": f"Price below lower band — Potential bounce"}
    if width < 0.05:
        return {"signal": "Squeeze", "color": "blue",
                "detail": f"Bands narrowing (width {width:.3f}) — Breakout imminent"}
    band_range = upper - lower
    if band_range > 0:
        position = (close - lower) / band_range
        if position > 0.8:
            return {"signal": "Near Upper", "color": "orange",
                    "detail": f"Price at {position:.0%} of bands — Approaching resistance"}
        if position < 0.2:
            return {"signal": "Near Lower", "color": "lightgreen",
                    "detail": f"Price at {position:.0%} of bands — Approaching support"}
    return {"signal": "Neutral", "color": "gray",
            "detail": f"Price within Bollinger Bands — Normal range"}


def get_volume_signal(current_vol, avg_vol):
    """Detect volume spikes."""
    if avg_vol is None or avg_vol == 0 or np.isnan(avg_vol):
        return {"signal": "Neutral", "color": "gray", "detail": "Volume data unavailable"}
    ratio = current_vol / avg_vol
    if ratio > 2.0:
        return {"signal": "Very High", "color": "red",
                "detail": f"Volume {ratio:.1f}x average — Major activity"}
    if ratio > 1.5:
        return {"signal": "High", "color": "orange",
                "detail": f"Volume {ratio:.1f}x average — Above normal"}
    if ratio < 0.5:
        return {"signal": "Very Low", "color": "blue",
                "detail": f"Volume {ratio:.1f}x average — Low interest"}
    return {"signal": "Normal", "color": "gray",
            "detail": f"Volume {ratio:.1f}x average — Normal activity"}


def get_adx_signal(adx):
    """Interpret ADX trend strength."""
    if adx is None or np.isnan(adx):
        return {"signal": "Neutral", "color": "gray", "detail": "ADX unavailable"}
    if adx > 50:
        return {"signal": "Very Strong Trend", "color": "green",
                "detail": f"ADX {adx:.1f} — Extremely strong trend"}
    if adx > 25:
        return {"signal": "Strong Trend", "color": "lightgreen",
                "detail": f"ADX {adx:.1f} — Trending market"}
    return {"signal": "Weak/No Trend", "color": "gray",
            "detail": f"ADX {adx:.1f} — Ranging/sideways market"}


# ============================================================
# SUPPORT / RESISTANCE (Classic Pivot Points)
# ============================================================

def compute_support_resistance(df):
    """Compute classic pivot points from the most recent completed day.
    Returns dict with Pivot, S1, S2, S3, R1, R2, R3."""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    h, l, c = float(prev['High']), float(prev['Low']), float(prev['Close'])
    pivot = (h + l + c) / 3
    return {
        'Pivot': round(pivot, 2),
        'R1': round(2 * pivot - l, 2),
        'R2': round(pivot + (h - l), 2),
        'R3': round(h + 2 * (pivot - l), 2),
        'S1': round(2 * pivot - h, 2),
        'S2': round(pivot - (h - l), 2),
        'S3': round(l - 2 * (h - pivot), 2),
    }


# ============================================================
# TRIGGER PRICES
# ============================================================

def compute_trigger_prices(close, atr, support_resistance, rsi=None, bb_lower=None, bb_upper=None):
    """Compute actionable trigger prices: buy, sell, stop loss, take profit.

    Args:
        close: current closing price
        atr: current ATR(14) value
        support_resistance: dict from compute_support_resistance()
        rsi: current RSI (optional, for adjustment)
        bb_lower: current Bollinger lower band (optional)
        bb_upper: current Bollinger upper band (optional)

    Returns dict with buy_price, sell_price, stop_loss, take_profit_1,
    take_profit_2, risk_reward_ratio.
    """
    if atr is None or np.isnan(atr) or atr == 0:
        atr = close * 0.015  # fallback: 1.5% of price

    s1 = support_resistance.get('S1', close - atr)
    r1 = support_resistance.get('R1', close + atr)

    # Buy price: midpoint between current price and S1, nudged by ATR
    buy_price = round(max(s1, close - 0.5 * atr), 2)

    # Sell price: midpoint between current price and R1
    sell_price = round(min(r1, close + 0.5 * atr), 2)

    # Stop loss: 1.5 ATR below buy price
    stop_loss = round(buy_price - 1.5 * atr, 2)

    # Take profit 1: 1.5 ATR above buy price (1:1 risk-reward from stop)
    take_profit_1 = round(buy_price + 1.5 * atr, 2)

    # Take profit 2: 3 ATR above buy price (2:1 risk-reward)
    take_profit_2 = round(buy_price + 3.0 * atr, 2)

    # Adjust with Bollinger if available
    if bb_lower is not None and not np.isnan(bb_lower):
        buy_price = round(max(buy_price, bb_lower), 2)
    if bb_upper is not None and not np.isnan(bb_upper):
        sell_price = round(min(sell_price, bb_upper), 2)

    # Risk-reward ratio
    risk = buy_price - stop_loss
    reward = take_profit_2 - buy_price
    risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    return {
        'buy_price': buy_price,
        'sell_price': sell_price,
        'stop_loss': stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'risk_reward_ratio': risk_reward,
    }
