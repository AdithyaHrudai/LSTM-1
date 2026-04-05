"""
Strategy Module for StockSense AI — Indian NSE Edition
📈 EMA 20 + RSI 14 Strategy (Optimized Version)

Indicators Used:
    - EMA 20 → Trend filter
    - RSI 14 → Momentum confirmation

Entry Rules (Long Only):
    - Close > EMA 20  (price above trend)
    - EMA 20 rising    (trend direction up)
    - RSI > 55         (bullish momentum confirmed)
    - RSI < 70         (not overbought — avoid late entries)

Exit Rules (any one triggers):
    - Close < EMA 20   (trend broken)
    - RSI < 45         (momentum weakness)
"""

import numpy as np
from config import CURRENCY_SYMBOL


# ============================================================
# EMA 20 + RSI 14 STRATEGY
# ============================================================

def evaluate_strategy(close, ema_20, prev_ema_20, rsi, volume=None, avg_volume=None):
    """
    Evaluate EMA 20 + RSI 14 strategy for a single point in time.

    Returns dict with: signal, strength, confidence, reason, details
    """
    # Validate inputs
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [close, ema_20, rsi]):
        return {
            "signal": "HOLD", "strength": 50, "confidence": "Low",
            "reason": "Insufficient data for strategy evaluation.",
            "close_vs_ema": "N/A", "ema_distance_pct": 0,
            "ema_trend": "N/A", "rsi": 0,
        }

    prev_ema_valid = prev_ema_20 is not None and not np.isnan(prev_ema_20)
    ema_rising = ema_20 > prev_ema_20 if prev_ema_valid else False

    above_ema = close > ema_20
    below_ema = close < ema_20
    ema_distance_pct = ((close - ema_20) / ema_20) * 100

    # Volume ratio (optional, for STRONG BUY confirmation)
    volume_ratio = None
    if volume and avg_volume and avg_volume > 0:
        volume_ratio = volume / avg_volume

    details = {
        "close_vs_ema": "Above" if above_ema else "Below",
        "ema_distance_pct": round(ema_distance_pct, 2),
        "ema_trend": "Rising ↑" if ema_rising else "Falling/Flat →",
        "rsi": round(rsi, 1),
        "volume_ratio": round(volume_ratio, 2) if volume_ratio else None,
    }

    # ══════════════════════════════════════
    # 🟢 STRONG BUY — All conditions + high volume
    # ══════════════════════════════════════
    if (above_ema and ema_rising and 55 < rsi < 70
            and volume_ratio is not None and volume_ratio > 1.3):
        strength = min(95, int(75 + (rsi - 55) * 1.0 + ema_distance_pct * 2))
        return {
            **details, "signal": "STRONG BUY",
            "strength": max(80, min(95, strength)),
            "confidence": "High",
            "reason": (
                "STRONG BUY: Price is above a rising EMA 20, RSI confirms bullish momentum "
                f"({rsi:.1f}, in the 55-70 sweet spot), and volume is {volume_ratio:.1f}x above average. "
                "All entry conditions are met with conviction."
            ),
        }

    # ══════════════════════════════════════
    # 🟢 BUY — Core entry conditions met
    # ══════════════════════════════════════
    if above_ema and ema_rising and 55 < rsi < 70:
        strength = int(65 + (rsi - 55) * 0.8 + ema_distance_pct * 1.5)
        return {
            **details, "signal": "BUY",
            "strength": max(65, min(79, strength)),
            "confidence": "Medium",
            "reason": (
                f"BUY: Price is above a rising EMA 20 and RSI is at {rsi:.1f} "
                "(bullish zone: 55-70). Entry conditions met — consider buying at next candle open."
            ),
        }

    # ══════════════════════════════════════
    # 🔴 STRONG SELL — Both exit conditions
    # ══════════════════════════════════════
    if below_ema and rsi < 45:
        strength = max(5, int(25 - abs(ema_distance_pct) * 2))
        return {
            **details, "signal": "STRONG SELL",
            "strength": max(5, min(25, strength)),
            "confidence": "High",
            "reason": (
                f"STRONG SELL: Price has broken below EMA 20 ({ema_distance_pct:+.2f}%) "
                f"AND RSI is weak at {rsi:.1f} (< 45). Both exit triggers fired — exit positions immediately."
            ),
        }

    # ══════════════════════════════════════
    # 🔴 SELL — Close below EMA 20
    # ══════════════════════════════════════
    if below_ema:
        strength = max(30, int(40 - abs(ema_distance_pct) * 1.5))
        return {
            **details, "signal": "SELL",
            "strength": max(30, min(44, strength)),
            "confidence": "Medium",
            "reason": (
                f"SELL: Price has closed below EMA 20 ({ema_distance_pct:+.2f}%) — "
                "trend exit triggered. Consider exiting long positions at next candle open."
            ),
        }

    # ══════════════════════════════════════
    # 🔴 SELL — RSI below 45
    # ══════════════════════════════════════
    if rsi < 45:
        strength = max(30, int(42 - (45 - rsi) * 0.8))
        return {
            **details, "signal": "SELL",
            "strength": max(30, min(44, strength)),
            "confidence": "Medium",
            "reason": (
                f"SELL: RSI has dropped to {rsi:.1f} (< 45) — momentum is weakening. "
                "Even though price is above EMA 20, consider exiting."
            ),
        }

    # ══════════════════════════════════════
    # 🟡 HOLD — Above EMA but overbought
    # ══════════════════════════════════════
    if above_ema and rsi >= 70:
        return {
            **details, "signal": "HOLD",
            "strength": 55, "confidence": "Medium",
            "reason": (
                f"HOLD: Price is above EMA 20 but RSI is overbought ({rsi:.1f} ≥ 70). "
                "Avoid new entries — wait for RSI to cool below 70."
            ),
        }

    # ══════════════════════════════════════
    # 🟡 HOLD — Above EMA but RSI not confirmed
    # ══════════════════════════════════════
    if above_ema and rsi <= 55:
        return {
            **details, "signal": "HOLD",
            "strength": min(64, int(55 + rsi * 0.1)),
            "confidence": "Low",
            "reason": (
                f"HOLD: Price is above EMA 20 but RSI ({rsi:.1f}) hasn't confirmed "
                "bullish momentum yet (needs > 55). Wait for RSI to cross above 55."
            ),
        }

    # ══════════════════════════════════════
    # 🟡 HOLD — Above EMA but EMA not rising
    # ══════════════════════════════════════
    if above_ema and not ema_rising:
        return {
            **details, "signal": "HOLD",
            "strength": 52, "confidence": "Low",
            "reason": (
                "HOLD: Price is above EMA 20 but the EMA is flat/falling — "
                "trend direction is uncertain. Wait for EMA to start rising."
            ),
        }

    # Default HOLD
    return {
        **details, "signal": "HOLD",
        "strength": 50, "confidence": "Low",
        "reason": "HOLD: No clear entry or exit signal. Wait for EMA 20 + RSI alignment.",
    }


# ============================================================
# TRIGGER PRICES (ATR-based)
# ============================================================

def compute_trigger_prices(close, atr, rsi):
    """Compute suggested entry, stop-loss, and target prices using ATR."""
    if atr is None or np.isnan(atr) or atr <= 0:
        atr = close * 0.02  # Fallback: 2% of price

    # Entry: slightly below current price if neutral, at current if BUY
    entry = round(close, 2)

    # Stop Loss: 1.5x ATR below entry
    stop_loss = round(close - 1.5 * atr, 2)

    # Targets: 1.5x ATR (1:1 RR) and 3x ATR (2:1 RR)
    target_1 = round(close + 1.5 * atr, 2)
    target_2 = round(close + 3.0 * atr, 2)

    risk = abs(close - stop_loss)
    rr_ratio = round((target_2 - close) / risk, 1) if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": round(risk, 2),
        "risk_reward": rr_ratio,
    }
