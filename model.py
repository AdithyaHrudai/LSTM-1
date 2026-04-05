"""
LSTM Model Module for StockSense AI — Indian NSE Edition
Handles model loading, data preparation, and predictions.
Single-market: India (NSE) only.
"""

import os
import glob
import json
import numpy as np
import keras
import joblib
import streamlit as st
from config import N_FEATURES, SEQUENCE_LENGTH, FEATURES


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_lstm_model():
    """Load the latest trained LSTM model and per-stock scalers from models/ folder.
    Returns (model, per_stock_scalers, model_info, error).

    per_stock_scalers is a dict: {ticker: {'feature_scaler': ..., 'target_scaler': ...}}
    For legacy models with universal scalers, wraps them in a compatible format.
    """
    model_info = {}
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')

        model_path = None
        scaler_path = None
        metrics_path = None

        # Search models/ folder for stage2_india_lstm or stage2_universal_lstm
        if os.path.isdir(models_dir):
            for prefix in ['stage2_india', 'stage2_universal', 'stage2']:
                model_glob = f'{prefix}_lstm_*.keras'
                model_files = sorted(glob.glob(os.path.join(models_dir, model_glob)))
                if not model_files:
                    model_files = sorted(glob.glob(os.path.join(models_dir, f'{prefix}*lstm*.keras')))
                if model_files:
                    model_path = model_files[-1]
                    for sf in sorted(glob.glob(os.path.join(models_dir, f'{prefix}*scalers*.pkl'))):
                        scaler_path = sf
                    for mf in sorted(glob.glob(os.path.join(models_dir, f'{prefix}*metrics*.json'))):
                        metrics_path = mf
                    break

        # Fallback: search root directory
        if model_path is None:
            root_models = sorted(glob.glob(os.path.join(script_dir, 'stage2*lstm*.keras')))
            root_scalers = sorted(glob.glob(os.path.join(script_dir, 'stage2*scalers*.pkl')))
            if root_models and root_scalers:
                model_path = root_models[-1]
                scaler_path = root_scalers[-1]

        if not model_path or not scaler_path:
            return None, None, {}, (
                "No model files found. Run `python train_model.py` first, "
                "or place .keras and .pkl files in the models/ folder."
            )

        # Load
        model = keras.models.load_model(model_path, compile=False)
        raw_scalers = joblib.load(scaler_path)

        # Detect format: per-stock dict vs legacy universal dict
        if 'feature_scaler' in raw_scalers and 'target_scaler' in raw_scalers:
            # Legacy universal scalers — wrap in per-stock format
            universal_entry = {
                'feature_scaler': raw_scalers['feature_scaler'],
                'target_scaler': raw_scalers['target_scaler'],
            }
            per_stock_scalers = {'_universal': universal_entry}
            model_info['scaler_type'] = 'universal'
        else:
            # New per-stock scalers
            per_stock_scalers = raw_scalers
            model_info['scaler_type'] = 'per_stock'

        model_info['model_file'] = os.path.basename(model_path)
        model_info['scaler_file'] = os.path.basename(scaler_path)

        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_info['metrics'] = json.load(f)

        expected_features = model.input_shape[-1]
        model_info['expected_features'] = expected_features

        return model, per_stock_scalers, model_info, None
    except Exception as e:
        return None, None, {}, str(e)


def get_scalers_for_ticker(per_stock_scalers, ticker):
    """Get (feature_scaler, target_scaler) for a given ticker.
    Falls back to universal scaler if per-stock not available."""
    if ticker in per_stock_scalers:
        s = per_stock_scalers[ticker]
        return s['feature_scaler'], s['target_scaler']
    if '_universal' in per_stock_scalers:
        s = per_stock_scalers['_universal']
        return s['feature_scaler'], s['target_scaler']
    # Last resort: use the first available stock's scaler
    first_key = next(iter(per_stock_scalers))
    s = per_stock_scalers[first_key]
    return s['feature_scaler'], s['target_scaler']


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data_for_prediction(data, feature_scaler, sentiment_compound=0.0, look_back=60):
    """Prepare data for LSTM prediction using the SAVED scalers.

    The model expects shape (batch, 60, N_FEATURES).
    Feature list depends on what the model was trained with.

    Args:
        data: DataFrame with OHLCV + technical indicator columns
        feature_scaler: the saved MinMaxScaler
        sentiment_compound: current sentiment compound score (-1 to +1)
        look_back: number of timesteps (60)
    """
    if len(data) < look_back:
        return None

    n_expected = feature_scaler.n_features_in_

    if n_expected == 9:
        # New 9-feature model: OHLCV + RSI_14 + MACD_hist + BB_width + sentiment
        required = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_hist', 'BB_width']
        for col in required:
            if col not in data.columns:
                return None
        ohlcv_tech = data[required].values[-look_back:]
        sentiment_col = np.full((look_back, 1), sentiment_compound)
        features = np.concatenate([ohlcv_tech, sentiment_col], axis=1)
    elif n_expected == 6:
        # Legacy 6-feature model: OHLCV + sentiment
        ohlcv = data[['Open', 'High', 'Low', 'Close', 'Volume']].values[-look_back:]
        sentiment_col = np.full((look_back, 1), sentiment_compound)
        features = np.concatenate([ohlcv, sentiment_col], axis=1)
    else:
        return None

    scaled = feature_scaler.transform(features)
    X_test = scaled.reshape(1, look_back, n_expected)
    return X_test


# ============================================================
# NEXT-DAY PREDICTION (Anchored LSTM)
# ============================================================

def make_prediction(model, X_test, target_scaler, data):
    """Make next-day price prediction using LSTM, anchored to current price.

    The raw LSTM output suffers from 'regression to the mean' — when the
    current price is near the top of the training range, the model drags
    predictions down regardless of actual market direction.

    Fix: Extract the LSTM's predicted *percentage change* (using the last
    training-day scaled value as reference) and apply it to the real
    current price. This preserves the LSTM's directional signal while
    keeping the magnitude realistic.
    """
    current_price = float(data['Close'].iloc[-1])
    closes = data['Close'].values

    # What the model predicts (scaled 0-1)
    predicted_scaled = model.predict(X_test, verbose=0)
    raw_pred = float(predicted_scaled[0][0])

    # What the model "sees" as the last day's close (scaled 0-1)
    last_close_scaled = float(target_scaler.transform([[current_price]])[0][0])

    # LSTM's predicted change in scaled space
    if last_close_scaled > 0.001:
        scaled_change_pct = (raw_pred - last_close_scaled) / last_close_scaled
    else:
        scaled_change_pct = 0.0

    # Adaptive cap based on stock's actual historical volatility
    if len(closes) >= 60:
        daily_returns = np.diff(closes[-60:]) / closes[-60:-1]
        daily_vol = float(np.std(daily_returns))
    else:
        daily_vol = 0.02
    max_move = max(0.005, min(0.03, daily_vol * 2.0))  # 2x daily vol, floor 0.5%, ceiling 3%

    scaled_change_pct = max(-max_move, min(max_move, scaled_change_pct))

    # Apply to actual current price
    predicted_price = current_price * (1 + scaled_change_pct)
    return round(predicted_price, 2)


# ============================================================
# 7-DAY FORECAST (LSTM-driven with decay)
# ============================================================

def make_7day_prediction(model, X_test, target_scaler, data, sentiment_score=50):
    """Make 7-day forecast using LSTM day-1 signal with momentum decay."""
    current_price = float(data['Close'].iloc[-1])
    closes = data['Close'].values

    # Extract LSTM's predicted percentage change (same logic as make_prediction)
    pred_scaled = model.predict(X_test, verbose=0)
    raw_pred = float(pred_scaled[0][0])
    last_close_scaled = float(target_scaler.transform([[current_price]])[0][0])

    if last_close_scaled > 0.001:
        lstm_daily_pct = (raw_pred - last_close_scaled) / last_close_scaled
    else:
        lstm_daily_pct = 0.0

    # Historical volatility
    if len(closes) >= 60:
        daily_returns = np.diff(closes[-60:]) / closes[-60:-1]
        avg_daily_vol = float(np.std(daily_returns))
    else:
        avg_daily_vol = 0.02

    # Adaptive cap: 2x daily vol, floor 0.5%, ceiling 3%
    max_move = max(0.005, min(0.03, avg_daily_vol * 2.0))
    lstm_daily_pct = max(-max_move, min(max_move, lstm_daily_pct))

    # Generate 7-day predictions with stronger decay
    # Day 1: full signal, Day 7: 20% of signal (aggressive decay to prevent compounding)
    predictions = []
    price = current_price

    for day in range(7):
        decay = max(0.20, 1.0 - day * 0.13)
        daily_move = lstm_daily_pct * decay

        # Also cap each day's move at 1x daily vol (not the full 2x)
        daily_cap = avg_daily_vol * 1.0
        daily_move = max(-daily_cap, min(daily_cap, daily_move))

        price = price * (1 + daily_move)
        predictions.append(round(price, 2))

    return predictions
