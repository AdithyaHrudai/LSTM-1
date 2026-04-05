"""
Unified LSTM Training Script for StockSense AI — Indian NSE Edition
===================================================================
Trains on all Indian NSE stocks with 9 features including technicals + sentiment.
Data: 2016-03-01 to 2026-03-01 (10 years)

Usage:
    python train_model.py

Output (in models/ folder):
    stage2_india_lstm_YYYYMMDD_HHMMSS.keras
    stage2_india_scalers_YYYYMMDD_HHMMSS.pkl
    stage2_india_architecture_YYYYMMDD_HHMMSS.txt
    stage2_india_metrics_YYYYMMDD_HHMMSS.json
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import json
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Import ta library for technical indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from config import (
    STOCK_UNIVERSE, TICKERS, FEATURES, N_FEATURES,
    SEQUENCE_LENGTH, DATA_START, DATA_END,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    get_display_name,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
PREFIX = 'stage2_india'

print("=" * 65)
print("  STOCKSENSE AI — UNIFIED INDIAN NSE MODEL TRAINER")
print(f"  Training Date : {TIMESTAMP}")
print(f"  Data Range    : {DATA_START}  ->  {DATA_END}")
print(f"  Stocks        : {len(TICKERS)} Indian NSE companies")
print(f"  Features      : {N_FEATURES} ({', '.join(FEATURES)})")
print(f"  Architecture  : LSTM({LSTM_UNITS_1}) -> LSTM({LSTM_UNITS_2}) -> Dense({DENSE_UNITS}) -> Dense(1)")
print(f"  GPU available : {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 65)


# ============================================================
# STEP 1 — Download Data
# ============================================================

def download_stock_data(ticker, start, end):
    """Download OHLCV data from Yahoo Finance."""
    name = get_display_name(ticker)
    print(f"  {name:35s} ({ticker:18s}) ...", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            print("EMPTY x")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        print(f"{len(df):>5d} rows  OK")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def add_technical_features(df):
    """Add RSI_14, MACD_hist, BB_width using ta library."""
    close = df['Close']
    high = df['High']
    low = df['Low']

    # RSI(14)
    rsi = RSIIndicator(close=close, window=14)
    df['RSI_14'] = rsi.rsi()

    # MACD histogram
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD_hist'] = macd.macd_diff()

    # Bollinger Band width
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df['BB_width'] = bb.bollinger_wband()

    return df


def add_sentiment_feature(df):
    """Build synthetic sentiment_10d_avg from price-action.
    Range: approximately -1 to +1."""
    returns = df['Close'].pct_change()
    raw = (returns / 0.03).clip(-1, 1)
    smoothed = raw.rolling(window=10, min_periods=1).mean()
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.05, len(smoothed))
    df['sentiment_10d_avg'] = (smoothed + noise).clip(-1, 1)
    return df


print(f"\n  STEP 1 / 7 — Downloading {len(TICKERS)} stocks ...")
all_stock_data = {}
for t in TICKERS:
    df = download_stock_data(t, DATA_START, DATA_END)
    if df is not None and len(df) > SEQUENCE_LENGTH + 100:
        df = add_technical_features(df)
        df = add_sentiment_feature(df)
        df.dropna(inplace=True)
        if len(df) > SEQUENCE_LENGTH + 50:
            all_stock_data[t] = df

print(f"\n  OK  {len(all_stock_data)} / {len(TICKERS)} stocks downloaded and processed")
if len(all_stock_data) < 10:
    raise SystemExit("  Too few stocks. Check internet connection and retry.")


# ============================================================
# STEP 2 — Fit Per-Stock Scalers
# ============================================================
print(f"\n  STEP 2 / 7 — Fitting per-stock scalers ...")

per_stock_scalers = {}
total_rows = 0
for ticker, df in all_stock_data.items():
    feat_sc = MinMaxScaler(feature_range=(0, 1))
    tgt_sc = MinMaxScaler(feature_range=(0, 1))
    feat_sc.fit(df[FEATURES].values)
    tgt_sc.fit(df[['Close']].values)
    per_stock_scalers[ticker] = {'feature_scaler': feat_sc, 'target_scaler': tgt_sc}
    total_rows += len(df)
    name = get_display_name(ticker)
    price_min = tgt_sc.data_min_[0]
    price_max = tgt_sc.data_max_[0]
    print(f"  {name:35s}: Rs.{price_min:>8.2f} -> Rs.{price_max:>8.2f}")

print(f"\n  Total rows fitted: {total_rows:,}")
print(f"  Stocks with scalers: {len(per_stock_scalers)}")


# ============================================================
# STEP 3 — Create Sequences
# ============================================================
print(f"\n  STEP 3 / 7 — Creating training sequences ...")


def create_sequences(df, feat_sc, tgt_sc, seq_len=60):
    feats = feat_sc.transform(df[FEATURES].values)
    tgts = tgt_sc.transform(df[['Close']].values).flatten()
    X, y = [], []
    for i in range(seq_len, len(feats)):
        X.append(feats[i - seq_len:i])
        y.append(tgts[i])
    return np.array(X), np.array(y)


all_X, all_y = [], []
for ticker, df in all_stock_data.items():
    scalers = per_stock_scalers[ticker]
    X, y = create_sequences(df, scalers['feature_scaler'], scalers['target_scaler'], SEQUENCE_LENGTH)
    all_X.append(X)
    all_y.append(y)
    name = get_display_name(ticker)
    print(f"  {name:35s}: {X.shape[0]:>5d} sequences")

X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
print(f"\n  Total sequences : {X_all.shape[0]:,}")

# Shuffle
rng = np.random.RandomState(42)
idx = np.arange(len(X_all))
rng.shuffle(idx)
X_all, y_all = X_all[idx], y_all[idx]

# 85 / 15 split
split = int(0.85 * len(X_all))
X_train, X_val = X_all[:split], X_all[split:]
y_train, y_val = y_all[:split], y_all[split:]
print(f"  Training        : {len(X_train):,}")
print(f"  Validation      : {len(X_val):,}")


# ============================================================
# STEP 4 — Build Model (upgraded architecture)
# ============================================================
print(f"\n  STEP 4 / 7 — Building LSTM model ...")

model = Sequential([
    LSTM(LSTM_UNITS_1, return_sequences=True,
         input_shape=(SEQUENCE_LENGTH, N_FEATURES)),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS_2, return_sequences=False),
    Dropout(DROPOUT_RATE),
    Dense(DENSE_UNITS, activation='relu'),
    Dense(1),
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae'],
)
model.summary()


# ============================================================
# STEP 5 — Train
# ============================================================
print(f"\n  STEP 5 / 7 — Training ...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

epochs_trained = len(history.history['loss'])
print(f"\n  Finished after {epochs_trained} epochs")


# ============================================================
# STEP 6 — Per-Stock Evaluation (using per-stock scalers)
# ============================================================
print(f"\n  STEP 6 / 7 — Per-stock evaluation ...")
print(f"  {'Stock':<36} {'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'R2':>8}")
print(f"  {'-' * 76}")

stock_metrics = {}
all_true_prices = []
all_pred_prices = []
for ticker, df in all_stock_data.items():
    scalers = per_stock_scalers[ticker]
    Xs, ys = create_sequences(df, scalers['feature_scaler'], scalers['target_scaler'], SEQUENCE_LENGTH)
    if len(Xs) == 0:
        continue
    ps = model.predict(Xs, verbose=0).flatten()
    pred = scalers['target_scaler'].inverse_transform(ps.reshape(-1, 1)).flatten()
    true = scalers['target_scaler'].inverse_transform(ys.reshape(-1, 1)).flatten()

    all_true_prices.extend(true.tolist())
    all_pred_prices.extend(pred.tolist())

    s_rmse = float(np.sqrt(mean_squared_error(true, pred)))
    s_mae = float(mean_absolute_error(true, pred))
    s_mape = float(np.mean(np.abs((true - pred) / true)) * 100)
    s_r2 = float(r2_score(true, pred))

    name = get_display_name(ticker)
    stock_metrics[ticker] = dict(
        rmse=round(s_rmse, 2), mae=round(s_mae, 2),
        mape=round(s_mape, 2), r2=round(s_r2, 4), name=name,
    )
    print(f"  {name:<36} Rs.{s_rmse:>8.2f} Rs.{s_mae:>8.2f} {s_mape:>7.2f}% {s_r2:>7.4f}")

avg_mape = np.mean([m['mape'] for m in stock_metrics.values()])
print(f"\n  Average MAPE : {avg_mape:.2f}%")

# Overall metrics from aggregated per-stock predictions
all_true_prices = np.array(all_true_prices)
all_pred_prices = np.array(all_pred_prices)
rmse = float(np.sqrt(mean_squared_error(all_true_prices, all_pred_prices)))
mae = float(mean_absolute_error(all_true_prices, all_pred_prices))
r2 = float(r2_score(all_true_prices, all_pred_prices))
mape = float(avg_mape)

print(f"\n  {'=' * 34}")
print(f"  OVERALL RESULTS")
print(f"  {'=' * 34}")
print(f"  RMSE  : Rs.{rmse:.2f}")
print(f"  MAE   : Rs.{mae:.2f}")
print(f"  R2    :  {r2:.4f}")
print(f"  MAPE  :  {mape:.2f}%")
print(f"  {'=' * 34}")

if mape < 3:
    print("  Warning: MAPE < 3% — may be slightly overfit")
elif mape <= 12:
    print("  OK: MAPE in realistic range (3 - 12%)")
else:
    print("  Warning: MAPE > 12% — model may be underfitting")


# ============================================================
# STEP 8 — Save Everything
# ============================================================
print(f"\n  STEP 7 / 7 — Saving model & scalers ...")

model_file = f"{PREFIX}_lstm_{TIMESTAMP}.keras"
scaler_file = f"{PREFIX}_scalers_{TIMESTAMP}.pkl"
arch_file = f"{PREFIX}_architecture_{TIMESTAMP}.txt"
metrics_file = f"{PREFIX}_metrics_{TIMESTAMP}.json"

model_path = os.path.join(MODELS_DIR, model_file)
scaler_path = os.path.join(MODELS_DIR, scaler_file)
arch_path = os.path.join(MODELS_DIR, arch_file)
metrics_path = os.path.join(MODELS_DIR, metrics_file)

# Save model
model.save(model_path)
print(f"  Model   -> {model_path}")

# Save per-stock scalers (keyed by ticker)
joblib.dump(per_stock_scalers, scaler_path)
print(f"  Scalers -> {scaler_path} ({len(per_stock_scalers)} per-stock scalers)")

# Save metrics JSON
metrics_json = {
    'timestamp': TIMESTAMP,
    'market': 'India',
    'currency': 'INR',
    'data_range': f"{DATA_START} to {DATA_END}",
    'stocks': sorted(all_stock_data.keys()),
    'stock_names': {t: get_display_name(t) for t in sorted(all_stock_data.keys())},
    'n_stocks': len(all_stock_data),
    'n_features': N_FEATURES,
    'features': FEATURES,
    'overall': dict(rmse=round(rmse, 2), mae=round(mae, 2),
                    r2=round(r2, 4), mape=round(mape, 2)),
    'per_stock': stock_metrics,
    'epochs_trained': epochs_trained,
    'best_val_loss': round(float(min(history.history['val_loss'])), 6),
    'architecture': {
        'lstm_1': LSTM_UNITS_1,
        'lstm_2': LSTM_UNITS_2,
        'dense': DENSE_UNITS,
        'dropout': DROPOUT_RATE,
        'sequence_length': SEQUENCE_LENGTH,
        'features': N_FEATURES,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    },
}
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=2)
print(f"  Metrics -> {metrics_path}")

# Save architecture text
stocks_bullet = '\n'.join(f'- {get_display_name(t)} ({t})' for t in sorted(all_stock_data.keys()))
arch_text = f"""StockSense AI — Indian NSE LSTM Model Architecture
=============================================

Model Type: Sequential LSTM (upgraded)
Market: India (NSE) — Currency: INR
Training Date: {TIMESTAMP}
Data Range: {DATA_START} to {DATA_END}
Training Stocks: {len(all_stock_data)} Indian companies
Features: {N_FEATURES} features
Target: Close (INR)

Architecture:
1. LSTM Layer 1: {LSTM_UNITS_1} units, return_sequences=True
2. Dropout: {DROPOUT_RATE}
3. LSTM Layer 2: {LSTM_UNITS_2} units, return_sequences=False
4. Dropout: {DROPOUT_RATE}
5. Dense: {DENSE_UNITS} units, ReLU activation
6. Dense Output: 1 unit

Training Parameters:
- Batch Size: {BATCH_SIZE}
- Learning Rate: {LEARNING_RATE}
- Epochs Trained: {epochs_trained}
- Early Stopping Patience: {PATIENCE}
- LR Reduction Patience: 8

Validation Performance:
- RMSE: Rs.{rmse:.2f}
- MAE: Rs.{mae:.2f}
- R2: {r2:.4f}
- MAPE: {mape:.2f}%

Features Used ({N_FEATURES}):
{chr(10).join(f'- {f}' for f in FEATURES)}

Stocks Trained On:
{stocks_bullet}

Usage:
1. model = keras.models.load_model('models/{model_file}', compile=False)
2. scalers = joblib.load('models/{scaler_file}')  # dict of per-stock scalers
3. feat_sc = scalers['TICKER.NS']['feature_scaler']
4. tgt_sc = scalers['TICKER.NS']['target_scaler']
5. Scale input (shape: batch, 60, {N_FEATURES}) with feat_sc
6. Inverse-transform predictions with tgt_sc (output in INR)
"""
with open(arch_path, 'w', encoding='utf-8') as f:
    f.write(arch_text)
print(f"  Arch    -> {arch_path}")

print(f"\n{'=' * 65}")
print(f"  TRAINING COMPLETE!")
print(f"{'=' * 65}")
print(f"  Model file  : {model_file}")
print(f"  Scaler file : {scaler_file}")
print(f"  MAPE        : {mape:.2f}%")
print(f"  Epochs      : {epochs_trained}")
print(f"  Features    : {N_FEATURES}")
print(f"\n  Next:  streamlit run app.py")
print(f"{'=' * 65}")
