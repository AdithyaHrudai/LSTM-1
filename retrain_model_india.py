"""
StockSense AI — Indian NSE LSTM Model Retrainer (Production)
=============================================================
Trains on the SAME 19 Indian stocks, SAME 9 features, SAME architecture
as the current production model. Only updates the data range.

Usage:
    python retrain_model_india.py

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

# ============================================================
# CONFIGURATION — MUST MATCH config.py and production model
# ============================================================

# Same 19 stocks as config.py STOCK_UNIVERSE
STOCKS = [
    'IRFC.NS', 'SUZLON.NS', 'NHPC.NS', 'NMDC.NS', 'PNB.NS',
    'CANBK.NS', 'UNIONBANK.NS', 'ADANIPOWER.NS', 'TATASTEEL.NS',
    'SAIL.NS', 'IOC.NS', 'COALINDIA.NS', 'BEL.NS', 'RECLTD.NS',
    'HINDUNILVR.NS', 'DLF.NS', 'PERSISTENT.NS', 'INDUSTOWER.NS',
    'GMRAIRPORT.NS', 'SUNPHARMA.NS',
]

STOCK_NAMES = {
    'IRFC.NS': 'Indian Railway Finance Corp',
    'SUZLON.NS': 'Suzlon Energy',
    'NHPC.NS': 'NHPC Limited',
    'NMDC.NS': 'NMDC Limited',
    'PNB.NS': 'Punjab National Bank',
    'CANBK.NS': 'Canara Bank',
    'UNIONBANK.NS': 'Union Bank of India',
    'ADANIPOWER.NS': 'Adani Power',
    'TATASTEEL.NS': 'Tata Steel',
    'SAIL.NS': 'Steel Authority of India',
    'IOC.NS': 'Indian Oil Corporation',
    'COALINDIA.NS': 'Coal India',
    'BEL.NS': 'Bharat Electronics',
    'RECLTD.NS': 'REC Limited',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'DLF.NS': 'DLF Limited',
    'PERSISTENT.NS': 'Persistent Systems',
    'INDUSTOWER.NS': 'Indus Towers',
    'GMRAIRPORT.NS': 'GMR Airports Infrastructure',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
}

# ── Data range: 10 years up to 27 March 2026 ──
DATA_START = '2016-02-27'
DATA_END   = '2026-03-28'   # yfinance end is exclusive, so this fetches up to 27 March

# ── Same 9 features as production ──
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume',
            'RSI_14', 'MACD_hist', 'BB_width', 'sentiment_10d_avg']
N_FEATURES = len(FEATURES)

SEQUENCE_LENGTH = 60

# ── Same architecture as production model ──
BATCH_SIZE    = 32
EPOCHS        = 150
LEARNING_RATE = 0.0003
PATIENCE      = 20
LSTM_UNITS_1  = 128
LSTM_UNITS_2  = 64
DENSE_UNITS   = 32
DROPOUT_RATE  = 0.30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
PREFIX = 'stage2_india'

print("=" * 60)
print("  STOCKSENSE AI — LSTM MODEL RETRAINER (Production)")
print(f"  Training Date : {TIMESTAMP}")
print(f"  Data Range    : {DATA_START}  →  2026-03-27")
print(f"  Stocks        : {len(STOCKS)} Indian companies (NSE)")
print(f"  Features      : {N_FEATURES}")
print(f"  Architecture  : LSTM({LSTM_UNITS_1}) → LSTM({LSTM_UNITS_2}) → Dense({DENSE_UNITS}) → Dense(1)")
print(f"  Currency      : INR (₹)")
print(f"  GPU available : {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 60)


# ============================================================
# STEP 1 — Download Data
# ============================================================
def download_stock_data(ticker, start, end):
    """Download OHLCV data from Yahoo Finance (NSE tickers)."""
    name = STOCK_NAMES.get(ticker, ticker)
    print(f"  {name:40s} ({ticker:18s}) ...", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            print("EMPTY ✗")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        print(f"{len(df):>5d} rows  ✓")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None


# ============================================================
# STEP 2 — Compute Technical Indicators (same as technical.py)
# ============================================================
def compute_features(df):
    """Add RSI_14, MACD_hist, BB_width, and sentiment_10d_avg to dataframe."""
    close = df['Close']

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - signal_line

    # Bollinger Band width
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_width'] = (upper - lower) / sma20

    # Synthetic sentiment (same as previous model)
    returns = close.pct_change()
    raw = (returns / 0.03).clip(-1, 1)
    smoothed = raw.rolling(window=10, min_periods=1).mean()
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.05, len(smoothed))
    df['sentiment_10d_avg'] = (smoothed + noise).clip(-1, 1)

    df.dropna(inplace=True)
    return df


print("\n📥  STEP 1 / 8 — Downloading stock data …")
all_stock_data = {}
for t in STOCKS:
    df = download_stock_data(t, DATA_START, DATA_END)
    if df is not None and len(df) > SEQUENCE_LENGTH + 50:
        df = compute_features(df)
        if len(df) > SEQUENCE_LENGTH + 50:
            all_stock_data[t] = df

print(f"\n  ✅  {len(all_stock_data)} / {len(STOCKS)} stocks downloaded")
if len(all_stock_data) < 15:
    raise SystemExit("❌  Too few stocks. Check internet connection and retry.")

# Print date range for each stock
print("\n  Data ranges:")
for t, df in sorted(all_stock_data.items()):
    name = STOCK_NAMES.get(t, t)
    print(f"    {name:40s} {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} rows)")


# ============================================================
# STEP 2 — Per-Stock Scalers (same format as production)
# ============================================================
print("\n📊  STEP 2 / 8 — Fitting per-stock scalers …")

per_stock_scalers = {}
for ticker, df in all_stock_data.items():
    feat_sc = MinMaxScaler(feature_range=(0, 1))
    tgt_sc  = MinMaxScaler(feature_range=(0, 1))
    feat_sc.fit(df[FEATURES].values)
    tgt_sc.fit(df[['Close']].values)
    per_stock_scalers[ticker] = {
        'feature_scaler': feat_sc,
        'target_scaler': tgt_sc,
    }
    name = STOCK_NAMES.get(ticker, ticker)
    print(f"  {name:40s} Close range: ₹{tgt_sc.data_min_[0]:.2f} → ₹{tgt_sc.data_max_[0]:.2f}")


# ============================================================
# STEP 3 — Create Sequences
# ============================================================
print("\n🔧  STEP 3 / 8 — Creating training sequences …")

def create_sequences(df, feat_sc, tgt_sc, seq_len=60):
    feats = feat_sc.transform(df[FEATURES].values)
    tgts  = tgt_sc.transform(df[['Close']].values).flatten()
    X, y = [], []
    for i in range(seq_len, len(feats)):
        X.append(feats[i - seq_len:i])
        y.append(tgts[i])
    return np.array(X), np.array(y)


all_X, all_y = [], []
for ticker, df in all_stock_data.items():
    sc = per_stock_scalers[ticker]
    X, y = create_sequences(df, sc['feature_scaler'], sc['target_scaler'], SEQUENCE_LENGTH)
    all_X.append(X)
    all_y.append(y)
    name = STOCK_NAMES.get(ticker, ticker)
    print(f"  {name:40s}: {X.shape[0]:>5d} sequences")

X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
print(f"\n  Total sequences : {X_all.shape[0]:,}")
print(f"  Shape           : {X_all.shape}")

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
# STEP 4 — Build Model (same architecture as production)
# ============================================================
print("\n🧠  STEP 4 / 8 — Building LSTM model …")

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
print(f"\n🚀  STEP 5 / 8 — Training (EarlyStopping patience={PATIENCE}) …")

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
best_val = min(history.history['val_loss'])
print(f"\n  Finished after {epochs_trained} epochs (best val_loss: {best_val:.6f})")


# ============================================================
# STEP 6 — Overall Evaluation
# ============================================================
print("\n📈  STEP 6 / 8 — Evaluating on validation set …")

y_pred_scaled = model.predict(X_val, verbose=0).flatten()

# Use the first stock's scaler for overall metrics (approximate)
first_ticker = list(all_stock_data.keys())[0]
ref_tgt_sc = per_stock_scalers[first_ticker]['target_scaler']
y_pred_approx = ref_tgt_sc.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true_approx = ref_tgt_sc.inverse_transform(y_val.reshape(-1, 1)).flatten()

rmse = float(np.sqrt(mean_squared_error(y_true_approx, y_pred_approx)))
mae  = float(mean_absolute_error(y_true_approx, y_pred_approx))
r2   = float(r2_score(y_true_approx, y_pred_approx))
mape = float(np.mean(np.abs((y_true_approx - y_pred_approx) /
                             np.where(y_true_approx == 0, 1, y_true_approx))) * 100)

print(f"\n  {'─' * 34}")
print(f"  VALIDATION RESULTS (INR)")
print(f"  {'─' * 34}")
print(f"  RMSE  : ₹{rmse:.2f}")
print(f"  MAE   : ₹{mae:.2f}")
print(f"  R²    :  {r2:.4f}")
print(f"  MAPE  :  {mape:.2f}%")
print(f"  {'─' * 34}")


# ============================================================
# STEP 7 — Per-Stock Evaluation
# ============================================================
print("\n📊  STEP 7 / 8 — Per-stock evaluation …")
print(f"  {'Stock':<42} {'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'R²':>8}")
print(f"  {'─' * 82}")

stock_metrics = {}
for ticker, df in all_stock_data.items():
    sc = per_stock_scalers[ticker]
    Xs, ys = create_sequences(df, sc['feature_scaler'], sc['target_scaler'], SEQUENCE_LENGTH)
    if len(Xs) == 0:
        continue
    ps = model.predict(Xs, verbose=0).flatten()
    pred = sc['target_scaler'].inverse_transform(ps.reshape(-1, 1)).flatten()
    true = sc['target_scaler'].inverse_transform(ys.reshape(-1, 1)).flatten()

    s_rmse = float(np.sqrt(mean_squared_error(true, pred)))
    s_mae  = float(mean_absolute_error(true, pred))
    s_mape = float(np.mean(np.abs((true - pred) / np.where(true == 0, 1, true))) * 100)
    s_r2   = float(r2_score(true, pred))

    name = STOCK_NAMES.get(ticker, ticker)
    stock_metrics[ticker] = dict(rmse=round(s_rmse, 2), mae=round(s_mae, 2),
                                  mape=round(s_mape, 2), r2=round(s_r2, 4),
                                  name=name)
    print(f"  {name:<42} ₹{s_rmse:>9.2f} ₹{s_mae:>9.2f} {s_mape:>7.2f}% {s_r2:>7.4f}")

avg_mape = np.mean([m['mape'] for m in stock_metrics.values()])
print(f"\n  Average MAPE : {avg_mape:.2f}%")


# ============================================================
# STEP 8 — Save Everything
# ============================================================
print("\n💾  STEP 8 / 8 — Saving model & scalers …")

model_file   = f"{PREFIX}_lstm_{TIMESTAMP}.keras"
scaler_file  = f"{PREFIX}_scalers_{TIMESTAMP}.pkl"
arch_file    = f"{PREFIX}_architecture_{TIMESTAMP}.txt"
metrics_file = f"{PREFIX}_metrics_{TIMESTAMP}.json"

model_path   = os.path.join(MODELS_DIR, model_file)
scaler_path  = os.path.join(MODELS_DIR, scaler_file)
arch_path    = os.path.join(MODELS_DIR, arch_file)
metrics_path = os.path.join(MODELS_DIR, metrics_file)

# Save model
model.save(model_path)
print(f"  Model   → {model_path}")

# Save per-stock scalers (same format model.py expects)
joblib.dump(per_stock_scalers, scaler_path)
print(f"  Scalers → {scaler_path}")

# Save metrics JSON
data_end_display = max(df.index[-1].strftime('%Y-%m-%d') for df in all_stock_data.values())
metrics_json = {
    'timestamp': TIMESTAMP,
    'market': 'India',
    'currency': 'INR',
    'data_range': f"{DATA_START} to {data_end_display}",
    'stocks': sorted(all_stock_data.keys()),
    'stock_names': {t: STOCK_NAMES.get(t, t) for t in sorted(all_stock_data.keys())},
    'n_stocks': len(all_stock_data),
    'n_features': N_FEATURES,
    'features': FEATURES,
    'overall': dict(rmse=round(rmse, 2), mae=round(mae, 2),
                    r2=round(r2, 4), mape=round(mape, 2)),
    'per_stock': stock_metrics,
    'epochs_trained': epochs_trained,
    'best_val_loss': round(float(best_val), 6),
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
print(f"  Metrics → {metrics_path}")

# Save architecture text
stocks_bullet = '\n'.join(f'- {STOCK_NAMES.get(t, t)} ({t})' for t in sorted(all_stock_data.keys()))
arch_text = f"""StockSense AI — Indian NSE LSTM Model Architecture
=============================================

Model Type: Sequential LSTM (upgraded)
Market: India (NSE) — Currency: INR
Training Date: {TIMESTAMP}
Data Range: {DATA_START} to {data_end_display}
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
- Open
- High
- Low
- Close
- Volume
- RSI_14
- MACD_hist
- BB_width
- sentiment_10d_avg

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
print(f"  ReadMe  → {arch_path}")

print(f"\n{'=' * 60}")
print(f"  ✅  MODEL RETRAINING COMPLETE!")
print(f"{'=' * 60}")
print(f"  Model file  : {model_file}")
print(f"  Scaler file : {scaler_file}")
print(f"  Data range  : {DATA_START} → {data_end_display}")
print(f"  MAPE        : {mape:.2f}%")
print(f"  Epochs      : {epochs_trained}")
print(f"\n  ➡️  Now run:  streamlit run app.py")
print(f"{'=' * 60}")
