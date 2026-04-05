"""
Central Configuration for StockSense AI — Indian NSE Edition
Stock universe, feature lists, constants, and search term mappings.
"""

# ============================================================
# STOCK UNIVERSE — 19 High-Volume Indian NSE Stocks (all >1M daily volume)
# ============================================================
STOCK_UNIVERSE = {
    "IRFC.NS": {
        "name": "Indian Railway Finance Corp",
        "sector": "Financial Services",
        "search_terms": "IRFC Indian Railway Finance",
        "domain": "irfc.nic.in",
    },
    "SUZLON.NS": {
        "name": "Suzlon Energy",
        "sector": "Renewable Energy",
        "search_terms": "Suzlon Energy wind",
        "domain": "suzlon.com",
    },
    "NHPC.NS": {
        "name": "NHPC Limited",
        "sector": "Power",
        "search_terms": "NHPC hydropower",
        "domain": "nhpcindia.com",
    },
    "NMDC.NS": {
        "name": "NMDC Limited",
        "sector": "Mining",
        "search_terms": "NMDC Iron Ore",
        "domain": "nmdc.co.in",
    },
    "PNB.NS": {
        "name": "Punjab National Bank",
        "sector": "Banking",
        "search_terms": "Punjab National Bank PNB",
        "domain": "pnbindia.in",
    },
    "CANBK.NS": {
        "name": "Canara Bank",
        "sector": "Banking",
        "search_terms": "Canara Bank",
        "domain": "canarabank.com",
    },
    "UNIONBANK.NS": {
        "name": "Union Bank of India",
        "sector": "Banking",
        "search_terms": "Union Bank of India",
        "domain": "unionbankofindia.co.in",
    },
    "ADANIPOWER.NS": {
        "name": "Adani Power",
        "sector": "Power",
        "search_terms": "Adani Power",
        "domain": "adanipower.com",
    },
    "TATASTEEL.NS": {
        "name": "Tata Steel",
        "sector": "Metals & Mining",
        "search_terms": "Tata Steel",
        "domain": "tatasteel.com",
    },
    "COALINDIA.NS": {
        "name": "Coal India",
        "sector": "Mining",
        "search_terms": "Coal India CIL",
        "domain": "coalindia.in",
    },
    "RECLTD.NS": {
        "name": "REC Limited",
        "sector": "Financial Services",
        "search_terms": "REC Limited Rural Electrification",
        "domain": "recindia.nic.in",
    },
    "IOC.NS": {
        "name": "Indian Oil Corporation",
        "sector": "Oil & Gas",
        "search_terms": "Indian Oil Corporation IOC",
        "domain": "iocl.com",
    },
    "BEL.NS": {
        "name": "Bharat Electronics",
        "sector": "Defence",
        "search_terms": "Bharat Electronics BEL defence",
        "domain": "bel-india.in",
    },
    "INDUSTOWER.NS": {
        "name": "Indus Towers",
        "sector": "Telecom Infrastructure",
        "search_terms": "Indus Towers Bharti Infratel",
        "domain": "industowers.com",
    },
    "DLF.NS": {
        "name": "DLF Limited",
        "sector": "Real Estate",
        "search_terms": "DLF real estate",
        "domain": "dlf.in",
    },
    "HINDUNILVR.NS": {
        "name": "Hindustan Unilever",
        "sector": "FMCG",
        "search_terms": "Hindustan Unilever HUL",
        "domain": "hul.co.in",
    },
    "PERSISTENT.NS": {
        "name": "Persistent Systems",
        "sector": "IT Services",
        "search_terms": "Persistent Systems",
        "domain": "persistent.com",
    },
    "SAIL.NS": {
        "name": "Steel Authority of India",
        "sector": "Metals & Mining",
        "search_terms": "SAIL Steel Authority",
        "domain": "sail.co.in",
    },
    "GMRAIRPORT.NS": {
        "name": "GMR Airports Infrastructure",
        "sector": "Infrastructure",
        "search_terms": "GMR Airports Infrastructure",
        "domain": "gmrgroup.in",
    },
    "SUNPHARMA.NS": {
        "name": "Sun Pharmaceutical Industries",
        "sector": "Pharmaceuticals",
        "search_terms": "Sun Pharma pharmaceutical",
        "domain": "sunpharma.com",
    },
}

# Ordered list of tickers for UI dropdown
TICKERS = sorted(STOCK_UNIVERSE.keys())

# ============================================================
# MODEL / TRAINING CONSTANTS
# ============================================================
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI_14', 'MACD_hist', 'BB_width', 'sentiment_10d_avg',
]
N_FEATURES = len(FEATURES)

SEQUENCE_LENGTH = 60
DATA_START = '2016-02-27'
DATA_END = '2026-03-27'

CURRENCY_SYMBOL = '\u20b9'  # ₹

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0003
PATIENCE = 20
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.30

# ============================================================
# RECOMMENDATION THRESHOLDS
# ============================================================
STRONG_BUY_THRESHOLD = 80
BUY_THRESHOLD = 65
HOLD_THRESHOLD = 45
SELL_THRESHOLD = 30
# Below SELL_THRESHOLD → STRONG SELL

# ============================================================
# HELPERS
# ============================================================

def get_display_name(ticker):
    """Return human-readable name for a ticker."""
    info = STOCK_UNIVERSE.get(ticker)
    return info["name"] if info else ticker


def get_search_term(ticker):
    """Return search terms for news/reddit queries."""
    info = STOCK_UNIVERSE.get(ticker)
    return info["search_terms"] if info else ticker.replace(".NS", "")


def get_sector(ticker):
    """Return sector for a ticker."""
    info = STOCK_UNIVERSE.get(ticker)
    return info["sector"] if info else "N/A"


def get_logo_url(ticker):
    """Return a logo URL via Google favicon service."""
    info = STOCK_UNIVERSE.get(ticker)
    if info and info.get("domain"):
        return f"https://www.google.com/s2/favicons?domain={info['domain']}&sz=128"
    return ""
