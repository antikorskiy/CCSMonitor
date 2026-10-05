# config.py

# Database and exchange
DB_PATH = "ccsmonitor.db"
BINANCE_BASE = "https://api.binance.com"

# Default asset and timeframe
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "5m"

# Supported symbols and timeframes
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "TONUSDT"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# How many candles to fetch per request (Binance limit = 1000)
FETCH_LIMIT = 1000

# Optional proxy, e.g. "socks5://127.0.0.1:1080"
PROXY = None

# Indicator windows
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14

# Feature windows
VOL_WINDOW = 20          # volatility window (std of returns)
VOLUME_WINDOW = 20       # window for volume_ratio

# Feature schema version — bump when the feature set changes
FEATURE_SCHEMA_VERSION = "v1"

# --- auto-sync ---
AUTO_SYNC_DEFAULT_ENABLED = True
AUTO_SYNC_DEFAULT_INTERVAL = 60     # seconds
AUTO_SYNC_MIN_INTERVAL = 5          # guard against too-frequent requests

# --- ML ---
ML_HORIZON_SECONDS = 3600        # prediction horizon (MVP = 1h)
ML_FLAT_THRESHOLD = 0.003        # ±0.3% — boundary between UP/DOWN and FLAT
ML_MIN_TRAIN_SAMPLES = 500       # do not train if fewer clean samples
ML_N_SPLITS = 5                  # walk-forward folds
ML_MODELS_DIR = "models"         # where to store trained boosters

# --- auto ML ---
AUTO_ML_DEFAULT_ENABLED = True
AUTO_PREDICT_ON_NEW_CANDLE = True
AUTO_PREDICT_MIN_INTERVAL_SEC = 60          # no more than once per minute
AUTO_TRAIN_MIN_INTERVAL_SEC = 6 * 3600      # at least 6 hours between trainings
AUTO_TRAIN_MIN_NEW_CANDLES = 500            # at least 500 new candles

# --- prediction tracking ---
PREDICTION_FLAT_EPS = 0.003   # threshold below which return is treated as flat

# --- prediction resolver ---
PREDICTION_RESOLVER_INTERVAL_SEC = 60   # how often to check PENDING predictions
PREDICTIONS_HISTORY_MAX = 100           # how many prediction arrows to draw on the chart

# --- Genome ---
GENOME_SIMILARITY_THRESHOLD = 0.90       # cosine similarity to attach to an existing genome
GENOME_MIN_SAMPLES = 60                  # minimum candles with features before building genomes
GENOME_REBUILD_EVERY_NEW_CANDLES = 5000  # full rebuild if this many new candles accumulated

GENOME_NAMES_A = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon",
                  "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda"]
GENOME_NAMES_B = ["trinity", "helix", "nexus", "prism", "vertex", "aurora",
                  "cascade", "vector", "quasar", "photon", "matrix", "lumen"]
GENOME_NAMES_C = ["core", "wave", "flame", "echo", "pulse", "spark",
                  "field", "node", "path", "ring", "flux", "seed"]

# --- Genome UI ---
GENOME_INSTANCES_PREVIEW_LIMIT = 200  # how many instances to show in the tree

# --- history ---
HISTORY_OPTIONS = ["1y", "2y", "3y", "5y", "Full"]
HISTORY_DEFAULT = "2y"
HISTORY_START_MS = 1502928000000   # 2017-08-17, Binance BTCUSDT launch
HISTORY_PAGE_SLEEP_SEC = 0.15      # sleep between kline pages
HISTORY_MAX_PAGES_PER_CYCLE = 100  # don't block the loop for too long in one go

# --- simulation ---
SIMULATION_DEFAULT_CAPITAL = 1000.0
SIMULATION_FEE_PCT = 0.001

# --- export ---
EXPORT_DEFAULT_DIR = "exports"
EXPORT_FORMATS = ["csv", "xlsx", "json"]

# --- network ---
NETWORK_REFRESH_MS = 1000                 # how often the Network tab refreshes
NETWORK_LATENCY_WINDOW = 100              # rolling window size for latency stats
NETWORK_REQUEST_TIMEOUT_SEC = 15
NETWORK_MAX_RETRIES = 5
NETWORK_INITIAL_BACKOFF_SEC = 1.0
NETWORK_MAX_BACKOFF_SEC = 60.0
NETWORK_429_FALLBACK_WAIT_SEC = 5         # if Retry-After header missing

PROXY_MODES = ["Direct", "HTTP", "HTTPS", "SOCKS4", "SOCKS5"]
PROXY_DEFAULT_MODE = "Direct"

# --- Multi-horizon ML ---
ML_HORIZONS = [
    ("5m",  300),
    ("15m", 900),
    ("1h",  3600),
    ("4h",  14400),
    ("24h", 86400),
]
ML_DEFAULT_HORIZON = 3600

ML_CALIBRATION = "isotonic"

SHAP_BACKGROUND_SIZE = 200
SHAP_MAX_ROWS = 500

CONSENSUS_MIN_CONFIDENCE = 0.55
CONSENSUS_TIMEFRAMES = ["5m", "15m", "1h"]
CONSENSUS_WEIGHTS = {"5m": 1.0, "15m": 1.5, "1h": 2.0}

# --- simulation extensions ---
SIMULATION_STOP_LOSS_PCT = 0.02
SIMULATION_TAKE_PROFIT_PCT = 0.04
SIMULATION_SIZING_MODE = "fixed"          # fixed | percent | kelly

SIMULATION_SIZING_OPTIONS = ["fixed", "percent", "kelly"]
SIMULATION_POSITION_PCT = 0.10            # for "percent" mode
SIMULATION_KELLY_FRACTION = 0.25          # fractional Kelly
SIMULATION_FIXED_SIZE_USDT = 100.0

# --- UI / alerts ---
ALERTS_ENABLED_DEFAULT = True
ALERTS_PREDICTION_MADE = True
ALERTS_PREDICTION_RESOLVED = True
ALERTS_ACCURACY_DROP_PCT = 0.40
ALERTS_WINDOW = 50

HOTKEYS = {
    "load": "<Control-l>",
    "sync_now": "<Control-r>",
    "train": "<Control-t>",
    "predict": "<Control-p>",
    "predict_all": "<Control-Shift-P>",
    "consensus": "<Control-Shift-C>",
    "genomes": "<Control-g>",
    "insights": "<Control-i>",
    "simulate": "<Control-s>",
    "export": "<Control-e>",
    "reset_zoom": "<Control-0>",
    "zoom_in": "<plus>",
    "zoom_out": "<minus>",
    "network": "<F1>",
    "proxy": "<F2>",
    "pipeline": "<F3>",
    "coverage": "<F4>",
    "database": "<F5>",
    "logs": "<F6>",
    "close": "<Escape>",
}