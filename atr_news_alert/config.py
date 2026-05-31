"""Central configuration for the ATR News-Triggered Alert System.

Edit these values to tune the system. Everything is plain Python so there is
no extra config format to learn.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"   # auditable run artifacts: manifest + snapshots
OUTCOMES_CSV = RESULTS_DIR / "outcomes.csv"  # decision feedback loop / calibration

WATCHLIST_CSV = CONFIG_DIR / "watchlist.csv"
PRICES_CSV = DATA_DIR / "prices.csv"     # OHLC + ATR per ticker/date
NEWS_CSV = DATA_DIR / "news.csv"         # ingested, de-duplicated articles
ALERTS_CSV = DATA_DIR / "alerts.csv"     # triggered alerts (the output)
BACKTEST_CSV = DATA_DIR / "backtest.csv"  # forward-return stats per ticker
NEWS_ARCHIVE_CSV = DATA_DIR / "news_archive.csv"  # append-only historical news store

# --- ATR / volatility parameters --------------------------------------------
ATR_PERIOD = 14            # lookback for Wilder's ATR
PRICE_LOOKBACK = "5y"      # how much history to pull for ATR + backtest
PRICE_INTERVAL = "1d"

# A bar is a "volatility breach" when the close-to-close move, measured in ATR
# units, is at least this large. 1.5 ATR is a meaningful move for most equities.
ATR_BREACH_MULT = 1.5

# --- News parameters ---------------------------------------------------------
NEWS_LOOKBACK_HOURS = 48   # only news newer than this counts toward a trigger
MAX_HEADLINES_PER_TICKER = 25

# Lightweight finance lexicon for headline sentiment (offline, no API key).
BULLISH_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps",
    "rally", "rallies", "record", "upgrade", "upgrades", "raises", "raise",
    "growth", "profit", "wins", "win", "approval", "approved", "partnership",
    "expands", "expansion", "outperform", "buyback", "dividend", "strong",
    "bullish", "gains", "gain", "rises", "rise", "tops", "boost", "acquire",
    "acquisition", "deal", "breakthrough",
}
BEARISH_WORDS = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls",
    "slump", "slumps", "downgrade", "downgrades", "cuts", "cut", "loss",
    "losses", "lawsuit", "probe", "investigation", "recall", "warning",
    "warns", "weak", "bearish", "decline", "declines", "slides", "sinks",
    "layoffs", "bankruptcy", "fraud", "halt", "halts", "delay", "delays",
    "fine", "fined", "subpoena", "selloff", "tumble", "tumbles",
}

# --- LLM sentiment (optional) ------------------------------------------------
# When True AND an API key is present in the environment, headlines are
# classified by an LLM (sentiment + category + confidence) instead of the
# offline lexicon. With no key, the system silently falls back to the lexicon.
#   * set ANTHROPIC_API_KEY -> uses Claude (default model below)
#   * or set OPENAI_API_KEY  -> uses OpenAI
USE_LLM = True
LLM_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap for classification
LLM_OPENAI_MODEL = "gpt-4o-mini"
LLM_MAX_HEADLINES = 60      # cap headlines sent per scan to control cost
LLM_BATCH_SIZE = 20         # headlines per API call
NEWS_CATEGORIES = [
    "Earnings", "M&A", "Product Launch", "Legal", "Regulation", "Macro",
    "Management", "Analyst Rating", "Other",
]

# --- Alert thresholds --------------------------------------------------------
# Severity is scored 0-10; only alerts at/above this level are surfaced.
MIN_ALERT_SEVERITY = 4.0

# --- Scheduler ---------------------------------------------------------------
SCAN_INTERVAL_MINUTES = 15   # used by `python main.py watch`

# --- Backtest ----------------------------------------------------------------
FORWARD_HORIZONS = [1, 5]   # trading days to measure forward returns over

# --- Portfolio simulation ----------------------------------------------------
# Event-driven, equal-weight, non-overlapping (one unit position per name).
PORTFOLIO_HOLD_DAYS = 5      # trading days to hold each breach trade
PORTFOLIO_COST_BPS = 5.0     # round-trip cost charged per turnover unit (entry/exit)
TRADING_DAYS_PER_YEAR = 252  # annualization factor for CAGR / Sharpe
BASELINE_SEED = 42           # RNG seed for the random-date baseline (reproducibility)

# --- Parameter sweep (robustness) --------------------------------------------
SWEEP_HOLD_DAYS = [3, 5, 10]       # default grid axes for `python main.py sweep`
SWEEP_COST_BPS = [0.0, 1.0, 5.0]

# --- Event study (signal mechanism) ------------------------------------------
# Days after a breach to trace the average cumulative return ("signal signature").
EVENT_STUDY_MAX_DAYS = 10
# Market regime = equal-weight universe index above/below this moving average.
REGIME_MA_WINDOW = 50

# --- Networking --------------------------------------------------------------
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; ATRNewsAlert/1.0)"
