"""
Validators and constants for the TradingView MCP server.

This module provides:
- Input validation functions for exchanges and timeframes
- Mappings between exchange names and TradingView screener types
- Timeframe conversion utilities
- Configuration constants
"""

from __future__ import annotations
import os
from typing import Set, Dict, Optional


# =============================================================================
# Timeframe Constants
# =============================================================================

# Valid timeframes supported by the server
ALLOWED_TIMEFRAMES: Set[str] = {"5m", "15m", "1h", "4h", "1D", "1W", "1M"}

# Mapping from our timeframe format to TradingView resolution format
# Used for API column suffixes (e.g., "close|240" for 4h)
TIMEFRAME_RESOLUTION_MAP: Dict[str, str] = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1D": "1D",
    "1W": "1W",
    "1M": "1M",
}


# =============================================================================
# Processing Constants
# =============================================================================

# Batch size for processing symbols in API calls
# TradingView has rate limits, so we process in batches
DEFAULT_BATCH_SIZE: int = 200

# Maximum number of symbols to process in a single scan
MAX_SYMBOLS_PER_SCAN: int = 500

# Default limits for various operations
DEFAULT_LIMIT: int = 25
MAX_LIMIT_STANDARD: int = 50
MAX_LIMIT_EXTENDED: int = 100


# =============================================================================
# Pattern Detection Thresholds
# =============================================================================

# Candle body ratio thresholds for pattern detection
STRONG_BODY_RATIO: float = 0.7
MODERATE_BODY_RATIO: float = 0.5
WEAK_BODY_RATIO: float = 0.3

# Volume thresholds
VOLUME_MINIMUM: int = 1000
VOLUME_DECENT: int = 5000
VOLUME_HIGH: int = 10000

# RSI thresholds
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0
RSI_NEUTRAL_HIGH: float = 55.0
RSI_NEUTRAL_LOW: float = 45.0

# Bollinger Band Width thresholds
BBW_HIGH_VOLATILITY: float = 0.05
BBW_MEDIUM_VOLATILITY: float = 0.02
BBW_SQUEEZE_THRESHOLD: float = 0.04

# ADX threshold for trend strength
ADX_STRONG_TREND: float = 25.0


# =============================================================================
# Exchange Mappings
# =============================================================================

# Maps exchange names to TradingView screener types
# Screener types: "crypto", "america", "turkey", "malaysia", "hongkong"
EXCHANGE_SCREENER: Dict[str, str] = {
    "all": "crypto",
    "huobi": "crypto",
    "kucoin": "crypto",
    "coinbase": "crypto",
    "gateio": "crypto",
    "binance": "crypto",
    "bitfinex": "crypto",
    "bitget": "crypto",
    "bybit": "crypto",
    "okx": "crypto",
    "bist": "turkey",
    "nasdaq": "america",
    # Malaysia Stock Market Support
    "bursa": "malaysia",
    "myx": "malaysia",
    "klse": "malaysia",
    "ace": "malaysia",      # ACE Market (Access, Certainty, Efficiency)
    "leap": "malaysia",     # LEAP Market (Leading Entrepreneur Accelerator Platform)
    # Hong Kong Stock Market Support
    "hkex": "hongkong",     # Hong Kong Exchange
    "hk": "hongkong",       # Hong Kong (alternate)
    "hsi": "hongkong",      # Hang Seng Index constituents
    "nyse": "america",
}

# Get absolute path to coinlist directory relative to this module
# This file is at: src/tradingview_mcp/core/utils/validators.py
# We want: src/tradingview_mcp/coinlist/
_this_file = __file__
_utils_dir = os.path.dirname(_this_file)  # core/utils
_core_dir = os.path.dirname(_utils_dir)   # core  
_package_dir = os.path.dirname(_core_dir) # tradingview_mcp
COINLIST_DIR = os.path.join(_package_dir, 'coinlist')


# =============================================================================
# Validation Functions
# =============================================================================

def sanitize_timeframe(tf: str, default: str = "5m") -> str:
    """
    Validate and normalize a timeframe string.

    Args:
        tf: Input timeframe string (e.g., "15m", "1h", "1D")
        default: Default value if input is invalid

    Returns:
        Valid timeframe string from ALLOWED_TIMEFRAMES, or default

    Example:
        >>> sanitize_timeframe("1h")
        '1h'
        >>> sanitize_timeframe("invalid")
        '5m'
    """
    if not tf:
        return default
    tfs = tf.strip()
    return tfs if tfs in ALLOWED_TIMEFRAMES else default


def sanitize_exchange(ex: str, default: str = "kucoin") -> str:
    """
    Validate and normalize an exchange name.

    Args:
        ex: Input exchange name (case-insensitive)
        default: Default value if input is invalid

    Returns:
        Lowercase exchange name from EXCHANGE_SCREENER, or default

    Example:
        >>> sanitize_exchange("BINANCE")
        'binance'
        >>> sanitize_exchange("unknown")
        'kucoin'
    """
    if not ex:
        return default
    exs = ex.strip().lower()
    return exs if exs in EXCHANGE_SCREENER else default


def tf_to_tv_resolution(tf: Optional[str]) -> Optional[str]:
    """
    Convert our timeframe format to TradingView resolution format.

    This is used when building column names for the TradingView API,
    which uses suffixes like "close|240" for 4-hour data.

    Args:
        tf: Timeframe string (e.g., "5m", "15m", "1h", "4h", "1D")

    Returns:
        TradingView resolution string, or None if invalid

    Example:
        >>> tf_to_tv_resolution("4h")
        '240'
        >>> tf_to_tv_resolution("1D")
        '1D'
        >>> tf_to_tv_resolution("invalid")
        None
    """
    if not tf:
        return None
    return TIMEFRAME_RESOLUTION_MAP.get(tf)
