"""
Input Sanitization and Validation for SigmaPilot MCP.

This module provides functions for validating and sanitizing user inputs
before they're used in analysis engines. All functions are pure and return
sanitized values with sensible defaults.

Key Features:
- Timeframe validation and normalization
- Exchange validation and normalization
- Symbol format validation
- Limit/count sanitization
- OHLCV data validation
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, Set

from .timeframes import TIMEFRAME_HIERARCHY


# =============================================================================
# Constants
# =============================================================================

# Valid timeframes (imported from timeframes module for single source of truth)
ALLOWED_TIMEFRAMES: Set[str] = set(TIMEFRAME_HIERARCHY)

# Exchange to screener type mapping
EXCHANGE_SCREENER: Dict[str, str] = {
    # Crypto exchanges
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
    # Stock exchanges
    "bist": "turkey",
    "nasdaq": "america",
    "nyse": "america",
    # Malaysia
    "bursa": "malaysia",
    "myx": "malaysia",
    "klse": "malaysia",
    "ace": "malaysia",
    "leap": "malaysia",
    # Hong Kong
    "hkex": "hongkong",
    "hk": "hongkong",
    "hsi": "hongkong",
}

# Default values
DEFAULT_TIMEFRAME: str = "1h"
DEFAULT_EXCHANGE: str = "binance"
DEFAULT_LIMIT: int = 25
MAX_LIMIT: int = 100
MIN_LIMIT: int = 1


# =============================================================================
# Timeframe Sanitization
# =============================================================================

def sanitize_timeframe(tf: str | None, default: str = DEFAULT_TIMEFRAME) -> str:
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
        '1h'
        >>> sanitize_timeframe(None)
        '1h'
    """
    if not tf:
        return default

    tfs = tf.strip()
    return tfs if tfs in ALLOWED_TIMEFRAMES else default


# =============================================================================
# Exchange Sanitization
# =============================================================================

def sanitize_exchange(ex: str | None, default: str = DEFAULT_EXCHANGE) -> str:
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
        'binance'
    """
    if not ex:
        return default

    exs = ex.strip().lower()
    return exs if exs in EXCHANGE_SCREENER else default


def get_screener_for_exchange(exchange: str) -> str:
    """
    Get the TradingView screener type for an exchange.

    Args:
        exchange: Exchange name (will be sanitized)

    Returns:
        Screener type (e.g., "crypto", "america", "turkey")

    Example:
        >>> get_screener_for_exchange("binance")
        'crypto'
        >>> get_screener_for_exchange("nasdaq")
        'america'
    """
    sanitized = sanitize_exchange(exchange)
    return EXCHANGE_SCREENER.get(sanitized, "crypto")


# =============================================================================
# Symbol Sanitization
# =============================================================================

def sanitize_symbol(symbol: str | None) -> str:
    """
    Validate and normalize a trading symbol.

    Handles formats like:
    - "BTCUSDT" -> "BTCUSDT"
    - "BTC/USDT" -> "BTCUSDT"
    - "btcusdt" -> "BTCUSDT"
    - "BINANCE:BTCUSDT" -> "BINANCE:BTCUSDT" (preserves exchange prefix)

    Args:
        symbol: Input symbol string

    Returns:
        Normalized symbol in uppercase, with slashes removed

    Example:
        >>> sanitize_symbol("btc/usdt")
        'BTCUSDT'
        >>> sanitize_symbol("BINANCE:BTCUSDT")
        'BINANCE:BTCUSDT'
    """
    if not symbol:
        return ""

    # Strip whitespace and convert to uppercase
    cleaned = symbol.strip().upper()

    # Remove common separators (but preserve exchange prefix colon)
    if ":" in cleaned:
        # Has exchange prefix - preserve it
        parts = cleaned.split(":", 1)
        exchange = parts[0]
        pair = parts[1].replace("/", "").replace("-", "").replace("_", "")
        return f"{exchange}:{pair}"
    else:
        # No exchange prefix - just clean the pair
        return cleaned.replace("/", "").replace("-", "").replace("_", "")


def parse_symbol(symbol: str) -> Tuple[str | None, str]:
    """
    Parse a symbol into exchange and pair components.

    Args:
        symbol: Symbol string (e.g., "BINANCE:BTCUSDT" or "BTCUSDT")

    Returns:
        Tuple of (exchange, pair). Exchange is None if not specified.

    Example:
        >>> parse_symbol("BINANCE:BTCUSDT")
        ('BINANCE', 'BTCUSDT')
        >>> parse_symbol("BTCUSDT")
        (None, 'BTCUSDT')
    """
    sanitized = sanitize_symbol(symbol)

    if ":" in sanitized:
        parts = sanitized.split(":", 1)
        return (parts[0], parts[1])
    else:
        return (None, sanitized)


# =============================================================================
# Limit/Count Sanitization
# =============================================================================

def sanitize_limit(
    limit: int | None,
    default: int = DEFAULT_LIMIT,
    max_limit: int = MAX_LIMIT,
    min_limit: int = MIN_LIMIT
) -> int:
    """
    Validate and clamp a limit/count value.

    Args:
        limit: Input limit value
        default: Default value if input is None
        max_limit: Maximum allowed value
        min_limit: Minimum allowed value

    Returns:
        Clamped integer within [min_limit, max_limit]

    Example:
        >>> sanitize_limit(50)
        50
        >>> sanitize_limit(200, max_limit=100)
        100
        >>> sanitize_limit(-5)
        1
        >>> sanitize_limit(None)
        25
    """
    if limit is None:
        return default

    try:
        value = int(limit)
    except (ValueError, TypeError):
        return default

    return max(min_limit, min(value, max_limit))


# =============================================================================
# OHLCV Data Validation
# =============================================================================

def validate_ohlcv_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a dictionary contains valid OHLCV data.

    Required keys: open, high, low, close
    Optional: volume

    Args:
        data: Dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, "error description") if invalid

    Example:
        >>> validate_ohlcv_data({"open": 100, "high": 110, "low": 95, "close": 105})
        (True, None)
        >>> validate_ohlcv_data({"open": 100})
        (False, 'Missing required key: high')
    """
    required_keys = ["open", "high", "low", "close"]

    # Check required keys exist
    for key in required_keys:
        if key not in data:
            return (False, f"Missing required key: {key}")

    # Check values are numeric and valid
    for key in required_keys:
        value = data[key]
        if value is None:
            return (False, f"Value for '{key}' is None")
        if not isinstance(value, (int, float)):
            return (False, f"Value for '{key}' is not numeric: {type(value).__name__}")
        if value < 0:
            return (False, f"Value for '{key}' is negative: {value}")

    # Validate OHLC relationships
    open_val = data["open"]
    high = data["high"]
    low = data["low"]
    close = data["close"]

    if high < low:
        return (False, f"High ({high}) is less than Low ({low})")

    if high < open_val or high < close:
        return (False, f"High ({high}) is less than Open ({open_val}) or Close ({close})")

    if low > open_val or low > close:
        return (False, f"Low ({low}) is greater than Open ({open_val}) or Close ({close})")

    # Validate volume if present
    if "volume" in data and data["volume"] is not None:
        volume = data["volume"]
        if not isinstance(volume, (int, float)):
            return (False, f"Volume is not numeric: {type(volume).__name__}")
        if volume < 0:
            return (False, f"Volume is negative: {volume}")

    return (True, None)


def validate_ohlcv_series(
    opens: list,
    highs: list,
    lows: list,
    closes: list,
    volumes: list | None = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate OHLCV data series (arrays/lists).

    Args:
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        volumes: Optional list of volumes

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> validate_ohlcv_series([100], [110], [95], [105])
        (True, None)
    """
    # Check all have same length
    lengths = [len(opens), len(highs), len(lows), len(closes)]
    if len(set(lengths)) != 1:
        return (False, f"Mismatched series lengths: O={lengths[0]}, H={lengths[1]}, L={lengths[2]}, C={lengths[3]}")

    if volumes is not None and len(volumes) != lengths[0]:
        return (False, f"Volume length ({len(volumes)}) doesn't match price length ({lengths[0]})")

    # Check not empty
    if lengths[0] == 0:
        return (False, "Empty OHLCV series")

    return (True, None)
