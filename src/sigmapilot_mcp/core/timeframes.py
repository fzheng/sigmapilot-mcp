"""
Multi-Timeframe Analysis Utilities for SigmaPilot MCP.

This module provides timeframe hierarchy management, weight calculations,
and utilities for multi-timeframe analysis across theory-based engines.

Key Features:
- Timeframe hierarchy from monthly to 5-minute
- Weight calculations for confidence adjustments
- Utilities for getting higher/lower timeframes
"""

from __future__ import annotations
from typing import List


# =============================================================================
# Timeframe Constants
# =============================================================================

# Ordered from highest to lowest timeframe (most significant first)
TIMEFRAME_HIERARCHY: List[str] = ["1M", "1W", "1D", "4h", "1h", "15m", "5m"]

# Weight factors for confidence calculation (W_time in the formula)
# Higher timeframes get higher weights as they're more significant
TIMEFRAME_WEIGHTS = {
    "1M": 1.00,   # Monthly - full weight
    "1W": 0.95,   # Weekly - slight discount
    "1D": 0.90,   # Daily - standard weight
    "4h": 0.85,   # 4-hour - good for swing
    "1h": 0.80,   # Hourly - intraday
    "15m": 0.75,  # 15-min - short-term
    "5m": 0.70,   # 5-min - scalping, lowest weight
}

# Default weight when timeframe is unknown
DEFAULT_TIMEFRAME_WEIGHT: float = 0.80

# Minimum bars required for analysis by timeframe
# Higher timeframes need fewer bars, lower need more for pattern detection
MINIMUM_BARS_BY_TIMEFRAME = {
    "1M": 24,     # 2 years of monthly data
    "1W": 52,     # 1 year of weekly data
    "1D": 100,    # ~4 months of daily data
    "4h": 150,    # ~25 days of 4h data
    "1h": 200,    # ~8 days of hourly data
    "15m": 200,   # ~2 days of 15m data
    "5m": 200,    # ~16 hours of 5m data
}

DEFAULT_MINIMUM_BARS: int = 100


# =============================================================================
# Timeframe Validation
# =============================================================================

def is_valid_timeframe(tf: str) -> bool:
    """
    Check if a timeframe string is valid.

    Args:
        tf: Timeframe string to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> is_valid_timeframe("1h")
        True
        >>> is_valid_timeframe("2h")
        False
    """
    return tf in TIMEFRAME_HIERARCHY


def get_timeframe_index(tf: str) -> int:
    """
    Get the index of a timeframe in the hierarchy.

    Args:
        tf: Timeframe string

    Returns:
        Index in TIMEFRAME_HIERARCHY (0 = highest), or -1 if not found

    Example:
        >>> get_timeframe_index("1D")
        2
        >>> get_timeframe_index("invalid")
        -1
    """
    try:
        return TIMEFRAME_HIERARCHY.index(tf)
    except ValueError:
        return -1


# =============================================================================
# Timeframe Weight Functions
# =============================================================================

def get_timeframe_weight(tf: str) -> float:
    """
    Get the confidence weight factor for a timeframe.

    This weight is used in the confidence formula as W_time.
    Higher timeframes receive higher weights since signals on
    larger timeframes are generally more significant.

    Args:
        tf: Timeframe string (e.g., "1h", "4h", "1D")

    Returns:
        Weight factor between 0.70 and 1.00

    Example:
        >>> get_timeframe_weight("1D")
        0.9
        >>> get_timeframe_weight("5m")
        0.7
    """
    return TIMEFRAME_WEIGHTS.get(tf, DEFAULT_TIMEFRAME_WEIGHT)


def get_minimum_bars(tf: str) -> int:
    """
    Get the minimum number of bars required for analysis on a timeframe.

    Args:
        tf: Timeframe string

    Returns:
        Minimum bar count required

    Example:
        >>> get_minimum_bars("1D")
        100
    """
    return MINIMUM_BARS_BY_TIMEFRAME.get(tf, DEFAULT_MINIMUM_BARS)


# =============================================================================
# Timeframe Navigation
# =============================================================================

def get_higher_timeframes(tf: str) -> List[str]:
    """
    Get all timeframes higher than the given one.

    Useful for multi-timeframe analysis where you want to check
    trend alignment on higher timeframes.

    Args:
        tf: Current timeframe string

    Returns:
        List of higher timeframes (ordered from highest to current-1)
        Empty list if tf is already the highest or invalid

    Example:
        >>> get_higher_timeframes("1h")
        ['1M', '1W', '1D', '4h']
        >>> get_higher_timeframes("1M")
        []
    """
    idx = get_timeframe_index(tf)
    if idx <= 0:  # Already highest or invalid
        return []
    return TIMEFRAME_HIERARCHY[:idx]


def get_lower_timeframes(tf: str) -> List[str]:
    """
    Get all timeframes lower than the given one.

    Useful for drill-down analysis or finding entry points
    on lower timeframes after identifying trend on higher.

    Args:
        tf: Current timeframe string

    Returns:
        List of lower timeframes (ordered from current+1 to lowest)
        Empty list if tf is already the lowest or invalid

    Example:
        >>> get_lower_timeframes("1h")
        ['15m', '5m']
        >>> get_lower_timeframes("5m")
        []
    """
    idx = get_timeframe_index(tf)
    if idx == -1 or idx >= len(TIMEFRAME_HIERARCHY) - 1:  # Lowest or invalid
        return []
    return TIMEFRAME_HIERARCHY[idx + 1:]


def get_adjacent_timeframes(tf: str) -> tuple[str | None, str | None]:
    """
    Get the immediately higher and lower timeframes.

    Args:
        tf: Current timeframe string

    Returns:
        Tuple of (higher_tf, lower_tf), with None for boundaries

    Example:
        >>> get_adjacent_timeframes("1h")
        ('4h', '15m')
        >>> get_adjacent_timeframes("1M")
        (None, '1W')
    """
    idx = get_timeframe_index(tf)
    if idx == -1:
        return (None, None)

    higher = TIMEFRAME_HIERARCHY[idx - 1] if idx > 0 else None
    lower = TIMEFRAME_HIERARCHY[idx + 1] if idx < len(TIMEFRAME_HIERARCHY) - 1 else None

    return (higher, lower)


# =============================================================================
# Timeframe Comparison
# =============================================================================

def is_higher_timeframe(tf1: str, tf2: str) -> bool:
    """
    Check if tf1 is a higher timeframe than tf2.

    Args:
        tf1: First timeframe
        tf2: Second timeframe

    Returns:
        True if tf1 is higher than tf2

    Example:
        >>> is_higher_timeframe("1D", "1h")
        True
        >>> is_higher_timeframe("5m", "1h")
        False
    """
    idx1 = get_timeframe_index(tf1)
    idx2 = get_timeframe_index(tf2)

    if idx1 == -1 or idx2 == -1:
        return False

    return idx1 < idx2  # Lower index = higher timeframe


def timeframe_ratio(higher_tf: str, lower_tf: str) -> float | None:
    """
    Calculate approximate ratio between two timeframes.

    Useful for determining how many lower-tf bars fit in one higher-tf bar.

    Args:
        higher_tf: Higher timeframe
        lower_tf: Lower timeframe

    Returns:
        Approximate ratio, or None if invalid timeframes

    Example:
        >>> timeframe_ratio("1h", "15m")
        4.0
        >>> timeframe_ratio("1D", "1h")
        24.0
    """
    # Approximate minutes per timeframe
    tf_minutes = {
        "1M": 43200,   # ~30 days
        "1W": 10080,   # 7 days
        "1D": 1440,    # 24 hours
        "4h": 240,
        "1h": 60,
        "15m": 15,
        "5m": 5,
    }

    if higher_tf not in tf_minutes or lower_tf not in tf_minutes:
        return None

    return tf_minutes[higher_tf] / tf_minutes[lower_tf]
