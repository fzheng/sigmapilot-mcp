"""
Technical Indicators Module.

This module provides functions for computing technical analysis indicators,
specifically focused on Bollinger Bands analysis and related metrics.

Key Functions:
- compute_change: Calculate percentage price change
- compute_bbw: Calculate Bollinger Band Width
- compute_bb_rating_signal: Generate trading signals from Bollinger Bands
- compute_metrics: Compute complete metrics from raw indicators
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple


def compute_change(open_price: float, close: float) -> float:
    """
    Calculate percentage change between open and close prices.

    Args:
        open_price: Opening price of the candle
        close: Closing price of the candle

    Returns:
        Percentage change as a float (e.g., 5.0 for 5% gain)
        Returns 0.0 if open_price is zero/falsy to avoid division by zero

    Example:
        >>> compute_change(100.0, 110.0)
        10.0
    """
    return ((close - open_price) / open_price) * 100 if open_price else 0.0


def compute_bbw(sma: float, bb_upper: float, bb_lower: float) -> Optional[float]:
    """
    Calculate Bollinger Band Width (BBW).

    BBW measures the width of Bollinger Bands relative to the middle band (SMA).
    Lower values indicate a "squeeze" (low volatility), which often precedes
    significant price movements.

    Formula: BBW = (Upper Band - Lower Band) / Middle Band

    Args:
        sma: Simple Moving Average (middle band, typically SMA20)
        bb_upper: Upper Bollinger Band value
        bb_lower: Lower Bollinger Band value

    Returns:
        BBW value as a float, or None if calculation is not possible

    Typical Values:
        - < 0.02: Extreme squeeze (very low volatility)
        - 0.02-0.05: Normal volatility
        - > 0.05: High volatility
    """
    if not sma:
        return None
    try:
        return (bb_upper - bb_lower) / sma
    except ZeroDivisionError:
        return None


def compute_bb_rating_signal(close: float, bb_upper: float, bb_middle: float, bb_lower: float) -> Tuple[int, str]:
    """
    Generate a rating and trading signal based on price position within Bollinger Bands.

    The rating system divides the Bollinger Bands into zones:
    - Above upper band: +3 (Strong Buy territory, but may be overbought)
    - Upper 50% of bands: +2 (Buy signal)
    - Above middle, lower 50%: +1 (Weak Buy)
    - At middle: 0 (Neutral)
    - Below middle, upper 50%: -1 (Weak Sell)
    - Lower 50% of bands: -2 (Sell signal)
    - Below lower band: -3 (Strong Sell territory, but may be oversold)

    Args:
        close: Current closing price
        bb_upper: Upper Bollinger Band
        bb_middle: Middle Bollinger Band (typically SMA20)
        bb_lower: Lower Bollinger Band

    Returns:
        Tuple of (rating: int, signal: str)
        - rating: Integer from -3 to +3
        - signal: "BUY", "SELL", or "NEUTRAL"
    """
    rating = 0

    # Determine position within Bollinger Bands
    if close > bb_upper:
        # Price is above upper band - extremely bullish or overbought
        rating = 3
    elif close > bb_middle + ((bb_upper - bb_middle) / 2):
        # Price is in upper 50% of upper half - bullish
        rating = 2
    elif close > bb_middle:
        # Price is above middle but in lower 50% of upper half
        rating = 1
    elif close < bb_lower:
        # Price is below lower band - extremely bearish or oversold
        rating = -3
    elif close < bb_middle - ((bb_middle - bb_lower) / 2):
        # Price is in lower 50% of lower half - bearish
        rating = -2
    elif close < bb_middle:
        # Price is below middle but in upper 50% of lower half
        rating = -1

    # Generate trading signal (only for strong +/-2 ratings)
    signal = "NEUTRAL"
    if rating == 2:
        signal = "BUY"
    elif rating == -2:
        signal = "SELL"

    return rating, signal


def compute_metrics(indicators: Dict) -> Optional[Dict]:
    """
    Compute complete trading metrics from raw TradingView indicators.

    This function combines price change, Bollinger Band analysis, and
    trading signals into a single metrics dictionary.

    Args:
        indicators: Dictionary containing raw indicator values:
            - "open": Opening price
            - "close": Closing price
            - "SMA20": 20-period Simple Moving Average
            - "BB.upper": Upper Bollinger Band
            - "BB.lower": Lower Bollinger Band

    Returns:
        Dictionary with computed metrics:
            - "price": Current price (rounded to 4 decimals)
            - "change": Percentage change (rounded to 3 decimals)
            - "bbw": Bollinger Band Width (rounded to 4 decimals)
            - "rating": Integer rating from -3 to +3
            - "signal": Trading signal ("BUY", "SELL", or "NEUTRAL")
        Returns None if required indicators are missing or invalid

    Example:
        >>> indicators = {"open": 100, "close": 105, "SMA20": 102,
        ...               "BB.upper": 110, "BB.lower": 94}
        >>> metrics = compute_metrics(indicators)
        >>> metrics["change"]
        5.0
    """
    try:
        # Extract required indicators
        open_price = indicators["open"]
        close = indicators["close"]
        sma = indicators["SMA20"]
        bb_upper = indicators["BB.upper"]
        bb_lower = indicators["BB.lower"]
        bb_middle = sma  # Middle band is the SMA

        # Compute individual metrics
        change = compute_change(open_price, close)
        bbw = compute_bbw(sma, bb_upper, bb_lower)
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)

        # Return rounded values
        return {
            "price": round(close, 4),
            "change": round(change, 3),
            "bbw": round(bbw, 4) if bbw is not None else None,
            "rating": rating,
            "signal": signal,
        }
    except (KeyError, TypeError):
        # Return None if required keys are missing or types are invalid
        return None
