"""
Ichimoku Kinko Hyo Analysis Engine.

This engine provides holistic trend/momentum/support analysis using the
Ichimoku Cloud system developed by Goichi Hosoda.

Components:
- Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
- Kijun-sen (Base Line): (26-period high + 26-period low) / 2
- Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
- Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 ahead
- Chikou Span (Lagging Span): Close plotted 26 periods back

Signals:
- Price vs Cloud: Above = bullish, Below = bearish, Inside = neutral
- TK Cross: Tenkan crosses Kijun (bullish/bearish)
- Future Cloud: Color indicates future trend expectation
- Chikou: Confirms if lagging span is above/below price

References:
- https://school.stockcharts.com/doku.php?id=technical_indicators:ichimoku_cloud
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

import numpy as np

from ..core.data_loader import OHLCVData
from ..core.confidence import (
    ConfidenceFactors,
    calculate_confidence,
)
from ..core.schemas import (
    AnalysisResult,
    build_analysis_result,
    build_no_signal_result,
    build_insufficient_data_result,
)


# =============================================================================
# Constants
# =============================================================================

# Base confidence for Ichimoku (from CLAUDE.md spec)
ICHIMOKU_BASE_CONFIDENCE = 65

# Default Ichimoku parameters (9, 26, 52)
DEFAULT_TENKAN_PERIOD = 9
DEFAULT_KIJUN_PERIOD = 26
DEFAULT_SENKOU_B_PERIOD = 52
DEFAULT_DISPLACEMENT = 26

# Minimum bars needed (need at least senkou_b_period + displacement)
MIN_BARS_REQUIRED = 78  # 52 + 26

TOOL_NAME = "ichimoku_insight"


class CloudPosition(str, Enum):
    """Price position relative to the cloud."""
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"


class CrossSignal(str, Enum):
    """TK cross signal type."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE = "none"


class CloudColor(str, Enum):
    """Future cloud color/direction."""
    BULLISH = "bullish"  # Span A > Span B
    BEARISH = "bearish"  # Span A < Span B
    MIXED = "mixed"


class ChikouConfirmation(str, Enum):
    """Chikou span confirmation status."""
    CONFIRMING = "confirming"
    NON_CONFIRMING = "non_confirming"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IchimokuValues:
    """Ichimoku indicator values at a specific point."""
    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float
    chikou: Optional[float] = None

    @property
    def cloud_top(self) -> float:
        """Upper boundary of the cloud."""
        return max(self.senkou_a, self.senkou_b)

    @property
    def cloud_bottom(self) -> float:
        """Lower boundary of the cloud."""
        return min(self.senkou_a, self.senkou_b)

    @property
    def cloud_color(self) -> CloudColor:
        """Current cloud color."""
        if self.senkou_a > self.senkou_b:
            return CloudColor.BULLISH
        elif self.senkou_a < self.senkou_b:
            return CloudColor.BEARISH
        return CloudColor.MIXED


@dataclass
class TKCross:
    """Tenkan-Kijun cross event."""
    signal: CrossSignal
    index: int
    timestamp: int
    tenkan_value: float
    kijun_value: float


@dataclass
class IchimokuAnalysis:
    """Complete Ichimoku analysis result."""
    price_vs_cloud: CloudPosition
    tk_cross: CrossSignal
    tk_cross_recent: Optional[TKCross] = None
    future_cloud: CloudColor = CloudColor.MIXED
    chikou_confirmation: ChikouConfirmation = ChikouConfirmation.NON_CONFIRMING
    overall_trend: str = "mixed"
    current_values: Optional[IchimokuValues] = None
    signals_aligned: int = 0  # Count of aligned bullish/bearish signals


# =============================================================================
# Ichimoku Calculations
# =============================================================================

def calculate_donchian_midpoint(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Calculate Donchian Channel midpoint (used for Tenkan, Kijun, Senkou B).

    Formula: (highest high + lowest low) / 2 over the period
    """
    n = len(highs)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        period_highs = highs[i - period + 1:i + 1]
        period_lows = lows[i - period + 1:i + 1]
        result[i] = (np.max(period_highs) + np.min(period_lows)) / 2

    return result


def calculate_ichimoku(
    data: OHLCVData,
    tenkan_period: int = DEFAULT_TENKAN_PERIOD,
    kijun_period: int = DEFAULT_KIJUN_PERIOD,
    senkou_b_period: int = DEFAULT_SENKOU_B_PERIOD,
    displacement: int = DEFAULT_DISPLACEMENT
) -> Dict[str, np.ndarray]:
    """
    Calculate all Ichimoku components.

    Returns:
        Dictionary with arrays for each component
    """
    highs = data.highs
    lows = data.lows
    closes = data.closes
    n = len(data)

    # Calculate Tenkan-sen (Conversion Line)
    tenkan = calculate_donchian_midpoint(highs, lows, tenkan_period)

    # Calculate Kijun-sen (Base Line)
    kijun = calculate_donchian_midpoint(highs, lows, kijun_period)

    # Calculate Senkou Span A (Leading Span A)
    # (Tenkan + Kijun) / 2, shifted forward by displacement
    senkou_a_values = (tenkan + kijun) / 2
    senkou_a = np.full(n + displacement, np.nan)
    senkou_a[displacement:n + displacement] = senkou_a_values

    # Calculate Senkou Span B (Leading Span B)
    # 52-period midpoint, shifted forward by displacement
    senkou_b_values = calculate_donchian_midpoint(highs, lows, senkou_b_period)
    senkou_b = np.full(n + displacement, np.nan)
    senkou_b[displacement:n + displacement] = senkou_b_values

    # Calculate Chikou Span (Lagging Span)
    # Close price shifted back by displacement
    chikou = np.full(n, np.nan)
    if n > displacement:
        chikou[:n - displacement] = closes[displacement:]

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a[:n],  # Trim to data length for current cloud
        "senkou_b": senkou_b[:n],
        "senkou_a_future": senkou_a,  # Full array including future projection
        "senkou_b_future": senkou_b,
        "chikou": chikou
    }


# =============================================================================
# Signal Analysis
# =============================================================================

def analyze_price_vs_cloud(
    close: float,
    senkou_a: float,
    senkou_b: float
) -> CloudPosition:
    """Determine price position relative to the cloud."""
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    if close > cloud_top:
        return CloudPosition.ABOVE
    elif close < cloud_bottom:
        return CloudPosition.BELOW
    else:
        return CloudPosition.INSIDE


def find_recent_tk_cross(
    tenkan: np.ndarray,
    kijun: np.ndarray,
    timestamps: np.ndarray,
    lookback: int = 10
) -> Optional[TKCross]:
    """Find the most recent TK cross within lookback period."""
    n = len(tenkan)
    if n < 2:
        return None

    # Look at recent bars
    start_idx = max(0, n - lookback)

    for i in range(n - 1, start_idx, -1):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            continue
        if np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1]):
            continue

        # Bullish cross: Tenkan crosses above Kijun
        if tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]:
            return TKCross(
                signal=CrossSignal.BULLISH,
                index=i,
                timestamp=int(timestamps[i]),
                tenkan_value=float(tenkan[i]),
                kijun_value=float(kijun[i])
            )

        # Bearish cross: Tenkan crosses below Kijun
        if tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]:
            return TKCross(
                signal=CrossSignal.BEARISH,
                index=i,
                timestamp=int(timestamps[i]),
                tenkan_value=float(tenkan[i]),
                kijun_value=float(kijun[i])
            )

    return None


def get_current_tk_relationship(
    tenkan: np.ndarray,
    kijun: np.ndarray
) -> CrossSignal:
    """Get current TK relationship (not necessarily a cross)."""
    # Get last valid values
    for i in range(len(tenkan) - 1, -1, -1):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            if tenkan[i] > kijun[i]:
                return CrossSignal.BULLISH
            elif tenkan[i] < kijun[i]:
                return CrossSignal.BEARISH
            break
    return CrossSignal.NONE


def analyze_future_cloud(
    senkou_a_future: np.ndarray,
    senkou_b_future: np.ndarray,
    current_length: int,
    displacement: int = DEFAULT_DISPLACEMENT
) -> CloudColor:
    """Analyze the color of the future cloud."""
    # Look at the projected cloud
    future_start = current_length
    future_end = min(len(senkou_a_future), current_length + displacement)

    if future_end <= future_start:
        return CloudColor.MIXED

    bullish_count = 0
    bearish_count = 0

    for i in range(future_start, future_end):
        if i < len(senkou_a_future) and i < len(senkou_b_future):
            if not np.isnan(senkou_a_future[i]) and not np.isnan(senkou_b_future[i]):
                if senkou_a_future[i] > senkou_b_future[i]:
                    bullish_count += 1
                else:
                    bearish_count += 1

    if bullish_count > bearish_count * 1.5:
        return CloudColor.BULLISH
    elif bearish_count > bullish_count * 1.5:
        return CloudColor.BEARISH
    return CloudColor.MIXED


def analyze_chikou(
    chikou: np.ndarray,
    closes: np.ndarray,
    price_vs_cloud: CloudPosition,
    displacement: int = DEFAULT_DISPLACEMENT
) -> ChikouConfirmation:
    """
    Analyze if Chikou span confirms the trend.

    Confirming bullish: Chikou above price from 26 periods ago
    Confirming bearish: Chikou below price from 26 periods ago
    """
    n = len(closes)
    if n <= displacement:
        return ChikouConfirmation.NON_CONFIRMING

    # Current chikou position (which is close from displacement periods ago)
    chikou_idx = n - displacement - 1
    if chikou_idx < 0 or chikou_idx >= len(closes):
        return ChikouConfirmation.NON_CONFIRMING

    # Compare chikou (current close) with price from 26 periods ago
    current_close = closes[-1]
    price_26_ago = closes[chikou_idx]

    if price_vs_cloud == CloudPosition.ABOVE:
        # For bullish, chikou (current price) should be above historical price
        if current_close > price_26_ago:
            return ChikouConfirmation.CONFIRMING
    elif price_vs_cloud == CloudPosition.BELOW:
        # For bearish, chikou should be below historical price
        if current_close < price_26_ago:
            return ChikouConfirmation.CONFIRMING

    return ChikouConfirmation.NON_CONFIRMING


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_ichimoku(
    data: OHLCVData,
    tenkan_period: int = DEFAULT_TENKAN_PERIOD,
    kijun_period: int = DEFAULT_KIJUN_PERIOD,
    senkou_b_period: int = DEFAULT_SENKOU_B_PERIOD,
    mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
) -> AnalysisResult:
    """
    Perform Ichimoku Kinko Hyo analysis on OHLCV data.

    Args:
        data: OHLCVData object with price history
        tenkan_period: Tenkan-sen period (default 9)
        kijun_period: Kijun-sen period (default 26)
        senkou_b_period: Senkou Span B period (default 52)
        mode: Analysis mode affecting confidence

    Returns:
        AnalysisResult with Ichimoku analysis
    """
    # Validate minimum data
    min_required = senkou_b_period + DEFAULT_DISPLACEMENT
    if len(data) < min_required:
        return build_insufficient_data_result(
            tool_name=TOOL_NAME,
            required_bars=min_required,
            available_bars=len(data)
        )

    # Calculate Ichimoku components
    ichimoku = calculate_ichimoku(
        data,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_b_period=senkou_b_period
    )

    # Get current values
    idx = len(data) - 1
    current_close = float(data.closes[idx])

    # Current Senkou values (for the current bar)
    # Note: Senkou is displaced, so current cloud uses values from 26 bars ago
    cloud_idx = idx
    if cloud_idx < len(ichimoku["senkou_a"]) and cloud_idx < len(ichimoku["senkou_b"]):
        senkou_a_current = ichimoku["senkou_a"][cloud_idx]
        senkou_b_current = ichimoku["senkou_b"][cloud_idx]
    else:
        senkou_a_current = np.nan
        senkou_b_current = np.nan

    # Analyze price vs cloud
    if np.isnan(senkou_a_current) or np.isnan(senkou_b_current):
        price_vs_cloud = CloudPosition.INSIDE  # Can't determine
    else:
        price_vs_cloud = analyze_price_vs_cloud(
            current_close,
            senkou_a_current,
            senkou_b_current
        )

    # Find TK cross
    tk_cross_recent = find_recent_tk_cross(
        ichimoku["tenkan"],
        ichimoku["kijun"],
        data.timestamps
    )
    tk_relationship = get_current_tk_relationship(
        ichimoku["tenkan"],
        ichimoku["kijun"]
    )

    # Analyze future cloud
    future_cloud = analyze_future_cloud(
        ichimoku["senkou_a_future"],
        ichimoku["senkou_b_future"],
        len(data)
    )

    # Analyze Chikou confirmation
    chikou_conf = analyze_chikou(
        ichimoku["chikou"],
        data.closes,
        price_vs_cloud
    )

    # Count aligned signals
    bullish_signals = 0
    bearish_signals = 0

    if price_vs_cloud == CloudPosition.ABOVE:
        bullish_signals += 1
    elif price_vs_cloud == CloudPosition.BELOW:
        bearish_signals += 1

    if tk_relationship == CrossSignal.BULLISH:
        bullish_signals += 1
    elif tk_relationship == CrossSignal.BEARISH:
        bearish_signals += 1

    if future_cloud == CloudColor.BULLISH:
        bullish_signals += 1
    elif future_cloud == CloudColor.BEARISH:
        bearish_signals += 1

    if chikou_conf == ChikouConfirmation.CONFIRMING:
        if price_vs_cloud == CloudPosition.ABOVE:
            bullish_signals += 1
        elif price_vs_cloud == CloudPosition.BELOW:
            bearish_signals += 1

    # Determine overall trend and status
    if bullish_signals >= 3:
        overall_trend = "bullish"
        status = "bullish"
        pattern_quality = 0.85 + (bullish_signals - 3) * 0.05  # Up to 1.0
    elif bearish_signals >= 3:
        overall_trend = "bearish"
        status = "bearish"
        pattern_quality = 0.85 + (bearish_signals - 3) * 0.05
    elif bullish_signals > bearish_signals:
        overall_trend = "mixed_bullish"
        status = "bullish"
        pattern_quality = 0.70
    elif bearish_signals > bullish_signals:
        overall_trend = "mixed_bearish"
        status = "bearish"
        pattern_quality = 0.70
    else:
        overall_trend = "mixed"
        status = "neutral"
        pattern_quality = 0.60

    # Apply mode adjustments
    mode_multiplier = {
        "conservative": 0.9,
        "balanced": 1.0,
        "aggressive": 1.1
    }.get(mode, 1.0)

    # Calculate confidence
    factors = ConfidenceFactors.from_timeframe(
        base=ICHIMOKU_BASE_CONFIDENCE * mode_multiplier,
        timeframe=data.timeframe,
        q_pattern=min(1.0, pattern_quality),
        v_conf=0.85  # Ichimoku doesn't use volume directly
    )
    confidence = calculate_confidence(factors)

    # Build invalidation
    if not np.isnan(senkou_a_current) and not np.isnan(senkou_b_current):
        cloud_top = max(senkou_a_current, senkou_b_current)
        cloud_bottom = min(senkou_a_current, senkou_b_current)

        if status == "bullish":
            invalidation = f"Price close below cloud bottom ({cloud_bottom:.2f})"
        elif status == "bearish":
            invalidation = f"Price close above cloud top ({cloud_top:.2f})"
        else:
            invalidation = "No clear invalidation - price inside cloud"
    else:
        invalidation = "Ichimoku cloud not yet formed"

    # Build LLM summary
    tk_text = f"TK {tk_relationship.value}" if tk_relationship != CrossSignal.NONE else "no TK cross"
    chikou_text = "Chikou confirming" if chikou_conf == ChikouConfirmation.CONFIRMING else "Chikou not confirming"

    summary = (
        f"Ichimoku on {data.timeframe}: Price {price_vs_cloud.value} cloud, "
        f"{tk_text}, future cloud {future_cloud.value}, {chikou_text}. "
        f"Overall: {overall_trend}."
    )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_detailed_ichimoku(
    data: OHLCVData,
    tenkan_period: int = DEFAULT_TENKAN_PERIOD,
    kijun_period: int = DEFAULT_KIJUN_PERIOD,
    senkou_b_period: int = DEFAULT_SENKOU_B_PERIOD
) -> Dict[str, Any]:
    """
    Get detailed Ichimoku analysis with all component values.

    Returns a dictionary with all Ichimoku components and signals.
    """
    min_required = senkou_b_period + DEFAULT_DISPLACEMENT
    if len(data) < min_required:
        return {
            "error": "Insufficient data",
            "required_bars": min_required,
            "available_bars": len(data)
        }

    ichimoku = calculate_ichimoku(
        data,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_b_period=senkou_b_period
    )

    idx = len(data) - 1

    return {
        "symbol": data.symbol,
        "timeframe": data.timeframe,
        "current_values": {
            "tenkan": float(ichimoku["tenkan"][idx]) if not np.isnan(ichimoku["tenkan"][idx]) else None,
            "kijun": float(ichimoku["kijun"][idx]) if not np.isnan(ichimoku["kijun"][idx]) else None,
            "senkou_a": float(ichimoku["senkou_a"][idx]) if idx < len(ichimoku["senkou_a"]) and not np.isnan(ichimoku["senkou_a"][idx]) else None,
            "senkou_b": float(ichimoku["senkou_b"][idx]) if idx < len(ichimoku["senkou_b"]) and not np.isnan(ichimoku["senkou_b"][idx]) else None,
        },
        "close": float(data.closes[idx]),
        "parameters": {
            "tenkan_period": tenkan_period,
            "kijun_period": kijun_period,
            "senkou_b_period": senkou_b_period,
            "displacement": DEFAULT_DISPLACEMENT
        }
    }
