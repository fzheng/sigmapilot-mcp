"""
Dow Theory Trend Analysis Engine.

This engine confirms primary trend via Higher Highs/Higher Lows (HH/HL)
or Lower Lows/Lower Highs (LL/LH) pattern analysis.

Based on Charles Dow's original theory principles:
1. The market discounts everything
2. The market has three trends (primary, secondary, minor)
3. Primary trends have three phases (accumulation, public participation, distribution)
4. Indices must confirm each other
5. Volume must confirm the trend
6. Trends persist until definitive reversal signals

References:
- https://en.wikipedia.org/wiki/Dow_theory
- Edwards & Magee, "Technical Analysis of Stock Trends"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

import numpy as np

from ..core.data_loader import OHLCVData
from ..core.confidence import (
    ConfidenceFactors,
    calculate_confidence,
    apply_no_signal_protocol,
    calculate_volume_confidence,
)
from ..core.schemas import (
    AnalysisResult,
    build_analysis_result,
    build_no_signal_result,
    build_insufficient_data_result,
)
from ..core.timeframes import get_minimum_bars
from ..core.errors import InsufficientDataError


# =============================================================================
# Constants
# =============================================================================

# Base confidence for Dow Theory (from CLAUDE.md spec)
DOW_THEORY_BASE_CONFIDENCE = 70

# Minimum swings needed to confirm a trend
DEFAULT_MIN_SWINGS = 2

# Default lookback period
DEFAULT_LOOKBACK = 200

# Minimum bars needed for analysis
MIN_BARS_REQUIRED = 50

# Swing point detection parameters
SWING_LOOKBACK = 5  # Bars to look back/forward for swing detection

TOOL_NAME = "dow_theory_trend"


class TrendDirection(str, Enum):
    """Trend direction classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class TrendPhase(str, Enum):
    """Dow Theory trend phases."""
    ACCUMULATION = "accumulation"
    PUBLIC_PARTICIPATION = "public_participation"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SwingPoint:
    """Represents a swing high or swing low point."""
    index: int
    price: float
    timestamp: int
    is_high: bool  # True for swing high, False for swing low
    volume: float = 0.0

    @property
    def type_str(self) -> str:
        return "high" if self.is_high else "low"


@dataclass
class TrendAnalysis:
    """Complete Dow Theory trend analysis result."""
    direction: TrendDirection
    phase: TrendPhase
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    pattern_quality: float = 0.85  # Q_pattern factor
    volume_confirms: bool = True
    invalidation_level: Optional[float] = None
    invalidation_description: str = ""
    rules_triggered: List[str] = field(default_factory=list)


# =============================================================================
# Swing Point Detection
# =============================================================================

def detect_swing_points(
    data: OHLCVData,
    lookback: int = SWING_LOOKBACK
) -> tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect swing highs and swing lows using a robust pivot method.

    A swing high is a bar with high greater than `lookback` bars on each side.
    A swing low is a bar with low less than `lookback` bars on each side.

    Args:
        data: OHLCV data
        lookback: Number of bars to look on each side

    Returns:
        Tuple of (swing_highs, swing_lows) lists
    """
    highs = data.highs
    lows = data.lows
    volumes = data.volumes
    timestamps = data.timestamps

    swing_highs: List[SwingPoint] = []
    swing_lows: List[SwingPoint] = []

    n = len(data)

    # Need at least 2*lookback + 1 bars
    if n < 2 * lookback + 1:
        return swing_highs, swing_lows

    for i in range(lookback, n - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break

        if is_swing_high:
            swing_highs.append(SwingPoint(
                index=i,
                price=float(highs[i]),
                timestamp=int(timestamps[i]),
                is_high=True,
                volume=float(volumes[i])
            ))

        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break

        if is_swing_low:
            swing_lows.append(SwingPoint(
                index=i,
                price=float(lows[i]),
                timestamp=int(timestamps[i]),
                is_high=False,
                volume=float(volumes[i])
            ))

    return swing_highs, swing_lows


# =============================================================================
# Trend Analysis
# =============================================================================

def analyze_trend_structure(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    min_swings: int = DEFAULT_MIN_SWINGS
) -> tuple[TrendDirection, float, List[str]]:
    """
    Analyze swing structure to determine trend direction.

    Bullish trend: Higher Highs (HH) and Higher Lows (HL)
    Bearish trend: Lower Lows (LL) and Lower Highs (LH)

    Args:
        swing_highs: List of swing high points
        swing_lows: List of swing low points
        min_swings: Minimum swings needed for confirmation

    Returns:
        Tuple of (direction, pattern_quality, rules_triggered)
    """
    rules_triggered: List[str] = []

    # Need enough swing points
    if len(swing_highs) < min_swings or len(swing_lows) < min_swings:
        return TrendDirection.SIDEWAYS, 0.7, ["Insufficient swing points"]

    # Analyze recent swing highs (most recent ones)
    recent_highs = swing_highs[-min_swings:]
    recent_lows = swing_lows[-min_swings:]

    # Check for Higher Highs
    hh_count = 0
    for i in range(1, len(recent_highs)):
        if recent_highs[i].price > recent_highs[i-1].price:
            hh_count += 1

    # Check for Higher Lows
    hl_count = 0
    for i in range(1, len(recent_lows)):
        if recent_lows[i].price > recent_lows[i-1].price:
            hl_count += 1

    # Check for Lower Lows
    ll_count = 0
    for i in range(1, len(recent_lows)):
        if recent_lows[i].price < recent_lows[i-1].price:
            ll_count += 1

    # Check for Lower Highs
    lh_count = 0
    for i in range(1, len(recent_highs)):
        if recent_highs[i].price < recent_highs[i-1].price:
            lh_count += 1

    # Determine trend
    required_confirmations = min_swings - 1  # Need n-1 confirmations for n swings

    # Strong bullish: HH + HL
    if hh_count >= required_confirmations and hl_count >= required_confirmations:
        rules_triggered.append("Higher Highs confirmed")
        rules_triggered.append("Higher Lows confirmed")
        quality = 1.0 if hh_count == hl_count == required_confirmations else 0.95
        return TrendDirection.BULLISH, quality, rules_triggered

    # Strong bearish: LL + LH
    if ll_count >= required_confirmations and lh_count >= required_confirmations:
        rules_triggered.append("Lower Lows confirmed")
        rules_triggered.append("Lower Highs confirmed")
        quality = 1.0 if ll_count == lh_count == required_confirmations else 0.95
        return TrendDirection.BEARISH, quality, rules_triggered

    # Partial bullish: HH without HL or HL without HH
    if hh_count >= required_confirmations or hl_count >= required_confirmations:
        if hh_count >= required_confirmations:
            rules_triggered.append("Higher Highs without HL confirmation")
        if hl_count >= required_confirmations:
            rules_triggered.append("Higher Lows without HH confirmation")
        return TrendDirection.BULLISH, 0.75, rules_triggered

    # Partial bearish: LL without LH or LH without LL
    if ll_count >= required_confirmations or lh_count >= required_confirmations:
        if ll_count >= required_confirmations:
            rules_triggered.append("Lower Lows without LH confirmation")
        if lh_count >= required_confirmations:
            rules_triggered.append("Lower Highs without LL confirmation")
        return TrendDirection.BEARISH, 0.75, rules_triggered

    # Sideways - no clear structure
    rules_triggered.append("No clear HH/HL or LL/LH structure")
    return TrendDirection.SIDEWAYS, 0.70, rules_triggered


def determine_trend_phase(
    data: OHLCVData,
    direction: TrendDirection,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint]
) -> TrendPhase:
    """
    Attempt to determine the current Dow Theory phase.

    Accumulation: Smart money buying, price base forming
    Public Participation: Trend recognition, momentum
    Distribution: Smart money selling, trend exhaustion

    This is a best-effort heuristic based on volume and price patterns.
    """
    if direction == TrendDirection.SIDEWAYS:
        return TrendPhase.UNKNOWN

    if len(data) < 50:
        return TrendPhase.UNKNOWN

    # Analyze volume trend
    volumes = data.volumes
    n = len(volumes)

    # Split into thirds for phase analysis
    first_third_vol = np.mean(volumes[:n//3])
    middle_third_vol = np.mean(volumes[n//3:2*n//3])
    last_third_vol = np.mean(volumes[2*n//3:])

    # Price analysis
    closes = data.closes
    first_third_price = np.mean(closes[:n//3])
    last_third_price = np.mean(closes[2*n//3:])

    if direction == TrendDirection.BULLISH:
        # Accumulation: Low volume, price basing
        if last_third_vol < middle_third_vol and last_third_price > first_third_price:
            # Volume declining but price rising slightly - early accumulation
            return TrendPhase.ACCUMULATION

        # Public participation: Rising volume, rising price
        if last_third_vol > first_third_vol and last_third_price > first_third_price:
            return TrendPhase.PUBLIC_PARTICIPATION

        # Distribution: High volume, price stalling
        if last_third_vol > middle_third_vol and abs(last_third_price - first_third_price) / first_third_price < 0.02:
            return TrendPhase.DISTRIBUTION

    elif direction == TrendDirection.BEARISH:
        # Distribution in bearish context
        if last_third_vol > first_third_vol:
            return TrendPhase.DISTRIBUTION

        # Public participation (markdown)
        if last_third_price < first_third_price:
            return TrendPhase.PUBLIC_PARTICIPATION

    return TrendPhase.UNKNOWN


def calculate_invalidation(
    direction: TrendDirection,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> tuple[Optional[float], str]:
    """
    Calculate the invalidation level for the current trend.

    For bullish: Invalidation is break below last higher low
    For bearish: Invalidation is break above last lower high
    """
    if direction == TrendDirection.BULLISH:
        if swing_lows:
            level = swing_lows[-1].price
            return level, f"Break below last HL at {level:.2f} invalidates bullish structure"

    elif direction == TrendDirection.BEARISH:
        if swing_highs:
            level = swing_highs[-1].price
            return level, f"Break above last LH at {level:.2f} invalidates bearish structure"

    return None, "No clear invalidation level"


def analyze_volume_confirmation(
    data: OHLCVData,
    direction: TrendDirection,
    swing_points: List[SwingPoint]
) -> tuple[bool, float]:
    """
    Check if volume confirms the trend direction.

    Bullish confirmation: Higher volume on up moves
    Bearish confirmation: Higher volume on down moves

    Returns:
        Tuple of (confirms, v_conf factor)
    """
    if not data.has_volume or len(swing_points) < 2:
        return True, 0.85  # Neutral - can't confirm

    volumes = data.volumes
    avg_volume = np.mean(volumes)

    # Get volume at swing points
    swing_volumes = [sp.volume for sp in swing_points[-4:]]  # Last 4 swings
    avg_swing_volume = np.mean(swing_volumes) if swing_volumes else avg_volume

    # Check if volume is expanding in trend direction
    if avg_swing_volume > avg_volume * 1.2:
        # Higher than average volume at swings - confirming
        return True, 1.0
    elif avg_swing_volume > avg_volume * 0.8:
        # Normal volume - neutral
        return True, 0.85
    else:
        # Below average volume - diverging
        return False, 0.70


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_dow_theory(
    data: OHLCVData,
    lookback: int = DEFAULT_LOOKBACK,
    min_swings: int = DEFAULT_MIN_SWINGS,
    mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
) -> AnalysisResult:
    """
    Perform Dow Theory trend analysis on OHLCV data.

    Args:
        data: OHLCVData object with price history
        lookback: Number of bars to analyze (default 200)
        min_swings: Minimum swings needed to confirm trend (default 2)
        mode: Analysis mode affecting confidence thresholds
            - conservative: Higher thresholds, fewer signals
            - balanced: Standard thresholds
            - aggressive: Lower thresholds, more signals

    Returns:
        AnalysisResult with trend analysis
    """
    # Validate minimum data
    if len(data) < MIN_BARS_REQUIRED:
        return build_insufficient_data_result(
            tool_name=TOOL_NAME,
            required_bars=MIN_BARS_REQUIRED,
            available_bars=len(data)
        )

    # Use only the lookback period if data is longer
    if len(data) > lookback:
        from ..core.data_loader import slice_ohlcv
        analysis_data = slice_ohlcv(data, start=-lookback)
    else:
        analysis_data = data

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(analysis_data)

    # Analyze trend structure
    direction, pattern_quality, rules_triggered = analyze_trend_structure(
        swing_highs, swing_lows, min_swings
    )

    # Determine trend phase
    phase = determine_trend_phase(analysis_data, direction, swing_highs, swing_lows)

    # Calculate invalidation
    invalidation_level, invalidation_desc = calculate_invalidation(
        direction, swing_highs, swing_lows,
        float(analysis_data.closes[-1]) if len(analysis_data) > 0 else 0
    )

    # Volume confirmation
    all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)
    volume_confirms, v_conf = analyze_volume_confirmation(analysis_data, direction, all_swings)

    # Apply mode adjustments
    mode_multiplier = {
        "conservative": 0.9,
        "balanced": 1.0,
        "aggressive": 1.1
    }.get(mode, 1.0)

    # Calculate confidence
    factors = ConfidenceFactors.from_timeframe(
        base=DOW_THEORY_BASE_CONFIDENCE * mode_multiplier,
        timeframe=data.timeframe,
        q_pattern=pattern_quality,
        v_conf=v_conf
    )
    confidence = calculate_confidence(factors)

    # Map direction to status
    if direction == TrendDirection.BULLISH:
        status = "bullish"
    elif direction == TrendDirection.BEARISH:
        status = "bearish"
    else:
        status = "neutral"

    # Build LLM summary
    if direction != TrendDirection.SIDEWAYS:
        vol_text = "confirmed by volume" if volume_confirms else "without volume confirmation"
        phase_text = f" in {phase.value} phase" if phase != TrendPhase.UNKNOWN else ""
        summary = (
            f"Dow Theory indicates {direction.value} trend{phase_text} on {data.timeframe}. "
            f"Structure shows {' and '.join(rules_triggered[:2])} {vol_text}. "
            f"Invalidation: {invalidation_desc}"
        )
    else:
        summary = (
            f"Dow Theory shows no clear trend on {data.timeframe}. "
            f"{' '.join(rules_triggered)}. "
            f"Wait for HH/HL or LL/LH structure to develop."
        )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation_desc
    )


def get_detailed_analysis(
    data: OHLCVData,
    lookback: int = DEFAULT_LOOKBACK,
    min_swings: int = DEFAULT_MIN_SWINGS
) -> Dict[str, Any]:
    """
    Get detailed Dow Theory analysis with all intermediate data.

    Returns a dictionary with:
    - trend_direction
    - trend_phase
    - swing_points (highs and lows)
    - invalidation
    - volume_confirmation
    - raw confidence factors

    Useful for debugging or advanced integrations.
    """
    if len(data) < MIN_BARS_REQUIRED:
        return {
            "error": "Insufficient data",
            "required_bars": MIN_BARS_REQUIRED,
            "available_bars": len(data)
        }

    # Use only the lookback period
    if len(data) > lookback:
        from ..core.data_loader import slice_ohlcv
        analysis_data = slice_ohlcv(data, start=-lookback)
    else:
        analysis_data = data

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(analysis_data)

    # Analyze trend
    direction, pattern_quality, rules_triggered = analyze_trend_structure(
        swing_highs, swing_lows, min_swings
    )

    phase = determine_trend_phase(analysis_data, direction, swing_highs, swing_lows)
    invalidation_level, invalidation_desc = calculate_invalidation(
        direction, swing_highs, swing_lows,
        float(analysis_data.closes[-1])
    )

    all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)
    volume_confirms, v_conf = analyze_volume_confirmation(analysis_data, direction, all_swings)

    return {
        "symbol": data.symbol,
        "timeframe": data.timeframe,
        "trend_direction": direction.value,
        "trend_phase": phase.value,
        "swing_highs": [
            {"index": sh.index, "price": sh.price, "timestamp": sh.timestamp}
            for sh in swing_highs[-5:]  # Last 5
        ],
        "swing_lows": [
            {"index": sl.index, "price": sl.price, "timestamp": sl.timestamp}
            for sl in swing_lows[-5:]  # Last 5
        ],
        "rules_triggered": rules_triggered,
        "pattern_quality": pattern_quality,
        "volume_confirms": volume_confirms,
        "v_conf": v_conf,
        "invalidation_level": invalidation_level,
        "invalidation_description": invalidation_desc,
        "bars_analyzed": len(analysis_data)
    }
