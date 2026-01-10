"""
Classical Chart Pattern Detection Engine.

This engine identifies classical chart patterns from price action,
based on the work of Edwards & Magee in "Technical Analysis of Stock Trends".

Patterns Detected:
- Head and Shoulders (Top and Inverse)
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- Ascending / Descending Triangle
- Symmetrical Triangle
- Rising / Falling Wedge
- Bull / Bear Flag
- Rectangle / Trading Range

References:
- Edwards & Magee, "Technical Analysis of Stock Trends"
- Thomas Bulkowski, "Encyclopedia of Chart Patterns"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Tuple
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

# Base confidence for Chart Patterns (from CLAUDE.md spec)
CHART_PATTERN_BASE_CONFIDENCE = 60

# Default lookback
DEFAULT_LOOKBACK = 300

# Minimum bars needed
MIN_BARS_REQUIRED = 50

# Swing detection parameters
SWING_LOOKBACK = 5

# Pattern tolerance (percentage)
LEVEL_TOLERANCE = 0.02  # 2% tolerance for price levels

TOOL_NAME = "chart_pattern_finder"


class PatternType(str, Enum):
    """Types of chart patterns."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    RECTANGLE = "rectangle"


class PatternDirection(str, Enum):
    """Pattern directional bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternStatus(str, Enum):
    """Pattern completion status."""
    FORMING = "forming"
    COMPLETE = "complete"
    CONFIRMED = "confirmed"  # Breakout occurred


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SwingPoint:
    """A swing high or low point."""
    index: int
    price: float
    timestamp: int
    is_high: bool


@dataclass
class ChartPattern:
    """Detected chart pattern."""
    pattern_type: PatternType
    direction: PatternDirection
    confidence: float
    status: PatternStatus
    key_levels: Dict[str, float]
    invalidation_level: float
    target_level: Optional[float]
    swing_points: List[SwingPoint]
    description: str


# =============================================================================
# Swing Point Detection
# =============================================================================

def detect_swings(
    data: OHLCVData,
    lookback: int = SWING_LOOKBACK
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect swing highs and lows using pivot method.
    """
    highs = data.highs
    lows = data.lows
    timestamps = data.timestamps
    n = len(data)

    swing_highs: List[SwingPoint] = []
    swing_lows: List[SwingPoint] = []

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
                is_high=True
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
                is_high=False
            ))

    return swing_highs, swing_lows


def prices_equal(p1: float, p2: float, tolerance: float = LEVEL_TOLERANCE) -> bool:
    """Check if two prices are approximately equal."""
    return abs(p1 - p2) / max(p1, p2) <= tolerance


# =============================================================================
# Pattern Detection Functions
# =============================================================================

def detect_double_top(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> Optional[ChartPattern]:
    """
    Detect Double Top pattern.

    Characteristics:
    - Two peaks at approximately same level
    - Trough (neckline) between them
    - Bearish reversal pattern
    """
    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return None

    # Get recent swing highs
    h1, h2 = swing_highs[-2], swing_highs[-1]

    # Check if peaks are at similar levels
    if not prices_equal(h1.price, h2.price, tolerance=0.03):
        return None

    # Find the low between them
    middle_lows = [l for l in swing_lows if h1.index < l.index < h2.index]
    if not middle_lows:
        return None

    neckline = min(l.price for l in middle_lows)

    # Determine status
    if current_price < neckline:
        status = PatternStatus.CONFIRMED
        confidence = 0.85
    elif current_price < h2.price:
        status = PatternStatus.COMPLETE
        confidence = 0.75
    else:
        return None  # Pattern invalid if price above peaks

    # Calculate target (measured move)
    pattern_height = ((h1.price + h2.price) / 2) - neckline
    target = neckline - pattern_height

    return ChartPattern(
        pattern_type=PatternType.DOUBLE_TOP,
        direction=PatternDirection.BEARISH,
        confidence=confidence,
        status=status,
        key_levels={
            "peak_1": h1.price,
            "peak_2": h2.price,
            "neckline": neckline
        },
        invalidation_level=max(h1.price, h2.price),
        target_level=target,
        swing_points=[h1, h2],
        description=f"Double Top pattern with peaks at {h1.price:.2f}/{h2.price:.2f}, neckline at {neckline:.2f}"
    )


def detect_double_bottom(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> Optional[ChartPattern]:
    """
    Detect Double Bottom pattern.

    Characteristics:
    - Two troughs at approximately same level
    - Peak (neckline) between them
    - Bullish reversal pattern
    """
    if len(swing_lows) < 2 or len(swing_highs) < 1:
        return None

    # Get recent swing lows
    l1, l2 = swing_lows[-2], swing_lows[-1]

    # Check if troughs are at similar levels
    if not prices_equal(l1.price, l2.price, tolerance=0.03):
        return None

    # Find the high between them
    middle_highs = [h for h in swing_highs if l1.index < h.index < l2.index]
    if not middle_highs:
        return None

    neckline = max(h.price for h in middle_highs)

    # Determine status
    if current_price > neckline:
        status = PatternStatus.CONFIRMED
        confidence = 0.85
    elif current_price > l2.price:
        status = PatternStatus.COMPLETE
        confidence = 0.75
    else:
        return None  # Pattern invalid if price below troughs

    # Calculate target (measured move)
    pattern_height = neckline - ((l1.price + l2.price) / 2)
    target = neckline + pattern_height

    return ChartPattern(
        pattern_type=PatternType.DOUBLE_BOTTOM,
        direction=PatternDirection.BULLISH,
        confidence=confidence,
        status=status,
        key_levels={
            "trough_1": l1.price,
            "trough_2": l2.price,
            "neckline": neckline
        },
        invalidation_level=min(l1.price, l2.price),
        target_level=target,
        swing_points=[l1, l2],
        description=f"Double Bottom pattern with troughs at {l1.price:.2f}/{l2.price:.2f}, neckline at {neckline:.2f}"
    )


def detect_head_and_shoulders(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> Optional[ChartPattern]:
    """
    Detect Head and Shoulders pattern.

    Characteristics:
    - Three peaks: left shoulder, head (highest), right shoulder
    - Two troughs forming neckline
    - Bearish reversal pattern
    """
    if len(swing_highs) < 3 or len(swing_lows) < 2:
        return None

    # Get three most recent swing highs
    ls, head, rs = swing_highs[-3], swing_highs[-2], swing_highs[-1]

    # Head must be higher than shoulders
    if head.price <= ls.price or head.price <= rs.price:
        return None

    # Shoulders should be at similar levels (within 5%)
    if not prices_equal(ls.price, rs.price, tolerance=0.05):
        return None

    # Find lows between shoulders and head (neckline points)
    left_lows = [l for l in swing_lows if ls.index < l.index < head.index]
    right_lows = [l for l in swing_lows if head.index < l.index < rs.index]

    if not left_lows or not right_lows:
        return None

    neckline_left = min(l.price for l in left_lows)
    neckline_right = min(l.price for l in right_lows)
    neckline = (neckline_left + neckline_right) / 2

    # Determine status
    if current_price < neckline:
        status = PatternStatus.CONFIRMED
        confidence = 0.90
    elif current_price < rs.price:
        status = PatternStatus.COMPLETE
        confidence = 0.80
    else:
        status = PatternStatus.FORMING
        confidence = 0.65

    # Calculate target
    pattern_height = head.price - neckline
    target = neckline - pattern_height

    return ChartPattern(
        pattern_type=PatternType.HEAD_AND_SHOULDERS,
        direction=PatternDirection.BEARISH,
        confidence=confidence,
        status=status,
        key_levels={
            "left_shoulder": ls.price,
            "head": head.price,
            "right_shoulder": rs.price,
            "neckline": neckline
        },
        invalidation_level=head.price,
        target_level=target,
        swing_points=[ls, head, rs],
        description=f"Head and Shoulders with head at {head.price:.2f}, neckline at {neckline:.2f}"
    )


def detect_inverse_head_and_shoulders(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> Optional[ChartPattern]:
    """
    Detect Inverse Head and Shoulders pattern.

    Characteristics:
    - Three troughs: left shoulder, head (lowest), right shoulder
    - Two peaks forming neckline
    - Bullish reversal pattern
    """
    if len(swing_lows) < 3 or len(swing_highs) < 2:
        return None

    # Get three most recent swing lows
    ls, head, rs = swing_lows[-3], swing_lows[-2], swing_lows[-1]

    # Head must be lower than shoulders
    if head.price >= ls.price or head.price >= rs.price:
        return None

    # Shoulders should be at similar levels
    if not prices_equal(ls.price, rs.price, tolerance=0.05):
        return None

    # Find highs between shoulders and head (neckline points)
    left_highs = [h for h in swing_highs if ls.index < h.index < head.index]
    right_highs = [h for h in swing_highs if head.index < h.index < rs.index]

    if not left_highs or not right_highs:
        return None

    neckline_left = max(h.price for h in left_highs)
    neckline_right = max(h.price for h in right_highs)
    neckline = (neckline_left + neckline_right) / 2

    # Determine status
    if current_price > neckline:
        status = PatternStatus.CONFIRMED
        confidence = 0.90
    elif current_price > rs.price:
        status = PatternStatus.COMPLETE
        confidence = 0.80
    else:
        status = PatternStatus.FORMING
        confidence = 0.65

    # Calculate target
    pattern_height = neckline - head.price
    target = neckline + pattern_height

    return ChartPattern(
        pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
        direction=PatternDirection.BULLISH,
        confidence=confidence,
        status=status,
        key_levels={
            "left_shoulder": ls.price,
            "head": head.price,
            "right_shoulder": rs.price,
            "neckline": neckline
        },
        invalidation_level=head.price,
        target_level=target,
        swing_points=[ls, head, rs],
        description=f"Inverse Head and Shoulders with head at {head.price:.2f}, neckline at {neckline:.2f}"
    )


def detect_ascending_triangle(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> Optional[ChartPattern]:
    """
    Detect Ascending Triangle pattern.

    Characteristics:
    - Flat/horizontal resistance (multiple highs at same level)
    - Rising support (higher lows)
    - Typically bullish continuation
    """
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return None

    recent_highs = swing_highs[-3:]
    recent_lows = swing_lows[-3:]

    # Check for flat resistance
    avg_high = np.mean([h.price for h in recent_highs])
    high_deviation = max(abs(h.price - avg_high) / avg_high for h in recent_highs)

    if high_deviation > 0.02:  # Highs not flat enough
        return None

    # Check for rising lows
    is_rising = all(
        recent_lows[i].price < recent_lows[i+1].price
        for i in range(len(recent_lows) - 1)
    )

    if not is_rising:
        return None

    resistance = avg_high
    support = recent_lows[-1].price

    # Determine status
    if current_price > resistance:
        status = PatternStatus.CONFIRMED
        confidence = 0.80
    else:
        status = PatternStatus.FORMING
        confidence = 0.70

    # Calculate target
    pattern_height = resistance - recent_lows[0].price
    target = resistance + pattern_height

    return ChartPattern(
        pattern_type=PatternType.ASCENDING_TRIANGLE,
        direction=PatternDirection.BULLISH,
        confidence=confidence,
        status=status,
        key_levels={
            "resistance": resistance,
            "support": support
        },
        invalidation_level=support,
        target_level=target,
        swing_points=recent_highs + recent_lows,
        description=f"Ascending Triangle with resistance at {resistance:.2f}"
    )


def detect_descending_triangle(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_price: float
) -> Optional[ChartPattern]:
    """
    Detect Descending Triangle pattern.

    Characteristics:
    - Flat/horizontal support (multiple lows at same level)
    - Falling resistance (lower highs)
    - Typically bearish continuation
    """
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return None

    recent_highs = swing_highs[-3:]
    recent_lows = swing_lows[-3:]

    # Check for flat support
    avg_low = np.mean([l.price for l in recent_lows])
    low_deviation = max(abs(l.price - avg_low) / avg_low for l in recent_lows)

    if low_deviation > 0.02:  # Lows not flat enough
        return None

    # Check for falling highs
    is_falling = all(
        recent_highs[i].price > recent_highs[i+1].price
        for i in range(len(recent_highs) - 1)
    )

    if not is_falling:
        return None

    support = avg_low
    resistance = recent_highs[-1].price

    # Determine status
    if current_price < support:
        status = PatternStatus.CONFIRMED
        confidence = 0.80
    else:
        status = PatternStatus.FORMING
        confidence = 0.70

    # Calculate target
    pattern_height = recent_highs[0].price - support
    target = support - pattern_height

    return ChartPattern(
        pattern_type=PatternType.DESCENDING_TRIANGLE,
        direction=PatternDirection.BEARISH,
        confidence=confidence,
        status=status,
        key_levels={
            "resistance": resistance,
            "support": support
        },
        invalidation_level=resistance,
        target_level=target,
        swing_points=recent_highs + recent_lows,
        description=f"Descending Triangle with support at {support:.2f}"
    )


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_chart_patterns(
    data: OHLCVData,
    lookback: int = DEFAULT_LOOKBACK,
    min_confidence: float = 0.70,
    mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
) -> AnalysisResult:
    """
    Detect classical chart patterns in OHLCV data.

    Args:
        data: OHLCVData object with price history
        lookback: Number of bars to analyze
        min_confidence: Minimum confidence to include pattern
        mode: Analysis mode affecting confidence

    Returns:
        AnalysisResult with detected patterns
    """
    # Validate minimum data
    if len(data) < MIN_BARS_REQUIRED:
        return build_insufficient_data_result(
            tool_name=TOOL_NAME,
            required_bars=MIN_BARS_REQUIRED,
            available_bars=len(data)
        )

    # Detect swing points
    swing_highs, swing_lows = detect_swings(data)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return build_no_signal_result(
            reason="Insufficient swing points detected for pattern analysis. Price action may be too choppy or data too short.",
            tool_name=TOOL_NAME
        )

    current_price = float(data.closes[-1])

    # Detect all patterns
    patterns: List[ChartPattern] = []

    # Run all detection functions
    detectors = [
        (detect_double_top, [swing_highs, swing_lows, current_price]),
        (detect_double_bottom, [swing_highs, swing_lows, current_price]),
        (detect_head_and_shoulders, [swing_highs, swing_lows, current_price]),
        (detect_inverse_head_and_shoulders, [swing_highs, swing_lows, current_price]),
        (detect_ascending_triangle, [swing_highs, swing_lows, current_price]),
        (detect_descending_triangle, [swing_highs, swing_lows, current_price]),
    ]

    for detector, args in detectors:
        pattern = detector(*args)
        if pattern and pattern.confidence >= min_confidence:
            patterns.append(pattern)

    # No patterns found
    if not patterns:
        return build_no_signal_result(
            reason="No high-confidence chart patterns detected in current price structure.",
            tool_name=TOOL_NAME
        )

    # Sort by confidence (highest first)
    patterns.sort(key=lambda p: p.confidence, reverse=True)

    # Determine overall bias
    bullish_patterns = [p for p in patterns if p.direction == PatternDirection.BULLISH]
    bearish_patterns = [p for p in patterns if p.direction == PatternDirection.BEARISH]

    if len(bullish_patterns) > len(bearish_patterns):
        status = "bullish"
    elif len(bearish_patterns) > len(bullish_patterns):
        status = "bearish"
    else:
        # Use highest confidence pattern
        status = patterns[0].direction.value

    # Pattern quality based on best pattern
    best_pattern = patterns[0]
    pattern_quality = best_pattern.confidence

    # Apply mode adjustments
    mode_multiplier = {
        "conservative": 0.9,
        "balanced": 1.0,
        "aggressive": 1.1
    }.get(mode, 1.0)

    # Calculate confidence
    factors = ConfidenceFactors.from_timeframe(
        base=CHART_PATTERN_BASE_CONFIDENCE * mode_multiplier,
        timeframe=data.timeframe,
        q_pattern=pattern_quality,
        v_conf=0.85  # Patterns don't directly use volume
    )
    confidence = calculate_confidence(factors)

    # Build invalidation
    invalidation = f"Invalidation at {best_pattern.invalidation_level:.2f} for {best_pattern.pattern_type.value}"

    # Build LLM summary
    pattern_names = [p.pattern_type.value.replace("_", " ").title() for p in patterns[:2]]
    target_str = f"{best_pattern.target_level:.2f}" if best_pattern.target_level else "N/A"
    summary = (
        f"Chart patterns on {data.timeframe}: {', '.join(pattern_names)} detected. "
        f"Primary pattern {best_pattern.status.value}. "
        f"Target: {target_str}."
    )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_all_patterns(
    data: OHLCVData,
    min_confidence: float = 0.60
) -> List[Dict[str, Any]]:
    """
    Get all detected patterns as a list of dictionaries.

    Returns detailed information about each pattern.
    """
    if len(data) < MIN_BARS_REQUIRED:
        return []

    swing_highs, swing_lows = detect_swings(data)
    current_price = float(data.closes[-1])

    patterns: List[ChartPattern] = []

    detectors = [
        detect_double_top,
        detect_double_bottom,
        detect_head_and_shoulders,
        detect_inverse_head_and_shoulders,
        detect_ascending_triangle,
        detect_descending_triangle,
    ]

    for detector in detectors:
        if detector in [detect_ascending_triangle, detect_descending_triangle]:
            pattern = detector(swing_highs, swing_lows, current_price)
        else:
            pattern = detector(swing_highs, swing_lows, current_price)

        if pattern and pattern.confidence >= min_confidence:
            patterns.append(pattern)

    return [
        {
            "pattern_type": p.pattern_type.value,
            "direction": p.direction.value,
            "confidence": p.confidence,
            "status": p.status.value,
            "key_levels": p.key_levels,
            "invalidation_level": p.invalidation_level,
            "target_level": p.target_level,
            "description": p.description
        }
        for p in patterns
    ]
