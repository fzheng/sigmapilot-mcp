"""
Harmonic Pattern Detector Engine.

Detects harmonic price patterns based on Fibonacci ratios:
- Gartley (primary)
- Bat
- Butterfly
- Crab

Theory Attribution:
    H.M. Gartley (1935) - "Profits in the Stock Market"
    Scott Carney - Extended harmonic patterns (Bat, Butterfly, Crab)

Base Confidence: 60 (per CLAUDE.md specification)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from sigmapilot_mcp.core.data_loader import OHLCVData
from sigmapilot_mcp.core.confidence import ConfidenceFactors, calculate_confidence
from sigmapilot_mcp.core.schemas import build_analysis_result, build_no_signal_result
from sigmapilot_mcp.core.timeframes import get_timeframe_weight

# Constants
TOOL_NAME = "Harmonic Patterns"
HARMONIC_BASE_CONFIDENCE = 60
MIN_BARS_REQUIRED = 30
SIGNAL_THRESHOLD = 60


class HarmonicType(Enum):
    """Types of harmonic patterns."""
    GARTLEY = "gartley"
    BAT = "bat"
    BUTTERFLY = "butterfly"
    CRAB = "crab"


class PatternDirection(Enum):
    """Pattern direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"


class PatternStatus(Enum):
    """Pattern completion status."""
    FORMING = "forming"
    COMPLETE = "complete"
    INVALID = "invalid"


# Fibonacci ratios for each pattern
# Format: {pattern: {leg: (min_ratio, ideal_ratio, max_ratio)}}
HARMONIC_RATIOS = {
    HarmonicType.GARTLEY: {
        "AB_XA": (0.618, 0.618, 0.618),  # AB retraces 61.8% of XA
        "BC_AB": (0.382, 0.618, 0.886),  # BC retraces 38.2-88.6% of AB
        "CD_BC": (1.27, 1.618, 1.618),   # CD extends 127-161.8% of BC
        "AD_XA": (0.786, 0.786, 0.786),  # D completes at 78.6% of XA
    },
    HarmonicType.BAT: {
        "AB_XA": (0.382, 0.5, 0.50),     # AB retraces 38.2-50% of XA
        "BC_AB": (0.382, 0.618, 0.886),  # BC retraces 38.2-88.6% of AB
        "CD_BC": (1.618, 2.0, 2.618),    # CD extends 161.8-261.8% of BC
        "AD_XA": (0.886, 0.886, 0.886),  # D completes at 88.6% of XA
    },
    HarmonicType.BUTTERFLY: {
        "AB_XA": (0.786, 0.786, 0.786),  # AB retraces 78.6% of XA
        "BC_AB": (0.382, 0.618, 0.886),  # BC retraces 38.2-88.6% of AB
        "CD_BC": (1.618, 2.0, 2.618),    # CD extends 161.8-261.8% of BC
        "AD_XA": (1.27, 1.414, 1.618),   # D extends 127-161.8% beyond X
    },
    HarmonicType.CRAB: {
        "AB_XA": (0.382, 0.618, 0.618),  # AB retraces 38.2-61.8% of XA
        "BC_AB": (0.382, 0.618, 0.886),  # BC retraces 38.2-88.6% of AB
        "CD_BC": (2.24, 3.14, 3.618),    # CD extends 224-361.8% of BC
        "AD_XA": (1.618, 1.618, 1.618),  # D extends 161.8% beyond X
    },
}


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    index: int
    price: float
    timestamp: int
    is_high: bool


@dataclass
class HarmonicPattern:
    """Represents a detected harmonic pattern."""
    pattern_type: HarmonicType
    direction: PatternDirection
    status: PatternStatus
    x: SwingPoint
    a: SwingPoint
    b: SwingPoint
    c: SwingPoint
    d: Optional[SwingPoint]
    prz_low: float   # Potential Reversal Zone low
    prz_high: float  # Potential Reversal Zone high
    confidence: float
    ratios: dict


def find_swing_points(
    data: OHLCVData,
    window: int = 5,
    max_points: int = 20
) -> List[SwingPoint]:
    """
    Find swing highs and lows.

    Args:
        data: OHLCV data
        window: Number of bars on each side to confirm swing
        max_points: Maximum number of recent points to return

    Returns:
        List of swing points sorted by index
    """
    highs = data.highs
    lows = data.lows
    points: List[SwingPoint] = []

    for i in range(window, len(highs) - window):
        # Check swing high
        is_swing_high = True
        for j in range(1, window + 1):
            if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                is_swing_high = False
                break

        if is_swing_high:
            points.append(SwingPoint(
                index=i,
                price=highs[i],
                timestamp=data.bars[i].timestamp,
                is_high=True
            ))

        # Check swing low
        is_swing_low = True
        for j in range(1, window + 1):
            if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                is_swing_low = False
                break

        if is_swing_low:
            points.append(SwingPoint(
                index=i,
                price=lows[i],
                timestamp=data.bars[i].timestamp,
                is_high=False
            ))

    # Sort by index and return most recent
    points.sort(key=lambda p: p.index)
    return points[-max_points:] if len(points) > max_points else points


def calculate_ratio(
    leg1_start: float,
    leg1_end: float,
    leg2_start: float,
    leg2_end: float
) -> float:
    """Calculate the ratio between two legs."""
    leg1 = abs(leg1_end - leg1_start)
    leg2 = abs(leg2_end - leg2_start)

    if leg1 == 0:
        return 0.0

    return leg2 / leg1


def check_ratio_tolerance(
    actual: float,
    target_min: float,
    target_ideal: float,
    target_max: float,
    tolerance: float = 0.03
) -> Tuple[bool, float]:
    """
    Check if ratio is within tolerance of target range.

    Returns:
        Tuple of (is_valid, quality_score)
    """
    # Expand range by tolerance
    min_allowed = target_min * (1 - tolerance)
    max_allowed = target_max * (1 + tolerance)

    if actual < min_allowed or actual > max_allowed:
        return False, 0.0

    # Calculate quality score (1.0 = ideal, lower for deviations)
    if actual <= target_ideal:
        # Between min and ideal
        if target_ideal == target_min:
            quality = 1.0
        else:
            quality = 1.0 - abs(actual - target_ideal) / (target_ideal - target_min) * 0.3
    else:
        # Between ideal and max
        if target_max == target_ideal:
            quality = 1.0
        else:
            quality = 1.0 - abs(actual - target_ideal) / (target_max - target_ideal) * 0.3

    return True, max(0.5, quality)


def validate_pattern(
    x: SwingPoint,
    a: SwingPoint,
    b: SwingPoint,
    c: SwingPoint,
    d_price: float,
    pattern_type: HarmonicType,
    direction: PatternDirection,
    tolerance: float
) -> Tuple[bool, float, dict]:
    """
    Validate if XABCD points form a valid harmonic pattern.

    Returns:
        Tuple of (is_valid, confidence, ratios_dict)
    """
    ratios = HARMONIC_RATIOS[pattern_type]
    ratio_results = {}
    total_quality = 0.0
    checks_passed = 0

    # AB/XA ratio
    ab_xa = calculate_ratio(x.price, a.price, a.price, b.price)
    target = ratios["AB_XA"]
    valid, quality = check_ratio_tolerance(ab_xa, target[0], target[1], target[2], tolerance)
    ratio_results["AB_XA"] = {"actual": ab_xa, "target": target[1], "valid": valid}
    if valid:
        total_quality += quality
        checks_passed += 1

    # BC/AB ratio
    bc_ab = calculate_ratio(a.price, b.price, b.price, c.price)
    target = ratios["BC_AB"]
    valid, quality = check_ratio_tolerance(bc_ab, target[0], target[1], target[2], tolerance)
    ratio_results["BC_AB"] = {"actual": bc_ab, "target": target[1], "valid": valid}
    if valid:
        total_quality += quality
        checks_passed += 1

    # CD/BC ratio
    cd_bc = calculate_ratio(b.price, c.price, c.price, d_price)
    target = ratios["CD_BC"]
    valid, quality = check_ratio_tolerance(cd_bc, target[0], target[1], target[2], tolerance)
    ratio_results["CD_BC"] = {"actual": cd_bc, "target": target[1], "valid": valid}
    if valid:
        total_quality += quality
        checks_passed += 1

    # AD/XA ratio (overall pattern)
    ad_xa = calculate_ratio(x.price, a.price, a.price, d_price)
    target = ratios["AD_XA"]
    valid, quality = check_ratio_tolerance(ad_xa, target[0], target[1], target[2], tolerance)
    ratio_results["AD_XA"] = {"actual": ad_xa, "target": target[1], "valid": valid}
    if valid:
        total_quality += quality
        checks_passed += 1

    # Pattern is valid if at least 3 of 4 ratios match
    is_valid = checks_passed >= 3
    confidence = total_quality / 4 if is_valid else 0.0

    return is_valid, confidence, ratio_results


def calculate_prz(
    x: SwingPoint,
    a: SwingPoint,
    c: SwingPoint,
    pattern_type: HarmonicType,
    direction: PatternDirection
) -> Tuple[float, float]:
    """
    Calculate Potential Reversal Zone (PRZ).

    Returns:
        Tuple of (prz_low, prz_high)
    """
    xa_move = abs(a.price - x.price)
    ratios = HARMONIC_RATIOS[pattern_type]["AD_XA"]

    if direction == PatternDirection.BULLISH:
        # D is below A for bullish
        prz_center = a.price - xa_move * ratios[1]
        prz_low = a.price - xa_move * ratios[2] * 1.05
        prz_high = a.price - xa_move * ratios[0] * 0.95
    else:
        # D is above A for bearish
        prz_center = a.price + xa_move * ratios[1]
        prz_low = a.price + xa_move * ratios[0] * 0.95
        prz_high = a.price + xa_move * ratios[2] * 1.05

    return min(prz_low, prz_high), max(prz_low, prz_high)


def detect_patterns(
    data: OHLCVData,
    swing_points: List[SwingPoint],
    tolerance: float = 0.05,
    min_conf: float = 0.6
) -> List[HarmonicPattern]:
    """
    Detect harmonic patterns from swing points.

    Args:
        data: OHLCV data
        swing_points: List of swing highs and lows
        tolerance: Ratio tolerance
        min_conf: Minimum confidence to include pattern

    Returns:
        List of detected patterns
    """
    patterns: List[HarmonicPattern] = []

    if len(swing_points) < 4:
        return patterns

    current_price = data.closes[-1]

    # Try different combinations of 4 points as XABC
    for i in range(len(swing_points) - 3):
        for j in range(i + 1, len(swing_points) - 2):
            for k in range(j + 1, len(swing_points) - 1):
                for l in range(k + 1, len(swing_points)):
                    x = swing_points[i]
                    a = swing_points[j]
                    b = swing_points[k]
                    c = swing_points[l]

                    # Check alternating pattern (high-low-high-low or low-high-low-high)
                    if not (x.is_high != a.is_high != b.is_high != c.is_high):
                        continue

                    # Determine direction
                    if x.is_high and not a.is_high:
                        # X is high, A is low -> bearish pattern (looking for high D)
                        direction = PatternDirection.BEARISH
                    elif not x.is_high and a.is_high:
                        # X is low, A is high -> bullish pattern (looking for low D)
                        direction = PatternDirection.BULLISH
                    else:
                        continue

                    # Try each pattern type
                    for pattern_type in HarmonicType:
                        # Calculate PRZ for D
                        prz_low, prz_high = calculate_prz(x, a, c, pattern_type, direction)

                        # Check if current price is in or near PRZ
                        in_prz = prz_low <= current_price <= prz_high
                        near_prz = (prz_low * 0.95 <= current_price <= prz_high * 1.05)

                        if not near_prz:
                            continue

                        # Validate pattern with current price as D
                        is_valid, confidence, ratios = validate_pattern(
                            x, a, b, c, current_price,
                            pattern_type, direction, tolerance
                        )

                        if is_valid and confidence >= min_conf:
                            # Create D point at current price
                            d_point = SwingPoint(
                                index=len(data.bars) - 1,
                                price=current_price,
                                timestamp=data.bars[-1].timestamp,
                                is_high=(direction == PatternDirection.BEARISH)
                            )

                            status = PatternStatus.COMPLETE if in_prz else PatternStatus.FORMING

                            patterns.append(HarmonicPattern(
                                pattern_type=pattern_type,
                                direction=direction,
                                status=status,
                                x=x,
                                a=a,
                                b=b,
                                c=c,
                                d=d_point,
                                prz_low=prz_low,
                                prz_high=prz_high,
                                confidence=confidence,
                                ratios=ratios
                            ))

    # Sort by confidence and return unique patterns
    patterns.sort(key=lambda p: p.confidence, reverse=True)

    # Remove duplicates (same type and direction at similar levels)
    unique: List[HarmonicPattern] = []
    for p in patterns:
        is_duplicate = False
        for u in unique:
            if (p.pattern_type == u.pattern_type and
                p.direction == u.direction and
                abs(p.prz_low - u.prz_low) / u.prz_low < 0.02):
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(p)

    return unique[:3]  # Return top 3


def analyze_harmonic(
    data: OHLCVData,
    tolerance: float = 0.05,
    min_conf: float = 0.60,
    mode: str = "balanced"
) -> dict:
    """
    Analyze market for harmonic patterns.

    Args:
        data: OHLCV data
        tolerance: Fibonacci ratio tolerance (default 5%)
        min_conf: Minimum confidence threshold
        mode: Analysis mode

    Returns:
        AnalysisResult dictionary
    """
    attribution = {
        "theory": "Harmonic Patterns",
        "authors": ["H.M. Gartley", "Scott Carney"],
        "reference": "Profits in the Stock Market (1935), Harmonic Trading (2004)"
    }

    # Check minimum data
    if len(data.bars) < MIN_BARS_REQUIRED:
        return build_no_signal_result(
            reason=f"Insufficient data: {len(data.bars)} bars (need {MIN_BARS_REQUIRED}+)",
            tool_name=TOOL_NAME
        )

    # Adjust parameters by mode
    if mode == "conservative":
        tolerance = 0.03
        min_conf = 0.70
        window = 7
    elif mode == "aggressive":
        tolerance = 0.07
        min_conf = 0.55
        window = 4
    else:  # balanced
        window = 5

    # Find swing points
    swing_points = find_swing_points(data, window=window)

    if len(swing_points) < 4:
        return build_no_signal_result(
            reason="Insufficient swing points for harmonic pattern detection",
            tool_name=TOOL_NAME
        )

    # Detect patterns
    patterns = detect_patterns(data, swing_points, tolerance, min_conf)

    if not patterns:
        return build_no_signal_result(
            reason="No valid harmonic patterns detected",
            tool_name=TOOL_NAME
        )

    # Use best pattern
    best = patterns[0]

    # Determine status
    if best.direction == PatternDirection.BULLISH:
        status = "bullish"
    else:
        status = "bearish"

    # Calculate confidence
    timeframe_weight = get_timeframe_weight(data.timeframe)

    # Quality based on pattern confidence and completion
    q_pattern = best.confidence
    if best.status == PatternStatus.COMPLETE:
        q_pattern = min(1.0, q_pattern * 1.1)

    factors = ConfidenceFactors(
        base=HARMONIC_BASE_CONFIDENCE,
        q_pattern=q_pattern,
        w_time=timeframe_weight,
        v_conf=0.85,  # Volume not primary in harmonics
        m_bonus=1.0
    )

    confidence = calculate_confidence(factors)

    # Build invalidation
    if best.direction == PatternDirection.BULLISH:
        invalidation = f"Break below PRZ ({best.prz_low:.2f}) invalidates bullish pattern"
    else:
        invalidation = f"Break above PRZ ({best.prz_high:.2f}) invalidates bearish pattern"

    # Build LLM summary
    pattern_name = best.pattern_type.value.title()
    status_str = "complete" if best.status == PatternStatus.COMPLETE else "forming"

    summary = (
        f"Harmonic Pattern: {best.direction.value.title()} {pattern_name} ({status_str}) "
        f"on {data.timeframe}. PRZ: {best.prz_low:.2f}-{best.prz_high:.2f}."
    )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_all_harmonic_patterns(
    data: OHLCVData,
    tolerance: float = 0.05
) -> dict:
    """
    Get detailed information about all detected harmonic patterns.

    Returns:
        Dictionary with all patterns and their details
    """
    swing_points = find_swing_points(data)
    patterns = detect_patterns(data, swing_points, tolerance, min_conf=0.5)

    return {
        "patterns": [
            {
                "type": p.pattern_type.value,
                "direction": p.direction.value,
                "status": p.status.value,
                "confidence": p.confidence,
                "prz": {
                    "low": p.prz_low,
                    "high": p.prz_high
                },
                "points": {
                    "X": {"index": p.x.index, "price": p.x.price},
                    "A": {"index": p.a.index, "price": p.a.price},
                    "B": {"index": p.b.index, "price": p.b.price},
                    "C": {"index": p.c.index, "price": p.c.price},
                    "D": {"index": p.d.index, "price": p.d.price} if p.d else None
                },
                "ratios": p.ratios
            }
            for p in patterns
        ],
        "swing_points_count": len(swing_points),
        "patterns_found": len(patterns)
    }
