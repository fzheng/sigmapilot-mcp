"""
Elliott Wave Analyzer Engine.

Detects Elliott Wave patterns (impulse and corrective waves) with strict rule validation.
Conservative approach - defaults to neutral when ambiguous.

Theory Attribution:
    Ralph Nelson Elliott (1871-1948)
    "The Wave Principle" - Market moves in repetitive wave patterns

Base Confidence: 60 (per CLAUDE.md specification)
Note: Elliott Wave is inherently subjective; strict rules enforced.
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
TOOL_NAME = "Elliott Wave Theory"
ELLIOTT_BASE_CONFIDENCE = 60
MIN_BARS_REQUIRED = 50
SIGNAL_THRESHOLD = 60


class WaveType(Enum):
    """Elliott Wave types."""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    UNKNOWN = "unknown"


class WaveLabel(Enum):
    """Wave position labels."""
    WAVE_1 = "1"
    WAVE_2 = "2"
    WAVE_3 = "3"
    WAVE_4 = "4"
    WAVE_5 = "5"
    WAVE_A = "A"
    WAVE_B = "B"
    WAVE_C = "C"
    UNKNOWN = "?"


class WaveDegree(Enum):
    """Wave degree/timeframe."""
    GRAND_SUPERCYCLE = "grand_supercycle"
    SUPERCYCLE = "supercycle"
    CYCLE = "cycle"
    PRIMARY = "primary"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    MINUTE = "minute"
    MINUETTE = "minuette"
    SUBMINUETTE = "subminuette"


@dataclass
class WavePoint:
    """Represents a wave pivot point."""
    index: int
    price: float
    timestamp: int
    label: WaveLabel


@dataclass
class WaveStructure:
    """Represents a detected wave structure."""
    wave_type: WaveType
    direction: str  # bullish or bearish
    current_wave: WaveLabel
    wave_points: List[WavePoint]
    confidence: float
    rules_satisfied: List[str]
    rules_violated: List[str]


def find_pivots(
    data: OHLCVData,
    window: int = 5
) -> Tuple[List[int], List[int]]:
    """
    Find pivot highs and lows.

    Returns:
        Tuple of (pivot_high_indices, pivot_low_indices)
    """
    highs = data.highs
    lows = data.lows

    pivot_highs = []
    pivot_lows = []

    for i in range(window, len(highs) - window):
        # Check pivot high
        is_pivot_high = True
        for j in range(1, window + 1):
            if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                is_pivot_high = False
                break
        if is_pivot_high:
            pivot_highs.append(i)

        # Check pivot low
        is_pivot_low = True
        for j in range(1, window + 1):
            if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                is_pivot_low = False
                break
        if is_pivot_low:
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def merge_pivots(
    pivot_highs: List[int],
    pivot_lows: List[int],
    highs: np.ndarray,
    lows: np.ndarray
) -> List[Tuple[int, str, float]]:
    """
    Merge pivot highs and lows into alternating sequence.

    Returns:
        List of (index, type, price) tuples
    """
    all_pivots = []

    for idx in pivot_highs:
        all_pivots.append((idx, "high", highs[idx]))

    for idx in pivot_lows:
        all_pivots.append((idx, "low", lows[idx]))

    # Sort by index
    all_pivots.sort(key=lambda x: x[0])

    # Remove consecutive same-type pivots (keep most extreme)
    if len(all_pivots) < 2:
        return all_pivots

    merged = [all_pivots[0]]
    for pivot in all_pivots[1:]:
        if pivot[1] == merged[-1][1]:
            # Same type - keep more extreme
            if pivot[1] == "high" and pivot[2] > merged[-1][2]:
                merged[-1] = pivot
            elif pivot[1] == "low" and pivot[2] < merged[-1][2]:
                merged[-1] = pivot
        else:
            merged.append(pivot)

    return merged


def validate_impulse_rules(
    waves: List[WavePoint],
    direction: str
) -> Tuple[List[str], List[str]]:
    """
    Validate Elliott Wave impulse rules.

    Rules:
    1. Wave 2 never retraces more than 100% of Wave 1
    2. Wave 3 is never the shortest among 1, 3, 5
    3. Wave 4 never enters Wave 1 territory

    Returns:
        Tuple of (satisfied_rules, violated_rules)
    """
    satisfied = []
    violated = []

    if len(waves) < 5:
        return satisfied, ["Insufficient waves for impulse pattern"]

    # Extract wave prices
    w0 = waves[0].price  # Start
    w1 = waves[1].price  # End of Wave 1
    w2 = waves[2].price  # End of Wave 2
    w3 = waves[3].price  # End of Wave 3
    w4 = waves[4].price  # End of Wave 4

    if direction == "bullish":
        # Rule 1: Wave 2 retracement
        wave1_move = w1 - w0
        wave2_retrace = w1 - w2

        if wave1_move > 0 and wave2_retrace < wave1_move:
            satisfied.append("Rule 1: Wave 2 retraces < 100% of Wave 1")
        else:
            violated.append("Rule 1: Wave 2 retraces >= 100% of Wave 1")

        # Rule 2: Wave 3 length
        wave1_len = abs(w1 - w0)
        wave3_len = abs(w3 - w2)

        if len(waves) >= 6:
            w5 = waves[5].price
            wave5_len = abs(w5 - w4)
            if wave3_len >= wave1_len or wave3_len >= wave5_len:
                satisfied.append("Rule 2: Wave 3 is not the shortest")
            else:
                violated.append("Rule 2: Wave 3 is the shortest wave")
        elif wave3_len >= wave1_len:
            satisfied.append("Rule 2: Wave 3 >= Wave 1 (partial check)")

        # Rule 3: Wave 4 doesn't enter Wave 1 territory
        if w4 > w1:
            violated.append("Rule 3: Wave 4 overlaps Wave 1")
        else:
            satisfied.append("Rule 3: Wave 4 doesn't overlap Wave 1")

    else:  # bearish
        # Rule 1: Wave 2 retracement
        wave1_move = w0 - w1
        wave2_retrace = w2 - w1

        if wave1_move > 0 and wave2_retrace < wave1_move:
            satisfied.append("Rule 1: Wave 2 retraces < 100% of Wave 1")
        else:
            violated.append("Rule 1: Wave 2 retraces >= 100% of Wave 1")

        # Rule 2: Wave 3 length
        wave1_len = abs(w0 - w1)
        wave3_len = abs(w2 - w3)

        if len(waves) >= 6:
            w5 = waves[5].price
            wave5_len = abs(w4 - w5)
            if wave3_len >= wave1_len or wave3_len >= wave5_len:
                satisfied.append("Rule 2: Wave 3 is not the shortest")
            else:
                violated.append("Rule 2: Wave 3 is the shortest wave")
        elif wave3_len >= wave1_len:
            satisfied.append("Rule 2: Wave 3 >= Wave 1 (partial check)")

        # Rule 3: Wave 4 doesn't enter Wave 1 territory
        if w4 < w1:
            violated.append("Rule 3: Wave 4 overlaps Wave 1")
        else:
            satisfied.append("Rule 3: Wave 4 doesn't overlap Wave 1")

    return satisfied, violated


def validate_corrective_rules(
    waves: List[WavePoint],
    direction: str
) -> Tuple[List[str], List[str]]:
    """
    Validate corrective wave rules (ABC pattern).

    Returns:
        Tuple of (satisfied_rules, violated_rules)
    """
    satisfied = []
    violated = []

    if len(waves) < 3:
        return satisfied, ["Insufficient waves for corrective pattern"]

    # Simple ABC validation
    w_start = waves[0].price
    w_a = waves[1].price
    w_b = waves[2].price

    if len(waves) >= 4:
        w_c = waves[3].price

        if direction == "bullish":  # Corrective in uptrend (moves down)
            # Wave B should retrace Wave A partially
            wave_a_move = w_start - w_a
            wave_b_retrace = w_b - w_a

            if wave_a_move > 0 and 0 < wave_b_retrace < wave_a_move:
                satisfied.append("Wave B partial retracement of Wave A")
            else:
                violated.append("Wave B retracement abnormal")

            # Wave C typically extends beyond Wave A
            if w_c < w_a:
                satisfied.append("Wave C extends beyond Wave A")
            else:
                violated.append("Wave C doesn't extend beyond Wave A")
        else:
            wave_a_move = w_a - w_start
            wave_b_retrace = w_a - w_b

            if wave_a_move > 0 and 0 < wave_b_retrace < wave_a_move:
                satisfied.append("Wave B partial retracement of Wave A")
            else:
                violated.append("Wave B retracement abnormal")

            if w_c > w_a:
                satisfied.append("Wave C extends beyond Wave A")
            else:
                violated.append("Wave C doesn't extend beyond Wave A")

    return satisfied, violated


def detect_wave_structure(
    data: OHLCVData,
    pivots: List[Tuple[int, str, float]],
    max_interpretations: int = 2
) -> List[WaveStructure]:
    """
    Detect potential wave structures from pivots.

    Args:
        data: OHLCV data
        pivots: Merged pivot points
        max_interpretations: Maximum number of interpretations to return

    Returns:
        List of detected wave structures
    """
    structures: List[WaveStructure] = []

    if len(pivots) < 5:
        return structures

    # Try to find impulse patterns (5-wave structure)
    # Look for bullish impulse (alternating low-high-low-high-low-high)
    for start_idx in range(len(pivots) - 5):
        subset = pivots[start_idx:start_idx + 6]

        # Check if starts with low (bullish) or high (bearish)
        if subset[0][1] == "low" and all(
            subset[i][1] == ("low" if i % 2 == 0 else "high")
            for i in range(6)
        ):
            # Potential bullish impulse
            wave_points = [
                WavePoint(
                    index=subset[i][0],
                    price=subset[i][2],
                    timestamp=data.bars[subset[i][0]].timestamp,
                    label=[WaveLabel.WAVE_1, WaveLabel.WAVE_1, WaveLabel.WAVE_2,
                           WaveLabel.WAVE_3, WaveLabel.WAVE_4, WaveLabel.WAVE_5][i]
                )
                for i in range(6)
            ]
            # Relabel: first is start (0), then 1,2,3,4,5
            wave_points[0].label = WaveLabel.UNKNOWN  # Start point

            satisfied, violated = validate_impulse_rules(wave_points, "bullish")

            if len(violated) == 0:
                confidence = 0.9
            elif len(violated) == 1:
                confidence = 0.7
            else:
                confidence = 0.5

            structures.append(WaveStructure(
                wave_type=WaveType.IMPULSE,
                direction="bullish",
                current_wave=WaveLabel.WAVE_5,
                wave_points=wave_points,
                confidence=confidence,
                rules_satisfied=satisfied,
                rules_violated=violated
            ))

        elif subset[0][1] == "high" and all(
            subset[i][1] == ("high" if i % 2 == 0 else "low")
            for i in range(6)
        ):
            # Potential bearish impulse
            wave_points = [
                WavePoint(
                    index=subset[i][0],
                    price=subset[i][2],
                    timestamp=data.bars[subset[i][0]].timestamp,
                    label=[WaveLabel.UNKNOWN, WaveLabel.WAVE_1, WaveLabel.WAVE_2,
                           WaveLabel.WAVE_3, WaveLabel.WAVE_4, WaveLabel.WAVE_5][i]
                )
                for i in range(6)
            ]

            satisfied, violated = validate_impulse_rules(wave_points, "bearish")

            if len(violated) == 0:
                confidence = 0.9
            elif len(violated) == 1:
                confidence = 0.7
            else:
                confidence = 0.5

            structures.append(WaveStructure(
                wave_type=WaveType.IMPULSE,
                direction="bearish",
                current_wave=WaveLabel.WAVE_5,
                wave_points=wave_points,
                confidence=confidence,
                rules_satisfied=satisfied,
                rules_violated=violated
            ))

    # Try to find ABC corrective patterns
    for start_idx in range(len(pivots) - 3):
        subset = pivots[start_idx:start_idx + 4]

        if subset[0][1] == "high":
            # Potential downward ABC correction
            wave_points = [
                WavePoint(
                    index=subset[i][0],
                    price=subset[i][2],
                    timestamp=data.bars[subset[i][0]].timestamp,
                    label=[WaveLabel.UNKNOWN, WaveLabel.WAVE_A,
                           WaveLabel.WAVE_B, WaveLabel.WAVE_C][i]
                )
                for i in range(4)
            ]

            satisfied, violated = validate_corrective_rules(wave_points, "bullish")

            if len(violated) == 0:
                confidence = 0.8
            elif len(violated) == 1:
                confidence = 0.6
            else:
                confidence = 0.4

            structures.append(WaveStructure(
                wave_type=WaveType.CORRECTIVE,
                direction="bearish",  # ABC in uptrend moves down
                current_wave=WaveLabel.WAVE_C,
                wave_points=wave_points,
                confidence=confidence,
                rules_satisfied=satisfied,
                rules_violated=violated
            ))

        elif subset[0][1] == "low":
            # Potential upward ABC correction
            wave_points = [
                WavePoint(
                    index=subset[i][0],
                    price=subset[i][2],
                    timestamp=data.bars[subset[i][0]].timestamp,
                    label=[WaveLabel.UNKNOWN, WaveLabel.WAVE_A,
                           WaveLabel.WAVE_B, WaveLabel.WAVE_C][i]
                )
                for i in range(4)
            ]

            satisfied, violated = validate_corrective_rules(wave_points, "bearish")

            if len(violated) == 0:
                confidence = 0.8
            elif len(violated) == 1:
                confidence = 0.6
            else:
                confidence = 0.4

            structures.append(WaveStructure(
                wave_type=WaveType.CORRECTIVE,
                direction="bullish",
                current_wave=WaveLabel.WAVE_C,
                wave_points=wave_points,
                confidence=confidence,
                rules_satisfied=satisfied,
                rules_violated=violated
            ))

    # Sort by confidence and recency, return top interpretations
    structures.sort(key=lambda s: (s.confidence, s.wave_points[-1].index), reverse=True)

    return structures[:max_interpretations]


def determine_current_position(
    data: OHLCVData,
    structures: List[WaveStructure]
) -> Tuple[Optional[WaveStructure], WaveLabel, str]:
    """
    Determine the most likely current wave position.

    Returns:
        Tuple of (best_structure, current_wave, status)
    """
    if not structures:
        return None, WaveLabel.UNKNOWN, "neutral"

    best = structures[0]

    # Determine if we're at the end of a pattern or mid-pattern
    current_wave = best.current_wave
    last_point_idx = best.wave_points[-1].index
    current_idx = len(data.bars) - 1
    bars_since_last = current_idx - last_point_idx

    # If many bars since last wave point, pattern may be stale
    if bars_since_last > 20:
        return best, WaveLabel.UNKNOWN, "neutral"

    # Determine status based on wave type and position
    if best.wave_type == WaveType.IMPULSE:
        if current_wave == WaveLabel.WAVE_5:
            # End of impulse - expect correction
            if best.direction == "bullish":
                status = "neutral"  # Expecting pullback
            else:
                status = "neutral"
        elif current_wave in [WaveLabel.WAVE_1, WaveLabel.WAVE_3]:
            status = best.direction
        elif current_wave in [WaveLabel.WAVE_2, WaveLabel.WAVE_4]:
            # In correction within trend
            status = "neutral"
        else:
            status = "neutral"
    else:  # Corrective
        if current_wave == WaveLabel.WAVE_C:
            # End of correction - expect trend resumption
            if best.direction == "bearish":
                status = "bullish"  # Uptrend should resume
            else:
                status = "bearish"
        else:
            status = "neutral"

    return best, current_wave, status


def analyze_elliott_wave(
    data: OHLCVData,
    degree: Optional[str] = None,
    max_interpretations: int = 2,
    mode: str = "balanced"
) -> dict:
    """
    Analyze market using Elliott Wave Theory.

    Args:
        data: OHLCV data
        degree: Optional wave degree hint
        max_interpretations: Maximum interpretations to consider
        mode: Analysis mode - "conservative", "balanced", or "aggressive"

    Returns:
        AnalysisResult dictionary
    """
    attribution = {
        "theory": "Elliott Wave Theory",
        "author": "Ralph Nelson Elliott",
        "reference": "The Wave Principle (1938)"
    }

    # Check minimum data
    if len(data.bars) < MIN_BARS_REQUIRED:
        return build_no_signal_result(
            reason=f"Insufficient data: {len(data.bars)} bars (need {MIN_BARS_REQUIRED}+)",
            tool_name=TOOL_NAME
        )

    # Find pivots
    pivot_window = 5 if mode == "aggressive" else 7 if mode == "balanced" else 10
    pivot_highs, pivot_lows = find_pivots(data, window=pivot_window)

    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return build_no_signal_result(
            reason="Insufficient pivot points for wave analysis",
            tool_name=TOOL_NAME
        )

    # Merge pivots
    merged = merge_pivots(pivot_highs, pivot_lows, data.highs, data.lows)

    if len(merged) < 5:
        return build_no_signal_result(
            reason="Insufficient alternating pivots for wave structure",
            tool_name=TOOL_NAME
        )

    # Detect wave structures
    structures = detect_wave_structure(data, merged, max_interpretations)

    if not structures:
        return build_no_signal_result(
            reason="No valid Elliott Wave patterns detected",
            tool_name=TOOL_NAME
        )

    # Determine current position
    best_structure, current_wave, status = determine_current_position(data, structures)

    if best_structure is None or best_structure.confidence < 0.5:
        return build_no_signal_result(
            reason="Wave pattern confidence too low for reliable signal",
            tool_name=TOOL_NAME
        )

    # Check for rule violations - be conservative
    if len(best_structure.rules_violated) > 0 and mode == "conservative":
        return build_no_signal_result(
            reason=f"Wave rules violated: {', '.join(best_structure.rules_violated)}",
            tool_name=TOOL_NAME
        )

    # Calculate confidence
    timeframe_weight = get_timeframe_weight(data.timeframe)

    # Quality based on rule satisfaction
    if len(best_structure.rules_violated) == 0:
        q_pattern = 0.95
    elif len(best_structure.rules_violated) == 1:
        q_pattern = 0.75
    else:
        q_pattern = 0.60

    # Mode adjustments
    if mode == "conservative":
        q_pattern *= 0.9
    elif mode == "aggressive":
        q_pattern = min(1.0, q_pattern * 1.05)

    factors = ConfidenceFactors(
        base=ELLIOTT_BASE_CONFIDENCE,
        q_pattern=q_pattern,
        w_time=timeframe_weight,
        v_conf=0.85,  # Volume not heavily used in Elliott
        m_bonus=1.0
    )

    confidence = calculate_confidence(factors)

    # Build invalidation
    if best_structure.wave_type == WaveType.IMPULSE:
        if best_structure.direction == "bullish":
            # Wave 4 low is key support
            wave4_price = best_structure.wave_points[4].price if len(best_structure.wave_points) > 4 else None
            if wave4_price:
                invalidation = f"Break below Wave 4 at {wave4_price:.2f} invalidates bullish impulse"
            else:
                invalidation = "Break below Wave 4 low invalidates bullish impulse"
        else:
            wave4_price = best_structure.wave_points[4].price if len(best_structure.wave_points) > 4 else None
            if wave4_price:
                invalidation = f"Break above Wave 4 at {wave4_price:.2f} invalidates bearish impulse"
            else:
                invalidation = "Break above Wave 4 high invalidates bearish impulse"
    else:
        # Corrective
        wave_a_price = best_structure.wave_points[1].price if len(best_structure.wave_points) > 1 else None
        if wave_a_price:
            invalidation = f"Break beyond Wave A at {wave_a_price:.2f} signals pattern failure"
        else:
            invalidation = "Break beyond Wave A signals pattern failure"

    # Build LLM summary
    wave_type_str = best_structure.wave_type.value.title()
    direction_str = best_structure.direction.title()

    if len(structures) > 1:
        summary = (
            f"Elliott Wave: {direction_str} {wave_type_str} pattern detected on {data.timeframe}. "
            f"Currently in Wave {current_wave.value}. "
            f"Note: {len(structures)} interpretations possible."
        )
    else:
        summary = (
            f"Elliott Wave: {direction_str} {wave_type_str} pattern on {data.timeframe}. "
            f"Currently in Wave {current_wave.value}. "
            f"Rules satisfied: {len(best_structure.rules_satisfied)}/{len(best_structure.rules_satisfied) + len(best_structure.rules_violated)}."
        )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_detailed_elliott(
    data: OHLCVData,
    max_interpretations: int = 3
) -> dict:
    """
    Get detailed Elliott Wave analysis with all interpretations.

    Returns:
        Dictionary with all detected structures and wave points
    """
    pivot_highs, pivot_lows = find_pivots(data)
    merged = merge_pivots(pivot_highs, pivot_lows, data.highs, data.lows)
    structures = detect_wave_structure(data, merged, max_interpretations)

    return {
        "interpretations": [
            {
                "wave_type": s.wave_type.value,
                "direction": s.direction,
                "current_wave": s.current_wave.value,
                "confidence": s.confidence,
                "rules_satisfied": s.rules_satisfied,
                "rules_violated": s.rules_violated,
                "wave_points": [
                    {
                        "label": wp.label.value,
                        "index": wp.index,
                        "price": wp.price,
                        "timestamp": wp.timestamp
                    }
                    for wp in s.wave_points
                ]
            }
            for s in structures
        ],
        "pivot_count": len(merged),
        "interpretation_count": len(structures)
    }
