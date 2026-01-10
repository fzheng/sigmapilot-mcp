"""
Chan Theory (Chanlun/缠论) Analyzer Engine.

Implements the Chan Theory technical analysis framework:
- Bi (笔/Stroke): Basic price movement between fractals
- Duan (段/Segment): Combination of strokes
- Zhongshu (中枢/Hub): Price consolidation zones
- Buy/Sell points based on structure

Theory Attribution:
    Chan (缠中说禅, 2006-2008)
    "Teaching You to Trade Stocks from Scratch" (教你炒股票)

Base Confidence: 55 (per CLAUDE.md specification)
Note: Simplified implementation without external library dependency.
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
TOOL_NAME = "Chan Theory (Chanlun)"
CHAN_BASE_CONFIDENCE = 55
MIN_BARS_REQUIRED = 30
SIGNAL_THRESHOLD = 60


class FractalType(Enum):
    """Fractal types (分型)."""
    TOP = "top"      # 顶分型
    BOTTOM = "bottom"  # 底分型


class TrendDirection(Enum):
    """Trend direction for strokes and segments."""
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


class SignalType(Enum):
    """Chan Theory signal types."""
    BUY_POINT_1 = "buy_point_1"   # 一买 - Bottom divergence
    BUY_POINT_2 = "buy_point_2"   # 二买 - Pullback after trend reversal
    BUY_POINT_3 = "buy_point_3"   # 三买 - Breakout above hub
    SELL_POINT_1 = "sell_point_1"  # 一卖 - Top divergence
    SELL_POINT_2 = "sell_point_2"  # 二卖 - Rally after trend reversal
    SELL_POINT_3 = "sell_point_3"  # 三卖 - Breakdown below hub


@dataclass
class Fractal:
    """Represents a fractal point (分型)."""
    fractal_type: FractalType
    index: int
    high: float
    low: float
    timestamp: int


@dataclass
class Stroke:
    """Represents a stroke (笔) - movement between fractals."""
    start_fractal: Fractal
    end_fractal: Fractal
    direction: TrendDirection
    bars_count: int


@dataclass
class Segment:
    """Represents a segment (段) - combination of strokes."""
    strokes: List[Stroke]
    direction: TrendDirection
    start_price: float
    end_price: float


@dataclass
class Hub:
    """Represents a hub/consolidation zone (中枢)."""
    high: float  # ZG - Hub high
    low: float   # ZD - Hub low
    start_idx: int
    end_idx: int
    strokes: List[Stroke]


@dataclass
class ChanSignal:
    """Represents a detected Chan Theory signal."""
    signal_type: SignalType
    bar_index: int
    price: float
    confidence: float
    description: str


def merge_k_lines(
    data: OHLCVData
) -> List[Tuple[int, float, float]]:
    """
    Merge inclusive K-lines (包含处理).

    When two consecutive bars have inclusive relationship (one contains the other),
    merge them based on trend direction.

    Returns:
        List of (index, high, low) after merging
    """
    if len(data.bars) < 2:
        return [(i, data.bars[i].high, data.bars[i].low) for i in range(len(data.bars))]

    merged = [(0, data.bars[0].high, data.bars[0].low)]

    for i in range(1, len(data.bars)):
        curr_high = data.bars[i].high
        curr_low = data.bars[i].low
        prev_high = merged[-1][1]
        prev_low = merged[-1][2]

        # Check for inclusion
        curr_contains_prev = curr_high >= prev_high and curr_low <= prev_low
        prev_contains_curr = prev_high >= curr_high and prev_low <= curr_low

        if curr_contains_prev or prev_contains_curr:
            # Determine trend direction from previous merged bars
            if len(merged) >= 2:
                trend_up = merged[-1][1] > merged[-2][1]
            else:
                trend_up = curr_high > prev_high

            if trend_up:
                # Merge upward - take higher high and higher low
                new_high = max(curr_high, prev_high)
                new_low = max(curr_low, prev_low)
            else:
                # Merge downward - take lower high and lower low
                new_high = min(curr_high, prev_high)
                new_low = min(curr_low, prev_low)

            merged[-1] = (i, new_high, new_low)
        else:
            merged.append((i, curr_high, curr_low))

    return merged


def detect_fractals(
    merged_bars: List[Tuple[int, float, float]]
) -> List[Fractal]:
    """
    Detect fractals (分型) from merged K-lines.

    A top fractal: middle bar has highest high
    A bottom fractal: middle bar has lowest low

    Returns:
        List of detected fractals
    """
    fractals: List[Fractal] = []

    if len(merged_bars) < 3:
        return fractals

    for i in range(1, len(merged_bars) - 1):
        idx, high, low = merged_bars[i]
        prev_idx, prev_high, prev_low = merged_bars[i - 1]
        next_idx, next_high, next_low = merged_bars[i + 1]

        # Top fractal
        if high > prev_high and high > next_high:
            fractals.append(Fractal(
                fractal_type=FractalType.TOP,
                index=idx,
                high=high,
                low=low,
                timestamp=idx  # Will be replaced with actual timestamp
            ))

        # Bottom fractal
        if low < prev_low and low < next_low:
            fractals.append(Fractal(
                fractal_type=FractalType.BOTTOM,
                index=idx,
                high=high,
                low=low,
                timestamp=idx
            ))

    return fractals


def build_strokes(
    fractals: List[Fractal],
    min_bars: int = 4
) -> List[Stroke]:
    """
    Build strokes (笔) from fractals.

    A valid stroke must:
    1. Connect opposite type fractals (top to bottom or bottom to top)
    2. Have at least min_bars between fractals

    Returns:
        List of valid strokes
    """
    strokes: List[Stroke] = []

    if len(fractals) < 2:
        return strokes

    # Filter to alternating fractals
    alternating: List[Fractal] = [fractals[0]]

    for f in fractals[1:]:
        if f.fractal_type != alternating[-1].fractal_type:
            # Check minimum distance
            if f.index - alternating[-1].index >= min_bars:
                alternating.append(f)
            else:
                # Replace if more extreme
                if f.fractal_type == FractalType.TOP and f.high > alternating[-1].high:
                    alternating[-1] = f
                elif f.fractal_type == FractalType.BOTTOM and f.low < alternating[-1].low:
                    alternating[-1] = f
        else:
            # Same type - keep more extreme
            if f.fractal_type == FractalType.TOP and f.high > alternating[-1].high:
                alternating[-1] = f
            elif f.fractal_type == FractalType.BOTTOM and f.low < alternating[-1].low:
                alternating[-1] = f

    # Build strokes from alternating fractals
    for i in range(len(alternating) - 1):
        start = alternating[i]
        end = alternating[i + 1]

        if start.fractal_type == FractalType.BOTTOM:
            direction = TrendDirection.UP
        else:
            direction = TrendDirection.DOWN

        strokes.append(Stroke(
            start_fractal=start,
            end_fractal=end,
            direction=direction,
            bars_count=end.index - start.index
        ))

    return strokes


def build_segments(
    strokes: List[Stroke]
) -> List[Segment]:
    """
    Build segments (段) from strokes.

    A segment consists of at least 3 strokes with the same overall direction.

    Returns:
        List of segments
    """
    segments: List[Segment] = []

    if len(strokes) < 3:
        return segments

    i = 0
    while i < len(strokes) - 2:
        # Look for segment pattern (3+ strokes)
        segment_strokes = [strokes[i]]
        direction = strokes[i].direction

        j = i + 1
        while j < len(strokes):
            # Alternate directions within segment
            segment_strokes.append(strokes[j])

            # Check if segment completes (odd number of strokes, ends in same direction)
            if len(segment_strokes) >= 3 and len(segment_strokes) % 2 == 1:
                if segment_strokes[-1].direction == direction:
                    # Valid segment found
                    if direction == TrendDirection.UP:
                        start_price = segment_strokes[0].start_fractal.low
                        end_price = segment_strokes[-1].end_fractal.high
                    else:
                        start_price = segment_strokes[0].start_fractal.high
                        end_price = segment_strokes[-1].end_fractal.low

                    segments.append(Segment(
                        strokes=segment_strokes.copy(),
                        direction=direction,
                        start_price=start_price,
                        end_price=end_price
                    ))
                    break
            j += 1

        i = max(i + 1, j - 1) if j < len(strokes) else i + 1

    return segments


def detect_hub(
    strokes: List[Stroke]
) -> Optional[Hub]:
    """
    Detect hub/consolidation zone (中枢) from strokes.

    A hub requires at least 3 overlapping strokes.
    Hub range = intersection of stroke ranges.

    Returns:
        Hub if detected, None otherwise
    """
    if len(strokes) < 3:
        return None

    # Find overlapping region of consecutive strokes
    for i in range(len(strokes) - 2):
        subset = strokes[i:i + 3]

        # Get the high-low range of each stroke
        ranges = []
        for s in subset:
            stroke_high = max(s.start_fractal.high, s.end_fractal.high)
            stroke_low = min(s.start_fractal.low, s.end_fractal.low)
            ranges.append((stroke_low, stroke_high))

        # Find intersection
        hub_low = max(r[0] for r in ranges)
        hub_high = min(r[1] for r in ranges)

        if hub_high > hub_low:
            # Valid hub found
            return Hub(
                high=hub_high,
                low=hub_low,
                start_idx=subset[0].start_fractal.index,
                end_idx=subset[-1].end_fractal.index,
                strokes=subset
            )

    return None


def detect_signals(
    data: OHLCVData,
    strokes: List[Stroke],
    segments: List[Segment],
    hub: Optional[Hub]
) -> List[ChanSignal]:
    """
    Detect Chan Theory buy/sell signals.

    Returns:
        List of detected signals
    """
    signals: List[ChanSignal] = []

    if len(strokes) < 2:
        return signals

    current_price = data.closes[-1]
    last_stroke = strokes[-1]

    # Buy Point 3 / Sell Point 3 - Hub breakout/breakdown
    if hub:
        if current_price > hub.high and last_stroke.direction == TrendDirection.UP:
            signals.append(ChanSignal(
                signal_type=SignalType.BUY_POINT_3,
                bar_index=len(data.bars) - 1,
                price=current_price,
                confidence=0.75,
                description=f"Breakout above hub ({hub.high:.2f})"
            ))

        if current_price < hub.low and last_stroke.direction == TrendDirection.DOWN:
            signals.append(ChanSignal(
                signal_type=SignalType.SELL_POINT_3,
                bar_index=len(data.bars) - 1,
                price=current_price,
                confidence=0.75,
                description=f"Breakdown below hub ({hub.low:.2f})"
            ))

    # Buy Point 2 / Sell Point 2 - Pullback signals
    if len(strokes) >= 3:
        # Check for bullish pullback (up-down-up pattern where pullback is weak)
        s1, s2, s3 = strokes[-3], strokes[-2], strokes[-1]

        if (s1.direction == TrendDirection.UP and
            s2.direction == TrendDirection.DOWN and
            s3.direction == TrendDirection.UP):

            # Pullback didn't break the start of s1
            if s2.end_fractal.low > s1.start_fractal.low:
                signals.append(ChanSignal(
                    signal_type=SignalType.BUY_POINT_2,
                    bar_index=s3.start_fractal.index,
                    price=s3.start_fractal.low,
                    confidence=0.70,
                    description="Higher low after upward stroke"
                ))

        # Check for bearish pullback
        if (s1.direction == TrendDirection.DOWN and
            s2.direction == TrendDirection.UP and
            s3.direction == TrendDirection.DOWN):

            if s2.end_fractal.high < s1.start_fractal.high:
                signals.append(ChanSignal(
                    signal_type=SignalType.SELL_POINT_2,
                    bar_index=s3.start_fractal.index,
                    price=s3.start_fractal.high,
                    confidence=0.70,
                    description="Lower high after downward stroke"
                ))

    # Buy Point 1 / Sell Point 1 - Divergence (simplified)
    # Check if latest stroke makes new extreme but momentum weakens
    if len(strokes) >= 2:
        prev_stroke = strokes[-2]
        curr_stroke = strokes[-1]

        if curr_stroke.direction == TrendDirection.DOWN:
            # Check for potential bottom
            if (curr_stroke.end_fractal.low <= prev_stroke.start_fractal.low and
                curr_stroke.bars_count < prev_stroke.bars_count):
                signals.append(ChanSignal(
                    signal_type=SignalType.BUY_POINT_1,
                    bar_index=curr_stroke.end_fractal.index,
                    price=curr_stroke.end_fractal.low,
                    confidence=0.65,
                    description="New low with weakening momentum"
                ))

        if curr_stroke.direction == TrendDirection.UP:
            if (curr_stroke.end_fractal.high >= prev_stroke.start_fractal.high and
                curr_stroke.bars_count < prev_stroke.bars_count):
                signals.append(ChanSignal(
                    signal_type=SignalType.SELL_POINT_1,
                    bar_index=curr_stroke.end_fractal.index,
                    price=curr_stroke.end_fractal.high,
                    confidence=0.65,
                    description="New high with weakening momentum"
                ))

    return signals


def analyze_chan_theory(
    data: OHLCVData,
    strictness: str = "balanced",
    mode: str = "balanced"
) -> dict:
    """
    Analyze market using Chan Theory (Chanlun).

    Args:
        data: OHLCV data
        strictness: "conservative", "balanced", or "aggressive"
        mode: Analysis mode (same as strictness for this engine)

    Returns:
        AnalysisResult dictionary
    """
    attribution = {
        "theory": "Chan Theory (Chanlun/缠论)",
        "author": "缠中说禅",
        "reference": "教你炒股票 (Teaching You to Trade Stocks)"
    }

    # Check minimum data
    if len(data.bars) < MIN_BARS_REQUIRED:
        return build_no_signal_result(
            reason=f"Insufficient data: {len(data.bars)} bars (need {MIN_BARS_REQUIRED}+)",
            tool_name=TOOL_NAME
        )

    # Merge K-lines
    merged = merge_k_lines(data)

    if len(merged) < 5:
        return build_no_signal_result(
            reason="Insufficient merged K-lines for Chan analysis",
            tool_name=TOOL_NAME
        )

    # Detect fractals
    fractals = detect_fractals(merged)

    if len(fractals) < 4:
        return build_no_signal_result(
            reason="Insufficient fractals detected",
            tool_name=TOOL_NAME
        )

    # Build strokes
    min_bars = 5 if strictness == "conservative" else 4 if strictness == "balanced" else 3
    strokes = build_strokes(fractals, min_bars=min_bars)

    if len(strokes) < 2:
        return build_no_signal_result(
            reason="Insufficient strokes for trend analysis",
            tool_name=TOOL_NAME
        )

    # Build segments
    segments = build_segments(strokes)

    # Detect hub
    hub = detect_hub(strokes)

    # Detect signals
    signals = detect_signals(data, strokes, segments, hub)

    # Determine current trend from latest stroke
    current_stroke = strokes[-1]
    current_trend = current_stroke.direction

    if not signals:
        # No clear signal but can still report structure
        status = "neutral"
        confidence_penalty = 0.8
    else:
        # Determine status from signals
        buy_signals = [s for s in signals if "buy" in s.signal_type.value]
        sell_signals = [s for s in signals if "sell" in s.signal_type.value]

        if buy_signals and not sell_signals:
            status = "bullish"
            confidence_penalty = 1.0
        elif sell_signals and not buy_signals:
            status = "bearish"
            confidence_penalty = 1.0
        else:
            status = "neutral"
            confidence_penalty = 0.85

    # Calculate confidence
    timeframe_weight = get_timeframe_weight(data.timeframe)

    # Quality based on structure clarity
    structure_quality = min(1.0, len(strokes) / 10)  # More strokes = clearer structure

    if strictness == "conservative":
        structure_quality *= 0.9
    elif strictness == "aggressive":
        structure_quality = min(1.0, structure_quality * 1.1)

    factors = ConfidenceFactors(
        base=CHAN_BASE_CONFIDENCE,
        q_pattern=structure_quality * confidence_penalty,
        w_time=timeframe_weight,
        v_conf=0.85,
        m_bonus=1.0
    )

    confidence = calculate_confidence(factors)

    # Build invalidation
    if current_trend == TrendDirection.UP:
        last_low = current_stroke.start_fractal.low
        invalidation = f"Break below {last_low:.2f} (last stroke low) invalidates bullish structure"
    else:
        last_high = current_stroke.start_fractal.high
        invalidation = f"Break above {last_high:.2f} (last stroke high) invalidates bearish structure"

    # Build LLM summary
    signal_strs = [f"{s.signal_type.value.replace('_', ' ').title()}" for s in signals[:2]]
    signal_part = f" Signals: {', '.join(signal_strs)}." if signal_strs else ""

    hub_part = f" Hub zone: {hub.low:.2f}-{hub.high:.2f}." if hub else ""

    summary = (
        f"Chan Theory on {data.timeframe}: Current trend {current_trend.value}. "
        f"{len(strokes)} strokes, {len(segments)} segments detected.{hub_part}{signal_part}"
    )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_detailed_chan(data: OHLCVData) -> dict:
    """
    Get detailed Chan Theory analysis with all structure components.

    Returns:
        Dictionary with fractals, strokes, segments, hub, and signals
    """
    merged = merge_k_lines(data)
    fractals = detect_fractals(merged)
    strokes = build_strokes(fractals)
    segments = build_segments(strokes)
    hub = detect_hub(strokes)
    signals = detect_signals(data, strokes, segments, hub)

    return {
        "fractals": [
            {
                "type": f.fractal_type.value,
                "index": f.index,
                "high": f.high,
                "low": f.low
            }
            for f in fractals
        ],
        "strokes": [
            {
                "direction": s.direction.value,
                "start_index": s.start_fractal.index,
                "end_index": s.end_fractal.index,
                "bars_count": s.bars_count
            }
            for s in strokes
        ],
        "segments": [
            {
                "direction": seg.direction.value,
                "start_price": seg.start_price,
                "end_price": seg.end_price,
                "stroke_count": len(seg.strokes)
            }
            for seg in segments
        ],
        "hub": {
            "detected": hub is not None,
            "high": hub.high if hub else None,
            "low": hub.low if hub else None,
            "start_idx": hub.start_idx if hub else None,
            "end_idx": hub.end_idx if hub else None
        },
        "signals": [
            {
                "type": sig.signal_type.value,
                "index": sig.bar_index,
                "price": sig.price,
                "confidence": sig.confidence,
                "description": sig.description
            }
            for sig in signals
        ],
        "counts": {
            "fractals": len(fractals),
            "strokes": len(strokes),
            "segments": len(segments),
            "signals": len(signals)
        }
    }
