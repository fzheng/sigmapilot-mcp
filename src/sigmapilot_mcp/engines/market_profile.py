"""
Market Profile Analyzer Engine.

Analyzes market structure using Market Profile concepts:
- Point of Control (POC): Price level with highest volume/time
- Value Area High (VAH) and Value Area Low (VAL)
- Profile shape analysis (normal, p-shape, b-shape, double distribution)
- Balance vs imbalance detection

Theory Attribution:
    J. Peter Steidlmayer (1980s)
    "Market Profile" - Understanding market structure through time/price

Base Confidence: 55 (per CLAUDE.md specification)
Note: True TPO requires tick data; this uses volume-at-price approximation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np

from sigmapilot_mcp.core.data_loader import OHLCVData
from sigmapilot_mcp.core.confidence import ConfidenceFactors, calculate_confidence
from sigmapilot_mcp.core.schemas import build_analysis_result, build_no_signal_result
from sigmapilot_mcp.core.timeframes import get_timeframe_weight

# Constants
TOOL_NAME = "Market Profile"
MARKET_PROFILE_BASE_CONFIDENCE = 55
MIN_BARS_REQUIRED = 20
SIGNAL_THRESHOLD = 60
DEFAULT_VALUE_AREA_PCT = 0.70  # 70% of volume defines value area


class ProfileShape(Enum):
    """Market Profile shapes."""
    NORMAL = "normal"           # Bell curve - balanced market
    P_SHAPE = "p_shape"         # High value area - selling pressure
    B_SHAPE = "b_shape"         # Low value area - buying pressure
    D_SHAPE = "d_shape"         # Double distribution - trend day
    WIDE = "wide"               # Wide range - high volatility
    NARROW = "narrow"           # Narrow range - consolidation


class MarketState(Enum):
    """Market state based on profile."""
    BALANCED = "balanced"       # Price rotating around POC
    IMBALANCED_UP = "imbalanced_up"    # Price above value area
    IMBALANCED_DOWN = "imbalanced_down"  # Price below value area
    BREAKOUT = "breakout"       # Price breaking out of range


class PricePosition(Enum):
    """Current price position relative to profile."""
    ABOVE_VAH = "above_vah"
    IN_VALUE_AREA = "in_value_area"
    AT_POC = "at_poc"
    BELOW_VAL = "below_val"


@dataclass
class VolumeLevel:
    """Volume at a specific price level."""
    price: float
    volume: float
    bar_count: int  # TPO approximation


@dataclass
class MarketProfileData:
    """Complete market profile analysis."""
    poc: float                    # Point of Control
    vah: float                    # Value Area High
    val: float                    # Value Area Low
    profile_high: float           # Profile range high
    profile_low: float            # Profile range low
    value_area_volume_pct: float  # % of volume in value area
    volume_levels: List[VolumeLevel]
    shape: ProfileShape
    bars_analyzed: int


def build_volume_profile(
    data: OHLCVData,
    num_levels: int = 50
) -> List[VolumeLevel]:
    """
    Build volume-at-price profile from OHLCV data.

    Uses volume distribution approximation: assigns each bar's volume
    across its price range proportionally.

    Args:
        data: OHLCV data
        num_levels: Number of price levels to divide range into

    Returns:
        List of VolumeLevel objects sorted by price
    """
    if len(data.bars) == 0:
        return []

    # Get price range
    all_highs = data.highs
    all_lows = data.lows
    profile_high = float(np.max(all_highs))
    profile_low = float(np.min(all_lows))

    if profile_high == profile_low:
        return [VolumeLevel(price=profile_high, volume=float(np.sum(data.volumes)), bar_count=len(data.bars))]

    # Create price levels
    price_step = (profile_high - profile_low) / num_levels
    levels: Dict[int, VolumeLevel] = {}

    for i in range(num_levels):
        level_price = profile_low + (i + 0.5) * price_step
        levels[i] = VolumeLevel(price=level_price, volume=0.0, bar_count=0)

    # Distribute each bar's volume across its price range
    for bar in data.bars:
        bar_high = bar.high
        bar_low = bar.low
        bar_volume = bar.volume
        bar_range = bar_high - bar_low

        if bar_range == 0:
            # Single price - assign to nearest level
            level_idx = int((bar_high - profile_low) / price_step)
            level_idx = max(0, min(num_levels - 1, level_idx))
            levels[level_idx].volume += bar_volume
            levels[level_idx].bar_count += 1
        else:
            # Distribute volume across levels within bar range
            for i in range(num_levels):
                level_low = profile_low + i * price_step
                level_high = level_low + price_step

                # Calculate overlap between bar and level
                overlap_low = max(bar_low, level_low)
                overlap_high = min(bar_high, level_high)

                if overlap_high > overlap_low:
                    overlap_pct = (overlap_high - overlap_low) / bar_range
                    levels[i].volume += bar_volume * overlap_pct
                    levels[i].bar_count += 1

    return [levels[i] for i in range(num_levels)]


def calculate_poc(volume_levels: List[VolumeLevel]) -> float:
    """
    Calculate Point of Control (price with highest volume).

    Returns:
        POC price level
    """
    if not volume_levels:
        return 0.0

    max_level = max(volume_levels, key=lambda l: l.volume)
    return max_level.price


def calculate_value_area(
    volume_levels: List[VolumeLevel],
    poc: float,
    value_area_pct: float = 0.70
) -> Tuple[float, float, float]:
    """
    Calculate Value Area High and Low.

    Value Area contains value_area_pct of total volume,
    built outward from POC.

    Returns:
        Tuple of (VAH, VAL, actual_pct_captured)
    """
    if not volume_levels:
        return poc, poc, 0.0

    total_volume = sum(l.volume for l in volume_levels)
    if total_volume == 0:
        return volume_levels[-1].price, volume_levels[0].price, 0.0

    target_volume = total_volume * value_area_pct

    # Find POC index
    sorted_levels = sorted(volume_levels, key=lambda l: l.price)
    poc_idx = 0
    for i, level in enumerate(sorted_levels):
        if abs(level.price - poc) < (sorted_levels[1].price - sorted_levels[0].price if len(sorted_levels) > 1 else 1):
            poc_idx = i
            break

    # Build value area outward from POC
    captured_volume = sorted_levels[poc_idx].volume
    low_idx = poc_idx
    high_idx = poc_idx

    while captured_volume < target_volume:
        # Check which side to expand
        can_go_low = low_idx > 0
        can_go_high = high_idx < len(sorted_levels) - 1

        if not can_go_low and not can_go_high:
            break

        if can_go_low and can_go_high:
            # Expand to side with more volume
            low_vol = sorted_levels[low_idx - 1].volume
            high_vol = sorted_levels[high_idx + 1].volume

            if high_vol >= low_vol:
                high_idx += 1
                captured_volume += high_vol
            else:
                low_idx -= 1
                captured_volume += low_vol
        elif can_go_high:
            high_idx += 1
            captured_volume += sorted_levels[high_idx].volume
        else:
            low_idx -= 1
            captured_volume += sorted_levels[low_idx].volume

    val = sorted_levels[low_idx].price
    vah = sorted_levels[high_idx].price
    actual_pct = captured_volume / total_volume if total_volume > 0 else 0.0

    return vah, val, actual_pct


def determine_shape(
    volume_levels: List[VolumeLevel],
    poc: float,
    vah: float,
    val: float
) -> ProfileShape:
    """
    Determine the profile shape based on volume distribution.

    Returns:
        ProfileShape enum
    """
    if not volume_levels:
        return ProfileShape.NORMAL

    sorted_levels = sorted(volume_levels, key=lambda l: l.price)
    total_levels = len(sorted_levels)

    if total_levels < 3:
        return ProfileShape.NORMAL

    # Divide into thirds
    third = total_levels // 3
    lower_vol = sum(l.volume for l in sorted_levels[:third])
    middle_vol = sum(l.volume for l in sorted_levels[third:2*third])
    upper_vol = sum(l.volume for l in sorted_levels[2*third:])

    total = lower_vol + middle_vol + upper_vol
    if total == 0:
        return ProfileShape.NORMAL

    lower_pct = lower_vol / total
    middle_pct = middle_vol / total
    upper_pct = upper_vol / total

    # Value area width relative to total range
    range_size = sorted_levels[-1].price - sorted_levels[0].price
    va_size = vah - val
    va_pct = va_size / range_size if range_size > 0 else 0.5

    # Determine shape
    if va_pct > 0.8:
        return ProfileShape.WIDE
    elif va_pct < 0.3:
        return ProfileShape.NARROW

    if upper_pct > 0.45:
        return ProfileShape.P_SHAPE  # Volume concentrated at top
    elif lower_pct > 0.45:
        return ProfileShape.B_SHAPE  # Volume concentrated at bottom

    # Check for double distribution
    if middle_pct < 0.2 and upper_pct > 0.3 and lower_pct > 0.3:
        return ProfileShape.D_SHAPE

    return ProfileShape.NORMAL


def determine_market_state(
    current_price: float,
    poc: float,
    vah: float,
    val: float,
    shape: ProfileShape
) -> Tuple[MarketState, PricePosition]:
    """
    Determine current market state and price position.

    Returns:
        Tuple of (MarketState, PricePosition)
    """
    # Determine price position
    va_range = vah - val
    poc_tolerance = va_range * 0.1 if va_range > 0 else 0

    if current_price > vah:
        position = PricePosition.ABOVE_VAH
    elif current_price < val:
        position = PricePosition.BELOW_VAL
    elif abs(current_price - poc) < poc_tolerance:
        position = PricePosition.AT_POC
    else:
        position = PricePosition.IN_VALUE_AREA

    # Determine market state
    if position == PricePosition.ABOVE_VAH:
        state = MarketState.IMBALANCED_UP
    elif position == PricePosition.BELOW_VAL:
        state = MarketState.IMBALANCED_DOWN
    elif position in [PricePosition.IN_VALUE_AREA, PricePosition.AT_POC]:
        state = MarketState.BALANCED
    else:
        state = MarketState.BALANCED

    # Check for breakout conditions
    if shape == ProfileShape.D_SHAPE:
        state = MarketState.BREAKOUT

    return state, position


def analyze_market_profile(
    data: OHLCVData,
    profile_period: str = "session",
    value_area_pct: float = 0.70,
    mode: str = "balanced"
) -> dict:
    """
    Analyze market using Market Profile concepts.

    Args:
        data: OHLCV data
        profile_period: Period for profile ("session", "day", "week")
        value_area_pct: Percentage for value area calculation
        mode: Analysis mode

    Returns:
        AnalysisResult dictionary
    """
    attribution = {
        "theory": "Market Profile",
        "author": "J. Peter Steidlmayer",
        "reference": "CBOT Market Profile Manual",
        "note": "Volume-at-price approximation (true TPO requires tick data)"
    }

    # Check minimum data
    if len(data.bars) < MIN_BARS_REQUIRED:
        return build_no_signal_result(
            reason=f"Insufficient data: {len(data.bars)} bars (need {MIN_BARS_REQUIRED}+)",
            tool_name=TOOL_NAME
        )

    # Check for volume data
    total_volume = float(np.sum(data.volumes))
    if total_volume == 0:
        return build_no_signal_result(
            reason="Volume data required for Market Profile analysis",
            tool_name=TOOL_NAME
        )

    # Build volume profile
    num_levels = 30 if mode == "conservative" else 50 if mode == "balanced" else 70
    volume_levels = build_volume_profile(data, num_levels=num_levels)

    if not volume_levels:
        return build_no_signal_result(
            reason="Could not build volume profile",
            tool_name=TOOL_NAME
        )

    # Calculate key levels
    poc = calculate_poc(volume_levels)
    vah, val, va_pct = calculate_value_area(volume_levels, poc, value_area_pct)

    # Determine shape
    shape = determine_shape(volume_levels, poc, vah, val)

    # Get current price and determine state
    current_price = data.closes[-1]
    market_state, price_position = determine_market_state(
        current_price, poc, vah, val, shape
    )

    # Determine status and bias
    if market_state == MarketState.IMBALANCED_UP:
        status = "bullish"
        bias_description = "Price above value area - bullish imbalance"
    elif market_state == MarketState.IMBALANCED_DOWN:
        status = "bearish"
        bias_description = "Price below value area - bearish imbalance"
    elif market_state == MarketState.BREAKOUT:
        # Determine direction from price vs POC
        if current_price > poc:
            status = "bullish"
            bias_description = "Breakout profile - bullish trend day"
        else:
            status = "bearish"
            bias_description = "Breakout profile - bearish trend day"
    else:
        status = "neutral"
        bias_description = "Price in value area - balanced/rotational"

    # Calculate confidence
    timeframe_weight = get_timeframe_weight(data.timeframe)

    # Quality based on profile clarity
    if shape == ProfileShape.NORMAL:
        q_pattern = 0.85
    elif shape in [ProfileShape.P_SHAPE, ProfileShape.B_SHAPE]:
        q_pattern = 0.80
    elif shape == ProfileShape.D_SHAPE:
        q_pattern = 0.75
    else:
        q_pattern = 0.70

    # Volume quality
    v_conf = min(1.0, va_pct + 0.2) if va_pct > 0.5 else 0.75

    # Mode adjustments
    if mode == "conservative":
        q_pattern *= 0.95
    elif mode == "aggressive":
        q_pattern = min(1.0, q_pattern * 1.05)

    factors = ConfidenceFactors(
        base=MARKET_PROFILE_BASE_CONFIDENCE,
        q_pattern=q_pattern,
        w_time=timeframe_weight,
        v_conf=v_conf,
        m_bonus=1.0
    )

    confidence = calculate_confidence(factors)

    # Build invalidation
    if status == "bullish":
        invalidation = f"Return below VAH ({vah:.2f}) weakens bullish bias; break below POC ({poc:.2f}) invalidates"
    elif status == "bearish":
        invalidation = f"Return above VAL ({val:.2f}) weakens bearish bias; break above POC ({poc:.2f}) invalidates"
    else:
        invalidation = f"Break above VAH ({vah:.2f}) = bullish; break below VAL ({val:.2f}) = bearish"

    # Build LLM summary
    summary = (
        f"Market Profile on {data.timeframe}: {shape.value.replace('_', ' ').title()} profile. "
        f"POC: {poc:.2f}, VAH: {vah:.2f}, VAL: {val:.2f}. "
        f"{bias_description}."
    )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_detailed_profile(
    data: OHLCVData,
    num_levels: int = 50,
    value_area_pct: float = 0.70
) -> dict:
    """
    Get detailed Market Profile data.

    Returns:
        Dictionary with complete profile data
    """
    volume_levels = build_volume_profile(data, num_levels=num_levels)
    poc = calculate_poc(volume_levels)
    vah, val, va_pct = calculate_value_area(volume_levels, poc, value_area_pct)
    shape = determine_shape(volume_levels, poc, vah, val)

    current_price = data.closes[-1]
    market_state, price_position = determine_market_state(
        current_price, poc, vah, val, shape
    )

    # Get high/low
    profile_high = max(l.price for l in volume_levels) if volume_levels else 0
    profile_low = min(l.price for l in volume_levels) if volume_levels else 0

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "profile_high": profile_high,
        "profile_low": profile_low,
        "value_area_pct": va_pct,
        "shape": shape.value,
        "market_state": market_state.value,
        "price_position": price_position.value,
        "current_price": current_price,
        "volume_levels": [
            {"price": l.price, "volume": l.volume, "bar_count": l.bar_count}
            for l in sorted(volume_levels, key=lambda x: x.price, reverse=True)[:20]  # Top 20 levels
        ],
        "total_levels": len(volume_levels),
        "bars_analyzed": len(data.bars)
    }
