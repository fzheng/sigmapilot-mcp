"""
Volume Spread Analysis (VSA) Engine.

This engine identifies smart-money bar signatures using Volume Spread Analysis,
a methodology developed by Tom Williams based on Richard Wyckoff's work.

VSA analyzes the relationship between:
- Volume (effort)
- Price spread (range)
- Close position within the bar

Key Patterns:
- Stopping Volume: High volume, narrow spread, close near high (bullish)
- Climactic Action: Extremely high volume, wide spread (potential reversal)
- No Demand: Low volume, narrow spread, up bar (bearish weakness)
- No Supply: Low volume, narrow spread, down bar (bullish strength)
- Test: Low volume retest of lows (bullish if successful)
- Upthrust: High volume, close near low after rally (bearish)

References:
- Tom Williams, "Master the Markets"
- https://en.wikipedia.org/wiki/Volume_spread_analysis
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

# Base confidence for VSA (from CLAUDE.md spec)
VSA_BASE_CONFIDENCE = 65

# Default lookback for average calculations
DEFAULT_LOOKBACK = 120

# Minimum bars needed for analysis
MIN_BARS_REQUIRED = 50

# Volume thresholds (relative to average)
ULTRA_HIGH_VOLUME = 2.0    # 200% of average
HIGH_VOLUME = 1.5          # 150% of average
ABOVE_AVERAGE_VOLUME = 1.2 # 120% of average
LOW_VOLUME = 0.7           # 70% of average
ULTRA_LOW_VOLUME = 0.5     # 50% of average

# Spread thresholds (relative to ATR)
WIDE_SPREAD = 1.5          # 150% of ATR
NARROW_SPREAD = 0.5        # 50% of ATR

# Close position thresholds
CLOSE_NEAR_HIGH = 0.7      # Close in upper 30%
CLOSE_NEAR_LOW = 0.3       # Close in lower 30%
CLOSE_MIDDLE = 0.5         # Middle of range

TOOL_NAME = "vsa_analyzer"


class VSASignalType(str, Enum):
    """Types of VSA signals."""
    STOPPING_VOLUME = "stopping_volume"
    CLIMACTIC_BUYING = "climactic_buying"
    CLIMACTIC_SELLING = "climactic_selling"
    NO_DEMAND = "no_demand"
    NO_SUPPLY = "no_supply"
    TEST = "test"
    UPTHRUST = "upthrust"
    SPRING = "spring"
    EFFORT_NO_RESULT = "effort_no_result"
    SIGN_OF_STRENGTH = "sign_of_strength"
    SIGN_OF_WEAKNESS = "sign_of_weakness"


class BackgroundBias(str, Enum):
    """Overall market background bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VSASignal:
    """A detected VSA signal."""
    signal_type: VSASignalType
    index: int
    timestamp: int
    confidence: float
    description: str
    is_bullish: bool
    volume_ratio: float
    spread_ratio: float
    close_position: float


@dataclass
class BarAnalysis:
    """Analysis of a single bar's VSA characteristics."""
    index: int
    is_up_bar: bool
    spread: float
    spread_ratio: float  # Relative to ATR
    volume: float
    volume_ratio: float  # Relative to average
    close_position: float  # 0=low, 1=high
    is_wide_spread: bool
    is_narrow_spread: bool
    is_high_volume: bool
    is_low_volume: bool


# =============================================================================
# Bar Analysis Functions
# =============================================================================

def calculate_close_position(open_price: float, high: float, low: float, close: float) -> float:
    """
    Calculate where the close is within the bar's range.

    Returns:
        0.0 = close at low
        0.5 = close in middle
        1.0 = close at high
    """
    bar_range = high - low
    if bar_range == 0:
        return 0.5
    return (close - low) / bar_range


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range."""
    n = len(highs)
    tr = np.zeros(n)

    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(n)
    atr[:period] = np.nan

    # Simple moving average of TR
    for i in range(period, n):
        atr[i] = np.mean(tr[i-period+1:i+1])

    return atr


def analyze_bar(
    idx: int,
    data: OHLCVData,
    avg_volume: float,
    atr: float
) -> BarAnalysis:
    """Analyze a single bar for VSA characteristics."""
    open_price = float(data.opens[idx])
    high = float(data.highs[idx])
    low = float(data.lows[idx])
    close = float(data.closes[idx])
    volume = float(data.volumes[idx])

    spread = high - low
    spread_ratio = spread / atr if atr > 0 else 1.0
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    close_position = calculate_close_position(open_price, high, low, close)

    return BarAnalysis(
        index=idx,
        is_up_bar=close > open_price,
        spread=spread,
        spread_ratio=spread_ratio,
        volume=volume,
        volume_ratio=volume_ratio,
        close_position=close_position,
        is_wide_spread=spread_ratio >= WIDE_SPREAD,
        is_narrow_spread=spread_ratio <= NARROW_SPREAD,
        is_high_volume=volume_ratio >= HIGH_VOLUME,
        is_low_volume=volume_ratio <= LOW_VOLUME
    )


# =============================================================================
# VSA Pattern Detection
# =============================================================================

def detect_stopping_volume(bar: BarAnalysis, prev_bars: List[BarAnalysis]) -> Optional[VSASignal]:
    """
    Detect Stopping Volume pattern.

    Characteristics:
    - High volume
    - Narrow spread
    - Close near high (for bullish)
    - After a down move
    """
    if not bar.is_high_volume:
        return None

    if bar.is_narrow_spread and bar.close_position >= CLOSE_NEAR_HIGH:
        # Check if preceded by down bars
        down_count = sum(1 for b in prev_bars[-3:] if not b.is_up_bar)
        if down_count >= 2:
            return VSASignal(
                signal_type=VSASignalType.STOPPING_VOLUME,
                index=bar.index,
                timestamp=0,  # Will be set later
                confidence=0.8 if bar.volume_ratio >= ULTRA_HIGH_VOLUME else 0.7,
                description="High volume with narrow spread closing near high after decline - potential buying absorption",
                is_bullish=True,
                volume_ratio=bar.volume_ratio,
                spread_ratio=bar.spread_ratio,
                close_position=bar.close_position
            )

    return None


def detect_no_demand(bar: BarAnalysis, prev_bars: List[BarAnalysis]) -> Optional[VSASignal]:
    """
    Detect No Demand pattern.

    Characteristics:
    - Low volume
    - Narrow spread
    - Up bar
    - Close in lower half
    """
    if not bar.is_low_volume:
        return None

    if bar.is_up_bar and bar.is_narrow_spread and bar.close_position < CLOSE_MIDDLE:
        return VSASignal(
            signal_type=VSASignalType.NO_DEMAND,
            index=bar.index,
            timestamp=0,
            confidence=0.7,
            description="Low volume up bar with narrow spread and weak close - lack of buying interest",
            is_bullish=False,
            volume_ratio=bar.volume_ratio,
            spread_ratio=bar.spread_ratio,
            close_position=bar.close_position
        )

    return None


def detect_no_supply(bar: BarAnalysis, prev_bars: List[BarAnalysis]) -> Optional[VSASignal]:
    """
    Detect No Supply pattern.

    Characteristics:
    - Low volume
    - Narrow spread
    - Down bar
    - Close in upper half
    """
    if not bar.is_low_volume:
        return None

    if not bar.is_up_bar and bar.is_narrow_spread and bar.close_position > CLOSE_MIDDLE:
        return VSASignal(
            signal_type=VSASignalType.NO_SUPPLY,
            index=bar.index,
            timestamp=0,
            confidence=0.7,
            description="Low volume down bar with narrow spread and strong close - lack of selling pressure",
            is_bullish=True,
            volume_ratio=bar.volume_ratio,
            spread_ratio=bar.spread_ratio,
            close_position=bar.close_position
        )

    return None


def detect_upthrust(bar: BarAnalysis, prev_bars: List[BarAnalysis]) -> Optional[VSASignal]:
    """
    Detect Upthrust pattern.

    Characteristics:
    - High volume
    - Wide spread or narrow
    - Close near low
    - Up bar that closes weak (or down bar)
    - After up move
    """
    if not bar.is_high_volume:
        return None

    if bar.close_position <= CLOSE_NEAR_LOW:
        # Check if preceded by up bars
        up_count = sum(1 for b in prev_bars[-3:] if b.is_up_bar)
        if up_count >= 2:
            return VSASignal(
                signal_type=VSASignalType.UPTHRUST,
                index=bar.index,
                timestamp=0,
                confidence=0.8 if bar.is_wide_spread else 0.7,
                description="High volume bar closing near low after rally - potential distribution/reversal",
                is_bullish=False,
                volume_ratio=bar.volume_ratio,
                spread_ratio=bar.spread_ratio,
                close_position=bar.close_position
            )

    return None


def detect_test(bar: BarAnalysis, prev_bars: List[BarAnalysis], recent_low: float, current_low: float) -> Optional[VSASignal]:
    """
    Detect Test pattern.

    Characteristics:
    - Low volume
    - Price dips to test previous low
    - Closes near high (successful test)
    """
    if not bar.is_low_volume:
        return None

    # Check if we're near a recent low (within 2%)
    low_tolerance = recent_low * 0.02
    if abs(current_low - recent_low) <= low_tolerance:
        if bar.close_position >= CLOSE_NEAR_HIGH:
            return VSASignal(
                signal_type=VSASignalType.TEST,
                index=bar.index,
                timestamp=0,
                confidence=0.75,
                description="Low volume test of support with strong close - bullish if successful",
                is_bullish=True,
                volume_ratio=bar.volume_ratio,
                spread_ratio=bar.spread_ratio,
                close_position=bar.close_position
            )

    return None


def detect_climactic_action(bar: BarAnalysis, is_after_trend: bool, trend_is_up: bool) -> Optional[VSASignal]:
    """
    Detect Climactic Buying or Selling.

    Characteristics:
    - Ultra high volume
    - Wide spread
    - After extended move
    """
    if bar.volume_ratio < ULTRA_HIGH_VOLUME:
        return None

    if bar.is_wide_spread and is_after_trend:
        if trend_is_up and bar.is_up_bar:
            return VSASignal(
                signal_type=VSASignalType.CLIMACTIC_BUYING,
                index=bar.index,
                timestamp=0,
                confidence=0.8,
                description="Climactic buying volume with wide spread - potential exhaustion/top",
                is_bullish=False,  # Potential reversal
                volume_ratio=bar.volume_ratio,
                spread_ratio=bar.spread_ratio,
                close_position=bar.close_position
            )
        elif not trend_is_up and not bar.is_up_bar:
            return VSASignal(
                signal_type=VSASignalType.CLIMACTIC_SELLING,
                index=bar.index,
                timestamp=0,
                confidence=0.8,
                description="Climactic selling volume with wide spread - potential exhaustion/bottom",
                is_bullish=True,  # Potential reversal
                volume_ratio=bar.volume_ratio,
                spread_ratio=bar.spread_ratio,
                close_position=bar.close_position
            )

    return None


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_vsa(
    data: OHLCVData,
    lookback: int = DEFAULT_LOOKBACK,
    mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
) -> AnalysisResult:
    """
    Perform Volume Spread Analysis on OHLCV data.

    Args:
        data: OHLCVData object with price history
        lookback: Number of bars for average calculations
        mode: Analysis mode affecting confidence

    Returns:
        AnalysisResult with VSA analysis
    """
    # Validate minimum data
    if len(data) < MIN_BARS_REQUIRED:
        return build_insufficient_data_result(
            tool_name=TOOL_NAME,
            required_bars=MIN_BARS_REQUIRED,
            available_bars=len(data)
        )

    # Check volume data availability
    if not data.has_volume:
        return build_no_signal_result(
            reason="No volume data available for VSA analysis. VSA requires reliable volume data.",
            tool_name=TOOL_NAME
        )

    # Calculate averages
    volumes = data.volumes
    avg_volume = np.mean(volumes[-lookback:]) if len(volumes) >= lookback else np.mean(volumes)

    atr_values = calculate_atr(data.highs, data.lows, data.closes)
    current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else np.nanmean(atr_values)

    # Analyze recent bars
    analysis_range = min(20, len(data))  # Look at last 20 bars for signals
    bar_analyses: List[BarAnalysis] = []

    for i in range(len(data) - analysis_range, len(data)):
        bar_analyses.append(analyze_bar(i, data, avg_volume, current_atr))

    # Detect signals
    signals: List[VSASignal] = []
    timestamps = data.timestamps

    # Determine recent trend
    recent_closes = data.closes[-20:]
    trend_is_up = recent_closes[-1] > recent_closes[0]
    is_after_trend = abs(recent_closes[-1] - recent_closes[0]) / recent_closes[0] > 0.05

    # Find recent swing low for test detection
    recent_lows = data.lows[-40:]
    swing_low = np.min(recent_lows)

    for i, bar in enumerate(bar_analyses):
        prev_bars = bar_analyses[max(0, i-5):i]
        bar_timestamp = int(timestamps[bar.index])

        # Check each pattern
        signal = detect_stopping_volume(bar, prev_bars)
        if signal:
            signal.timestamp = bar_timestamp
            signals.append(signal)
            continue

        signal = detect_no_demand(bar, prev_bars)
        if signal:
            signal.timestamp = bar_timestamp
            signals.append(signal)
            continue

        signal = detect_no_supply(bar, prev_bars)
        if signal:
            signal.timestamp = bar_timestamp
            signals.append(signal)
            continue

        signal = detect_upthrust(bar, prev_bars)
        if signal:
            signal.timestamp = bar_timestamp
            signals.append(signal)
            continue

        signal = detect_test(bar, prev_bars, swing_low, float(data.lows[bar.index]))
        if signal:
            signal.timestamp = bar_timestamp
            signals.append(signal)
            continue

        signal = detect_climactic_action(bar, is_after_trend, trend_is_up)
        if signal:
            signal.timestamp = bar_timestamp
            signals.append(signal)

    # Determine background bias
    bullish_signals = sum(1 for s in signals if s.is_bullish)
    bearish_signals = sum(1 for s in signals if not s.is_bullish)

    if bullish_signals > bearish_signals + 1:
        background_bias = BackgroundBias.BULLISH
        status = "bullish"
    elif bearish_signals > bullish_signals + 1:
        background_bias = BackgroundBias.BEARISH
        status = "bearish"
    else:
        background_bias = BackgroundBias.NEUTRAL
        status = "neutral"

    # Calculate pattern quality based on signal count and recency
    if signals:
        recent_signals = [s for s in signals if s.index >= len(data) - 5]
        if recent_signals:
            avg_confidence = np.mean([s.confidence for s in recent_signals])
            pattern_quality = avg_confidence
        else:
            pattern_quality = 0.7
    else:
        pattern_quality = 0.5

    # Volume confidence based on data quality
    volume_std = np.std(volumes[-lookback:]) / avg_volume if avg_volume > 0 else 0
    v_conf = 0.9 if volume_std > 0.3 else 0.85  # Higher variance = more useful for VSA

    # Apply mode adjustments
    mode_multiplier = {
        "conservative": 0.9,
        "balanced": 1.0,
        "aggressive": 1.1
    }.get(mode, 1.0)

    # Calculate confidence
    factors = ConfidenceFactors.from_timeframe(
        base=VSA_BASE_CONFIDENCE * mode_multiplier,
        timeframe=data.timeframe,
        q_pattern=pattern_quality,
        v_conf=v_conf
    )
    confidence = calculate_confidence(factors)

    # Build invalidation
    if status == "bullish":
        invalidation = "Break below recent swing low with high volume would invalidate bullish VSA signals"
    elif status == "bearish":
        invalidation = "Break above recent swing high with high volume would invalidate bearish VSA signals"
    else:
        invalidation = "No clear VSA bias to invalidate"

    # Build LLM summary
    if signals:
        recent_signal_names = [s.signal_type.value.replace("_", " ").title() for s in signals[-3:]]
        summary = (
            f"VSA on {data.timeframe}: {len(signals)} signal(s) detected. "
            f"Recent: {', '.join(recent_signal_names)}. "
            f"Background bias: {background_bias.value}."
        )
    else:
        summary = (
            f"VSA on {data.timeframe}: No significant signals detected. "
            f"Volume patterns inconclusive for directional bias."
        )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_detailed_vsa(
    data: OHLCVData,
    lookback: int = DEFAULT_LOOKBACK
) -> Dict[str, Any]:
    """
    Get detailed VSA analysis with all signals.

    Returns a dictionary with all detected signals and analysis details.
    """
    if len(data) < MIN_BARS_REQUIRED:
        return {
            "error": "Insufficient data",
            "required_bars": MIN_BARS_REQUIRED,
            "available_bars": len(data)
        }

    if not data.has_volume:
        return {"error": "No volume data available"}

    # Run analysis and collect signals
    volumes = data.volumes
    avg_volume = np.mean(volumes[-lookback:])

    atr_values = calculate_atr(data.highs, data.lows, data.closes)
    current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else np.nanmean(atr_values)

    # Last bar analysis
    last_bar = analyze_bar(len(data) - 1, data, avg_volume, current_atr)

    return {
        "symbol": data.symbol,
        "timeframe": data.timeframe,
        "avg_volume": float(avg_volume),
        "current_atr": float(current_atr),
        "last_bar": {
            "is_up_bar": last_bar.is_up_bar,
            "spread_ratio": last_bar.spread_ratio,
            "volume_ratio": last_bar.volume_ratio,
            "close_position": last_bar.close_position,
            "is_high_volume": last_bar.is_high_volume,
            "is_low_volume": last_bar.is_low_volume,
            "is_wide_spread": last_bar.is_wide_spread,
            "is_narrow_spread": last_bar.is_narrow_spread
        }
    }
