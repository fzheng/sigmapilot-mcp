"""
Wyckoff Phase Detector Engine.

Detects Wyckoff market phases (Accumulation, Distribution, Markup, Markdown)
and key Wyckoff events (SC, AR, ST, LPS, SOS, UT, UTAD, Spring, etc.).

Theory Attribution:
    Richard D. Wyckoff (1873-1934)
    "The Wyckoff Method" - Understanding supply/demand through price and volume

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
TOOL_NAME = "Wyckoff Method"
WYCKOFF_BASE_CONFIDENCE = 60
MIN_BARS_REQUIRED = 100  # Need sufficient data to identify phases
SIGNAL_THRESHOLD = 60


class WyckoffPhase(Enum):
    """Wyckoff market phases."""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class WyckoffStage(Enum):
    """Wyckoff phase stages (A-E for Accumulation/Distribution)."""
    PHASE_A = "A"  # Stopping action
    PHASE_B = "B"  # Building cause
    PHASE_C = "C"  # Test (Spring/UTAD)
    PHASE_D = "D"  # Breakout
    PHASE_E = "E"  # Trending
    UNKNOWN = "unknown"


class WyckoffEvent(Enum):
    """Key Wyckoff events."""
    # Accumulation events
    PS = "preliminary_support"
    SC = "selling_climax"
    AR = "automatic_rally"
    ST = "secondary_test"
    SPRING = "spring"
    LPS = "last_point_of_support"
    SOS = "sign_of_strength"
    BU = "backup"

    # Distribution events
    PSY = "preliminary_supply"
    BC = "buying_climax"
    AR_DIST = "automatic_reaction"
    ST_DIST = "secondary_test_distribution"
    UT = "upthrust"
    UTAD = "upthrust_after_distribution"
    LPSY = "last_point_of_supply"
    SOW = "sign_of_weakness"


@dataclass
class WyckoffEventDetection:
    """Represents a detected Wyckoff event."""
    event: WyckoffEvent
    bar_index: int
    timestamp: int
    price: float
    volume_ratio: float  # Relative to average
    confidence: float


@dataclass
class TradingRange:
    """Represents a trading range for Wyckoff analysis."""
    start_idx: int
    end_idx: int
    high: float
    low: float
    range_bars: int
    avg_volume: float


def detect_trading_range(
    data: OHLCVData,
    lookback: int = 100,
    range_threshold: float = 0.15
) -> Optional[TradingRange]:
    """
    Detect if price is in a trading range (consolidation).

    Args:
        data: OHLCV data
        lookback: Number of bars to analyze
        range_threshold: Maximum range as percentage of price for consolidation

    Returns:
        TradingRange if detected, None otherwise
    """
    if len(data.bars) < lookback:
        lookback = len(data.bars)

    recent_bars = data.bars[-lookback:]
    highs = np.array([b.high for b in recent_bars])
    lows = np.array([b.low for b in recent_bars])
    volumes = np.array([b.volume for b in recent_bars])

    range_high = np.max(highs)
    range_low = np.min(lows)
    range_size = (range_high - range_low) / range_low

    # Check if range is consolidation (not too wide)
    if range_size > range_threshold:
        return None

    return TradingRange(
        start_idx=len(data.bars) - lookback,
        end_idx=len(data.bars) - 1,
        high=range_high,
        low=range_low,
        range_bars=lookback,
        avg_volume=float(np.mean(volumes))
    )


def detect_climax_volume(
    volumes: np.ndarray,
    idx: int,
    lookback: int = 20,
    threshold: float = 2.0
) -> bool:
    """Detect if bar has climactic volume (significantly above average)."""
    if idx < lookback:
        return False

    avg_vol = np.mean(volumes[idx-lookback:idx])
    if avg_vol == 0:
        return False

    return volumes[idx] >= avg_vol * threshold


def detect_volume_dry_up(
    volumes: np.ndarray,
    idx: int,
    lookback: int = 20,
    threshold: float = 0.5
) -> bool:
    """Detect if volume is drying up (significantly below average)."""
    if idx < lookback:
        return False

    avg_vol = np.mean(volumes[idx-lookback:idx])
    if avg_vol == 0:
        return False

    return volumes[idx] <= avg_vol * threshold


def find_swing_lows(
    lows: np.ndarray,
    window: int = 5
) -> List[int]:
    """Find swing low indices."""
    swing_lows = []
    for i in range(window, len(lows) - window):
        if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, window+1)):
            swing_lows.append(i)
    return swing_lows


def find_swing_highs(
    highs: np.ndarray,
    window: int = 5
) -> List[int]:
    """Find swing high indices."""
    swing_highs = []
    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, window+1)):
            swing_highs.append(i)
    return swing_highs


def detect_wyckoff_events(
    data: OHLCVData,
    trading_range: Optional[TradingRange] = None
) -> List[WyckoffEventDetection]:
    """
    Detect Wyckoff events within the data.

    Args:
        data: OHLCV data
        trading_range: Optional pre-detected trading range

    Returns:
        List of detected Wyckoff events
    """
    events: List[WyckoffEventDetection] = []

    if len(data.bars) < MIN_BARS_REQUIRED:
        return events

    closes = data.closes
    highs = data.highs
    lows = data.lows
    volumes = data.volumes

    avg_volume = np.mean(volumes[-50:]) if len(volumes) >= 50 else np.mean(volumes)

    swing_lows = find_swing_lows(lows)
    swing_highs = find_swing_highs(highs)

    # Detect Selling Climax (SC) - Sharp drop with high volume
    for i in range(20, len(data.bars) - 5):
        # Check for sharp price drop
        if i >= 5:
            price_change = (closes[i] - closes[i-5]) / closes[i-5]
            if price_change < -0.05:  # 5%+ drop
                if detect_climax_volume(volumes, i):
                    # Check if followed by bounce
                    if i + 3 < len(closes) and closes[i+3] > closes[i]:
                        vol_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                        events.append(WyckoffEventDetection(
                            event=WyckoffEvent.SC,
                            bar_index=i,
                            timestamp=data.bars[i].timestamp,
                            price=closes[i],
                            volume_ratio=vol_ratio,
                            confidence=0.7
                        ))

    # Detect Buying Climax (BC) - Sharp rise with high volume
    for i in range(20, len(data.bars) - 5):
        if i >= 5:
            price_change = (closes[i] - closes[i-5]) / closes[i-5]
            if price_change > 0.05:  # 5%+ rise
                if detect_climax_volume(volumes, i):
                    # Check if followed by pullback
                    if i + 3 < len(closes) and closes[i+3] < closes[i]:
                        vol_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                        events.append(WyckoffEventDetection(
                            event=WyckoffEvent.BC,
                            bar_index=i,
                            timestamp=data.bars[i].timestamp,
                            price=closes[i],
                            volume_ratio=vol_ratio,
                            confidence=0.7
                        ))

    # Detect Spring - False breakdown below support with quick recovery
    if trading_range:
        range_low = trading_range.low
        for idx in swing_lows:
            if lows[idx] < range_low * 0.99:  # Broke below support
                # Check for quick recovery
                if idx + 3 < len(closes):
                    if closes[idx + 3] > range_low:  # Recovered
                        # Volume should not be climactic during spring
                        if not detect_climax_volume(volumes, idx):
                            vol_ratio = volumes[idx] / avg_volume if avg_volume > 0 else 1.0
                            events.append(WyckoffEventDetection(
                                event=WyckoffEvent.SPRING,
                                bar_index=idx,
                                timestamp=data.bars[idx].timestamp,
                                price=lows[idx],
                                volume_ratio=vol_ratio,
                                confidence=0.75
                            ))

    # Detect Upthrust (UT) - False breakout above resistance with quick reversal
    if trading_range:
        range_high = trading_range.high
        for idx in swing_highs:
            if highs[idx] > range_high * 1.01:  # Broke above resistance
                # Check for quick reversal
                if idx + 3 < len(closes):
                    if closes[idx + 3] < range_high:  # Reversed
                        vol_ratio = volumes[idx] / avg_volume if avg_volume > 0 else 1.0
                        events.append(WyckoffEventDetection(
                            event=WyckoffEvent.UT,
                            bar_index=idx,
                            timestamp=data.bars[idx].timestamp,
                            price=highs[idx],
                            volume_ratio=vol_ratio,
                            confidence=0.75
                        ))

    # Detect Sign of Strength (SOS) - Strong move up with expanding volume
    for i in range(30, len(data.bars) - 3):
        if i >= 5:
            price_change = (closes[i] - closes[i-5]) / closes[i-5]
            if price_change > 0.03:  # 3%+ rise
                # Volume should be expanding
                recent_vol = np.mean(volumes[i-3:i+1])
                prior_vol = np.mean(volumes[i-8:i-3])
                if prior_vol > 0 and recent_vol > prior_vol * 1.3:
                    vol_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                    events.append(WyckoffEventDetection(
                        event=WyckoffEvent.SOS,
                        bar_index=i,
                        timestamp=data.bars[i].timestamp,
                        price=closes[i],
                        volume_ratio=vol_ratio,
                        confidence=0.7
                    ))

    # Detect Sign of Weakness (SOW) - Strong move down with expanding volume
    for i in range(30, len(data.bars) - 3):
        if i >= 5:
            price_change = (closes[i] - closes[i-5]) / closes[i-5]
            if price_change < -0.03:  # 3%+ drop
                # Volume should be expanding
                recent_vol = np.mean(volumes[i-3:i+1])
                prior_vol = np.mean(volumes[i-8:i-3])
                if prior_vol > 0 and recent_vol > prior_vol * 1.3:
                    vol_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                    events.append(WyckoffEventDetection(
                        event=WyckoffEvent.SOW,
                        bar_index=i,
                        timestamp=data.bars[i].timestamp,
                        price=closes[i],
                        volume_ratio=vol_ratio,
                        confidence=0.7
                    ))

    # Detect Last Point of Support (LPS) - Higher low on declining volume
    for i in range(len(swing_lows) - 1):
        idx1, idx2 = swing_lows[i], swing_lows[i + 1]
        if idx2 < len(lows):
            # Check for higher low
            if lows[idx2] > lows[idx1]:
                # Check for declining volume
                if detect_volume_dry_up(volumes, idx2):
                    vol_ratio = volumes[idx2] / avg_volume if avg_volume > 0 else 1.0
                    events.append(WyckoffEventDetection(
                        event=WyckoffEvent.LPS,
                        bar_index=idx2,
                        timestamp=data.bars[idx2].timestamp,
                        price=lows[idx2],
                        volume_ratio=vol_ratio,
                        confidence=0.65
                    ))

    return events


def determine_phase(
    data: OHLCVData,
    events: List[WyckoffEventDetection],
    trading_range: Optional[TradingRange]
) -> Tuple[WyckoffPhase, WyckoffStage, float]:
    """
    Determine the current Wyckoff phase based on detected events.

    Returns:
        Tuple of (phase, stage, quality score)
    """
    if len(data.bars) < MIN_BARS_REQUIRED:
        return WyckoffPhase.UNKNOWN, WyckoffStage.UNKNOWN, 0.5

    closes = data.closes

    # Calculate overall trend direction
    sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
    sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
    current_price = closes[-1]

    # Score accumulators
    accumulation_score = 0.0
    distribution_score = 0.0

    # Count bullish vs bearish events
    bullish_events = [WyckoffEvent.SC, WyckoffEvent.SPRING, WyckoffEvent.LPS, WyckoffEvent.SOS]
    bearish_events = [WyckoffEvent.BC, WyckoffEvent.UT, WyckoffEvent.UTAD, WyckoffEvent.SOW, WyckoffEvent.LPSY]

    recent_events = [e for e in events if e.bar_index > len(data.bars) - 50]

    for event in recent_events:
        if event.event in bullish_events:
            accumulation_score += event.confidence
        elif event.event in bearish_events:
            distribution_score += event.confidence

    # Determine phase
    phase = WyckoffPhase.UNKNOWN
    stage = WyckoffStage.UNKNOWN
    quality = 0.5

    if trading_range:
        # In a trading range - accumulation or distribution
        if accumulation_score > distribution_score * 1.3:
            phase = WyckoffPhase.ACCUMULATION
            quality = min(1.0, 0.5 + accumulation_score * 0.1)

            # Determine stage
            has_spring = any(e.event == WyckoffEvent.SPRING for e in recent_events)
            has_sos = any(e.event == WyckoffEvent.SOS for e in recent_events)
            has_sc = any(e.event == WyckoffEvent.SC for e in events)

            if has_sos and current_price > trading_range.high:
                stage = WyckoffStage.PHASE_E
            elif has_sos:
                stage = WyckoffStage.PHASE_D
            elif has_spring:
                stage = WyckoffStage.PHASE_C
            elif has_sc:
                stage = WyckoffStage.PHASE_B
            else:
                stage = WyckoffStage.PHASE_A

        elif distribution_score > accumulation_score * 1.3:
            phase = WyckoffPhase.DISTRIBUTION
            quality = min(1.0, 0.5 + distribution_score * 0.1)

            # Determine stage
            has_ut = any(e.event in [WyckoffEvent.UT, WyckoffEvent.UTAD] for e in recent_events)
            has_sow = any(e.event == WyckoffEvent.SOW for e in recent_events)
            has_bc = any(e.event == WyckoffEvent.BC for e in events)

            if has_sow and current_price < trading_range.low:
                stage = WyckoffStage.PHASE_E
            elif has_sow:
                stage = WyckoffStage.PHASE_D
            elif has_ut:
                stage = WyckoffStage.PHASE_C
            elif has_bc:
                stage = WyckoffStage.PHASE_B
            else:
                stage = WyckoffStage.PHASE_A
    else:
        # Not in range - markup or markdown
        if current_price > sma_20 > sma_50:
            phase = WyckoffPhase.MARKUP
            quality = 0.7
            stage = WyckoffStage.PHASE_E
        elif current_price < sma_20 < sma_50:
            phase = WyckoffPhase.MARKDOWN
            quality = 0.7
            stage = WyckoffStage.PHASE_E

    return phase, stage, quality


def analyze_wyckoff(
    data: OHLCVData,
    range_lookback: int = 200,
    mode: str = "balanced"
) -> dict:
    """
    Analyze market using Wyckoff Method.

    Args:
        data: OHLCV data
        range_lookback: Number of bars to look for trading range
        mode: Analysis mode - "conservative", "balanced", or "aggressive"

    Returns:
        AnalysisResult dictionary
    """
    attribution = {
        "theory": "Wyckoff Method",
        "author": "Richard D. Wyckoff",
        "reference": "The Wyckoff Method - Stock Market Science and Technique"
    }

    # Check minimum data requirements
    if len(data.bars) < MIN_BARS_REQUIRED:
        return build_no_signal_result(
            reason=f"Insufficient data: {len(data.bars)} bars (need {MIN_BARS_REQUIRED}+)",
            tool_name=TOOL_NAME
        )

    # Check for valid volume data
    volumes = data.volumes
    if np.sum(volumes) == 0:
        return build_no_signal_result(
            reason="Volume data required for Wyckoff analysis",
            tool_name=TOOL_NAME
        )

    # Detect trading range
    trading_range = detect_trading_range(data, lookback=min(range_lookback, len(data.bars)))

    # Detect Wyckoff events
    events = detect_wyckoff_events(data, trading_range)

    # Determine phase and stage
    phase, stage, quality = determine_phase(data, events, trading_range)

    if phase == WyckoffPhase.UNKNOWN:
        return build_no_signal_result(
            reason="Unable to determine Wyckoff phase - insufficient structural evidence",
            tool_name=TOOL_NAME
        )

    # Determine status based on phase
    if phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]:
        status = "bullish"
    elif phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]:
        status = "bearish"
    else:
        status = "neutral"

    # Calculate confidence
    timeframe_weight = get_timeframe_weight(data.timeframe)

    # Volume confidence based on event detection quality
    event_confidences = [e.confidence for e in events] if events else [0.85]
    v_conf = min(1.0, np.mean(event_confidences) + 0.15)

    # Mode adjustments
    mode_multiplier = 1.0
    if mode == "conservative":
        mode_multiplier = 0.9
    elif mode == "aggressive":
        mode_multiplier = 1.05

    factors = ConfidenceFactors(
        base=WYCKOFF_BASE_CONFIDENCE,
        q_pattern=quality * mode_multiplier,
        w_time=timeframe_weight,
        v_conf=v_conf,
        m_bonus=1.0
    )

    confidence = calculate_confidence(factors)

    # Build invalidation
    if phase == WyckoffPhase.ACCUMULATION:
        if trading_range:
            invalidation = f"Break below {trading_range.low:.2f} (range support) invalidates accumulation"
        else:
            invalidation = "Break below recent swing low invalidates bullish phase"
    elif phase == WyckoffPhase.DISTRIBUTION:
        if trading_range:
            invalidation = f"Break above {trading_range.high:.2f} (range resistance) invalidates distribution"
        else:
            invalidation = "Break above recent swing high invalidates bearish phase"
    elif phase == WyckoffPhase.MARKUP:
        sma_50 = float(np.mean(data.closes[-50:])) if len(data.closes) >= 50 else data.closes[-1]
        invalidation = f"Break below {sma_50:.2f} (50-period MA) signals trend weakness"
    else:  # MARKDOWN
        sma_50 = float(np.mean(data.closes[-50:])) if len(data.closes) >= 50 else data.closes[-1]
        invalidation = f"Break above {sma_50:.2f} (50-period MA) signals trend weakness"

    # Build LLM summary
    event_names = [e.event.value.replace("_", " ").title() for e in events[-3:]]
    event_str = f" Recent events: {', '.join(event_names)}." if event_names else ""

    summary = (
        f"Wyckoff {phase.value.title()} Phase {stage.value} detected on {data.timeframe}. "
        f"{'Trading range identified.' if trading_range else 'Trending phase.'}{event_str}"
    )

    return build_analysis_result(
        status=status,
        confidence=confidence,
        tool_name=TOOL_NAME,
        llm_summary=summary,
        invalidation=invalidation
    )


def get_detailed_wyckoff(
    data: OHLCVData,
    range_lookback: int = 200
) -> dict:
    """
    Get detailed Wyckoff analysis with all detected events.

    Args:
        data: OHLCV data
        range_lookback: Number of bars to look for trading range

    Returns:
        Dictionary with detailed phase, stage, events, and trading range info
    """
    trading_range = detect_trading_range(data, lookback=min(range_lookback, len(data.bars)))
    events = detect_wyckoff_events(data, trading_range)
    phase, stage, quality = determine_phase(data, events, trading_range)

    return {
        "phase": phase.value,
        "stage": stage.value,
        "quality": quality,
        "trading_range": {
            "detected": trading_range is not None,
            "high": trading_range.high if trading_range else None,
            "low": trading_range.low if trading_range else None,
            "bars": trading_range.range_bars if trading_range else None
        },
        "events": [
            {
                "event": e.event.value,
                "bar_index": e.bar_index,
                "timestamp": e.timestamp,
                "price": e.price,
                "volume_ratio": e.volume_ratio,
                "confidence": e.confidence
            }
            for e in events
        ],
        "event_count": len(events)
    }
