"""
Confidence Calculation Module for SigmaPilot MCP.

This module implements the confidence formula from the CLAUDE.md specification:

    Confidence = Base × Q_pattern × W_time × V_conf × M_bonus

Where:
- Base: Raw confidence from pattern detection (0-100)
- Q_pattern: Pattern quality factor (0.5-1.0)
- W_time: Timeframe weight factor (0.7-1.0)
- V_conf: Volume confidence factor (0.7-1.0)
- M_bonus: Multi-engine agreement bonus (1.0-1.15)

Key Features:
- Confidence factor dataclass for clean parameter passing
- No Signal Protocol implementation (confidence < 60 = neutral)
- Clamping to ensure valid 0-100 range
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Literal

from .timeframes import get_timeframe_weight


# =============================================================================
# Constants
# =============================================================================

# Threshold for generating a signal (bullish/bearish)
# Below this, status must be "neutral" (No Signal Protocol)
SIGNAL_THRESHOLD: int = 60

# Confidence bounds
MIN_CONFIDENCE: int = 0
MAX_CONFIDENCE: int = 100

# Default factor values (multipliers)
DEFAULT_Q_PATTERN: float = 0.85  # Pattern quality (neutral assumption)
DEFAULT_W_TIME: float = 0.85    # Will be overridden by timeframe
DEFAULT_V_CONF: float = 0.85    # Volume confidence (neutral)
DEFAULT_M_BONUS: float = 1.0    # No multi-engine bonus by default

# Factor bounds
MIN_Q_PATTERN: float = 0.5
MAX_Q_PATTERN: float = 1.0
MIN_V_CONF: float = 0.7
MAX_V_CONF: float = 1.0
MIN_M_BONUS: float = 1.0
MAX_M_BONUS: float = 1.15


# =============================================================================
# Confidence Factors Dataclass
# =============================================================================

@dataclass
class ConfidenceFactors:
    """
    Container for all factors used in confidence calculation.

    The confidence formula is:
        Confidence = base × q_pattern × w_time × v_conf × m_bonus

    Attributes:
        base: Raw confidence from pattern detection (0-100)
        q_pattern: Pattern quality factor (0.5-1.0)
                   - 1.0 = Perfect pattern
                   - 0.85 = Typical pattern
                   - 0.5 = Marginal pattern
        w_time: Timeframe weight (0.7-1.0)
                - 1.0 = Monthly
                - 0.7 = 5-minute
        v_conf: Volume confidence (0.7-1.0)
                - 1.0 = Volume confirms signal
                - 0.85 = Volume neutral
                - 0.7 = Volume uncertain/missing
        m_bonus: Multi-engine agreement bonus (1.0-1.15)
                 - 1.0 = Single engine
                 - 1.05 = 2 engines agree
                 - 1.10 = 3 engines agree
                 - 1.15 = 4+ engines agree

    Example:
        >>> factors = ConfidenceFactors(base=75, q_pattern=0.9, w_time=0.85)
        >>> calculate_confidence(factors)
        54
    """
    base: float = 0.0
    q_pattern: float = DEFAULT_Q_PATTERN
    w_time: float = DEFAULT_W_TIME
    v_conf: float = DEFAULT_V_CONF
    m_bonus: float = DEFAULT_M_BONUS

    def __post_init__(self):
        """Validate and clamp factor values to valid ranges."""
        self.base = max(0.0, min(100.0, self.base))
        self.q_pattern = max(MIN_Q_PATTERN, min(MAX_Q_PATTERN, self.q_pattern))
        self.v_conf = max(MIN_V_CONF, min(MAX_V_CONF, self.v_conf))
        self.m_bonus = max(MIN_M_BONUS, min(MAX_M_BONUS, self.m_bonus))
        # w_time is not clamped here as it comes from timeframes module

    @classmethod
    def from_timeframe(
        cls,
        base: float,
        timeframe: str,
        q_pattern: float = DEFAULT_Q_PATTERN,
        v_conf: float = DEFAULT_V_CONF,
        m_bonus: float = DEFAULT_M_BONUS
    ) -> "ConfidenceFactors":
        """
        Create factors with timeframe weight auto-calculated.

        Args:
            base: Raw confidence (0-100)
            timeframe: Timeframe string (e.g., "1h", "4h")
            q_pattern: Pattern quality factor
            v_conf: Volume confidence factor
            m_bonus: Multi-engine bonus

        Returns:
            ConfidenceFactors instance with w_time set from timeframe
        """
        w_time = get_timeframe_weight(timeframe)
        return cls(
            base=base,
            q_pattern=q_pattern,
            w_time=w_time,
            v_conf=v_conf,
            m_bonus=m_bonus
        )


# =============================================================================
# Confidence Calculation
# =============================================================================

def calculate_confidence(factors: ConfidenceFactors) -> int:
    """
    Calculate final confidence score using the formula.

    Formula: Confidence = Base × Q_pattern × W_time × V_conf × M_bonus

    Args:
        factors: ConfidenceFactors instance with all factor values

    Returns:
        Integer confidence value clamped to 0-100

    Example:
        >>> factors = ConfidenceFactors(base=80, q_pattern=0.9, w_time=0.85, v_conf=0.9, m_bonus=1.0)
        >>> calculate_confidence(factors)
        55
    """
    raw = (
        factors.base
        * factors.q_pattern
        * factors.w_time
        * factors.v_conf
        * factors.m_bonus
    )

    # Clamp to valid range and round to integer
    clamped = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, raw))
    return int(round(clamped))


def calculate_confidence_simple(
    base: float,
    timeframe: str = "1D",
    q_pattern: float = DEFAULT_Q_PATTERN,
    v_conf: float = DEFAULT_V_CONF,
    m_bonus: float = DEFAULT_M_BONUS
) -> int:
    """
    Convenience function to calculate confidence without creating a dataclass.

    Args:
        base: Raw confidence from pattern detection (0-100)
        timeframe: Timeframe string for weight calculation
        q_pattern: Pattern quality factor (0.5-1.0)
        v_conf: Volume confidence factor (0.7-1.0)
        m_bonus: Multi-engine bonus (1.0-1.15)

    Returns:
        Integer confidence value clamped to 0-100

    Example:
        >>> calculate_confidence_simple(80, "1D", 0.9, 0.9, 1.0)
        52
    """
    factors = ConfidenceFactors.from_timeframe(
        base=base,
        timeframe=timeframe,
        q_pattern=q_pattern,
        v_conf=v_conf,
        m_bonus=m_bonus
    )
    return calculate_confidence(factors)


# =============================================================================
# No Signal Protocol
# =============================================================================

StatusType = Literal["bullish", "bearish", "neutral"]


def apply_no_signal_protocol(
    confidence: int,
    status: StatusType
) -> Tuple[int, StatusType]:
    """
    Apply the No Signal Protocol.

    If confidence is below the threshold, the status must be "neutral"
    regardless of what the analysis detected. This prevents weak signals
    from being acted upon.

    Args:
        confidence: Calculated confidence score (0-100)
        status: Detected status ("bullish", "bearish", or "neutral")

    Returns:
        Tuple of (confidence, adjusted_status)
        - If confidence >= SIGNAL_THRESHOLD: returns original status
        - If confidence < SIGNAL_THRESHOLD: forces "neutral"

    Example:
        >>> apply_no_signal_protocol(75, "bullish")
        (75, 'bullish')
        >>> apply_no_signal_protocol(55, "bullish")
        (55, 'neutral')
    """
    if confidence < SIGNAL_THRESHOLD:
        return (confidence, "neutral")
    return (confidence, status)


def is_signal_valid(confidence: int) -> bool:
    """
    Check if confidence is high enough for a valid signal.

    Args:
        confidence: Confidence score (0-100)

    Returns:
        True if confidence >= SIGNAL_THRESHOLD

    Example:
        >>> is_signal_valid(65)
        True
        >>> is_signal_valid(55)
        False
    """
    return confidence >= SIGNAL_THRESHOLD


# =============================================================================
# Multi-Engine Bonus Calculation
# =============================================================================

def calculate_multi_engine_bonus(agreeing_engines: int) -> float:
    """
    Calculate the multi-engine agreement bonus.

    When multiple analysis engines agree on direction, confidence
    gets a bonus multiplier.

    Args:
        agreeing_engines: Number of engines that agree (1-N)

    Returns:
        Bonus multiplier (1.0-1.15)
        - 1 engine: 1.00 (no bonus)
        - 2 engines: 1.05
        - 3 engines: 1.10
        - 4+ engines: 1.15 (max)

    Example:
        >>> calculate_multi_engine_bonus(1)
        1.0
        >>> calculate_multi_engine_bonus(3)
        1.1
        >>> calculate_multi_engine_bonus(5)
        1.15
    """
    if agreeing_engines <= 1:
        return 1.0
    elif agreeing_engines == 2:
        return 1.05
    elif agreeing_engines == 3:
        return 1.10
    else:
        return 1.15


# =============================================================================
# Volume Confidence Helpers
# =============================================================================

def calculate_volume_confidence(
    current_volume: float,
    average_volume: float,
    volume_confirms_direction: bool = True
) -> float:
    """
    Calculate volume confidence factor based on volume analysis.

    Args:
        current_volume: Current bar/period volume
        average_volume: Average volume (e.g., 20-period SMA)
        volume_confirms_direction: Whether volume supports the price direction

    Returns:
        V_conf factor (0.7-1.0)

    Example:
        >>> calculate_volume_confidence(1500000, 1000000, True)
        1.0
        >>> calculate_volume_confidence(500000, 1000000, False)
        0.7
    """
    if average_volume <= 0:
        return DEFAULT_V_CONF  # Can't calculate, use neutral

    volume_ratio = current_volume / average_volume

    # High volume confirming direction = highest confidence
    if volume_ratio >= 1.5 and volume_confirms_direction:
        return 1.0
    elif volume_ratio >= 1.0 and volume_confirms_direction:
        return 0.95
    elif volume_ratio >= 0.8:
        return 0.85  # Normal volume, neutral
    elif volume_ratio >= 0.5:
        return 0.80  # Low volume
    else:
        return 0.70  # Very low volume, minimum confidence


def get_default_volume_confidence(has_volume_data: bool) -> float:
    """
    Get default volume confidence when detailed analysis isn't available.

    Args:
        has_volume_data: Whether volume data is available

    Returns:
        Default V_conf value

    Example:
        >>> get_default_volume_confidence(True)
        0.85
        >>> get_default_volume_confidence(False)
        0.7
    """
    return DEFAULT_V_CONF if has_volume_data else MIN_V_CONF
