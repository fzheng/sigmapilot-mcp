"""
Tests for the confidence calculation module.

Tests cover:
- Confidence formula calculations
- Factor clamping and validation
- No Signal Protocol implementation
- Multi-engine bonus calculations
- Volume confidence helpers
"""

import pytest
from sigmapilot_mcp.core.confidence import (
    ConfidenceFactors,
    calculate_confidence,
    calculate_confidence_simple,
    apply_no_signal_protocol,
    is_signal_valid,
    calculate_multi_engine_bonus,
    calculate_volume_confidence,
    get_default_volume_confidence,
    SIGNAL_THRESHOLD,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    DEFAULT_Q_PATTERN,
    DEFAULT_V_CONF,
)


class TestConfidenceFactors:
    """Tests for ConfidenceFactors dataclass."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        factors = ConfidenceFactors()
        assert factors.base == 0.0
        assert factors.q_pattern == DEFAULT_Q_PATTERN
        assert factors.v_conf == DEFAULT_V_CONF
        assert factors.m_bonus == 1.0

    def test_custom_values(self):
        """Test that custom values are preserved."""
        factors = ConfidenceFactors(
            base=80,
            q_pattern=0.9,
            w_time=0.85,
            v_conf=0.95,
            m_bonus=1.1
        )
        assert factors.base == 80
        assert factors.q_pattern == 0.9
        assert factors.w_time == 0.85
        assert factors.v_conf == 0.95
        assert factors.m_bonus == 1.1

    def test_base_clamping_high(self):
        """Test that base is clamped to 100."""
        factors = ConfidenceFactors(base=150)
        assert factors.base == 100.0

    def test_base_clamping_low(self):
        """Test that base is clamped to 0."""
        factors = ConfidenceFactors(base=-20)
        assert factors.base == 0.0

    def test_q_pattern_clamping_high(self):
        """Test that q_pattern is clamped to 1.0."""
        factors = ConfidenceFactors(q_pattern=1.5)
        assert factors.q_pattern == 1.0

    def test_q_pattern_clamping_low(self):
        """Test that q_pattern is clamped to 0.5."""
        factors = ConfidenceFactors(q_pattern=0.3)
        assert factors.q_pattern == 0.5

    def test_v_conf_clamping_high(self):
        """Test that v_conf is clamped to 1.0."""
        factors = ConfidenceFactors(v_conf=1.5)
        assert factors.v_conf == 1.0

    def test_v_conf_clamping_low(self):
        """Test that v_conf is clamped to 0.7."""
        factors = ConfidenceFactors(v_conf=0.5)
        assert factors.v_conf == 0.7

    def test_m_bonus_clamping_high(self):
        """Test that m_bonus is clamped to 1.15."""
        factors = ConfidenceFactors(m_bonus=1.5)
        assert factors.m_bonus == 1.15

    def test_m_bonus_clamping_low(self):
        """Test that m_bonus is clamped to 1.0."""
        factors = ConfidenceFactors(m_bonus=0.8)
        assert factors.m_bonus == 1.0

    def test_from_timeframe_daily(self):
        """Test creating factors from timeframe."""
        factors = ConfidenceFactors.from_timeframe(base=80, timeframe="1D")
        assert factors.base == 80
        assert factors.w_time == 0.90  # Daily weight

    def test_from_timeframe_hourly(self):
        """Test creating factors from hourly timeframe."""
        factors = ConfidenceFactors.from_timeframe(base=80, timeframe="1h")
        assert factors.w_time == 0.80  # Hourly weight

    def test_from_timeframe_5m(self):
        """Test creating factors from 5m timeframe."""
        factors = ConfidenceFactors.from_timeframe(base=80, timeframe="5m")
        assert factors.w_time == 0.70  # 5m weight (lowest)


class TestCalculateConfidence:
    """Tests for confidence calculation."""

    def test_basic_formula(self):
        """Test basic confidence formula calculation."""
        factors = ConfidenceFactors(
            base=100,
            q_pattern=1.0,
            w_time=1.0,
            v_conf=1.0,
            m_bonus=1.0
        )
        result = calculate_confidence(factors)
        assert result == 100

    def test_formula_with_multipliers(self):
        """Test formula with various multipliers."""
        factors = ConfidenceFactors(
            base=80,
            q_pattern=0.9,  # 80 * 0.9 = 72
            w_time=1.0,
            v_conf=1.0,
            m_bonus=1.0
        )
        result = calculate_confidence(factors)
        assert result == 72

    def test_all_multipliers(self):
        """Test with all multipliers applied."""
        # 80 * 0.9 * 0.85 * 0.9 * 1.1 = 60.588
        factors = ConfidenceFactors(
            base=80,
            q_pattern=0.9,
            w_time=0.85,
            v_conf=0.9,
            m_bonus=1.1
        )
        result = calculate_confidence(factors)
        assert result == 61  # Rounded

    def test_clamping_at_100(self):
        """Test that result is clamped to 100."""
        factors = ConfidenceFactors(
            base=100,
            q_pattern=1.0,
            w_time=1.0,
            v_conf=1.0,
            m_bonus=1.15  # Would give 115
        )
        result = calculate_confidence(factors)
        assert result == 100

    def test_clamping_at_0(self):
        """Test that result is clamped to 0."""
        factors = ConfidenceFactors(
            base=0,
            q_pattern=0.5,
            w_time=0.7,
            v_conf=0.7,
            m_bonus=1.0
        )
        result = calculate_confidence(factors)
        assert result == 0

    def test_returns_integer(self):
        """Test that result is always an integer."""
        factors = ConfidenceFactors(
            base=77,
            q_pattern=0.87,
            w_time=0.83,
            v_conf=0.91,
            m_bonus=1.05
        )
        result = calculate_confidence(factors)
        assert isinstance(result, int)


class TestCalculateConfidenceSimple:
    """Tests for the simple confidence calculation function."""

    def test_basic_calculation(self):
        """Test basic calculation without explicit factors."""
        result = calculate_confidence_simple(base=100, timeframe="1D")
        # 100 * 0.85 * 0.90 * 0.85 * 1.0 = 65.025
        assert result == 65

    def test_with_custom_factors(self):
        """Test with custom factor values."""
        result = calculate_confidence_simple(
            base=80,
            timeframe="1h",
            q_pattern=0.9,
            v_conf=0.95,
            m_bonus=1.1
        )
        # 80 * 0.9 * 0.8 * 0.95 * 1.1 = 60.192
        assert result == 60


class TestNoSignalProtocol:
    """Tests for the No Signal Protocol implementation."""

    def test_confidence_above_threshold_bullish(self):
        """Test that bullish status is preserved above threshold."""
        confidence, status = apply_no_signal_protocol(75, "bullish")
        assert confidence == 75
        assert status == "bullish"

    def test_confidence_above_threshold_bearish(self):
        """Test that bearish status is preserved above threshold."""
        confidence, status = apply_no_signal_protocol(65, "bearish")
        assert confidence == 65
        assert status == "bearish"

    def test_confidence_at_threshold(self):
        """Test behavior at exactly 60 (threshold)."""
        confidence, status = apply_no_signal_protocol(60, "bullish")
        assert confidence == 60
        assert status == "bullish"

    def test_confidence_below_threshold_forces_neutral(self):
        """Test that status is forced to neutral below threshold."""
        confidence, status = apply_no_signal_protocol(59, "bullish")
        assert confidence == 59
        assert status == "neutral"

    def test_confidence_below_threshold_bearish(self):
        """Test that bearish is also forced to neutral below threshold."""
        confidence, status = apply_no_signal_protocol(55, "bearish")
        assert confidence == 55
        assert status == "neutral"

    def test_already_neutral(self):
        """Test that neutral status is preserved."""
        confidence, status = apply_no_signal_protocol(45, "neutral")
        assert confidence == 45
        assert status == "neutral"

    def test_is_signal_valid_above_threshold(self):
        """Test is_signal_valid returns True above threshold."""
        assert is_signal_valid(65) is True
        assert is_signal_valid(60) is True
        assert is_signal_valid(100) is True

    def test_is_signal_valid_below_threshold(self):
        """Test is_signal_valid returns False below threshold."""
        assert is_signal_valid(59) is False
        assert is_signal_valid(50) is False
        assert is_signal_valid(0) is False


class TestMultiEngineBonus:
    """Tests for multi-engine bonus calculations."""

    def test_single_engine_no_bonus(self):
        """Test that single engine gets no bonus."""
        assert calculate_multi_engine_bonus(1) == 1.0

    def test_zero_engines(self):
        """Test that zero engines gets no bonus."""
        assert calculate_multi_engine_bonus(0) == 1.0

    def test_two_engines(self):
        """Test two engines agreement bonus."""
        assert calculate_multi_engine_bonus(2) == 1.05

    def test_three_engines(self):
        """Test three engines agreement bonus."""
        assert calculate_multi_engine_bonus(3) == 1.10

    def test_four_engines_max(self):
        """Test that four engines gets max bonus."""
        assert calculate_multi_engine_bonus(4) == 1.15

    def test_five_engines_capped(self):
        """Test that bonus is capped at 1.15."""
        assert calculate_multi_engine_bonus(5) == 1.15
        assert calculate_multi_engine_bonus(10) == 1.15


class TestVolumeConfidence:
    """Tests for volume confidence calculations."""

    def test_high_volume_confirming(self):
        """Test high volume confirming direction."""
        result = calculate_volume_confidence(
            current_volume=1500000,
            average_volume=1000000,
            volume_confirms_direction=True
        )
        assert result == 1.0

    def test_normal_volume_confirming(self):
        """Test normal volume confirming direction."""
        result = calculate_volume_confidence(
            current_volume=1000000,
            average_volume=1000000,
            volume_confirms_direction=True
        )
        assert result == 0.95

    def test_low_volume(self):
        """Test low volume scenario."""
        result = calculate_volume_confidence(
            current_volume=600000,
            average_volume=1000000,
            volume_confirms_direction=True
        )
        assert result == 0.80

    def test_very_low_volume(self):
        """Test very low volume scenario."""
        result = calculate_volume_confidence(
            current_volume=300000,
            average_volume=1000000,
            volume_confirms_direction=True
        )
        assert result == 0.70

    def test_zero_average_volume(self):
        """Test handling of zero average volume."""
        result = calculate_volume_confidence(
            current_volume=1000000,
            average_volume=0,
            volume_confirms_direction=True
        )
        assert result == DEFAULT_V_CONF

    def test_default_volume_confidence_with_data(self):
        """Test default volume confidence when data is available."""
        assert get_default_volume_confidence(True) == DEFAULT_V_CONF

    def test_default_volume_confidence_no_data(self):
        """Test default volume confidence when no data."""
        assert get_default_volume_confidence(False) == 0.7


class TestConfidenceThresholds:
    """Tests for confidence threshold constants."""

    def test_signal_threshold_value(self):
        """Test signal threshold is 60."""
        assert SIGNAL_THRESHOLD == 60

    def test_min_confidence_value(self):
        """Test min confidence is 0."""
        assert MIN_CONFIDENCE == 0

    def test_max_confidence_value(self):
        """Test max confidence is 100."""
        assert MAX_CONFIDENCE == 100
