"""
Tests for the timeframes module.

Tests cover:
- Timeframe hierarchy and weights
- Timeframe validation
- Higher/lower timeframe navigation
- Timeframe comparison utilities
"""

import pytest
from sigmapilot_mcp.core.timeframes import (
    TIMEFRAME_HIERARCHY,
    TIMEFRAME_WEIGHTS,
    MINIMUM_BARS_BY_TIMEFRAME,
    is_valid_timeframe,
    get_timeframe_index,
    get_timeframe_weight,
    get_minimum_bars,
    get_higher_timeframes,
    get_lower_timeframes,
    get_adjacent_timeframes,
    is_higher_timeframe,
    timeframe_ratio,
    DEFAULT_TIMEFRAME_WEIGHT,
    DEFAULT_MINIMUM_BARS,
)


class TestTimeframeConstants:
    """Tests for timeframe constants."""

    def test_hierarchy_order(self):
        """Test that hierarchy is ordered from highest to lowest."""
        expected = ["1M", "1W", "1D", "4h", "1h", "15m", "5m"]
        assert TIMEFRAME_HIERARCHY == expected

    def test_all_timeframes_have_weights(self):
        """Test that all timeframes in hierarchy have weights."""
        for tf in TIMEFRAME_HIERARCHY:
            assert tf in TIMEFRAME_WEIGHTS, f"Missing weight for {tf}"

    def test_weights_decrease_with_timeframe(self):
        """Test that weights decrease as timeframe gets smaller."""
        for i in range(len(TIMEFRAME_HIERARCHY) - 1):
            higher_tf = TIMEFRAME_HIERARCHY[i]
            lower_tf = TIMEFRAME_HIERARCHY[i + 1]
            assert TIMEFRAME_WEIGHTS[higher_tf] >= TIMEFRAME_WEIGHTS[lower_tf], \
                f"Weight for {higher_tf} should be >= {lower_tf}"

    def test_monthly_has_highest_weight(self):
        """Test that monthly has weight 1.0."""
        assert TIMEFRAME_WEIGHTS["1M"] == 1.0

    def test_5m_has_lowest_weight(self):
        """Test that 5m has lowest weight."""
        assert TIMEFRAME_WEIGHTS["5m"] == 0.7

    def test_minimum_bars_defined(self):
        """Test that minimum bars are defined for all timeframes."""
        for tf in TIMEFRAME_HIERARCHY:
            assert tf in MINIMUM_BARS_BY_TIMEFRAME, f"Missing minimum bars for {tf}"


class TestTimeframeValidation:
    """Tests for timeframe validation functions."""

    def test_is_valid_timeframe_valid(self):
        """Test valid timeframes return True."""
        for tf in TIMEFRAME_HIERARCHY:
            assert is_valid_timeframe(tf) is True

    def test_is_valid_timeframe_invalid(self):
        """Test invalid timeframes return False."""
        assert is_valid_timeframe("2h") is False
        assert is_valid_timeframe("1d") is False  # Case sensitive
        assert is_valid_timeframe("") is False
        assert is_valid_timeframe("invalid") is False

    def test_get_timeframe_index_valid(self):
        """Test getting index for valid timeframes."""
        assert get_timeframe_index("1M") == 0
        assert get_timeframe_index("1W") == 1
        assert get_timeframe_index("1D") == 2
        assert get_timeframe_index("4h") == 3
        assert get_timeframe_index("1h") == 4
        assert get_timeframe_index("15m") == 5
        assert get_timeframe_index("5m") == 6

    def test_get_timeframe_index_invalid(self):
        """Test getting index for invalid timeframe returns -1."""
        assert get_timeframe_index("invalid") == -1
        assert get_timeframe_index("") == -1


class TestTimeframeWeights:
    """Tests for timeframe weight functions."""

    def test_get_weight_for_all_timeframes(self):
        """Test getting weight for all valid timeframes."""
        for tf in TIMEFRAME_HIERARCHY:
            weight = get_timeframe_weight(tf)
            assert 0.0 <= weight <= 1.0
            assert weight == TIMEFRAME_WEIGHTS[tf]

    def test_get_weight_invalid_returns_default(self):
        """Test that invalid timeframe returns default weight."""
        weight = get_timeframe_weight("invalid")
        assert weight == DEFAULT_TIMEFRAME_WEIGHT

    def test_specific_weights(self):
        """Test specific weight values."""
        assert get_timeframe_weight("1D") == 0.90
        assert get_timeframe_weight("4h") == 0.85
        assert get_timeframe_weight("1h") == 0.80


class TestMinimumBars:
    """Tests for minimum bars functions."""

    def test_get_minimum_bars_monthly(self):
        """Test minimum bars for monthly."""
        assert get_minimum_bars("1M") == 24

    def test_get_minimum_bars_daily(self):
        """Test minimum bars for daily."""
        assert get_minimum_bars("1D") == 100

    def test_get_minimum_bars_hourly(self):
        """Test minimum bars for hourly."""
        assert get_minimum_bars("1h") == 200

    def test_get_minimum_bars_invalid(self):
        """Test minimum bars for invalid timeframe returns default."""
        assert get_minimum_bars("invalid") == DEFAULT_MINIMUM_BARS


class TestTimeframeNavigation:
    """Tests for timeframe navigation functions."""

    def test_get_higher_timeframes_1h(self):
        """Test getting higher timeframes for 1h."""
        higher = get_higher_timeframes("1h")
        assert higher == ["1M", "1W", "1D", "4h"]

    def test_get_higher_timeframes_monthly(self):
        """Test that monthly has no higher timeframes."""
        higher = get_higher_timeframes("1M")
        assert higher == []

    def test_get_higher_timeframes_invalid(self):
        """Test that invalid timeframe returns empty list."""
        higher = get_higher_timeframes("invalid")
        assert higher == []

    def test_get_lower_timeframes_1h(self):
        """Test getting lower timeframes for 1h."""
        lower = get_lower_timeframes("1h")
        assert lower == ["15m", "5m"]

    def test_get_lower_timeframes_5m(self):
        """Test that 5m has no lower timeframes."""
        lower = get_lower_timeframes("5m")
        assert lower == []

    def test_get_lower_timeframes_invalid(self):
        """Test that invalid timeframe returns empty list."""
        lower = get_lower_timeframes("invalid")
        assert lower == []

    def test_get_adjacent_timeframes_middle(self):
        """Test getting adjacent timeframes for middle timeframe."""
        higher, lower = get_adjacent_timeframes("1h")
        assert higher == "4h"
        assert lower == "15m"

    def test_get_adjacent_timeframes_highest(self):
        """Test getting adjacent timeframes for highest."""
        higher, lower = get_adjacent_timeframes("1M")
        assert higher is None
        assert lower == "1W"

    def test_get_adjacent_timeframes_lowest(self):
        """Test getting adjacent timeframes for lowest."""
        higher, lower = get_adjacent_timeframes("5m")
        assert higher == "15m"
        assert lower is None

    def test_get_adjacent_timeframes_invalid(self):
        """Test getting adjacent timeframes for invalid."""
        higher, lower = get_adjacent_timeframes("invalid")
        assert higher is None
        assert lower is None


class TestTimeframeComparison:
    """Tests for timeframe comparison functions."""

    def test_is_higher_timeframe_true(self):
        """Test is_higher_timeframe returns True correctly."""
        assert is_higher_timeframe("1D", "1h") is True
        assert is_higher_timeframe("1M", "5m") is True
        assert is_higher_timeframe("4h", "15m") is True

    def test_is_higher_timeframe_false(self):
        """Test is_higher_timeframe returns False correctly."""
        assert is_higher_timeframe("1h", "1D") is False
        assert is_higher_timeframe("5m", "1M") is False
        assert is_higher_timeframe("15m", "4h") is False

    def test_is_higher_timeframe_equal(self):
        """Test is_higher_timeframe with equal timeframes."""
        assert is_higher_timeframe("1h", "1h") is False

    def test_is_higher_timeframe_invalid(self):
        """Test is_higher_timeframe with invalid timeframes."""
        assert is_higher_timeframe("invalid", "1h") is False
        assert is_higher_timeframe("1h", "invalid") is False


class TestTimeframeRatio:
    """Tests for timeframe ratio calculations."""

    def test_ratio_1h_to_15m(self):
        """Test ratio between 1h and 15m."""
        ratio = timeframe_ratio("1h", "15m")
        assert ratio == 4.0

    def test_ratio_1D_to_1h(self):
        """Test ratio between 1D and 1h."""
        ratio = timeframe_ratio("1D", "1h")
        assert ratio == 24.0

    def test_ratio_4h_to_15m(self):
        """Test ratio between 4h and 15m."""
        ratio = timeframe_ratio("4h", "15m")
        assert ratio == 16.0

    def test_ratio_1W_to_1D(self):
        """Test ratio between 1W and 1D."""
        ratio = timeframe_ratio("1W", "1D")
        assert ratio == 7.0

    def test_ratio_invalid_returns_none(self):
        """Test ratio with invalid timeframe returns None."""
        assert timeframe_ratio("invalid", "1h") is None
        assert timeframe_ratio("1h", "invalid") is None
        assert timeframe_ratio("invalid", "invalid") is None
