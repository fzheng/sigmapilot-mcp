"""
Unit tests for the server module.

Tests cover:
- Helper functions: _map_indicators, _percent_change, _tf_to_tv_resolution
- Input validation and parameter handling
- Type definitions: IndicatorMap, Row, MultiRow
"""

import pytest
from tradingview_mcp.server import (
    _map_indicators,
    _percent_change,
    _tf_to_tv_resolution,
    IndicatorMap,
)


# =============================================================================
# Tests for _map_indicators
# =============================================================================

class TestMapIndicators:
    """Tests for the _map_indicators function."""

    def test_maps_all_fields(self):
        """Test that all indicator fields are mapped correctly."""
        raw = {
            "open": 100.0,
            "close": 105.0,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
            "EMA9": 101.0,
            "EMA21": 100.5,
            "EMA50": 99.0,
            "RSI": 65.0,
            "ATR": 2.5,
            "volume": 1000000.0,
        }

        result = _map_indicators(raw)

        assert result["open"] == 100.0
        assert result["close"] == 105.0
        assert result["SMA20"] == 102.0
        assert result["BB_upper"] == 110.0
        assert result["BB_lower"] == 94.0
        assert result["EMA9"] == 101.0
        assert result["EMA21"] == 100.5
        assert result["EMA50"] == 99.0
        assert result["RSI"] == 65.0
        assert result["ATR"] == 2.5
        assert result["volume"] == 1000000.0

    def test_handles_bb_upper_alternate_key(self):
        """Test handling of BB_upper as alternate key for BB.upper."""
        raw = {
            "open": 100.0,
            "close": 105.0,
            "BB_upper": 110.0,  # Alternate key format
            "BB_lower": 94.0,
        }

        result = _map_indicators(raw)
        assert result["BB_upper"] == 110.0
        assert result["BB_lower"] == 94.0

    def test_prefers_dot_notation(self):
        """Test that BB.upper takes precedence over BB_upper."""
        raw = {
            "BB.upper": 110.0,
            "BB_upper": 108.0,  # Should be ignored
            "BB.lower": 94.0,
            "BB_lower": 92.0,  # Should be ignored
        }

        result = _map_indicators(raw)
        assert result["BB_upper"] == 110.0
        assert result["BB_lower"] == 94.0

    def test_handles_missing_fields(self):
        """Test handling of missing fields (returns None for missing)."""
        raw = {
            "open": 100.0,
            "close": 105.0,
        }

        result = _map_indicators(raw)
        assert result["open"] == 100.0
        assert result["close"] == 105.0
        assert result.get("SMA20") is None
        assert result.get("RSI") is None

    def test_empty_dict(self):
        """Test handling of empty raw dict."""
        result = _map_indicators({})
        assert result.get("open") is None
        assert result.get("close") is None


# =============================================================================
# Tests for _percent_change
# =============================================================================

class TestPercentChange:
    """Tests for the _percent_change function."""

    def test_positive_change(self):
        """Test positive percentage change."""
        result = _percent_change(100.0, 110.0)
        assert result == 10.0

    def test_negative_change(self):
        """Test negative percentage change."""
        result = _percent_change(100.0, 90.0)
        assert result == -10.0

    def test_zero_change(self):
        """Test zero change when open equals close."""
        result = _percent_change(100.0, 100.0)
        assert result == 0.0

    def test_none_open(self):
        """Test handling of None open price."""
        result = _percent_change(None, 100.0)
        assert result is None

    def test_none_close(self):
        """Test handling of None close price."""
        result = _percent_change(100.0, None)
        assert result is None

    def test_zero_open(self):
        """Test handling of zero open price."""
        result = _percent_change(0, 100.0)
        assert result is None

    def test_both_none(self):
        """Test handling of both values being None."""
        result = _percent_change(None, None)
        assert result is None


# =============================================================================
# Tests for _tf_to_tv_resolution
# =============================================================================

class TestTfToTvResolution:
    """Tests for the _tf_to_tv_resolution function."""

    def test_minute_timeframes(self):
        """Test minute-based timeframe conversion."""
        assert _tf_to_tv_resolution("5m") == "5"
        assert _tf_to_tv_resolution("15m") == "15"

    def test_hour_timeframes(self):
        """Test hour-based timeframe conversion."""
        assert _tf_to_tv_resolution("1h") == "60"
        assert _tf_to_tv_resolution("4h") == "240"

    def test_daily_weekly_monthly(self):
        """Test daily, weekly, monthly conversion."""
        assert _tf_to_tv_resolution("1D") == "1D"
        assert _tf_to_tv_resolution("1W") == "1W"
        assert _tf_to_tv_resolution("1M") == "1M"

    def test_none_input(self):
        """Test handling of None input."""
        result = _tf_to_tv_resolution(None)
        assert result is None

    def test_empty_string(self):
        """Test handling of empty string."""
        result = _tf_to_tv_resolution("")
        assert result is None

    def test_invalid_timeframe(self):
        """Test handling of invalid timeframe."""
        result = _tf_to_tv_resolution("invalid")
        assert result is None

    def test_case_sensitivity(self):
        """Test that timeframes are case-sensitive."""
        # Lowercase 'd' should not work
        result = _tf_to_tv_resolution("1d")
        assert result is None


# =============================================================================
# Tests for Type Definitions
# =============================================================================

class TestTypedDictDefinitions:
    """Tests for TypedDict type definitions."""

    def test_indicator_map_creation(self):
        """Test creating an IndicatorMap."""
        indicator_map: IndicatorMap = {
            "open": 100.0,
            "close": 105.0,
            "SMA20": 102.0,
            "BB_upper": 110.0,
            "BB_lower": 94.0,
            "EMA9": 101.0,
            "EMA21": 100.5,
            "EMA50": 99.0,
            "RSI": 65.0,
            "ATR": 2.5,
            "volume": 1000000.0,
        }

        assert indicator_map["open"] == 100.0
        assert indicator_map["close"] == 105.0

    def test_indicator_map_partial(self):
        """Test that IndicatorMap accepts partial data (total=False)."""
        # TypedDict with total=False allows missing keys
        indicator_map: IndicatorMap = {
            "open": 100.0,
            "close": 105.0,
        }

        assert indicator_map["open"] == 100.0
        # Other keys are optional
        assert indicator_map.get("SMA20") is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestServerIntegration:
    """Integration tests for server helper functions."""

    def test_full_indicator_pipeline(self):
        """Test complete indicator processing pipeline."""
        # Raw data from TradingView
        raw_data = {
            "open": 100.0,
            "close": 110.0,
            "SMA20": 105.0,
            "BB.upper": 115.0,
            "BB.lower": 95.0,
            "EMA9": 108.0,
            "EMA21": 106.0,
            "EMA50": 102.0,
            "RSI": 72.0,
            "ATR": 3.5,
            "volume": 5000000.0,
        }

        # Map indicators
        mapped = _map_indicators(raw_data)

        # Calculate percent change
        change = _percent_change(mapped.get("open"), mapped.get("close"))

        assert change == 10.0  # 10% gain
        assert mapped["BB_upper"] == 115.0
        assert mapped["RSI"] == 72.0

    def test_timeframe_resolution_all_valid(self):
        """Test all valid timeframe conversions."""
        timeframes = {
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1D": "1D",
            "1W": "1W",
            "1M": "1M",
        }

        for tf, expected in timeframes.items():
            result = _tf_to_tv_resolution(tf)
            assert result == expected, f"Failed for {tf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
