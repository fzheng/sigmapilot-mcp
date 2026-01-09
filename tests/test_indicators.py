"""
Unit tests for the indicators module.

Tests cover:
- compute_change: Percentage change calculations
- compute_bbw: Bollinger Band Width calculations
- compute_bb_rating_signal: Rating and signal generation
- compute_metrics: Full metrics computation
"""

import pytest
from sigmapilot_mcp.core.services.indicators import (
    compute_change,
    compute_bbw,
    compute_bb_rating_signal,
    compute_metrics,
)


# =============================================================================
# Tests for compute_change
# =============================================================================

class TestComputeChange:
    """Tests for the compute_change function."""

    def test_positive_change(self):
        """Test positive price change calculation."""
        # Price went from 100 to 110 = 10% gain
        result = compute_change(100.0, 110.0)
        assert result == 10.0

    def test_negative_change(self):
        """Test negative price change calculation."""
        # Price went from 100 to 90 = -10% loss
        result = compute_change(100.0, 90.0)
        assert result == -10.0

    def test_no_change(self):
        """Test zero change when open equals close."""
        result = compute_change(100.0, 100.0)
        assert result == 0.0

    def test_zero_open_price(self):
        """Test handling of zero open price (division by zero protection)."""
        result = compute_change(0.0, 100.0)
        assert result == 0.0

    def test_small_price_change(self):
        """Test small fractional price changes."""
        # 0.5% change
        result = compute_change(100.0, 100.5)
        assert abs(result - 0.5) < 0.0001

    def test_large_price_change(self):
        """Test large price changes (100%+ gain)."""
        # Price doubled = 100% gain
        result = compute_change(50.0, 100.0)
        assert result == 100.0

    def test_crypto_decimals(self):
        """Test with typical crypto decimal prices."""
        # Common scenario: BTC price movement
        result = compute_change(42500.0, 43000.0)
        expected = ((43000.0 - 42500.0) / 42500.0) * 100
        assert abs(result - expected) < 0.0001


# =============================================================================
# Tests for compute_bbw
# =============================================================================

class TestComputeBbw:
    """Tests for the compute_bbw (Bollinger Band Width) function."""

    def test_normal_bbw_calculation(self):
        """Test normal BBW calculation."""
        # BBW = (upper - lower) / sma
        sma = 100.0
        bb_upper = 110.0
        bb_lower = 90.0
        result = compute_bbw(sma, bb_upper, bb_lower)
        # (110 - 90) / 100 = 0.2
        assert result == 0.2

    def test_tight_bands(self):
        """Test tight Bollinger Bands (squeeze scenario)."""
        sma = 100.0
        bb_upper = 102.0
        bb_lower = 98.0
        result = compute_bbw(sma, bb_upper, bb_lower)
        # (102 - 98) / 100 = 0.04
        assert result == 0.04

    def test_wide_bands(self):
        """Test wide Bollinger Bands (high volatility)."""
        sma = 100.0
        bb_upper = 130.0
        bb_lower = 70.0
        result = compute_bbw(sma, bb_upper, bb_lower)
        # (130 - 70) / 100 = 0.6
        assert result == 0.6

    def test_zero_sma(self):
        """Test handling of zero SMA (returns None)."""
        result = compute_bbw(0.0, 110.0, 90.0)
        assert result is None

    def test_none_sma(self):
        """Test handling of None SMA."""
        result = compute_bbw(None, 110.0, 90.0)
        assert result is None

    def test_false_sma(self):
        """Test handling of falsy SMA value."""
        result = compute_bbw(0, 110.0, 90.0)
        assert result is None


# =============================================================================
# Tests for compute_bb_rating_signal
# =============================================================================

class TestComputeBbRatingSignal:
    """Tests for the compute_bb_rating_signal function."""

    def test_above_upper_band(self):
        """Test rating when price is above upper band (+3 Strong Buy)."""
        close = 115.0
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == 3
        assert signal == "NEUTRAL"  # Signal only triggers at +/-2

    def test_upper_half_of_bands(self):
        """Test rating when price is in upper 50% of bands (+2 Buy)."""
        close = 107.0  # Above bb_middle + ((bb_upper - bb_middle) / 2) = 105
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == 2
        assert signal == "BUY"

    def test_above_middle(self):
        """Test rating when price is just above middle (+1 Weak Buy)."""
        close = 102.0
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == 1
        assert signal == "NEUTRAL"

    def test_below_lower_band(self):
        """Test rating when price is below lower band (-3 Strong Sell)."""
        close = 85.0
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == -3
        assert signal == "NEUTRAL"

    def test_lower_half_of_bands(self):
        """Test rating when price is in lower 50% of bands (-2 Sell)."""
        close = 93.0  # Below bb_middle - ((bb_middle - bb_lower) / 2) = 95
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == -2
        assert signal == "SELL"

    def test_below_middle(self):
        """Test rating when price is just below middle (-1 Weak Sell)."""
        close = 98.0
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == -1
        assert signal == "NEUTRAL"

    def test_at_middle(self):
        """Test rating when price is exactly at middle (0 Neutral)."""
        close = 100.0
        bb_upper = 110.0
        bb_middle = 100.0
        bb_lower = 90.0
        rating, signal = compute_bb_rating_signal(close, bb_upper, bb_middle, bb_lower)
        assert rating == 0
        assert signal == "NEUTRAL"


# =============================================================================
# Tests for compute_metrics
# =============================================================================

class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_complete_metrics(self):
        """Test full metrics computation with valid data."""
        indicators = {
            "open": 100.0,
            "close": 105.0,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)

        assert result is not None
        assert "price" in result
        assert "change" in result
        assert "bbw" in result
        assert "rating" in result
        assert "signal" in result

        assert result["price"] == 105.0
        assert result["change"] == 5.0  # 5% gain
        # BBW = (110 - 94) / 102 ≈ 0.1569
        assert abs(result["bbw"] - 0.1569) < 0.001

    def test_missing_open(self):
        """Test handling of missing open price."""
        indicators = {
            "close": 105.0,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        assert result is None

    def test_missing_close(self):
        """Test handling of missing close price."""
        indicators = {
            "open": 100.0,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        assert result is None

    def test_missing_sma(self):
        """Test handling of missing SMA20."""
        indicators = {
            "open": 100.0,
            "close": 105.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        assert result is None

    def test_missing_bb_upper(self):
        """Test handling of missing BB.upper."""
        indicators = {
            "open": 100.0,
            "close": 105.0,
            "SMA20": 102.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        assert result is None

    def test_missing_bb_lower(self):
        """Test handling of missing BB.lower."""
        indicators = {
            "open": 100.0,
            "close": 105.0,
            "SMA20": 102.0,
            "BB.upper": 110.0,
        }
        result = compute_metrics(indicators)
        assert result is None

    def test_empty_indicators(self):
        """Test handling of empty indicators dict."""
        result = compute_metrics({})
        assert result is None

    def test_type_error_handling(self):
        """Test handling of invalid types in indicators."""
        indicators = {
            "open": "invalid",
            "close": 105.0,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        # Should return None due to TypeError in calculation
        assert result is None

    def test_price_rounding(self):
        """Test that price is properly rounded to 4 decimal places."""
        indicators = {
            "open": 100.0,
            "close": 105.12345678,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        assert result is not None
        assert result["price"] == 105.1235  # Rounded to 4 decimal places

    def test_change_rounding(self):
        """Test that change percentage is properly rounded to 3 decimal places."""
        indicators = {
            "open": 100.0,
            "close": 103.33333333,
            "SMA20": 102.0,
            "BB.upper": 110.0,
            "BB.lower": 94.0,
        }
        result = compute_metrics(indicators)
        assert result is not None
        # 3.33333...% rounded to 3 decimal places
        assert abs(result["change"] - 3.333) < 0.001


# =============================================================================
# Integration Tests
# =============================================================================

class TestIndicatorsIntegration:
    """Integration tests combining multiple indicator functions."""

    def test_full_analysis_bullish(self):
        """Test a complete bullish scenario."""
        indicators = {
            "open": 100.0,
            "close": 108.0,  # Above upper half of bands
            "SMA20": 100.0,
            "BB.upper": 110.0,
            "BB.lower": 90.0,
        }

        result = compute_metrics(indicators)
        assert result is not None
        assert result["change"] == 8.0  # 8% gain
        assert result["rating"] == 2  # Upper half = Buy signal
        assert result["signal"] == "BUY"
        assert result["bbw"] == 0.2  # Normal volatility

    def test_full_analysis_bearish(self):
        """Test a complete bearish scenario."""
        indicators = {
            "open": 100.0,
            "close": 92.0,  # In lower half of bands
            "SMA20": 100.0,
            "BB.upper": 110.0,
            "BB.lower": 90.0,
        }

        result = compute_metrics(indicators)
        assert result is not None
        assert result["change"] == -8.0  # 8% loss
        assert result["rating"] == -2  # Lower half = Sell signal
        assert result["signal"] == "SELL"

    def test_squeeze_detection(self):
        """Test Bollinger Band squeeze scenario."""
        indicators = {
            "open": 100.0,
            "close": 100.5,
            "SMA20": 100.0,
            "BB.upper": 102.0,  # Very tight bands
            "BB.lower": 98.0,
        }

        result = compute_metrics(indicators)
        assert result is not None
        assert result["bbw"] == 0.04  # Low BBW = squeeze
        # Tight bands often precede breakouts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
