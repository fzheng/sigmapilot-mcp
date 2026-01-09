"""
Unit tests for the shared tools module.

Tests cover:
- BatchResult dataclass behavior
- fetch_analysis with various edge cases
- fetch_trending_analysis with batch processing
- All tool functions (get_top_gainers, get_top_losers, etc.)
- Error handling and partial failures
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock the mcp and tradingview modules before importing tools
mock_mcp = MagicMock()
mock_ta = MagicMock()
mock_ta.get_multiple_analysis = MagicMock()

if 'mcp' not in sys.modules:
    sys.modules['mcp'] = mock_mcp
    sys.modules['mcp.server'] = MagicMock()
    sys.modules['mcp.server.fastmcp'] = MagicMock()

if 'tradingview_ta' not in sys.modules:
    sys.modules['tradingview_ta'] = mock_ta

from sigmapilot_mcp.core.services.tools import (
    BatchResult,
    fetch_analysis,
    fetch_trending_analysis,
    get_top_gainers,
    get_top_losers,
    get_bollinger_scan,
    get_rating_filter,
    get_coin_analysis,
    get_exchanges_list,
)


# =============================================================================
# Tests for BatchResult
# =============================================================================

class TestBatchResult:
    """Tests for the BatchResult dataclass."""

    def test_default_values(self):
        """Test BatchResult default initialization."""
        result = BatchResult()
        assert result.data == []
        assert result.failed_batches == 0
        assert result.total_batches == 0
        assert result.errors == []

    def test_partial_failure_detection(self):
        """Test partial_failure property."""
        # No failure
        result = BatchResult(failed_batches=0, total_batches=3)
        assert result.partial_failure is False

        # Partial failure
        result = BatchResult(failed_batches=1, total_batches=3)
        assert result.partial_failure is True

        # Total failure
        result = BatchResult(failed_batches=3, total_batches=3)
        assert result.partial_failure is False

    def test_total_failure_detection(self):
        """Test total_failure property."""
        # No failure
        result = BatchResult(failed_batches=0, total_batches=3)
        assert result.total_failure is False

        # Partial failure
        result = BatchResult(failed_batches=1, total_batches=3)
        assert result.total_failure is False

        # Total failure
        result = BatchResult(failed_batches=3, total_batches=3)
        assert result.total_failure is True

    def test_zero_batches_edge_case(self):
        """Test edge case with zero batches."""
        result = BatchResult(failed_batches=0, total_batches=0)
        assert result.partial_failure is False
        assert result.total_failure is False

    def test_data_accumulation(self):
        """Test that data can be accumulated."""
        result = BatchResult()
        result.data.append({"symbol": "TEST"})
        assert len(result.data) == 1

    def test_errors_accumulation(self):
        """Test that errors can be accumulated."""
        result = BatchResult()
        result.errors.append("Error 1")
        result.errors.append("Error 2")
        assert len(result.errors) == 2


# =============================================================================
# Tests for Tool Functions (using BatchResult)
# =============================================================================

class TestGetTopGainers:
    """Tests for the get_top_gainers function."""

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_returns_gainers_from_batch(self, mock_fetch):
        """Test that results are returned from fetch_trending_analysis."""
        # Results are pre-sorted by fetch_trending_analysis
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "B", "changePercent": 10.0},
            {"symbol": "A", "changePercent": 5.0},
            {"symbol": "C", "changePercent": 2.0},
        ])

        data, warning = get_top_gainers("kucoin", "15m", 10)

        # Should return what fetch_trending_analysis provided
        assert len(data) == 3
        assert data[0]["symbol"] == "B"  # Highest gainer
        assert warning is None

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_respects_limit(self, mock_fetch):
        """Test that limit is respected."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": f"SYM{i}", "changePercent": float(i)} for i in range(20)
        ])

        data, _ = get_top_gainers("kucoin", "15m", 5)

        assert len(data) == 5

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_partial_failure_warning(self, mock_fetch):
        """Test warning message on partial failure."""
        mock_fetch.return_value = BatchResult(
            data=[{"symbol": "A", "changePercent": 5.0}],
            failed_batches=1,
            total_batches=3,
            errors=["Batch 2 failed"]
        )

        data, warning = get_top_gainers("kucoin", "15m", 10)

        assert warning is not None
        assert "Partial data" in warning

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_total_failure_warning(self, mock_fetch):
        """Test warning message on total failure."""
        mock_fetch.return_value = BatchResult(
            data=[],
            failed_batches=3,
            total_batches=3,
            errors=["Error 1", "Error 2", "Error 3"]
        )

        data, warning = get_top_gainers("kucoin", "15m", 10)

        assert warning is not None
        assert "All API requests failed" in warning

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_limit_clamping_lower(self, mock_fetch):
        """Test that limit is clamped to minimum 1."""
        mock_fetch.return_value = BatchResult(data=[])

        # Should not raise, limit should be clamped to 1
        data, _ = get_top_gainers("kucoin", "15m", -5)

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_limit_clamping_upper(self, mock_fetch):
        """Test that limit is clamped to maximum 50."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": f"SYM{i}", "changePercent": float(i)} for i in range(60)
        ])

        data, _ = get_top_gainers("kucoin", "15m", 100)
        # Even with 60 items in data, should be limited to 50
        assert len(data) <= 50


class TestGetTopLosers:
    """Tests for the get_top_losers function."""

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_returns_sorted_losers(self, mock_fetch):
        """Test that results are sorted by change ascending."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "changePercent": -5.0},
            {"symbol": "B", "changePercent": -10.0},
            {"symbol": "C", "changePercent": -2.0},
        ])

        data, _ = get_top_losers("kucoin", "15m", 10)

        # Should be sorted ascending (most negative first)
        assert data[0]["changePercent"] <= data[1]["changePercent"]

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_empty_results(self, mock_fetch):
        """Test handling of empty results."""
        mock_fetch.return_value = BatchResult(data=[])

        data, _ = get_top_losers("kucoin", "15m", 10)
        assert data == []


class TestGetBollingerScan:
    """Tests for the get_bollinger_scan function."""

    @patch('sigmapilot_mcp.core.services.tools.fetch_analysis')
    def test_filters_by_bbw_threshold(self, mock_fetch):
        """Test that results are filtered by BBW threshold."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "bbw": 0.02},  # Below threshold
            {"symbol": "B", "bbw": 0.05},  # Above threshold
            {"symbol": "C", "bbw": 0.01},  # Below threshold
        ])

        data, _ = get_bollinger_scan("kucoin", "4h", bbw_threshold=0.04, limit=10)

        # Only symbols with bbw < 0.04 and > 0
        assert len(data) == 2
        assert all(d["bbw"] < 0.04 for d in data)

    @patch('sigmapilot_mcp.core.services.tools.fetch_analysis')
    def test_excludes_zero_bbw(self, mock_fetch):
        """Test that zero BBW values are excluded."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "bbw": 0.02},
            {"symbol": "B", "bbw": 0},  # Zero - should be excluded
            {"symbol": "C", "bbw": 0.01},
        ])

        data, _ = get_bollinger_scan("kucoin", "4h", bbw_threshold=0.04, limit=10)

        assert len(data) == 2
        assert all(d["bbw"] > 0 for d in data)

    @patch('sigmapilot_mcp.core.services.tools.fetch_analysis')
    def test_sorted_by_bbw(self, mock_fetch):
        """Test that results are sorted by BBW ascending."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "bbw": 0.03},
            {"symbol": "B", "bbw": 0.01},
            {"symbol": "C", "bbw": 0.02},
        ])

        data, _ = get_bollinger_scan("kucoin", "4h", bbw_threshold=0.04, limit=10)

        assert data[0]["bbw"] <= data[1]["bbw"] <= data[2]["bbw"]

    @patch('sigmapilot_mcp.core.services.tools.fetch_analysis')
    def test_excludes_none_bbw(self, mock_fetch):
        """Test that None BBW values are excluded."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "bbw": 0.02},
            {"symbol": "B", "bbw": None},  # None - should be excluded
            {"symbol": "C", "bbw": 0.01},
        ])

        data, _ = get_bollinger_scan("kucoin", "4h", bbw_threshold=0.04, limit=10)

        assert len(data) == 2


class TestGetRatingFilter:
    """Tests for the get_rating_filter function."""

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_rating_clamping_lower(self, mock_fetch):
        """Test that rating is clamped to minimum -3."""
        mock_fetch.return_value = BatchResult(data=[])

        # Should call with clamped rating
        get_rating_filter("kucoin", "15m", rating=-10)

        # Verify the call was made with clamped rating
        call_args = mock_fetch.call_args
        assert call_args[1]['rating_filter'] == -3

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_rating_clamping_upper(self, mock_fetch):
        """Test that rating is clamped to maximum +3."""
        mock_fetch.return_value = BatchResult(data=[])

        get_rating_filter("kucoin", "15m", rating=10)

        call_args = mock_fetch.call_args
        assert call_args[1]['rating_filter'] == 3

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_returns_filtered_results(self, mock_fetch):
        """Test that results are returned correctly."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "changePercent": 5.0, "rating": 2},
            {"symbol": "B", "changePercent": 3.0, "rating": 2},
        ])

        data, _ = get_rating_filter("kucoin", "15m", rating=2, limit=10)
        assert len(data) == 2


class TestGetCoinAnalysis:
    """Tests for the get_coin_analysis function."""

    @patch('sigmapilot_mcp.core.services.tools.TRADINGVIEW_TA_AVAILABLE', False)
    def test_tradingview_not_available(self):
        """Test error when TradingView library is not available."""
        result = get_coin_analysis("BTCUSDT", "kucoin", "15m")

        assert "error" in result
        assert "not available" in result["error"]


class TestGetExchangesList:
    """Tests for the get_exchanges_list function."""

    def test_returns_all_categories(self):
        """Test that all exchange categories are returned."""
        result = get_exchanges_list()

        assert "crypto" in result
        assert "us_stocks" in result
        assert "turkey" in result
        assert "malaysia" in result
        assert "hongkong" in result
        assert "timeframes" in result

    def test_crypto_exchanges_present(self):
        """Test that crypto exchanges are included."""
        result = get_exchanges_list()

        assert "kucoin" in result["crypto"]
        assert "binance" in result["crypto"]

    def test_timeframes_complete(self):
        """Test that all timeframes are listed."""
        result = get_exchanges_list()

        expected_timeframes = ["5m", "15m", "1h", "4h", "1D", "1W", "1M"]
        assert result["timeframes"] == expected_timeframes

    def test_us_stocks_present(self):
        """Test that US stock exchanges are included."""
        result = get_exchanges_list()

        assert "nasdaq" in result["us_stocks"]

    def test_turkey_exchange_present(self):
        """Test that Turkey exchange is included."""
        result = get_exchanges_list()

        assert "bist" in result["turkey"]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for various edge cases."""

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_empty_exchange_string(self, mock_fetch):
        """Test handling of empty exchange string."""
        mock_fetch.return_value = BatchResult(data=[])
        # Should use default exchange (kucoin)
        get_top_gainers("", "15m", 10)

        call_args = mock_fetch.call_args
        assert call_args[0][0] == "kucoin"

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_empty_timeframe_string(self, mock_fetch):
        """Test handling of empty timeframe string."""
        mock_fetch.return_value = BatchResult(data=[])
        # Should use default timeframe
        get_top_gainers("kucoin", "", 10)

        call_args = mock_fetch.call_args
        assert call_args[1]['timeframe'] == "5m"  # default

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_invalid_exchange(self, mock_fetch):
        """Test handling of invalid exchange name."""
        mock_fetch.return_value = BatchResult(data=[])
        # Should sanitize to default
        get_top_gainers("invalid_exchange", "15m", 10)

        call_args = mock_fetch.call_args
        assert call_args[0][0] == "kucoin"

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_case_insensitive_exchange_upper(self, mock_fetch):
        """Test that exchange names are case-insensitive (uppercase)."""
        mock_fetch.return_value = BatchResult(data=[])

        get_top_gainers("KUCOIN", "15m", 10)
        call_args = mock_fetch.call_args
        assert call_args[0][0] == "kucoin"

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_case_insensitive_exchange_mixed(self, mock_fetch):
        """Test that exchange names are case-insensitive (mixed case)."""
        mock_fetch.return_value = BatchResult(data=[])

        get_top_gainers("KuCoin", "15m", 10)
        call_args = mock_fetch.call_args
        assert call_args[0][0] == "kucoin"

    @patch('sigmapilot_mcp.core.services.tools.fetch_analysis')
    def test_bollinger_with_negative_threshold(self, mock_fetch):
        """Test bollinger scan with negative threshold."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "bbw": 0.02},
        ])

        # Negative threshold should return nothing (no BBW < negative)
        data, _ = get_bollinger_scan("kucoin", "4h", bbw_threshold=-0.01, limit=10)
        assert data == []

    @patch('sigmapilot_mcp.core.services.tools.fetch_trending_analysis')
    def test_zero_limit(self, mock_fetch):
        """Test with zero limit (should clamp to 1)."""
        mock_fetch.return_value = BatchResult(data=[
            {"symbol": "A", "changePercent": 5.0}
        ])

        data, _ = get_top_gainers("kucoin", "15m", 0)
        # Limit clamped to 1
        assert len(data) <= 1


# =============================================================================
# Integration-like Tests
# =============================================================================

class TestToolsIntegration:
    """Integration-style tests for the tools module."""

    def test_batch_result_with_all_fields(self):
        """Test BatchResult with all fields populated."""
        result = BatchResult(
            data=[{"symbol": "TEST", "changePercent": 1.0}],
            failed_batches=1,
            total_batches=5,
            errors=["Error 1"]
        )

        assert len(result.data) == 1
        assert result.failed_batches == 1
        assert result.total_batches == 5
        assert result.partial_failure is True
        assert result.total_failure is False
        assert len(result.errors) == 1

    def test_exchanges_list_structure(self):
        """Test that exchanges list has correct structure."""
        result = get_exchanges_list()

        # Check all keys exist
        required_keys = ["crypto", "us_stocks", "turkey", "malaysia", "hongkong", "timeframes"]
        for key in required_keys:
            assert key in result

        # Check values are lists
        for key in required_keys:
            assert isinstance(result[key], list)

        # Check timeframes are valid
        valid_timeframes = {"5m", "15m", "1h", "4h", "1D", "1W", "1M"}
        for tf in result["timeframes"]:
            assert tf in valid_timeframes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
