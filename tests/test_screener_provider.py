"""
Unit tests for the screener_provider module.

Tests cover:
- Timeframe resolution conversion
- fetch_screener_indicators function
- fetch_screener_multi_changes function
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from tradingview_mcp.core.services.screener_provider import (
    _tf_to_tv_resolution,
    fetch_screener_indicators,
    fetch_screener_multi_changes,
)


def create_mock_query(return_data):
    """Helper to create a mock Query object with chained methods."""
    mock_query_instance = MagicMock()
    mock_query_instance.set_markets.return_value = mock_query_instance
    mock_query_instance.select.return_value = mock_query_instance
    mock_query_instance.set_tickers.return_value = mock_query_instance
    mock_query_instance.where.return_value = mock_query_instance
    mock_query_instance.limit.return_value = mock_query_instance
    mock_query_instance.get_scanner_data.return_value = return_data
    return mock_query_instance


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


# =============================================================================
# Tests for fetch_screener_indicators
# =============================================================================

class TestFetchScreenerIndicators:
    """Tests for the fetch_screener_indicators function."""

    def test_returns_empty_list_on_empty_dataframe(self):
        """Test returns empty list when DataFrame is empty."""
        mock_query = create_mock_query((0, pd.DataFrame()))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_indicators("kucoin", symbols=["KUCOIN:BTCUSDT"])
                assert result == []

    def test_returns_empty_list_on_none_dataframe(self):
        """Test returns empty list when DataFrame is None."""
        mock_query = create_mock_query((0, None))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_indicators("kucoin")
                assert result == []

    def test_processes_dataframe_rows(self):
        """Test that DataFrame rows are processed correctly."""
        mock_df = pd.DataFrame({
            'ticker': ['KUCOIN:BTCUSDT', 'KUCOIN:ETHUSDT'],
            'open': [100.0, 200.0],
            'close': [105.0, 210.0],
            'SMA20': [102.0, 205.0],
            'BB.upper': [110.0, 220.0],
            'BB.lower': [94.0, 190.0],
            'EMA50': [101.0, 202.0],
            'RSI': [65.0, 55.0],
            'volume': [1000000.0, 500000.0],
        })

        mock_query = create_mock_query((2, mock_df))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_indicators("kucoin", symbols=["KUCOIN:BTCUSDT", "KUCOIN:ETHUSDT"])

                assert len(result) == 2
                assert result[0]['symbol'] == 'KUCOIN:BTCUSDT'
                assert result[0]['indicators']['open'] == 100.0
                assert result[1]['symbol'] == 'KUCOIN:ETHUSDT'

    def test_applies_timeframe_suffix(self):
        """Test that timeframe suffix is applied to columns."""
        mock_df = pd.DataFrame({
            'ticker': ['KUCOIN:BTCUSDT'],
            'open|240': [100.0],
            'close|240': [105.0],
            'SMA20|240': [102.0],
            'BB.upper|240': [110.0],
            'BB.lower|240': [94.0],
            'EMA50|240': [101.0],
            'RSI|240': [65.0],
            'volume|240': [1000000.0],
        })

        mock_query = create_mock_query((1, mock_df))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_indicators("kucoin", symbols=["KUCOIN:BTCUSDT"], timeframe="4h")

                assert len(result) == 1
                # Column names should be normalized back to base names
                assert result[0]['indicators']['open'] == 100.0
                assert result[0]['indicators']['close'] == 105.0


# =============================================================================
# Tests for fetch_screener_multi_changes
# =============================================================================

class TestFetchScreenerMultiChanges:
    """Tests for the fetch_screener_multi_changes function."""

    def test_returns_empty_on_empty_dataframe(self):
        """Test returns empty list when DataFrame is empty."""
        mock_query = create_mock_query((0, pd.DataFrame()))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_multi_changes("kucoin")
                assert result == []

    def test_default_timeframes(self):
        """Test that default timeframes are used when none specified."""
        mock_query = create_mock_query((0, pd.DataFrame()))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                # Should not raise, uses default timeframes
                result = fetch_screener_multi_changes("kucoin", timeframes=None)
                assert result == []

    def test_calculates_percentage_changes(self):
        """Test that percentage changes are calculated correctly."""
        mock_df = pd.DataFrame({
            'ticker': ['KUCOIN:BTCUSDT'],
            'open|60': [100.0],
            'close|60': [110.0],
            'open|240': [95.0],
            'close|240': [105.0],
            'SMA20|240': [100.0],
            'BB.upper|240': [110.0],
            'BB.lower|240': [90.0],
            'volume|240': [1000000.0],
        })

        mock_query = create_mock_query((1, mock_df))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_multi_changes(
                    "kucoin",
                    timeframes=['1h', '4h'],
                    base_timeframe='4h'
                )

                assert len(result) == 1
                assert result[0]['symbol'] == 'KUCOIN:BTCUSDT'
                # 1h: (110 - 100) / 100 * 100 = 10%
                assert result[0]['changes']['1h'] == 10.0
                # 4h: (105 - 95) / 95 * 100 ≈ 10.526%
                assert abs(result[0]['changes']['4h'] - 10.526) < 0.01

    def test_handles_zero_open_price(self):
        """Test that zero open price returns None for change."""
        mock_df = pd.DataFrame({
            'ticker': ['KUCOIN:BTCUSDT'],
            'open|240': [0.0],
            'close|240': [105.0],
            'SMA20|240': [100.0],
            'BB.upper|240': [110.0],
            'BB.lower|240': [90.0],
            'volume|240': [1000000.0],
        })

        mock_query = create_mock_query((1, mock_df))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_multi_changes(
                    "kucoin",
                    timeframes=['4h'],
                    base_timeframe='4h'
                )

                assert len(result) == 1
                # Zero open price should result in None change
                assert result[0]['changes']['4h'] is None

    def test_includes_base_indicators(self):
        """Test that base indicators are included in result."""
        mock_df = pd.DataFrame({
            'ticker': ['KUCOIN:BTCUSDT'],
            'open|240': [100.0],
            'close|240': [105.0],
            'SMA20|240': [102.0],
            'BB.upper|240': [110.0],
            'BB.lower|240': [94.0],
            'volume|240': [1000000.0],
        })

        mock_query = create_mock_query((1, mock_df))
        mock_query_class = MagicMock(return_value=mock_query)

        with patch('tradingview_screener.Query', mock_query_class):
            with patch('tradingview_screener.column.Column'):
                result = fetch_screener_multi_changes(
                    "kucoin",
                    timeframes=['4h'],
                    base_timeframe='4h'
                )

                assert len(result) == 1
                assert 'base_indicators' in result[0]
                assert result[0]['base_indicators']['open'] == 100.0
                assert result[0]['base_indicators']['close'] == 105.0
                assert result[0]['base_indicators']['SMA20'] == 102.0
                assert result[0]['base_indicators']['volume'] == 1000000.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestScreenerProviderIntegration:
    """Integration tests for screener_provider module."""

    def test_timeframe_resolution_used_in_columns(self):
        """Test that timeframe resolution is applied to column names."""
        # This tests the internal logic without actually calling the API
        from tradingview_mcp.core.utils.validators import tf_to_tv_resolution

        # When timeframe is 4h, resolution should be 240
        resolution = tf_to_tv_resolution("4h")
        assert resolution == "240"

        # Column names should use this suffix
        expected_column = f"close|{resolution}"
        assert expected_column == "close|240"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
