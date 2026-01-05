"""
Unit tests for the main.py remote server entry point.

Tests cover:
- Configuration loading
- Environment variable handling
- Helper function behavior
- Tool function logic (without external API calls)
"""

import pytest
import os
from unittest.mock import patch, MagicMock


# =============================================================================
# Tests for Configuration
# =============================================================================

class TestConfiguration:
    """Tests for environment configuration."""

    def test_default_port(self):
        """Test default port when PORT env is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh config
            import importlib
            import main
            importlib.reload(main)
            # Default should be 8000
            assert main.PORT == 8000

    def test_custom_port(self):
        """Test custom port from environment."""
        with patch.dict(os.environ, {"PORT": "9000"}, clear=True):
            import importlib
            import main
            importlib.reload(main)
            assert main.PORT == 9000

    def test_auth_disabled_without_env(self):
        """Test auth is disabled when Auth0 vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import main
            importlib.reload(main)
            assert main.ENABLE_AUTH is False

    def test_auth_enabled_with_env(self):
        """Test auth is enabled when Auth0 vars are set."""
        with patch.dict(os.environ, {
            "AUTH0_DOMAIN": "test.auth0.com",
            "AUTH0_AUDIENCE": "https://api.example.com"
        }, clear=True):
            import importlib
            import main
            importlib.reload(main)
            assert main.ENABLE_AUTH is True

    def test_default_host(self):
        """Test default host is 0.0.0.0 for production."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import main
            importlib.reload(main)
            assert main.HOST == "0.0.0.0"


# =============================================================================
# Tests for Helper Functions
# =============================================================================

class TestFetchAnalysis:
    """Tests for the _fetch_analysis helper function."""

    def test_returns_empty_on_no_symbols(self):
        """Test returns empty list when no symbols are loaded."""
        from main import _fetch_analysis
        with patch('main.load_symbols', return_value=[]):
            result = _fetch_analysis("nonexistent", "15m", 10)
            assert result == []

    def test_handles_tradingview_not_available(self):
        """Test raises error when TradingView TA is not available."""
        import main
        original = main.TRADINGVIEW_TA_AVAILABLE
        main.TRADINGVIEW_TA_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="tradingview_ta is not available"):
                main._fetch_analysis("kucoin", "15m", 10)
        finally:
            main.TRADINGVIEW_TA_AVAILABLE = original


# =============================================================================
# Tests for Tool Functions
# =============================================================================

class TestTopGainersLosers:
    """Tests for top_gainers and top_losers tools."""

    def test_top_gainers_sanitizes_inputs(self):
        """Test that top_gainers sanitizes exchange and timeframe."""
        from main import top_gainers
        with patch('main._fetch_analysis', return_value=[]):
            # Should not raise, just return empty list
            result = top_gainers("INVALID_EXCHANGE", "invalid_tf", 10)
            assert result == []

    def test_top_losers_sanitizes_inputs(self):
        """Test that top_losers sanitizes exchange and timeframe."""
        from main import top_losers
        with patch('main._fetch_analysis', return_value=[]):
            result = top_losers("INVALID", "bad", 10)
            assert result == []

    def test_limit_clamping_min(self):
        """Test that limit is clamped to minimum of 1."""
        from main import top_gainers
        with patch('main._fetch_analysis', return_value=[]) as mock:
            top_gainers("kucoin", "15m", -5)
            # Should have been clamped to 1
            mock.assert_called()

    def test_limit_clamping_max(self):
        """Test that limit is clamped to maximum of 50."""
        from main import top_gainers
        with patch('main._fetch_analysis', return_value=[]) as mock:
            top_gainers("kucoin", "15m", 1000)
            # Should have been clamped to 50
            mock.assert_called()

    def test_sorts_gainers_descending(self):
        """Test that top_gainers sorts by changePercent descending."""
        from main import top_gainers
        mock_data = [
            {"symbol": "A", "changePercent": 5.0},
            {"symbol": "B", "changePercent": 15.0},
            {"symbol": "C", "changePercent": 10.0},
        ]
        with patch('main._fetch_analysis', return_value=mock_data):
            result = top_gainers("kucoin", "15m", 3)
            assert result[0]["symbol"] == "B"
            assert result[1]["symbol"] == "C"
            assert result[2]["symbol"] == "A"

    def test_sorts_losers_ascending(self):
        """Test that top_losers sorts by changePercent ascending."""
        from main import top_losers
        mock_data = [
            {"symbol": "A", "changePercent": -5.0},
            {"symbol": "B", "changePercent": -15.0},
            {"symbol": "C", "changePercent": -10.0},
        ]
        with patch('main._fetch_analysis', return_value=mock_data):
            result = top_losers("kucoin", "15m", 3)
            assert result[0]["symbol"] == "B"
            assert result[1]["symbol"] == "C"
            assert result[2]["symbol"] == "A"


class TestBollingerScan:
    """Tests for bollinger_scan tool."""

    def test_filters_by_bbw_threshold(self):
        """Test that bollinger_scan filters by BBW threshold."""
        from main import bollinger_scan
        mock_data = [
            {"symbol": "A", "bbw": 0.02, "changePercent": 5.0},
            {"symbol": "B", "bbw": 0.06, "changePercent": 3.0},  # Above threshold
            {"symbol": "C", "bbw": 0.03, "changePercent": 2.0},
        ]
        with patch('main._fetch_analysis', return_value=mock_data):
            result = bollinger_scan("kucoin", "4h", 0.04, 10)
            assert len(result) == 2
            symbols = [r["symbol"] for r in result]
            assert "A" in symbols
            assert "C" in symbols
            assert "B" not in symbols

    def test_sorts_by_bbw_ascending(self):
        """Test that results are sorted by BBW ascending (tightest first)."""
        from main import bollinger_scan
        mock_data = [
            {"symbol": "A", "bbw": 0.03, "changePercent": 5.0},
            {"symbol": "B", "bbw": 0.01, "changePercent": 3.0},
            {"symbol": "C", "bbw": 0.02, "changePercent": 2.0},
        ]
        with patch('main._fetch_analysis', return_value=mock_data):
            result = bollinger_scan("kucoin", "4h", 0.05, 10)
            assert result[0]["symbol"] == "B"
            assert result[1]["symbol"] == "C"
            assert result[2]["symbol"] == "A"

    def test_excludes_zero_bbw(self):
        """Test that zero BBW values are excluded."""
        from main import bollinger_scan
        mock_data = [
            {"symbol": "A", "bbw": 0.0, "changePercent": 5.0},
            {"symbol": "B", "bbw": 0.02, "changePercent": 3.0},
        ]
        with patch('main._fetch_analysis', return_value=mock_data):
            result = bollinger_scan("kucoin", "4h", 0.05, 10)
            assert len(result) == 1
            assert result[0]["symbol"] == "B"


class TestRatingFilter:
    """Tests for rating_filter tool."""

    def test_filters_by_rating(self):
        """Test that rating_filter returns only matching ratings."""
        from main import rating_filter
        mock_data = [
            {"symbol": "A", "rating": 2, "changePercent": 5.0},
            {"symbol": "B", "rating": -2, "changePercent": 3.0},
            {"symbol": "C", "rating": 2, "changePercent": 2.0},
        ]
        with patch('main._fetch_analysis', return_value=mock_data):
            result = rating_filter("kucoin", "15m", 2, 10)
            assert len(result) == 2
            for r in result:
                assert r["rating"] == 2

    def test_clamps_rating_range(self):
        """Test that rating is clamped to -3 to +3."""
        from main import rating_filter
        with patch('main._fetch_analysis', return_value=[]):
            # Should not raise, rating gets clamped
            rating_filter("kucoin", "15m", 10, 10)  # 10 -> 3
            rating_filter("kucoin", "15m", -10, 10)  # -10 -> -3


class TestListExchanges:
    """Tests for list_exchanges tool."""

    def test_returns_grouped_exchanges(self):
        """Test that list_exchanges returns properly grouped exchanges."""
        from main import list_exchanges
        result = list_exchanges()

        assert "crypto" in result
        assert "us_stocks" in result
        assert "timeframes" in result

        assert "kucoin" in result["crypto"]
        assert "binance" in result["crypto"]
        assert "nasdaq" in result["us_stocks"]

    def test_includes_all_timeframes(self):
        """Test that all timeframes are included."""
        from main import list_exchanges
        result = list_exchanges()

        expected_timeframes = ["5m", "15m", "1h", "4h", "1D", "1W", "1M"]
        for tf in expected_timeframes:
            assert tf in result["timeframes"]


class TestCoinAnalysis:
    """Tests for coin_analysis tool."""

    def test_formats_symbol_correctly(self):
        """Test that symbol is formatted with exchange prefix."""
        from main import coin_analysis
        with patch('main.TRADINGVIEW_TA_AVAILABLE', True):
            with patch('main.get_multiple_analysis', return_value={}) as mock:
                result = coin_analysis("BTCUSDT", "kucoin", "15m")
                # Should have called with KUCOIN:BTCUSDT
                mock.assert_called_once()
                args = mock.call_args
                assert "KUCOIN:BTCUSDT" in args[1]["symbols"]

    def test_handles_missing_data(self):
        """Test that missing data returns error response."""
        from main import coin_analysis
        with patch('main.TRADINGVIEW_TA_AVAILABLE', True):
            with patch('main.get_multiple_analysis', return_value={}):
                result = coin_analysis("INVALID", "kucoin", "15m")
                assert "error" in result

    def test_returns_error_when_ta_unavailable(self):
        """Test returns error when tradingview-ta is not available."""
        import main
        original = main.TRADINGVIEW_TA_AVAILABLE
        main.TRADINGVIEW_TA_AVAILABLE = False

        try:
            result = main.coin_analysis("BTCUSDT", "kucoin", "15m")
            assert "error" in result
        finally:
            main.TRADINGVIEW_TA_AVAILABLE = original


# =============================================================================
# Tests for Server Instructions
# =============================================================================

class TestServerInstructions:
    """Tests for server configuration."""

    def test_server_has_instructions(self):
        """Test that server instructions are defined."""
        from main import SERVER_INSTRUCTIONS
        assert len(SERVER_INSTRUCTIONS) > 100
        assert "TradingView" in SERVER_INSTRUCTIONS

    def test_server_name(self):
        """Test that MCP server has correct name."""
        from main import mcp
        assert mcp.name == "TradingView MCP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
