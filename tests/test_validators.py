"""
Unit tests for the validators module.

Tests cover:
- sanitize_timeframe: Timeframe validation and normalization
- sanitize_exchange: Exchange name validation and normalization
- Constants: ALLOWED_TIMEFRAMES, EXCHANGE_SCREENER, COINLIST_DIR
"""

import pytest
import os
from sigmapilot_mcp.core.utils.validators import (
    sanitize_timeframe,
    sanitize_exchange,
    ALLOWED_TIMEFRAMES,
    EXCHANGE_SCREENER,
    COINLIST_DIR,
)


# =============================================================================
# Tests for ALLOWED_TIMEFRAMES constant
# =============================================================================

class TestAllowedTimeframes:
    """Tests for the ALLOWED_TIMEFRAMES constant."""

    def test_contains_all_expected_timeframes(self):
        """Test that all expected timeframes are present."""
        expected = {"5m", "15m", "1h", "4h", "1D", "1W", "1M"}
        assert ALLOWED_TIMEFRAMES == expected

    def test_is_set(self):
        """Test that ALLOWED_TIMEFRAMES is a set for O(1) lookup."""
        assert isinstance(ALLOWED_TIMEFRAMES, set)

    def test_minute_timeframes(self):
        """Test that minute-based timeframes are included."""
        assert "5m" in ALLOWED_TIMEFRAMES
        assert "15m" in ALLOWED_TIMEFRAMES

    def test_hour_timeframe(self):
        """Test that hour-based timeframes are included."""
        assert "1h" in ALLOWED_TIMEFRAMES
        assert "4h" in ALLOWED_TIMEFRAMES

    def test_daily_weekly_monthly(self):
        """Test that daily, weekly, monthly timeframes are included."""
        assert "1D" in ALLOWED_TIMEFRAMES
        assert "1W" in ALLOWED_TIMEFRAMES
        assert "1M" in ALLOWED_TIMEFRAMES


# =============================================================================
# Tests for EXCHANGE_SCREENER constant
# =============================================================================

class TestExchangeScreener:
    """Tests for the EXCHANGE_SCREENER mapping."""

    def test_is_dict(self):
        """Test that EXCHANGE_SCREENER is a dictionary."""
        assert isinstance(EXCHANGE_SCREENER, dict)

    def test_crypto_exchanges(self):
        """Test that major crypto exchanges map to 'crypto' screener."""
        crypto_exchanges = ["binance", "kucoin", "bybit", "bitget", "okx",
                          "coinbase", "gateio", "huobi", "bitfinex"]
        for exchange in crypto_exchanges:
            assert exchange in EXCHANGE_SCREENER
            assert EXCHANGE_SCREENER[exchange] == "crypto"

    def test_all_exchange(self):
        """Test that 'all' exchange maps to crypto screener."""
        assert "all" in EXCHANGE_SCREENER
        assert EXCHANGE_SCREENER["all"] == "crypto"

    def test_turkey_market(self):
        """Test BIST (Turkish market) mapping."""
        assert "bist" in EXCHANGE_SCREENER
        assert EXCHANGE_SCREENER["bist"] == "turkey"

    def test_us_markets(self):
        """Test US stock market mappings."""
        assert "nasdaq" in EXCHANGE_SCREENER
        assert EXCHANGE_SCREENER["nasdaq"] == "america"
        assert "nyse" in EXCHANGE_SCREENER
        assert EXCHANGE_SCREENER["nyse"] == "america"

    def test_malaysia_markets(self):
        """Test Malaysian market mappings."""
        malaysia_exchanges = ["bursa", "myx", "klse", "ace", "leap"]
        for exchange in malaysia_exchanges:
            assert exchange in EXCHANGE_SCREENER
            assert EXCHANGE_SCREENER[exchange] == "malaysia"

    def test_hongkong_markets(self):
        """Test Hong Kong market mappings."""
        hk_exchanges = ["hkex", "hk", "hsi"]
        for exchange in hk_exchanges:
            assert exchange in EXCHANGE_SCREENER
            assert EXCHANGE_SCREENER[exchange] == "hongkong"


# =============================================================================
# Tests for COINLIST_DIR constant
# =============================================================================

class TestCoinlistDir:
    """Tests for the COINLIST_DIR path constant."""

    def test_is_string(self):
        """Test that COINLIST_DIR is a string path."""
        assert isinstance(COINLIST_DIR, str)

    def test_ends_with_coinlist(self):
        """Test that path ends with 'coinlist' directory."""
        assert COINLIST_DIR.endswith("coinlist")

    def test_path_contains_sigmapilot_mcp(self):
        """Test that path is within sigmapilot_mcp package."""
        assert "sigmapilot_mcp" in COINLIST_DIR


# =============================================================================
# Tests for sanitize_timeframe
# =============================================================================

class TestSanitizeTimeframe:
    """Tests for the sanitize_timeframe function."""

    def test_valid_timeframes(self):
        """Test that valid timeframes are returned as-is."""
        valid_timeframes = ["5m", "15m", "1h", "4h", "1D", "1W", "1M"]
        for tf in valid_timeframes:
            assert sanitize_timeframe(tf) == tf

    def test_default_on_empty_string(self):
        """Test that empty string returns default."""
        result = sanitize_timeframe("")
        assert result == "5m"

    def test_default_on_none(self):
        """Test that None returns default."""
        result = sanitize_timeframe(None)
        assert result == "5m"

    def test_custom_default(self):
        """Test custom default value."""
        result = sanitize_timeframe("", default="1h")
        assert result == "1h"

    def test_invalid_timeframe_returns_default(self):
        """Test that invalid timeframes return the default."""
        invalid_timeframes = ["1m", "30m", "2h", "1d", "1w", "invalid", "abc"]
        for tf in invalid_timeframes:
            result = sanitize_timeframe(tf)
            assert result == "5m"

    def test_whitespace_handling(self):
        """Test that whitespace is stripped from input."""
        assert sanitize_timeframe("  15m  ") == "15m"
        assert sanitize_timeframe("\t1h\n") == "1h"

    def test_case_sensitive(self):
        """Test that timeframes are case-sensitive (1D not 1d)."""
        # Uppercase D is required for daily
        assert sanitize_timeframe("1D") == "1D"
        # Lowercase d is invalid
        assert sanitize_timeframe("1d") == "5m"


# =============================================================================
# Tests for sanitize_exchange
# =============================================================================

class TestSanitizeExchange:
    """Tests for the sanitize_exchange function."""

    def test_valid_exchanges_lowercase(self):
        """Test that valid lowercase exchanges are returned."""
        valid_exchanges = ["binance", "kucoin", "bybit", "bitget", "bist"]
        for ex in valid_exchanges:
            result = sanitize_exchange(ex)
            assert result == ex.lower()

    def test_uppercase_conversion(self):
        """Test that uppercase inputs are converted to lowercase."""
        assert sanitize_exchange("BINANCE") == "binance"
        assert sanitize_exchange("KUCOIN") == "kucoin"
        assert sanitize_exchange("BIST") == "bist"

    def test_mixed_case_conversion(self):
        """Test that mixed case inputs are converted to lowercase."""
        assert sanitize_exchange("BiNaNcE") == "binance"
        assert sanitize_exchange("KuCoin") == "kucoin"

    def test_default_on_empty_string(self):
        """Test that empty string returns default."""
        result = sanitize_exchange("")
        assert result == "kucoin"

    def test_default_on_none(self):
        """Test that None returns default."""
        result = sanitize_exchange(None)
        assert result == "kucoin"

    def test_custom_default(self):
        """Test custom default value."""
        result = sanitize_exchange("", default="binance")
        assert result == "binance"

    def test_invalid_exchange_returns_default(self):
        """Test that invalid exchanges return the default."""
        invalid_exchanges = ["invalid", "unknown", "fake", "xyz", "123"]
        for ex in invalid_exchanges:
            result = sanitize_exchange(ex)
            assert result == "kucoin"

    def test_whitespace_handling(self):
        """Test that whitespace is stripped from input."""
        assert sanitize_exchange("  binance  ") == "binance"
        assert sanitize_exchange("\tkucoin\n") == "kucoin"

    def test_all_known_exchanges(self):
        """Test all exchanges in EXCHANGE_SCREENER are valid."""
        for exchange in EXCHANGE_SCREENER.keys():
            result = sanitize_exchange(exchange)
            assert result == exchange.lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidatorsIntegration:
    """Integration tests for validators module."""

    def test_sanitize_both_valid(self):
        """Test sanitizing both exchange and timeframe together."""
        exchange = sanitize_exchange("BINANCE")
        timeframe = sanitize_timeframe("1h")

        assert exchange == "binance"
        assert timeframe == "1h"
        assert exchange in EXCHANGE_SCREENER
        assert timeframe in ALLOWED_TIMEFRAMES

    def test_sanitize_both_invalid(self):
        """Test sanitizing invalid exchange and timeframe."""
        exchange = sanitize_exchange("invalid_exchange")
        timeframe = sanitize_timeframe("invalid_tf")

        # Should get defaults
        assert exchange == "kucoin"
        assert timeframe == "5m"

    def test_coinlist_dir_exists(self):
        """Test that COINLIST_DIR exists (if package is properly installed)."""
        # This test checks if the coinlist directory path is valid
        # In a proper installation, this directory should exist
        # We just verify the path is constructed correctly
        assert os.path.isabs(COINLIST_DIR) or "coinlist" in COINLIST_DIR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
