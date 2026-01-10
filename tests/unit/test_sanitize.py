"""
Tests for the sanitize module.

Tests cover:
- Timeframe sanitization
- Exchange sanitization
- Symbol sanitization
- Limit sanitization
- OHLCV data validation
"""

import pytest
from sigmapilot_mcp.core.sanitize import (
    sanitize_timeframe,
    sanitize_exchange,
    sanitize_symbol,
    sanitize_limit,
    parse_symbol,
    get_screener_for_exchange,
    validate_ohlcv_data,
    validate_ohlcv_series,
    ALLOWED_TIMEFRAMES,
    EXCHANGE_SCREENER,
    DEFAULT_TIMEFRAME,
    DEFAULT_EXCHANGE,
    DEFAULT_LIMIT,
    MAX_LIMIT,
)


class TestSanitizeTimeframe:
    """Tests for sanitize_timeframe function."""

    def test_valid_timeframes(self):
        """Test that valid timeframes are returned unchanged."""
        for tf in ALLOWED_TIMEFRAMES:
            assert sanitize_timeframe(tf) == tf

    def test_invalid_timeframe_returns_default(self):
        """Test that invalid timeframes return default."""
        assert sanitize_timeframe("invalid") == DEFAULT_TIMEFRAME
        assert sanitize_timeframe("2h") == DEFAULT_TIMEFRAME
        assert sanitize_timeframe("1d") == DEFAULT_TIMEFRAME  # Case sensitive

    def test_none_returns_default(self):
        """Test that None returns default."""
        assert sanitize_timeframe(None) == DEFAULT_TIMEFRAME

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        assert sanitize_timeframe("") == DEFAULT_TIMEFRAME

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        assert sanitize_timeframe("  1h  ") == "1h"
        assert sanitize_timeframe("\t4h\n") == "4h"

    def test_custom_default(self):
        """Test custom default value."""
        assert sanitize_timeframe("invalid", default="1D") == "1D"
        assert sanitize_timeframe(None, default="4h") == "4h"


class TestSanitizeExchange:
    """Tests for sanitize_exchange function."""

    def test_valid_exchanges(self):
        """Test that valid exchanges are returned in lowercase."""
        assert sanitize_exchange("binance") == "binance"
        assert sanitize_exchange("BINANCE") == "binance"
        assert sanitize_exchange("Binance") == "binance"

    def test_all_known_exchanges(self):
        """Test all known exchanges work."""
        for exchange in EXCHANGE_SCREENER.keys():
            result = sanitize_exchange(exchange)
            assert result == exchange.lower()

    def test_invalid_exchange_returns_default(self):
        """Test that invalid exchanges return default."""
        assert sanitize_exchange("unknown") == DEFAULT_EXCHANGE
        assert sanitize_exchange("xyz") == DEFAULT_EXCHANGE

    def test_none_returns_default(self):
        """Test that None returns default."""
        assert sanitize_exchange(None) == DEFAULT_EXCHANGE

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        assert sanitize_exchange("") == DEFAULT_EXCHANGE

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        assert sanitize_exchange("  binance  ") == "binance"
        assert sanitize_exchange("\tkucoin\n") == "kucoin"

    def test_custom_default(self):
        """Test custom default value."""
        assert sanitize_exchange("invalid", default="kucoin") == "kucoin"


class TestGetScreenerForExchange:
    """Tests for get_screener_for_exchange function."""

    def test_crypto_exchanges(self):
        """Test crypto exchanges return crypto screener."""
        assert get_screener_for_exchange("binance") == "crypto"
        assert get_screener_for_exchange("kucoin") == "crypto"
        assert get_screener_for_exchange("bybit") == "crypto"

    def test_stock_exchanges(self):
        """Test stock exchanges return correct screener."""
        assert get_screener_for_exchange("nasdaq") == "america"
        assert get_screener_for_exchange("nyse") == "america"
        assert get_screener_for_exchange("bist") == "turkey"

    def test_malaysia_exchanges(self):
        """Test Malaysia exchanges."""
        assert get_screener_for_exchange("bursa") == "malaysia"
        assert get_screener_for_exchange("klse") == "malaysia"

    def test_hongkong_exchanges(self):
        """Test Hong Kong exchanges."""
        assert get_screener_for_exchange("hkex") == "hongkong"
        assert get_screener_for_exchange("hk") == "hongkong"


class TestSanitizeSymbol:
    """Tests for sanitize_symbol function."""

    def test_basic_symbol(self):
        """Test basic symbol conversion."""
        assert sanitize_symbol("btcusdt") == "BTCUSDT"
        assert sanitize_symbol("BTCUSDT") == "BTCUSDT"

    def test_symbol_with_slash(self):
        """Test symbols with slash separator."""
        assert sanitize_symbol("BTC/USDT") == "BTCUSDT"
        assert sanitize_symbol("btc/usdt") == "BTCUSDT"

    def test_symbol_with_dash(self):
        """Test symbols with dash separator."""
        assert sanitize_symbol("BTC-USDT") == "BTCUSDT"

    def test_symbol_with_underscore(self):
        """Test symbols with underscore separator."""
        assert sanitize_symbol("BTC_USDT") == "BTCUSDT"

    def test_symbol_with_exchange_prefix(self):
        """Test symbols with exchange prefix are preserved."""
        assert sanitize_symbol("BINANCE:BTCUSDT") == "BINANCE:BTCUSDT"
        assert sanitize_symbol("binance:btcusdt") == "BINANCE:BTCUSDT"
        assert sanitize_symbol("BINANCE:BTC/USDT") == "BINANCE:BTCUSDT"

    def test_none_returns_empty(self):
        """Test that None returns empty string."""
        assert sanitize_symbol(None) == ""

    def test_empty_string(self):
        """Test empty string handling."""
        assert sanitize_symbol("") == ""

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert sanitize_symbol("  BTCUSDT  ") == "BTCUSDT"


class TestParseSymbol:
    """Tests for parse_symbol function."""

    def test_symbol_with_exchange(self):
        """Test parsing symbol with exchange prefix."""
        exchange, pair = parse_symbol("BINANCE:BTCUSDT")
        assert exchange == "BINANCE"
        assert pair == "BTCUSDT"

    def test_symbol_without_exchange(self):
        """Test parsing symbol without exchange prefix."""
        exchange, pair = parse_symbol("BTCUSDT")
        assert exchange is None
        assert pair == "BTCUSDT"

    def test_lowercase_input(self):
        """Test lowercase input is uppercased."""
        exchange, pair = parse_symbol("kucoin:ethusdt")
        assert exchange == "KUCOIN"
        assert pair == "ETHUSDT"


class TestSanitizeLimit:
    """Tests for sanitize_limit function."""

    def test_valid_limit(self):
        """Test valid limits are returned unchanged."""
        assert sanitize_limit(25) == 25
        assert sanitize_limit(50) == 50

    def test_limit_clamped_to_max(self):
        """Test limits are clamped to max."""
        assert sanitize_limit(200) == MAX_LIMIT
        assert sanitize_limit(500) == MAX_LIMIT

    def test_limit_clamped_to_min(self):
        """Test limits are clamped to min."""
        assert sanitize_limit(0) == 1
        assert sanitize_limit(-5) == 1

    def test_none_returns_default(self):
        """Test None returns default."""
        assert sanitize_limit(None) == DEFAULT_LIMIT

    def test_custom_max(self):
        """Test custom max limit."""
        assert sanitize_limit(75, max_limit=50) == 50
        assert sanitize_limit(30, max_limit=50) == 30

    def test_custom_min(self):
        """Test custom min limit."""
        assert sanitize_limit(3, min_limit=5) == 5
        assert sanitize_limit(10, min_limit=5) == 10

    def test_float_converted_to_int(self):
        """Test float is converted to int."""
        assert sanitize_limit(25.7) == 25
        assert sanitize_limit(25.2) == 25

    def test_invalid_type_returns_default(self):
        """Test invalid type returns default."""
        assert sanitize_limit("invalid") == DEFAULT_LIMIT


class TestValidateOHLCVData:
    """Tests for validate_ohlcv_data function."""

    def test_valid_ohlcv(self):
        """Test validation of valid OHLCV data."""
        data = {"open": 100, "high": 110, "low": 95, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is True
        assert error is None

    def test_valid_ohlcv_with_volume(self):
        """Test validation with volume."""
        data = {"open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000000}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is True

    def test_missing_open(self):
        """Test validation catches missing open."""
        data = {"high": 110, "low": 95, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "open" in error

    def test_missing_high(self):
        """Test validation catches missing high."""
        data = {"open": 100, "low": 95, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "high" in error

    def test_none_value(self):
        """Test validation catches None values."""
        data = {"open": 100, "high": None, "low": 95, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "None" in error

    def test_non_numeric_value(self):
        """Test validation catches non-numeric values."""
        data = {"open": "100", "high": 110, "low": 95, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "numeric" in error

    def test_negative_value(self):
        """Test validation catches negative values."""
        data = {"open": 100, "high": 110, "low": -5, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "negative" in error

    def test_high_less_than_low(self):
        """Test validation catches high < low."""
        data = {"open": 100, "high": 90, "low": 95, "close": 100}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "High" in error and "Low" in error

    def test_high_less_than_open(self):
        """Test validation catches high < open."""
        data = {"open": 100, "high": 95, "low": 90, "close": 92}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False

    def test_low_greater_than_close(self):
        """Test validation catches low > close."""
        data = {"open": 100, "high": 110, "low": 108, "close": 105}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False

    def test_negative_volume(self):
        """Test validation catches negative volume."""
        data = {"open": 100, "high": 110, "low": 95, "close": 105, "volume": -1000}
        is_valid, error = validate_ohlcv_data(data)
        assert is_valid is False
        assert "Volume" in error


class TestValidateOHLCVSeries:
    """Tests for validate_ohlcv_series function."""

    def test_valid_series(self):
        """Test validation of valid OHLCV series."""
        is_valid, error = validate_ohlcv_series(
            opens=[100, 102],
            highs=[110, 112],
            lows=[95, 97],
            closes=[105, 108]
        )
        assert is_valid is True
        assert error is None

    def test_valid_series_with_volume(self):
        """Test validation with volume series."""
        is_valid, error = validate_ohlcv_series(
            opens=[100, 102],
            highs=[110, 112],
            lows=[95, 97],
            closes=[105, 108],
            volumes=[1000, 2000]
        )
        assert is_valid is True

    def test_mismatched_lengths(self):
        """Test validation catches mismatched lengths."""
        is_valid, error = validate_ohlcv_series(
            opens=[100, 102, 104],  # 3 elements
            highs=[110, 112],       # 2 elements
            lows=[95, 97],
            closes=[105, 108]
        )
        assert is_valid is False
        assert "length" in error.lower()

    def test_mismatched_volume_length(self):
        """Test validation catches mismatched volume length."""
        is_valid, error = validate_ohlcv_series(
            opens=[100, 102],
            highs=[110, 112],
            lows=[95, 97],
            closes=[105, 108],
            volumes=[1000]  # Wrong length
        )
        assert is_valid is False
        assert "Volume" in error

    def test_empty_series(self):
        """Test validation catches empty series."""
        is_valid, error = validate_ohlcv_series(
            opens=[],
            highs=[],
            lows=[],
            closes=[]
        )
        assert is_valid is False
        assert "Empty" in error
