"""
Pytest configuration and shared fixtures for SigmaPilot MCP tests.

This module provides common fixtures and configuration used across all test modules.
"""

import pytest
import os
import tempfile


# =============================================================================
# Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )


# =============================================================================
# Common Fixtures
# =============================================================================

@pytest.fixture
def sample_indicators():
    """Provide sample indicator data for testing."""
    return {
        "open": 100.0,
        "close": 105.0,
        "high": 108.0,
        "low": 98.0,
        "SMA20": 102.0,
        "BB.upper": 110.0,
        "BB.lower": 94.0,
        "EMA9": 103.0,
        "EMA21": 101.5,
        "EMA50": 99.0,
        "EMA200": 95.0,
        "RSI": 65.0,
        "ATR": 2.5,
        "MACD.macd": 1.5,
        "MACD.signal": 1.2,
        "ADX": 28.0,
        "Stoch.K": 75.0,
        "Stoch.D": 70.0,
        "volume": 1000000.0,
    }


@pytest.fixture
def sample_bullish_indicators():
    """Provide sample bullish indicator data."""
    return {
        "open": 100.0,
        "close": 115.0,  # Strong bullish move
        "high": 116.0,
        "low": 99.0,
        "SMA20": 100.0,
        "BB.upper": 110.0,
        "BB.lower": 90.0,
        "EMA9": 112.0,
        "EMA21": 108.0,
        "EMA50": 102.0,
        "RSI": 75.0,  # Overbought
        "ATR": 4.0,
        "volume": 2500000.0,  # High volume
    }


@pytest.fixture
def sample_bearish_indicators():
    """Provide sample bearish indicator data."""
    return {
        "open": 100.0,
        "close": 88.0,  # Strong bearish move
        "high": 101.0,
        "low": 87.0,
        "SMA20": 100.0,
        "BB.upper": 110.0,
        "BB.lower": 90.0,
        "EMA9": 90.0,
        "EMA21": 94.0,
        "EMA50": 98.0,
        "RSI": 25.0,  # Oversold
        "ATR": 5.0,
        "volume": 3000000.0,  # High volume on sell-off
    }


@pytest.fixture
def sample_squeeze_indicators():
    """Provide sample Bollinger Band squeeze indicators."""
    return {
        "open": 100.0,
        "close": 100.5,
        "high": 101.0,
        "low": 99.5,
        "SMA20": 100.0,
        "BB.upper": 102.0,  # Very tight bands
        "BB.lower": 98.0,
        "EMA9": 100.2,
        "EMA21": 100.0,
        "EMA50": 99.8,
        "RSI": 50.0,  # Neutral
        "ATR": 0.8,  # Low volatility
        "volume": 500000.0,
    }


@pytest.fixture
def temp_coinlist_dir():
    """Create a temporary directory with mock coinlist files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock exchange files
        exchanges = {
            "kucoin": ["KUCOIN:BTCUSDT", "KUCOIN:ETHUSDT", "KUCOIN:SOLUSDT"],
            "binance": ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT"],
            "bist": ["BIST:AKBNK", "BIST:THYAO", "BIST:ISCTR"],
        }

        for exchange, symbols in exchanges.items():
            filepath = os.path.join(tmpdir, f"{exchange}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(symbols))

        yield tmpdir


@pytest.fixture
def mock_tradingview_response():
    """Provide mock TradingView API response data."""
    class MockIndicators:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key, default)

    class MockAnalysis:
        def __init__(self, indicators_data):
            self.indicators = indicators_data

    return {
        "KUCOIN:BTCUSDT": MockAnalysis({
            "open": 42000.0,
            "close": 43500.0,
            "high": 43800.0,
            "low": 41800.0,
            "SMA20": 42500.0,
            "BB.upper": 44000.0,
            "BB.lower": 41000.0,
            "EMA9": 43200.0,
            "EMA21": 42800.0,
            "EMA50": 42000.0,
            "RSI": 68.0,
            "ATR": 800.0,
            "volume": 15000000.0,
        }),
        "KUCOIN:ETHUSDT": MockAnalysis({
            "open": 2200.0,
            "close": 2280.0,
            "high": 2300.0,
            "low": 2180.0,
            "SMA20": 2250.0,
            "BB.upper": 2350.0,
            "BB.lower": 2150.0,
            "EMA9": 2270.0,
            "EMA21": 2240.0,
            "EMA50": 2200.0,
            "RSI": 62.0,
            "ATR": 50.0,
            "volume": 8000000.0,
        }),
    }


# =============================================================================
# Helper Functions for Tests
# =============================================================================

@pytest.fixture
def assert_valid_metrics():
    """Fixture providing a function to validate metrics structure."""
    def _assert_valid_metrics(metrics):
        """Assert that metrics dict has expected structure."""
        assert metrics is not None
        assert "price" in metrics
        assert "change" in metrics
        assert "bbw" in metrics
        assert "rating" in metrics
        assert "signal" in metrics

        assert isinstance(metrics["price"], (int, float))
        assert isinstance(metrics["change"], (int, float))
        assert metrics["bbw"] is None or isinstance(metrics["bbw"], (int, float))
        assert isinstance(metrics["rating"], int)
        assert metrics["rating"] >= -3 and metrics["rating"] <= 3
        assert metrics["signal"] in ("BUY", "SELL", "NEUTRAL")

    return _assert_valid_metrics


@pytest.fixture
def assert_valid_row():
    """Fixture providing a function to validate Row structure."""
    def _assert_valid_row(row):
        """Assert that row dict has expected structure."""
        assert "symbol" in row
        assert "changePercent" in row
        assert "indicators" in row

        assert isinstance(row["symbol"], str)
        assert isinstance(row["changePercent"], (int, float))
        assert isinstance(row["indicators"], dict)

    return _assert_valid_row
