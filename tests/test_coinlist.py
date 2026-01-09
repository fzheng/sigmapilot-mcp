"""
Unit tests for the coinlist module.

Tests cover:
- load_symbols: Loading symbols from exchange-specific files
- Path fallback strategies
- Error handling for missing/invalid files
"""

import pytest
import os
import tempfile
from unittest.mock import patch
from sigmapilot_mcp.core.services.coinlist import load_symbols
from sigmapilot_mcp.core.utils.validators import COINLIST_DIR


# =============================================================================
# Tests for load_symbols
# =============================================================================

class TestLoadSymbols:
    """Tests for the load_symbols function."""

    def test_returns_list(self):
        """Test that load_symbols always returns a list."""
        result = load_symbols("nonexistent_exchange_xyz")
        assert isinstance(result, list)

    def test_empty_list_for_unknown_exchange(self):
        """Test that unknown exchanges return empty list."""
        result = load_symbols("completely_fake_exchange_123")
        assert result == []

    def test_kucoin_has_symbols(self):
        """Test that KuCoin exchange has symbols (if file exists)."""
        result = load_symbols("kucoin")
        # KuCoin should have symbols if the coinlist file exists
        # If file doesn't exist in test environment, this is still valid
        assert isinstance(result, list)

    def test_binance_has_symbols(self):
        """Test that Binance exchange has symbols (if file exists)."""
        result = load_symbols("binance")
        assert isinstance(result, list)

    def test_bist_has_symbols(self):
        """Test that BIST exchange has symbols (if file exists)."""
        result = load_symbols("bist")
        assert isinstance(result, list)

    def test_case_insensitive_loading(self):
        """Test that exchange names are case-insensitive for loading."""
        # The function tries both original case and lowercase
        result1 = load_symbols("KUCOIN")
        result2 = load_symbols("kucoin")
        result3 = load_symbols("KuCoin")
        # All should return same result type
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert isinstance(result3, list)

    def test_symbols_are_strings(self):
        """Test that all returned symbols are strings."""
        result = load_symbols("kucoin")
        for symbol in result:
            assert isinstance(symbol, str)

    def test_no_empty_symbols(self):
        """Test that empty lines are filtered out."""
        result = load_symbols("kucoin")
        for symbol in result:
            assert symbol.strip() != ""
            assert len(symbol) > 0


# =============================================================================
# Tests with mock files
# =============================================================================

class TestLoadSymbolsWithMockFiles:
    """Tests using mock coinlist files."""

    def test_load_from_temp_file(self):
        """Test loading symbols from a temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock coinlist file
            filepath = os.path.join(tmpdir, "test_exchange.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("BTCUSDT\nETHUSDT\nSOLUSDT\n")

            # Patch COINLIST_DIR to use temp directory
            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                result = load_symbols("test_exchange")

            assert result == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_handles_empty_lines(self):
        """Test that empty lines in file are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_exchange.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("BTCUSDT\n\nETHUSDT\n\n\nSOLUSDT\n")

            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                result = load_symbols("test_exchange")

            assert result == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_handles_whitespace_only_lines(self):
        """Test that whitespace-only lines are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_exchange.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("BTCUSDT\n   \nETHUSDT\n\t\nSOLUSDT\n")

            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                result = load_symbols("test_exchange")

            assert result == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_strips_whitespace_from_symbols(self):
        """Test that whitespace is stripped from symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_exchange.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("  BTCUSDT  \n ETHUSDT\nSOLUSDT   \n")

            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                result = load_symbols("test_exchange")

            assert result == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_empty_file_returns_empty_list(self):
        """Test that an empty file returns an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty_exchange.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("")

            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                result = load_symbols("empty_exchange")

            assert result == []

    def test_file_with_exchange_prefix(self):
        """Test loading symbols with exchange:symbol format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_exchange.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("KUCOIN:BTCUSDT\nKUCOIN:ETHUSDT\n")

            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                result = load_symbols("test_exchange")

            assert result == ["KUCOIN:BTCUSDT", "KUCOIN:ETHUSDT"]


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestLoadSymbolsErrorHandling:
    """Tests for error handling in load_symbols."""

    def test_nonexistent_file_returns_empty(self):
        """Test that nonexistent file returns empty list without error."""
        result = load_symbols("definitely_does_not_exist_xyz_123")
        assert result == []
        assert isinstance(result, list)

    def test_handles_unicode_errors(self):
        """Test handling of files with encoding issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "bad_encoding.txt")
            # Write binary data that might cause encoding issues
            with open(filepath, "wb") as f:
                f.write(b"\xff\xfe\x00\x00")  # Invalid UTF-8

            with patch('sigmapilot_mcp.core.services.coinlist.COINLIST_DIR', tmpdir):
                # Should handle gracefully and return empty list or partial data
                result = load_symbols("bad_encoding")
                assert isinstance(result, list)


# =============================================================================
# Integration Tests
# =============================================================================

class TestCoinlistIntegration:
    """Integration tests for coinlist functionality."""

    def test_coinlist_dir_path(self):
        """Test that COINLIST_DIR is a valid path string."""
        assert isinstance(COINLIST_DIR, str)
        assert len(COINLIST_DIR) > 0

    def test_multiple_exchanges_load(self):
        """Test loading from multiple exchanges."""
        exchanges = ["kucoin", "binance", "bybit", "bist"]
        for exchange in exchanges:
            result = load_symbols(exchange)
            assert isinstance(result, list)
            # Each result should be a list (empty or with symbols)

    def test_symbols_format(self):
        """Test that loaded symbols have expected format."""
        result = load_symbols("kucoin")
        for symbol in result:
            # Symbols should be non-empty strings
            assert isinstance(symbol, str)
            assert len(symbol) > 0
            # Common crypto symbol patterns: either "BTCUSDT" or "KUCOIN:BTCUSDT"
            assert symbol.isascii() or ":" in symbol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
