"""
Coinlist Management Module.

This module handles loading trading symbol lists for various exchanges.
Symbol lists are stored as text files in the coinlist directory, with
one symbol per line.

File Format:
    - One symbol per line
    - Symbols may include exchange prefix (e.g., "KUCOIN:BTCUSDT")
    - Empty lines are ignored
    - Whitespace is stripped from symbols
"""

from __future__ import annotations
import os
from typing import List
from ..utils.validators import COINLIST_DIR


def load_symbols(exchange: str) -> List[str]:
    """
    Load trading symbols for a given exchange from coinlist files.

    This function attempts to load symbols from text files in the coinlist
    directory. It uses multiple fallback strategies to find the file:
    1. Direct path using COINLIST_DIR constant
    2. Lowercase version of exchange name
    3. Relative path fallbacks for development environments

    Args:
        exchange: Exchange name (case-insensitive, e.g., "kucoin", "BINANCE")

    Returns:
        List of symbol strings, or empty list if file not found/empty

    Example:
        >>> symbols = load_symbols("kucoin")
        >>> len(symbols) > 0
        True
        >>> "KUCOIN:BTCUSDT" in symbols or "BTCUSDT" in symbols
        True

    File Location:
        Symbols are loaded from: {COINLIST_DIR}/{exchange}.txt
        Each line should contain one symbol (e.g., "KUCOIN:BTCUSDT")
    """
    # Try multiple possible paths for robustness
    # This handles different installation scenarios (pip install vs local dev)
    possible_paths = [
        # Primary: Use the calculated COINLIST_DIR from validators
        os.path.join(COINLIST_DIR, f"{exchange}.txt"),
        # Try lowercase version (exchanges are case-insensitive)
        os.path.join(COINLIST_DIR, f"{exchange.lower()}.txt"),
        # Fallback: relative to this file (for development)
        os.path.join(os.path.dirname(__file__), "..", "..", "coinlist", f"{exchange}.txt"),
        # Another fallback with lowercase
        os.path.join(os.path.dirname(__file__), "..", "..", "coinlist", f"{exchange.lower()}.txt")
    ]

    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Parse lines, strip whitespace, filter empty lines
                symbols = [line.strip() for line in content.split('\n') if line.strip()]
                if symbols:  # Only return if we actually got symbols
                    return symbols
        except (FileNotFoundError, IOError, UnicodeDecodeError):
            # Try next path in fallback chain
            continue

    # If all paths fail, return empty list (graceful degradation)
    return []
