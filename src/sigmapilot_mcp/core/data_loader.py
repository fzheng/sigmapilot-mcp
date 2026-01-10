"""
OHLCV Data Loading Utilities for SigmaPilot MCP.

This module provides unified data fetching, caching, and validation
for OHLCV (Open, High, Low, Close, Volume) market data.

Key Features:
- OHLCVBar and OHLCVData dataclasses for type-safe data handling
- NumPy array accessors for efficient calculations
- Data validation and minimum bar requirements
- CSV fixture loading for testing
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from pathlib import Path
import csv

import numpy as np

from .errors import InsufficientDataError, DataError
from .timeframes import get_minimum_bars
from .sanitize import validate_ohlcv_series


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OHLCVBar:
    """
    Single OHLCV bar/candle data.

    Attributes:
        timestamp: Unix timestamp in seconds
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        """Validate bar data after initialization."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
        if self.high < max(self.open, self.close):
            raise ValueError(f"High ({self.high}) must be >= Open ({self.open}) and Close ({self.close})")
        if self.low > min(self.open, self.close):
            raise ValueError(f"Low ({self.low}) must be <= Open ({self.open}) and Close ({self.close})")

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (green) candle."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish (red) candle."""
        return self.close < self.open

    @property
    def body_size(self) -> float:
        """Absolute size of the candle body."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Full range from high to low."""
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        """Ratio of body to full range (0-1)."""
        if self.range_size == 0:
            return 0.0
        return self.body_size / self.range_size

    @property
    def upper_wick(self) -> float:
        """Size of upper wick/shadow."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Size of lower wick/shadow."""
        return min(self.open, self.close) - self.low


@dataclass
class OHLCVData:
    """
    Collection of OHLCV bars for a symbol/timeframe.

    Provides both list-based access to individual bars and
    NumPy array accessors for efficient vectorized calculations.

    Attributes:
        symbol: Trading symbol (e.g., "BINANCE:BTCUSDT")
        timeframe: Timeframe string (e.g., "1h", "4h")
        bars: List of OHLCVBar objects (oldest first)
    """
    symbol: str
    timeframe: str
    bars: List[OHLCVBar] = field(default_factory=list)

    # Cached numpy arrays (lazily computed)
    _opens: Optional[np.ndarray] = field(default=None, repr=False)
    _highs: Optional[np.ndarray] = field(default=None, repr=False)
    _lows: Optional[np.ndarray] = field(default=None, repr=False)
    _closes: Optional[np.ndarray] = field(default=None, repr=False)
    _volumes: Optional[np.ndarray] = field(default=None, repr=False)
    _timestamps: Optional[np.ndarray] = field(default=None, repr=False)

    def __len__(self) -> int:
        """Return number of bars."""
        return len(self.bars)

    def __getitem__(self, index: int) -> OHLCVBar:
        """Get bar by index."""
        return self.bars[index]

    def __iter__(self) -> Iterator[OHLCVBar]:
        """Iterate over bars."""
        return iter(self.bars)

    def _invalidate_cache(self):
        """Clear cached arrays when bars change."""
        self._opens = None
        self._highs = None
        self._lows = None
        self._closes = None
        self._volumes = None
        self._timestamps = None

    def append(self, bar: OHLCVBar):
        """Append a bar and invalidate cache."""
        self.bars.append(bar)
        self._invalidate_cache()

    # =========================================================================
    # NumPy Array Accessors (for vectorized calculations)
    # =========================================================================

    @property
    def opens(self) -> np.ndarray:
        """Get all open prices as numpy array."""
        if self._opens is None:
            self._opens = np.array([bar.open for bar in self.bars], dtype=np.float64)
        return self._opens

    @property
    def highs(self) -> np.ndarray:
        """Get all high prices as numpy array."""
        if self._highs is None:
            self._highs = np.array([bar.high for bar in self.bars], dtype=np.float64)
        return self._highs

    @property
    def lows(self) -> np.ndarray:
        """Get all low prices as numpy array."""
        if self._lows is None:
            self._lows = np.array([bar.low for bar in self.bars], dtype=np.float64)
        return self._lows

    @property
    def closes(self) -> np.ndarray:
        """Get all close prices as numpy array."""
        if self._closes is None:
            self._closes = np.array([bar.close for bar in self.bars], dtype=np.float64)
        return self._closes

    @property
    def volumes(self) -> np.ndarray:
        """Get all volumes as numpy array."""
        if self._volumes is None:
            self._volumes = np.array([bar.volume for bar in self.bars], dtype=np.float64)
        return self._volumes

    @property
    def timestamps(self) -> np.ndarray:
        """Get all timestamps as numpy array."""
        if self._timestamps is None:
            self._timestamps = np.array([bar.timestamp for bar in self.bars], dtype=np.int64)
        return self._timestamps

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def latest(self) -> Optional[OHLCVBar]:
        """Get the most recent bar."""
        return self.bars[-1] if self.bars else None

    @property
    def oldest(self) -> Optional[OHLCVBar]:
        """Get the oldest bar."""
        return self.bars[0] if self.bars else None

    @property
    def current_price(self) -> Optional[float]:
        """Get the current (latest close) price."""
        return self.bars[-1].close if self.bars else None

    @property
    def has_volume(self) -> bool:
        """Check if volume data is available (non-zero)."""
        if not self.bars:
            return False
        return any(bar.volume > 0 for bar in self.bars)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_minimum_bars(data: OHLCVData, required: Optional[int] = None) -> bool:
    """
    Check if data has enough bars for analysis.

    Args:
        data: OHLCVData to validate
        required: Minimum bars required (uses timeframe default if None)

    Returns:
        True if data has enough bars

    Raises:
        InsufficientDataError: If not enough bars available
    """
    min_required = required or get_minimum_bars(data.timeframe)
    available = len(data)

    if available < min_required:
        raise InsufficientDataError(
            required=min_required,
            available=available
        )

    return True


def ensure_minimum_bars(data: OHLCVData, required: Optional[int] = None) -> OHLCVData:
    """
    Validate and return data if it has enough bars.

    Args:
        data: OHLCVData to validate
        required: Minimum bars required

    Returns:
        The same OHLCVData if valid

    Raises:
        InsufficientDataError: If not enough bars
    """
    validate_minimum_bars(data, required)
    return data


# =============================================================================
# CSV Fixture Loading (for testing)
# =============================================================================

def load_ohlcv_from_csv(
    filepath: str | Path,
    symbol: str = "TEST:SYMBOL",
    timeframe: str = "1h"
) -> OHLCVData:
    """
    Load OHLCV data from a CSV file.

    Expected CSV format (with header):
        timestamp,open,high,low,close,volume

    Args:
        filepath: Path to the CSV file
        symbol: Symbol to assign to the data
        timeframe: Timeframe to assign to the data

    Returns:
        OHLCVData populated from the CSV

    Raises:
        DataError: If file cannot be read or parsed

    Example:
        >>> data = load_ohlcv_from_csv("tests/fixtures/btc_1h_uptrend.csv")
        >>> len(data)
        100
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise DataError(f"CSV file not found: {filepath}")

    bars = []

    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                bar = OHLCVBar(
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0))
                )
                bars.append(bar)

    except KeyError as e:
        raise DataError(f"CSV missing required column: {e}")
    except ValueError as e:
        raise DataError(f"CSV contains invalid data: {e}")
    except Exception as e:
        raise DataError(f"Failed to read CSV: {e}")

    return OHLCVData(symbol=symbol, timeframe=timeframe, bars=bars)


def create_ohlcv_from_arrays(
    symbol: str,
    timeframe: str,
    timestamps: List[int],
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: Optional[List[float]] = None
) -> OHLCVData:
    """
    Create OHLCVData from parallel arrays.

    Useful for converting from indicator library formats.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        timestamps: List of unix timestamps
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        volumes: Optional list of volumes (defaults to 0)

    Returns:
        OHLCVData populated from the arrays

    Raises:
        DataError: If arrays have mismatched lengths
    """
    # Validate series lengths
    is_valid, error = validate_ohlcv_series(opens, highs, lows, closes, volumes)
    if not is_valid:
        raise DataError(error or "Invalid OHLCV series")

    if len(timestamps) != len(opens):
        raise DataError(f"Timestamp length ({len(timestamps)}) doesn't match price length ({len(opens)})")

    # Default volumes to 0 if not provided
    if volumes is None:
        volumes = [0.0] * len(opens)

    bars = [
        OHLCVBar(
            timestamp=ts,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v
        )
        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes)
    ]

    return OHLCVData(symbol=symbol, timeframe=timeframe, bars=bars)


# =============================================================================
# Data Transformation Utilities
# =============================================================================

def slice_ohlcv(data: OHLCVData, start: int = 0, end: Optional[int] = None) -> OHLCVData:
    """
    Create a new OHLCVData with a subset of bars.

    Args:
        data: Source OHLCVData
        start: Start index (inclusive)
        end: End index (exclusive), None for end

    Returns:
        New OHLCVData with sliced bars
    """
    sliced_bars = data.bars[start:end]
    return OHLCVData(
        symbol=data.symbol,
        timeframe=data.timeframe,
        bars=sliced_bars
    )


def get_latest_n_bars(data: OHLCVData, n: int) -> OHLCVData:
    """
    Get the most recent N bars.

    Args:
        data: Source OHLCVData
        n: Number of bars to get

    Returns:
        New OHLCVData with latest N bars
    """
    if n >= len(data):
        return data
    return slice_ohlcv(data, start=-n)
