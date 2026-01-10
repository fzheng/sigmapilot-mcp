# Deterministic OHLCV Test Fixtures

This directory contains deterministic OHLCV data files for testing theory-based analysis engines.

## Purpose

These fixtures provide reproducible test data with known patterns for validating:
- Dow Theory trend detection
- Ichimoku calculations
- Chart pattern recognition
- Other theory-based analysis engines

## Files

### btc_1h_uptrend.csv
- **Symbol**: BTC (simulated)
- **Timeframe**: 1 hour
- **Pattern**: Clear uptrend with higher highs and higher lows
- **Bars**: 30
- **Price Range**: $42,000 → $49,500
- **Expected Analysis**:
  - Dow Theory: Bullish (primary uptrend)
  - Trend: Strong upward momentum
  - Volume: Increasing with price (confirmation)

### eth_4h_downtrend.csv
- **Symbol**: ETH (simulated)
- **Timeframe**: 4 hours
- **Pattern**: Clear downtrend with lower highs and lower lows
- **Bars**: 25
- **Price Range**: $2,500 → $1,620
- **Expected Analysis**:
  - Dow Theory: Bearish (primary downtrend)
  - Trend: Strong downward momentum
  - Volume: Increasing on declines (confirmation)

### sol_1d_range.csv
- **Symbol**: SOL (simulated)
- **Timeframe**: 1 day
- **Pattern**: Range-bound consolidation
- **Bars**: 30
- **Price Range**: $97 - $108 (oscillating around $102-103)
- **Expected Analysis**:
  - Dow Theory: Neutral (no clear trend)
  - Trend: Sideways/consolidation
  - Pattern: Potential rectangle formation

## CSV Format

All files follow this format:
```csv
timestamp,open,high,low,close,volume
1704067200,42000.00,42500.00,41800.00,42300.00,1500000
```

- **timestamp**: Unix timestamp in seconds
- **open**: Opening price
- **high**: Highest price in the period
- **low**: Lowest price in the period
- **close**: Closing price
- **volume**: Trading volume

## Usage in Tests

```python
from sigmapilot_mcp.core.data_loader import load_ohlcv_from_csv

# Load fixture
data = load_ohlcv_from_csv(
    "tests/fixtures/deterministic_ohlcv/btc_1h_uptrend.csv",
    symbol="TEST:BTCUSDT",
    timeframe="1h"
)

# Use in tests
assert len(data) == 30
assert data.closes[-1] > data.closes[0]  # Price went up
```

## Data Integrity

All fixtures maintain:
- Valid OHLC relationships (High >= Open/Close >= Low)
- Realistic price movements
- Consistent timestamp intervals
- Positive volume values

## Adding New Fixtures

When adding new fixtures:
1. Use descriptive names: `{symbol}_{timeframe}_{pattern}.csv`
2. Include at least 20-30 bars for pattern analysis
3. Document expected analysis results above
4. Ensure OHLC relationships are valid
5. Use realistic price and volume ranges
