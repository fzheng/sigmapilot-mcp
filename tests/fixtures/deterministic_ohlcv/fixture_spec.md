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
- **Bars**: 100
- **Price Range**: $40,000 → $60,000
- **Expected Analysis**:
  - Dow Theory: Bullish (primary uptrend)
  - Trend: Strong upward momentum
  - Volume: Increasing with price (confirmation)

### eth_4h_downtrend.csv
- **Symbol**: ETH (simulated)
- **Timeframe**: 4 hours
- **Pattern**: Clear downtrend with lower highs and lower lows
- **Bars**: 100
- **Price Range**: $2,500 → $500
- **Expected Analysis**:
  - Dow Theory: Bearish (primary downtrend)
  - Trend: Strong downward momentum
  - Volume: Increasing on declines (confirmation)

### sol_1d_range.csv
- **Symbol**: SOL (simulated)
- **Timeframe**: 1 day
- **Pattern**: Range-bound consolidation
- **Bars**: 100
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

### btc_double_top.csv
- **Symbol**: BTC (simulated)
- **Timeframe**: 1 hour
- **Pattern**: Double Top reversal pattern
- **Bars**: 100
- **Price Range**: $40,000 → $45,200 → $34,300
- **Structure**:
  - First peak: ~$45,000 (bar 26-27)
  - Pullback to ~$43,000 (bar 36)
  - Second peak: ~$45,000 (bar 47-48)
  - Breakdown below neckline with continuation
- **Expected Analysis**:
  - Chart Patterns: Double Top detected, bearish
  - Dow Theory: Bearish (lower high on second peak)
  - VSA: Distribution signals on peaks

### eth_head_shoulders.csv
- **Symbol**: ETH (simulated)
- **Timeframe**: 4 hours
- **Pattern**: Head & Shoulders reversal pattern
- **Bars**: 100
- **Price Range**: $2,000 → $2,500 → $1,510
- **Structure**:
  - Left Shoulder: ~$2,300 (bar 15-16)
  - Head: ~$2,500 (bar 33-34)
  - Right Shoulder: ~$2,400 (bar 55-56)
  - Neckline: ~$2,170
  - Breakdown with continuation
- **Expected Analysis**:
  - Chart Patterns: H&S detected, bearish
  - Dow Theory: Bearish (lower high on right shoulder)

### sol_double_bottom.csv
- **Symbol**: SOL (simulated)
- **Timeframe**: 4 hours
- **Pattern**: Double Bottom reversal pattern (bullish)
- **Bars**: 100
- **Price Range**: $120 → $88 → $175
- **Structure**:
  - First bottom: ~$90 (bar 30)
  - Rally to ~$108 (bar 40)
  - Second bottom: ~$90 (bar 57)
  - Breakout above neckline with continuation
- **Expected Analysis**:
  - Chart Patterns: Double Bottom detected, bullish
  - Dow Theory: Bullish (higher low confirmed)

### btc_wyckoff_accumulation.csv
- **Symbol**: BTC (simulated)
- **Timeframe**: 4 hours
- **Pattern**: Wyckoff Accumulation structure
- **Bars**: 100
- **Price Range**: $45,000 → $41,000 → $53,300
- **Structure**:
  - Selling Climax: ~$42,100 (bar 15)
  - Trading Range: $41,000 - $43,000 (bars 16-40)
  - Spring: ~$41,000 (bar 40-41) with high volume
  - Sign of Strength: breakout with volume (bars 41-60)
  - Markup phase: (bars 60-100)
- **Expected Analysis**:
  - Wyckoff: Accumulation phase detected
  - VSA: Stopping volume, spring, SOS signals
  - Dow Theory: Bullish (after markup begins)

## Adding New Fixtures

When adding new fixtures:
1. Use descriptive names: `{symbol}_{timeframe}_{pattern}.csv`
2. Include at least 20-30 bars for pattern analysis
3. Document expected analysis results above
4. Ensure OHLC relationships are valid
5. Use realistic price and volume ranges
