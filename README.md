# TradingView MCP Server

A Model Context Protocol (MCP) server that provides real-time cryptocurrency and stock market analysis using TradingView data. Designed for use with Claude Desktop and other MCP-compatible AI assistants.

## Features

### Market Screening
- **Top Gainers/Losers** - Find best and worst performing assets across exchanges
- **Bollinger Band Scan** - Detect squeeze patterns for potential breakouts
- **Rating Filter** - Filter assets by Bollinger Band position (-3 to +3 rating)

### Technical Analysis
- **Coin Analysis** - Comprehensive single-asset analysis with all indicators
- **Pattern Recognition** - Detect consecutive bullish/bearish candle patterns
- **Volume Breakout Scanner** - Find high-volume price breakouts

### Indicators Included
- Bollinger Bands (BBW, position, rating)
- Moving Averages (SMA20, EMA9, EMA21, EMA50, EMA200)
- RSI, MACD, ADX, Stochastic
- ATR (Average True Range)
- Volume analysis

### Supported Markets

| Market Type | Exchanges |
|-------------|-----------|
| Crypto | KuCoin, Binance, Bybit, Bitget, OKX, Coinbase, Gate.io, Huobi, Bitfinex |
| US Stocks | NASDAQ, NYSE |
| Turkish Stocks | BIST |
| Malaysian Stocks | Bursa, KLSE, ACE, LEAP |
| Hong Kong Stocks | HKEX, HSI |

### Timeframes
`5m` `15m` `1h` `4h` `1D` `1W` `1M`

## Quick Start

### Claude Desktop Setup

1. **Install UV Package Manager:**
   ```bash
   # macOS
   brew install uv

   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Add to Claude Desktop config:**

   Config location:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "tradingview-mcp": {
         "command": "uv",
         "args": [
           "tool", "run", "--from",
           "git+https://github.com/atilaahmettaner/tradingview-mcp.git",
           "tradingview-mcp"
         ]
       }
     }
   }
   ```

3. **Restart Claude Desktop**

## Available Tools

| Tool | Description |
|------|-------------|
| `top_gainers` | Top performing assets by exchange/timeframe |
| `top_losers` | Worst performing assets by exchange/timeframe |
| `bollinger_scan` | Find assets with Bollinger Band squeeze |
| `rating_filter` | Filter by BB rating (-3 to +3) |
| `coin_analysis` | Complete technical analysis for a symbol |
| `consecutive_candles_scan` | Detect candle patterns |
| `advanced_candle_pattern` | Multi-timeframe pattern analysis |
| `volume_breakout_scanner` | Detect volume + price breakouts |
| `volume_confirmation_analysis` | Volume confirmation for a symbol |
| `smart_volume_scanner` | Combined volume + technical filter |

## Bollinger Band Rating System

| Rating | Meaning | Signal |
|--------|---------|--------|
| +3 | Above upper band | Strong momentum (may be overbought) |
| +2 | Upper 50% of bands | BUY |
| +1 | Above middle line | Weak bullish |
| 0 | At middle line | NEUTRAL |
| -1 | Below middle line | Weak bearish |
| -2 | Lower 50% of bands | SELL |
| -3 | Below lower band | Strong momentum (may be oversold) |

## Example Queries

```
"Show me top 10 crypto gainers on KuCoin in 15 minutes"
"Find coins with Bollinger Band squeeze on Binance"
"Analyze BTCUSDT with all technical indicators"
"Find stocks with strong buy signals on NASDAQ"
"Show volume breakouts on Bybit"
```

## Development

```bash
# Clone and install
git clone https://github.com/atilaahmettaner/tradingview-mcp.git
cd tradingview-mcp
uv sync

# Run tests
uv run pytest tests/ -v

# Run locally
uv run python src/tradingview_mcp/server.py
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Usage Examples](docs/EXAMPLES.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Development TODO](docs/TODO.md)

## License

MIT License - see [LICENSE](LICENSE)

## Support

- [GitHub Issues](https://github.com/atilaahmettaner/tradingview-mcp/issues)
