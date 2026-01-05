# TradingView MCP Server

A remote Model Context Protocol (MCP) server that provides real-time cryptocurrency and stock market analysis using TradingView data. Deployable to Railway with Auth0 authentication for secure access.

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

### Option 1: Remote Deployment (Recommended)

Deploy as a secure remote MCP server with Auth0 authentication. Once deployed, connect from **any MCP-compatible AI platform**:
- **Claude.ai** - Via [Connectors](https://claude.ai/settings/connectors) ([documentation](https://support.claude.com/en/articles/11724452-using-the-connectors-directory-to-extend-claude-s-capabilities))
- **ChatGPT** - Via MCP plugin support
- **Other AI platforms** - Any service supporting MCP protocol

#### Prerequisites
- [Railway account](https://railway.app/) (free tier available)
- [Auth0 account](https://auth0.com/) (free tier available)
- GitHub account

#### Deploy Steps

1. **Fork/Clone this repository to GitHub**

2. **Set up Auth0**
   - Create account at [auth0.com](https://auth0.com)
   - Create API: Dashboard > Applications > APIs > Create API
   - Note your `Domain` and `API Identifier`

3. **Deploy to Railway**
   - Connect your GitHub repo to Railway
   - Add environment variables:
     ```
     AUTH0_DOMAIN=your-tenant.auth0.com
     AUTH0_AUDIENCE=https://your-api-identifier
     RESOURCE_SERVER_URL=https://your-app.up.railway.app/mcp
     ```

4. **Connect to AI Platform**

   **Claude.ai (Web):**
   - Go to [claude.ai/settings/connectors](https://claude.ai/settings/connectors)
   - Add your MCP server URL: `https://your-app.up.railway.app/mcp`
   - Authenticate with Auth0 when prompted

   **Claude Desktop:**
   ```json
   {
     "mcpServers": {
       "tradingview": {
         "url": "https://your-app.up.railway.app/mcp",
         "transport": "streamable-http"
       }
     }
   }
   ```

See [Remote Deployment Guide](docs/REMOTE_DEPLOYMENT.md) for detailed instructions.

### Option 2: Local Installation

For local development or direct Claude Desktop connection (stdio mode).

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
           "git+https://github.com/fzheng/tradingview-mcp.git",
           "tradingview-mcp"
         ]
       }
     }
   }
   ```

3. **Restart Claude Desktop**

## Available Tools

### Remote Server (main.py)

| Tool | Description |
|------|-------------|
| `top_gainers` | Top performing assets by exchange/timeframe |
| `top_losers` | Worst performing assets by exchange/timeframe |
| `bollinger_scan` | Find assets with Bollinger Band squeeze |
| `rating_filter` | Filter by BB rating (-3 to +3) |
| `coin_analysis` | Complete technical analysis for a symbol |
| `list_exchanges` | List all supported exchanges |

### Local Server (server.py) - Additional Tools

| Tool | Description |
|------|-------------|
| `consecutive_candles_scan` | Detect bullish/bearish candle patterns |
| `advanced_candle_pattern` | Advanced pattern recognition with scoring |
| `volume_breakout_scanner` | Find high-volume price breakouts |
| `volume_confirmation_analysis` | Analyze volume confirmation for a symbol |
| `smart_volume_scanner` | Intelligent volume-based scanning |

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
git clone https://github.com/fzheng/tradingview-mcp.git
cd tradingview-mcp
uv sync

# Run tests
make test

# Run locally (stdio mode)
uv run python src/tradingview_mcp/server.py

# Run as remote server (HTTP mode with optional auth)
uv run python main.py
```

## Architecture

```
┌─────────────────┐      HTTPS       ┌─────────────────┐
│   Claude.ai     │ ───────────────► │  Railway Server │
│   ChatGPT       │  + OAuth Token   │  (main.py)      │
│   AI Platforms  │                  └────────┬────────┘
└─────────────────┘                           │
                                              ▼
                                     ┌─────────────────┐
                                     │    Auth0        │
                                     │  Token Verify   │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │  TradingView    │
                                     │  Public APIs    │
                                     └─────────────────┘
```

## Documentation

- [Remote Deployment Guide](docs/REMOTE_DEPLOYMENT.md)
- [Installation Guide](docs/INSTALLATION.md)
- [Usage Examples](docs/EXAMPLES.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Development TODO](docs/TODO.md)

## License

MIT License - see [LICENSE](LICENSE)

## Support

- [GitHub Issues](https://github.com/fzheng/tradingview-mcp/issues)
