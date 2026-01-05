# Installation Guide - TradingView MCP Server v1.1.0

This guide covers both remote deployment (recommended) and local installation options.

## Prerequisites

- **Python 3.11+** (required for mcp-oauth)
- **UV Package Manager** (for dependency management)
- **Claude Desktop** (for MCP integration)
- **Internet Connection** (for TradingView data access)

For remote deployment, you'll also need:
- **Railway account** (free tier available)
- **Auth0 account** (free tier available)

## Option 1: Remote Deployment (Recommended)

Deploy as a secure remote MCP server accessible from anywhere.

See [Remote Deployment Guide](REMOTE_DEPLOYMENT.md) for complete instructions.

### Quick Summary

1. Fork this repository to your GitHub
2. Create Auth0 API at [auth0.com](https://auth0.com)
3. Deploy to Railway and add environment variables
4. Connect Claude Desktop to your remote URL

## Option 2: Local Installation

### Step 1: Install UV Package Manager

#### macOS (Homebrew):
```bash
brew install uv
```

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Verify Installation:
```bash
uv --version
```

### Step 2: Configure Claude Desktop

#### Find Config File:

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

#### Add MCP Server (Remote):
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

#### Add MCP Server (Local via Git):
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

### Step 3: Restart Claude Desktop

1. Completely quit Claude Desktop
2. Restart the application
3. Wait 30-60 seconds for initialization

### Step 4: Verify Installation

Ask Claude:
```
"Can you show me the available TradingView tools?"
```

You should see tools like:
- `top_gainers`
- `top_losers`
- `bollinger_scan`
- `coin_analysis`
- `list_exchanges`

## Local Development Setup

For modifying the code or contributing:

```bash
# Clone the repository
git clone https://github.com/fzheng/tradingview-mcp.git
cd tradingview-mcp

# Install dependencies
uv sync

# Run tests
make test

# Run locally (stdio mode for Claude Desktop)
uv run python src/tradingview_mcp/server.py

# Run as remote server (HTTP mode)
uv run python main.py
```

### Windows Development Config

```json
{
  "mcpServers": {
    "tradingview-mcp-local": {
      "command": "C:\\Users\\YOUR_USERNAME\\tradingview-mcp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\YOUR_USERNAME\\tradingview-mcp\\src\\tradingview_mcp\\server.py"],
      "cwd": "C:\\Users\\YOUR_USERNAME\\tradingview-mcp"
    }
  }
}
```

### macOS/Linux Development Config

```json
{
  "mcpServers": {
    "tradingview-mcp-local": {
      "command": "uv",
      "args": ["run", "python", "src/tradingview_mcp/server.py"],
      "cwd": "/path/to/tradingview-mcp"
    }
  }
}
```

## Testing Your Installation

### Basic Functionality Test

1. **Market Screening:**
   ```
   "Show me top 5 crypto gainers on KuCoin in 15 minutes"
   ```

2. **Technical Analysis:**
   ```
   "Analyze Bitcoin with all technical indicators"
   ```

3. **Pattern Detection:**
   ```
   "Find coins with Bollinger Band squeeze on Binance"
   ```

### Expected Results

- Real-time price data
- Technical indicator values (RSI, EMA, ATR, etc.)
- Bollinger Band ratings (-3 to +3)
- Trading signals (BUY, SELL, NEUTRAL)

## Troubleshooting

### "Command not found: uv"

UV is not installed or not in PATH.

**Windows:**
```powershell
# Re-run installation
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart PowerShell and check
uv --version
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Claude doesn't see the MCP server

1. Check JSON syntax in config file
2. Verify file paths are correct
3. Restart Claude Desktop completely
4. Wait 30-60 seconds for initialization
5. Check Claude Desktop logs: Settings → Developer → View Logs

### "No data found" errors

- Try different exchanges (KuCoin usually works best)
- Use standard symbols (e.g., "BTCUSDT", "ETHUSDT")
- Check timeframe format ("15m", "1h", "1D")

### Connection timeout (remote server)

- Verify Railway deployment is running
- Check environment variables are set correctly
- Ensure `RESOURCE_SERVER_URL` matches your Railway URL

## Getting Help

- **GitHub Issues:** [Report bugs](https://github.com/fzheng/tradingview-mcp/issues)
- **Remote Deployment:** See [REMOTE_DEPLOYMENT.md](REMOTE_DEPLOYMENT.md)
- **Usage Examples:** See [EXAMPLES.md](EXAMPLES.md)
