"""
TradingView MCP Server - Remote Entry Point with Auth0 Authentication.

This module provides a remote-deployable MCP server with OAuth authentication
via Auth0. It wraps the existing TradingView analysis tools and exposes them
securely over HTTP.

Usage:
    Local development:
        uv run python main.py

    Production (Railway):
        Automatically runs via Procfile or railway.json

Environment Variables Required:
    AUTH0_DOMAIN: Your Auth0 domain (e.g., your-tenant.auth0.com)
    AUTH0_AUDIENCE: API identifier from Auth0 (e.g., https://tradingview-mcp.example.com)
    RESOURCE_SERVER_URL: Public URL of this server (e.g., https://your-app.railway.app/mcp)
    PORT: Server port (default: 8000, Railway sets this automatically)
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from starlette.responses import JSONResponse
from starlette.requests import Request

# Auth0 JWT verification
try:
    from tradingview_mcp.core.utils.auth import create_auth0_verifier, Auth0TokenVerifier
    AUTH0_VERIFIER_AVAILABLE = True
except ImportError:
    AUTH0_VERIFIER_AVAILABLE = False

# Import TradingView analysis functions
from tradingview_mcp.core.services.indicators import compute_metrics
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.utils.validators import (
    sanitize_timeframe,
    sanitize_exchange,
    EXCHANGE_SCREENER,
    BBW_HIGH_VOLATILITY,
    BBW_MEDIUM_VOLATILITY,
    ADX_STRONG_TREND,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)

# TradingView libraries
try:
    from tradingview_ta import get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

# Load environment variables
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "")
RESOURCE_SERVER_URL = os.getenv("RESOURCE_SERVER_URL", "http://localhost:8000/mcp")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Determine if we should enable auth (disabled for local development)
ENABLE_AUTH = bool(AUTH0_DOMAIN and AUTH0_AUDIENCE)

# =============================================================================
# Auth0 Token Verification Setup
# =============================================================================

token_verifier: Auth0TokenVerifier | None = None

if ENABLE_AUTH and AUTH0_VERIFIER_AVAILABLE:
    try:
        # Create Auth0 JWT verifier using JWKS
        token_verifier = create_auth0_verifier()
    except ValueError as e:
        print(f"⚠️  Auth0 configuration error: {e}")
        ENABLE_AUTH = False

# =============================================================================
# MCP Server Initialization
# =============================================================================

# Server instructions for AI assistants
SERVER_INSTRUCTIONS = """
TradingView MCP Server - Real-time Cryptocurrency and Stock Market Analysis

This server provides technical analysis tools powered by TradingView data.

Available Tools:
- top_gainers: Find best performing assets on an exchange
- top_losers: Find worst performing assets on an exchange
- bollinger_scan: Detect Bollinger Band squeeze patterns
- rating_filter: Filter by Bollinger Band rating (-3 to +3)
- coin_analysis: Complete technical analysis for a symbol
- list_exchanges: List all supported exchanges and timeframes

Supported Exchanges:
- Crypto: KuCoin, Binance, Bybit, Bitget, OKX, Coinbase, Gate.io, Huobi, Bitfinex
- Stocks: NASDAQ, NYSE, BIST (Turkey), HKEX (Hong Kong), Bursa (Malaysia)

Timeframes: 5m, 15m, 1h, 4h, 1D, 1W, 1M
"""


def create_mcp_server() -> FastMCP:
    """
    Create and configure the MCP server with optional Auth0 authentication.

    Returns:
        Configured FastMCP server instance
    """
    # Build auth settings if authentication is enabled
    auth_settings = None
    if ENABLE_AUTH and token_verifier is not None:
        auth_settings = AuthSettings(
            issuer_url=AnyHttpUrl(f"https://{AUTH0_DOMAIN}/"),
            resource_server_url=AnyHttpUrl(RESOURCE_SERVER_URL),
            required_scopes=["openid", "profile", "email"],
        )

    # Create the FastMCP server with all configuration
    server = FastMCP(
        name="TradingView MCP",
        instructions=SERVER_INSTRUCTIONS,
        host=HOST,
        port=PORT,
        token_verifier=token_verifier if ENABLE_AUTH else None,
        auth=auth_settings,
    )

    return server


# Create the MCP server instance
mcp = create_mcp_server()


# =============================================================================
# Helper Functions
# =============================================================================

def _fetch_analysis(exchange: str, timeframe: str, limit: int = 50):
    """Fetch analysis data for an exchange."""
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is not available")

    symbols = load_symbols(exchange)
    if not symbols:
        return []

    symbols = symbols[:limit * 2]
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    try:
        analysis = get_multiple_analysis(
            screener=screener,
            interval=timeframe,
            symbols=symbols
        )
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")

    rows = []
    for key, value in analysis.items():
        if value is None:
            continue

        try:
            indicators = value.indicators
            metrics = compute_metrics(indicators)

            if not metrics or metrics.get('bbw') is None:
                continue

            rows.append({
                "symbol": key,
                "changePercent": metrics['change'],
                "price": metrics['price'],
                "bbw": metrics['bbw'],
                "rating": metrics['rating'],
                "signal": metrics['signal'],
                "indicators": {
                    "RSI": indicators.get("RSI"),
                    "EMA50": indicators.get("EMA50"),
                    "volume": indicators.get("volume"),
                }
            })
        except (TypeError, KeyError):
            continue

    return rows


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def top_gainers(exchange: str = "kucoin", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """
    Get top gaining assets on an exchange.

    Args:
        exchange: Exchange name (kucoin, binance, bybit, nasdaq, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
        limit: Number of results (max 50)

    Returns:
        List of top gainers with price, change %, and indicators
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)
    limit = max(1, min(limit, 50))

    rows = _fetch_analysis(exchange, timeframe, limit)
    rows.sort(key=lambda x: x["changePercent"], reverse=True)

    return rows[:limit]


@mcp.tool()
def top_losers(exchange: str = "kucoin", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """
    Get top losing assets on an exchange.

    Args:
        exchange: Exchange name (kucoin, binance, bybit, nasdaq, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
        limit: Number of results (max 50)

    Returns:
        List of top losers with price, change %, and indicators
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)
    limit = max(1, min(limit, 50))

    rows = _fetch_analysis(exchange, timeframe, limit)
    rows.sort(key=lambda x: x["changePercent"])

    return rows[:limit]


@mcp.tool()
def bollinger_scan(
    exchange: str = "kucoin",
    timeframe: str = "4h",
    bbw_threshold: float = 0.04,
    limit: int = 25
) -> list[dict]:
    """
    Scan for assets with Bollinger Band squeeze (low BBW).

    A low BBW indicates consolidation and potential breakout.

    Args:
        exchange: Exchange name
        timeframe: Time interval
        bbw_threshold: Maximum BBW value (default 0.04 for squeeze)
        limit: Number of results (max 50)

    Returns:
        List of assets with tight Bollinger Bands
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 50))

    rows = _fetch_analysis(exchange, timeframe, limit * 2)

    # Filter by BBW threshold
    filtered = [r for r in rows if r["bbw"] and r["bbw"] < bbw_threshold and r["bbw"] > 0]
    filtered.sort(key=lambda x: x["bbw"])

    return filtered[:limit]


@mcp.tool()
def rating_filter(
    exchange: str = "kucoin",
    timeframe: str = "15m",
    rating: int = 2,
    limit: int = 25
) -> list[dict]:
    """
    Filter assets by Bollinger Band rating.

    Rating scale:
        +3: Above upper band (strong momentum, may be overbought)
        +2: Upper 50% of bands (BUY signal)
        +1: Above middle line (weak bullish)
         0: At middle line (neutral)
        -1: Below middle line (weak bearish)
        -2: Lower 50% of bands (SELL signal)
        -3: Below lower band (strong momentum, may be oversold)

    Args:
        exchange: Exchange name
        timeframe: Time interval
        rating: Target rating (-3 to +3)
        limit: Number of results (max 50)

    Returns:
        List of assets matching the rating
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))

    rows = _fetch_analysis(exchange, timeframe, limit * 3)

    # Filter by rating
    filtered = [r for r in rows if r["rating"] == rating]
    filtered.sort(key=lambda x: abs(x["changePercent"]), reverse=True)

    return filtered[:limit]


@mcp.tool()
def coin_analysis(
    symbol: str,
    exchange: str = "kucoin",
    timeframe: str = "15m"
) -> dict:
    """
    Get detailed technical analysis for a specific symbol.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT, ETHUSDT)
        exchange: Exchange name
        timeframe: Time interval

    Returns:
        Comprehensive analysis including price, Bollinger Bands, RSI, MACD, etc.
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "TradingView TA library not available"}

    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)

    # Format symbol
    if ":" not in symbol:
        full_symbol = f"{exchange.upper()}:{symbol.upper()}"
    else:
        full_symbol = symbol.upper()

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    try:
        analysis = get_multiple_analysis(
            screener=screener,
            interval=timeframe,
            symbols=[full_symbol]
        )

        if full_symbol not in analysis or analysis[full_symbol] is None:
            return {
                "error": f"No data found for {symbol}",
                "symbol": symbol,
                "exchange": exchange
            }

        data = analysis[full_symbol]
        indicators = data.indicators
        metrics = compute_metrics(indicators)

        if not metrics:
            return {"error": f"Could not compute metrics for {symbol}"}

        # Build comprehensive response
        return {
            "symbol": full_symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "price_data": {
                "current_price": metrics['price'],
                "change_percent": metrics['change'],
                "open": indicators.get("open"),
                "high": indicators.get("high"),
                "low": indicators.get("low"),
                "close": indicators.get("close"),
                "volume": indicators.get("volume"),
            },
            "bollinger_analysis": {
                "rating": metrics['rating'],
                "signal": metrics['signal'],
                "bbw": metrics['bbw'],
                "volatility": "High" if metrics['bbw'] > BBW_HIGH_VOLATILITY else
                             "Medium" if metrics['bbw'] > BBW_MEDIUM_VOLATILITY else "Low",
            },
            "technical_indicators": {
                "rsi": round(indicators.get("RSI", 0), 2),
                "rsi_signal": "Overbought" if indicators.get("RSI", 0) > RSI_OVERBOUGHT else
                             "Oversold" if indicators.get("RSI", 0) < RSI_OVERSOLD else "Neutral",
                "ema9": indicators.get("EMA9"),
                "ema21": indicators.get("EMA21"),
                "ema50": indicators.get("EMA50"),
                "atr": indicators.get("ATR"),
                "adx": round(indicators.get("ADX", 0), 2),
                "trend_strength": "Strong" if indicators.get("ADX", 0) > ADX_STRONG_TREND else "Weak",
            },
            "market_sentiment": {
                "momentum": "Bullish" if metrics['change'] > 0 else "Bearish",
                "overall_signal": metrics['signal'],
            }
        }

    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "symbol": symbol,
            "exchange": exchange
        }


@mcp.tool()
def list_exchanges() -> dict:
    """
    List all supported exchanges and their market types.

    Returns:
        Dictionary of exchanges grouped by market type
    """
    exchanges_by_type = {}
    for exchange, screener in EXCHANGE_SCREENER.items():
        if screener not in exchanges_by_type:
            exchanges_by_type[screener] = []
        exchanges_by_type[screener].append(exchange)

    return {
        "crypto": exchanges_by_type.get("crypto", []),
        "us_stocks": exchanges_by_type.get("america", []),
        "turkey": exchanges_by_type.get("turkey", []),
        "malaysia": exchanges_by_type.get("malaysia", []),
        "hongkong": exchanges_by_type.get("hongkong", []),
        "timeframes": ["5m", "15m", "1h", "4h", "1D", "1W", "1M"],
    }


# =============================================================================
# Health Check Endpoint
# =============================================================================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for Railway and other deployment platforms."""
    return JSONResponse({
        "status": "healthy",
        "service": "tradingview-mcp",
        "version": "1.1.0",
    })


@mcp.custom_route("/", methods=["GET"])
async def root_health(request: Request) -> JSONResponse:
    """Root endpoint returns health status for convenience."""
    return JSONResponse({
        "status": "healthy",
        "service": "tradingview-mcp",
        "version": "1.1.0",
    })


# =============================================================================
# Server Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    # Check if running in auth mode
    if ENABLE_AUTH:
        print(f"[AUTH] Auth0 authentication enabled")
        print(f"   Domain: {AUTH0_DOMAIN}")
        print(f"   Audience: {AUTH0_AUDIENCE}")
    else:
        print("[WARN] Running without authentication (development mode)")
        print("   Set AUTH0_DOMAIN and AUTH0_AUDIENCE for production")

    print(f"[START] TradingView MCP Server on {HOST}:{PORT}")
    print(f"   Endpoint: {RESOURCE_SERVER_URL}")
    print(f"   Health check: http://{HOST}:{PORT}/health")

    # Run with streamable HTTP transport (includes custom routes)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
