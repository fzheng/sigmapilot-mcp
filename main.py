"""
SigmaPilot MCP Server - Remote Entry Point with Auth0 Authentication.

This module provides a remote-deployable MCP server with OAuth authentication
via Auth0. It wraps the existing market analysis tools and exposes them
securely over HTTP.

Usage:
    Local development:
        uv run python main.py

    Production (Railway):
        Automatically runs via Procfile or railway.json

Environment Variables Required:
    AUTH0_DOMAIN: Your Auth0 domain (e.g., your-tenant.auth0.com)
    AUTH0_AUDIENCE: API identifier from Auth0 (e.g., https://sigmapilot-mcp.example.com)
    RESOURCE_SERVER_URL: Public URL of this server (e.g., https://your-app.railway.app/mcp)
    PORT: Server port (default: 8000, Railway sets this automatically)
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from starlette.responses import JSONResponse
from starlette.requests import Request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Auth0 JWT verification
try:
    from sigmapilot_mcp.core.utils.auth import create_auth0_verifier, Auth0TokenVerifier
    AUTH0_VERIFIER_AVAILABLE = True
except ImportError:
    AUTH0_VERIFIER_AVAILABLE = False

# Import shared tools implementation
from sigmapilot_mcp.core.services.tools import (
    get_top_gainers,
    get_top_losers,
    get_bollinger_scan,
    get_rating_filter,
    get_coin_analysis,
    get_exchanges_list,
)

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
SigmaPilot MCP Server - Real-time Cryptocurrency and Stock Market Analysis

This server provides AI-powered technical analysis tools for market intelligence.

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
        name="SigmaPilot MCP",
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
# MCP Tools - Using shared implementation
# =============================================================================

@mcp.tool()
def top_gainers(exchange: str = "kucoin", timeframe: str = "15m", limit: int = 25) -> dict:
    """
    Get top gaining assets on an exchange.

    Args:
        exchange: Exchange name (kucoin, binance, bybit, nasdaq, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
        limit: Number of results (max 50)

    Returns:
        Dictionary with 'data' list and optional 'warning' for partial failures
    """
    data, warning = get_top_gainers(exchange, timeframe, limit)
    result = {"data": data}
    if warning:
        result["warning"] = warning
        logger.warning(f"top_gainers: {warning}")
    return result


@mcp.tool()
def top_losers(exchange: str = "kucoin", timeframe: str = "15m", limit: int = 25) -> dict:
    """
    Get top losing assets on an exchange.

    Args:
        exchange: Exchange name (kucoin, binance, bybit, nasdaq, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
        limit: Number of results (max 50)

    Returns:
        Dictionary with 'data' list and optional 'warning' for partial failures
    """
    data, warning = get_top_losers(exchange, timeframe, limit)
    result = {"data": data}
    if warning:
        result["warning"] = warning
        logger.warning(f"top_losers: {warning}")
    return result


@mcp.tool()
def bollinger_scan(
    exchange: str = "kucoin",
    timeframe: str = "4h",
    bbw_threshold: float = 0.04,
    limit: int = 25
) -> dict:
    """
    Scan for assets with Bollinger Band squeeze (low BBW).

    A low BBW indicates consolidation and potential breakout.

    Args:
        exchange: Exchange name
        timeframe: Time interval
        bbw_threshold: Maximum BBW value (default 0.04 for squeeze)
        limit: Number of results (max 50)

    Returns:
        Dictionary with 'data' list and optional 'warning' for partial failures
    """
    data, warning = get_bollinger_scan(exchange, timeframe, bbw_threshold, limit)
    result = {"data": data}
    if warning:
        result["warning"] = warning
        logger.warning(f"bollinger_scan: {warning}")
    return result


@mcp.tool()
def rating_filter(
    exchange: str = "kucoin",
    timeframe: str = "15m",
    rating: int = 2,
    limit: int = 25
) -> dict:
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
        Dictionary with 'data' list and optional 'warning' for partial failures
    """
    data, warning = get_rating_filter(exchange, timeframe, rating, limit)
    result = {"data": data}
    if warning:
        result["warning"] = warning
        logger.warning(f"rating_filter: {warning}")
    return result


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
    return get_coin_analysis(symbol, exchange, timeframe)


@mcp.tool()
def list_exchanges() -> dict:
    """
    List all supported exchanges and their market types.

    Returns:
        Dictionary of exchanges grouped by market type
    """
    return get_exchanges_list()


# =============================================================================
# Health Check Endpoint
# =============================================================================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for Railway and other deployment platforms."""
    return JSONResponse({
        "status": "healthy",
        "service": "sigmapilot-mcp",
        "version": "1.2.0",
    })


@mcp.custom_route("/", methods=["GET"])
async def root_health(request: Request) -> JSONResponse:
    """Root endpoint returns health status for convenience."""
    return JSONResponse({
        "status": "healthy",
        "service": "sigmapilot-mcp",
        "version": "1.2.0",
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

    print(f"[START] SigmaPilot MCP Server on {HOST}:{PORT}")
    print(f"   Endpoint: {RESOURCE_SERVER_URL}")
    print(f"   Health check: http://{HOST}:{PORT}/health")

    # Run with streamable HTTP transport (includes custom routes)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
