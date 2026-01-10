"""
SigmaPilot MCP Server - Unified Entry Point.

This module provides the main MCP server implementation for cryptocurrency and
stock market analysis. It offers tools for market screening, technical analysis,
and pattern detection.

The server supports both local (stdio) and remote (HTTP with Auth0) modes.

Features:
    - Top gainers/losers screening by exchange and timeframe
    - Bollinger Band analysis and squeeze detection
    - Candle pattern scanning (consecutive and advanced patterns)
    - Volume analysis (breakout detection and smart scanning)
    - Single coin detailed analysis
    - TradingView recommendation signals
    - Pivot point analysis

Tools (10 total):
    1. top_gainers - Top gaining assets
    2. top_losers - Top losing assets
    3. bollinger_scan - Bollinger Band squeeze detection
    4. rating_filter - Filter by BB rating
    5. coin_analysis - Detailed single-symbol analysis
    6. candle_pattern_scanner - Candle pattern detection
    7. volume_scanner - Volume breakout and smart scanning
    8. volume_analysis - Single-symbol volume confirmation
    9. pivot_points_scanner - Pivot point level analysis
    10. tradingview_recommendation - TradingView signals

Usage:
    # Run as stdio server (for Claude Desktop)
    uv run python src/sigmapilot_mcp/server.py

    # Run as HTTP server (local development)
    uv run python src/sigmapilot_mcp/server.py streamable-http --port 8000

    # Run as HTTP server with Auth0 (production)
    AUTH0_DOMAIN=your-tenant.auth0.com AUTH0_AUDIENCE=https://api.example.com \
        uv run python src/sigmapilot_mcp/server.py streamable-http --port 8000

Environment Variables:
    DEBUG_MCP: Enable debug logging (set to any value)
    HOST: Server host for HTTP mode (default: 0.0.0.0)
    PORT: Server port for HTTP mode (default: 8000)
    AUTH0_DOMAIN: Auth0 tenant domain (enables authentication)
    AUTH0_AUDIENCE: Auth0 API audience identifier
    RESOURCE_SERVER_URL: Public URL for OAuth resource server

    Rate Limiting (TradingView API):
    TV_BATCH_SIZE: Symbols per API batch (default: 50)
    TV_BATCH_DELAY: Seconds between batches (default: 0.5)
    TV_MAX_RETRIES: Max retry attempts on rate limit (default: 3)
    TV_RETRY_BASE_DELAY: Base delay for exponential backoff (default: 2.0)
    TV_MAX_SYMBOLS: Max symbols to scan per request (default: 300)
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Auth0 authentication (optional)
try:
    from mcp.server.auth.settings import AuthSettings
    from sigmapilot_mcp.core.utils.auth import create_auth0_verifier, Auth0TokenVerifier
    AUTH0_AVAILABLE = True
except ImportError:
    AUTH0_AVAILABLE = False
    AuthSettings = None
    Auth0TokenVerifier = None

# Starlette for custom routes (optional, for HTTP mode)
try:
    from starlette.responses import JSONResponse
    from starlette.requests import Request
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    JSONResponse = None
    Request = None

# Import core analysis modules
from sigmapilot_mcp.core.services.indicators import compute_metrics
from sigmapilot_mcp.core.services.coinlist import load_symbols
from sigmapilot_mcp.core.utils.validators import (
    sanitize_timeframe,
    sanitize_exchange,
    tf_to_tv_resolution,
    EXCHANGE_SCREENER,
    ALLOWED_TIMEFRAMES,
    DEFAULT_BATCH_SIZE,
    MAX_SYMBOLS_PER_SCAN,
    STRONG_BODY_RATIO,
    MODERATE_BODY_RATIO,
    VOLUME_MINIMUM,
    VOLUME_DECENT,
    VOLUME_HIGH,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_NEUTRAL_HIGH,
    RSI_NEUTRAL_LOW,
    BBW_HIGH_VOLATILITY,
    BBW_MEDIUM_VOLATILITY,
    ADX_STRONG_TREND,
)

# =============================================================================
# Optional Dependencies
# =============================================================================
# These libraries provide TradingView data access. The server gracefully handles
# their absence by disabling features that depend on them.

try:
    from tradingview_ta import TA_Handler, get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

# Note: tradingview_screener (Query, Column) is imported locally in functions
# that use it, with their own error handling

# =============================================================================
# Rate Limiting Configuration
# =============================================================================
# TradingView has undocumented rate limits. These settings help avoid hitting them.
# Based on community findings: >7 concurrent connections trigger 429 errors.

import time
import random

# Configurable via environment variables
API_BATCH_SIZE = int(os.environ.get("TV_BATCH_SIZE", "50"))  # Symbols per batch (reduced from 100-200)
API_BATCH_DELAY = float(os.environ.get("TV_BATCH_DELAY", "0.5"))  # Seconds between batches
API_MAX_RETRIES = int(os.environ.get("TV_MAX_RETRIES", "3"))  # Max retry attempts
API_RETRY_BASE_DELAY = float(os.environ.get("TV_RETRY_BASE_DELAY", "2.0"))  # Base delay for exponential backoff
API_MAX_SYMBOLS_PER_SCAN = int(os.environ.get("TV_MAX_SYMBOLS", "300"))  # Max symbols to scan per request

# Track last request time for global throttling
_last_api_request_time: float = 0.0
_api_request_lock = None  # Will use threading.Lock for thread safety

try:
    import threading
    _api_request_lock = threading.Lock()
except ImportError:
    pass


def _throttled_api_call(api_func, *args, **kwargs):
    """
    Execute an API call with rate limiting and retry logic.

    Features:
    - Global throttling: Ensures minimum delay between all API calls
    - Exponential backoff: Retries failed requests with increasing delays
    - Jitter: Adds randomness to prevent thundering herd

    Args:
        api_func: The API function to call (e.g., get_multiple_analysis)
        *args, **kwargs: Arguments to pass to the API function

    Returns:
        The result of the API call

    Raises:
        The last exception if all retries fail
    """
    global _last_api_request_time

    last_exception = None

    for attempt in range(API_MAX_RETRIES):
        try:
            # Global throttling - ensure minimum time between requests
            if _api_request_lock:
                with _api_request_lock:
                    elapsed = time.time() - _last_api_request_time
                    if elapsed < API_BATCH_DELAY:
                        sleep_time = API_BATCH_DELAY - elapsed
                        # Add jitter (10-30% random variation)
                        jitter = sleep_time * random.uniform(0.1, 0.3)
                        time.sleep(sleep_time + jitter)
                    _last_api_request_time = time.time()
            else:
                # Fallback without lock
                time.sleep(API_BATCH_DELAY)

            # Make the API call
            result = api_func(*args, **kwargs)
            return result

        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if it's a rate limit error (worth retrying)
            is_rate_limit = any(term in error_str for term in [
                "429", "too many", "rate limit", "max sessions",
                "expecting value"  # Empty response often means rate limited
            ])

            if is_rate_limit and attempt < API_MAX_RETRIES - 1:
                # Exponential backoff with jitter
                delay = API_RETRY_BASE_DELAY * (2 ** attempt)
                jitter = delay * random.uniform(0.1, 0.3)
                total_delay = delay + jitter

                logger.warning(
                    f"Rate limit detected (attempt {attempt + 1}/{API_MAX_RETRIES}). "
                    f"Retrying in {total_delay:.1f}s..."
                )
                time.sleep(total_delay)
            elif not is_rate_limit:
                # Non-rate-limit error, don't retry
                raise

    # All retries exhausted
    if last_exception:
        raise last_exception


def _get_batch_config():
    """Get current batch configuration for logging/debugging."""
    return {
        "batch_size": API_BATCH_SIZE,
        "batch_delay": API_BATCH_DELAY,
        "max_retries": API_MAX_RETRIES,
        "retry_base_delay": API_RETRY_BASE_DELAY,
        "max_symbols": API_MAX_SYMBOLS_PER_SCAN
    }


# =============================================================================
# Type Definitions
# =============================================================================

class IndicatorMap(TypedDict, total=False):
    """Type definition for technical indicator data from TradingView."""
    open: Optional[float]
    close: Optional[float]
    SMA20: Optional[float]
    BB_upper: Optional[float]
    BB_lower: Optional[float]
    EMA9: Optional[float]
    EMA21: Optional[float]
    EMA50: Optional[float]
    RSI: Optional[float]
    ATR: Optional[float]
    volume: Optional[float]


class Row(TypedDict):
    """Single timeframe analysis result for a symbol."""
    symbol: str
    changePercent: float
    indicators: IndicatorMap


class MultiRow(TypedDict):
    """Multi-timeframe analysis result with price changes across periods."""
    symbol: str
    changes: dict[str, Optional[float]]  # Timeframe -> percent change
    base_indicators: IndicatorMap


# =============================================================================
# Helper Functions
# =============================================================================

def _percent_change(o: Optional[float], c: Optional[float]) -> Optional[float]:
    """
    Calculate percentage change between open and close prices.

    Args:
        o: Open price
        c: Close price

    Returns:
        Percentage change as float, or None if calculation not possible

    Example:
        >>> _percent_change(100.0, 105.0)
        5.0
    """
    try:
        if o in (None, 0) or c is None:
            return None
        return (c - o) / o * 100
    except Exception:
        return None


# Alias for backward compatibility - tf_to_tv_resolution converts user-friendly
# timeframes (e.g., "15m", "1h") to TradingView's internal format (e.g., "15", "60")
_tf_to_tv_resolution = tf_to_tv_resolution


def _classify_api_error(error: Exception) -> str:
    """
    Classify TradingView API errors into user-friendly messages.

    Args:
        error: The exception raised during API call

    Returns:
        A descriptive error message based on the error type
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # JSON decode errors - empty or invalid response
    if "expecting value" in error_str or "jsondecodeerror" in error_type.lower():
        return "TradingView API returned empty response. This may indicate rate limiting or temporary unavailability. Please wait a moment and try again."

    # Connection errors
    if any(term in error_str for term in ["connection", "timeout", "timed out", "refused"]):
        return f"Network error connecting to TradingView API: {error}. Please check your internet connection."

    # HTTP errors
    if "429" in error_str or "too many requests" in error_str:
        return "TradingView API rate limit exceeded. Please wait 30-60 seconds before retrying."

    if "403" in error_str or "forbidden" in error_str:
        return "TradingView API access denied. The API may be temporarily blocking requests."

    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return f"TradingView API server error ({error_str[:50]}...). Please try again later."

    # Symbol-related errors
    if "symbol" in error_str and ("not found" in error_str or "invalid" in error_str):
        return f"Invalid symbol or exchange: {error}"

    # Generic fallback with original error
    return f"TradingView API error: {error}"


def _fetch_bollinger_analysis(exchange: str, timeframe: str = "4h", limit: int = 50, bbw_filter: float = None) -> List[Row]:
    """
    Fetch Bollinger Band analysis data for symbols on an exchange.

    Uses tradingview_ta library to get real-time indicator data and computes
    Bollinger Band metrics including band width (BBW), rating, and signal.

    Args:
        exchange: Exchange name (e.g., "KUCOIN", "BINANCE")
        timeframe: TradingView interval (e.g., "4h", "1D")
        limit: Maximum number of results to return
        bbw_filter: Optional BBW threshold - only return symbols with BBW below this value
                    (used for squeeze detection)

    Returns:
        List of Row objects sorted by change percentage (highest first)

    Raises:
        RuntimeError: If tradingview_ta is not available or API call fails
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")

    # Load symbols from coinlist files
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    # Limit symbols for performance (use configured max)
    max_symbols = min(len(symbols), API_MAX_SYMBOLS_PER_SCAN, limit * 3)
    symbols = symbols[:max_symbols]

    # Get screener type based on exchange
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    rows: List[Row] = []

    # Process in batches with rate limiting
    total_batches = (len(symbols) + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    failed_batches = 0
    last_error = None

    for i in range(0, len(symbols), API_BATCH_SIZE):
        batch_symbols = symbols[i:i + API_BATCH_SIZE]

        try:
            analysis = _throttled_api_call(
                get_multiple_analysis,
                screener=screener,
                interval=timeframe,
                symbols=batch_symbols
            )
        except Exception as e:
            failed_batches += 1
            last_error = e
            logger.warning(f"Bollinger batch {i // API_BATCH_SIZE + 1}/{total_batches} failed: {_classify_api_error(e)}")
            continue

        for key, value in analysis.items():
            try:
                if value is None:
                    continue

                indicators = value.indicators
                metrics = compute_metrics(indicators)

                if not metrics or metrics.get('bbw') is None:
                    continue

                # Apply BBW filter if specified
                if bbw_filter is not None and (metrics['bbw'] >= bbw_filter or metrics['bbw'] <= 0):
                    continue

                # Check if we have required indicators
                if not (indicators.get("EMA50") and indicators.get("RSI")):
                    continue

                rows.append(Row(
                    symbol=key,
                    changePercent=metrics['change'],
                    indicators=IndicatorMap(
                        open=metrics.get('open'),
                        close=metrics.get('price'),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA9=indicators.get("EMA9"),
                        EMA21=indicators.get("EMA21"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        ATR=indicators.get("ATR"),
                        volume=indicators.get("volume"),
                    )
                ))

            except (TypeError, ZeroDivisionError, KeyError):
                continue

    # If all batches failed, raise an error
    if failed_batches == total_batches and last_error is not None:
        raise RuntimeError(_classify_api_error(last_error))

    # Sort by change percentage in descending order (highest gainers first)
    rows.sort(key=lambda x: x["changePercent"], reverse=True)

    # Return the requested limit
    return rows[:limit]


def _fetch_trending_analysis(exchange: str, timeframe: str = "5m", filter_type: str = "", rating_filter: int = None, limit: int = 50) -> List[Row]:
    """
    Fetch trending coins with technical analysis data.

    Processes symbols in batches to handle large exchanges while respecting
    TradingView API rate limits. Supports filtering by Bollinger Band rating.

    Args:
        exchange: Exchange name (e.g., "KUCOIN", "BINANCE")
        timeframe: TradingView interval (e.g., "5m", "15m", "1h")
        filter_type: Optional filter - use "rating" to enable rating_filter
        rating_filter: When filter_type="rating", only return symbols with this
                       Bollinger Band rating (-3 to +3)
        limit: Maximum number of results to return

    Returns:
        List of Row objects sorted by change percentage (highest first)

    Raises:
        RuntimeError: If tradingview_ta is not available or no symbols found
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")

    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    # Limit symbols to configured max
    symbols = symbols[:API_MAX_SYMBOLS_PER_SCAN]

    all_coins = []
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    # Process symbols in batches with rate limiting
    failed_batches = 0
    total_batches = (len(symbols) + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    last_error = None

    for i in range(0, len(symbols), API_BATCH_SIZE):
        batch_symbols = symbols[i:i + API_BATCH_SIZE]

        try:
            analysis = _throttled_api_call(
                get_multiple_analysis,
                screener=screener,
                interval=timeframe,
                symbols=batch_symbols
            )
        except Exception as e:
            failed_batches += 1
            last_error = e
            logger.warning(f"Trending batch {i // API_BATCH_SIZE + 1}/{total_batches} failed: {_classify_api_error(e)}")
            continue  # If this batch fails, move to the next one

        # Process coins in this batch
        for key, value in analysis.items():
            try:
                if value is None:
                    continue
                    
                indicators = value.indicators
                metrics = compute_metrics(indicators)
                
                if not metrics or metrics.get('bbw') is None:
                    continue
                
                # Apply rating filter if specified
                if filter_type == "rating" and rating_filter is not None:
                    if metrics['rating'] != rating_filter:
                        continue
                
                all_coins.append(Row(
                    symbol=key,
                    changePercent=metrics['change'],
                    indicators=IndicatorMap(
                        open=metrics.get('open'),
                        close=metrics.get('price'),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA9=indicators.get("EMA9"),
                        EMA21=indicators.get("EMA21"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        ATR=indicators.get("ATR"),
                        volume=indicators.get("volume"),
                    )
                ))
                
            except (TypeError, ZeroDivisionError, KeyError):
                continue

    # If all batches failed, raise an error with the last error message
    if failed_batches == total_batches and last_error is not None:
        raise RuntimeError(_classify_api_error(last_error))

    # Sort all coins by change percentage
    all_coins.sort(key=lambda x: x["changePercent"], reverse=True)

    return all_coins[:limit]
def _fetch_multi_changes(exchange: str, timeframes: List[str] | None, base_timeframe: str = "4h", limit: int | None = None, cookies: Any | None = None) -> List[MultiRow]:
    """
    Fetch price changes across multiple timeframes using tradingview-screener.

    This function queries TradingView's screener API to get OHLC data across
    multiple timeframes simultaneously, allowing comparison of performance
    across different periods.

    Args:
        exchange: Exchange name (e.g., "KUCOIN", "BINANCE")
        timeframes: List of timeframes to analyze (e.g., ["15m", "1h", "4h", "1D"])
                    If None, defaults to ["15m", "1h", "4h", "1D"]
        base_timeframe: Primary timeframe for indicator calculations
        limit: Maximum number of symbols to return
        cookies: Optional cookies for authenticated requests

    Returns:
        List of MultiRow objects containing symbol, changes per timeframe,
        and base indicators

    Raises:
        RuntimeError: If tradingview-screener is not available
    """
    try:
        from tradingview_screener import Query
        from tradingview_screener.column import Column
    except Exception as e:
        raise RuntimeError("tradingview-screener missing; run `uv sync`.") from e

    tfs = timeframes or ["15m", "1h", "4h", "1D"]
    suffix_map: dict[str, str] = {}
    for tf in tfs:
        s = _tf_to_tv_resolution(tf)
        if s:
            suffix_map[tf] = s
    if not suffix_map:
        suffix_map = {base_timeframe: _tf_to_tv_resolution(base_timeframe) or "240"}

    base_suffix = _tf_to_tv_resolution(base_timeframe) or next(iter(suffix_map.values()))
    cols: list[str] = []
    seen: set[str] = set()
    for tf, s in suffix_map.items():
        for c in (f"open|{s}", f"close|{s}"):
            if c not in seen:
                cols.append(c)
                seen.add(c)
    for c in (f"SMA20|{base_suffix}", f"BB.upper|{base_suffix}", f"BB.lower|{base_suffix}", f"volume|{base_suffix}"):
        if c not in seen:
            cols.append(c)
            seen.add(c)

    q = Query().set_markets("crypto").select(*cols)
    if exchange:
        q = q.where(Column("exchange") == exchange.upper())
    if limit:
        q = q.limit(int(limit))

    _total, df = q.get_scanner_data(cookies=cookies)
    if df is None or df.empty:
        return []

    out: List[MultiRow] = []
    for _, r in df.iterrows():
        symbol = r.get("ticker")
        changes: dict[str, Optional[float]] = {}
        for tf, s in suffix_map.items():
            o = r.get(f"open|{s}")
            c = r.get(f"close|{s}")
            changes[tf] = _percent_change(o, c)
        base_ind = IndicatorMap(
            open=r.get(f"open|{base_suffix}"),
            close=r.get(f"close|{base_suffix}"),
            SMA20=r.get(f"SMA20|{base_suffix}"),
            BB_upper=r.get(f"BB.upper|{base_suffix}"),
            BB_lower=r.get(f"BB.lower|{base_suffix}"),
            volume=r.get(f"volume|{base_suffix}"),
        )
        out.append(MultiRow(symbol=symbol, changes=changes, base_indicators=base_ind))
    return out


# =============================================================================
# MCP Server Configuration
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
- candle_pattern_scanner: Detect bullish/bearish candle patterns
- volume_scanner: Volume breakout detection with RSI filtering
- volume_analysis: Detailed volume analysis for a symbol
- pivot_points_scanner: Find coins near pivot point levels
- tradingview_recommendation: TradingView buy/sell recommendations

Supported Exchanges:
- Crypto: KuCoin, Binance, Bybit, Bitget, OKX, Coinbase, Gate.io, Huobi, Bitfinex
- Stocks: NASDAQ, NYSE, BIST (Turkey), HKEX (Hong Kong), Bursa (Malaysia)

Timeframes: 5m, 15m, 1h, 4h, 1D, 1W, 1M
"""

# Configuration from environment
AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN", "")
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE", "")
RESOURCE_SERVER_URL = os.environ.get("RESOURCE_SERVER_URL", "http://localhost:8000/mcp")
# HTTP server configuration (Railway sets PORT automatically)
HTTP_HOST = os.environ.get("HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("PORT", "8000"))

# Global token verifier (set when auth is enabled)
_token_verifier: Optional[Any] = None


def create_mcp_server(
    enable_auth: bool = False,
    host: Optional[str] = None,
    port: Optional[int] = None
) -> FastMCP:
    """
    Create and configure the MCP server with optional Auth0 authentication.

    Args:
        enable_auth: Enable Auth0 authentication (requires environment variables)
        host: Server host for HTTP mode (default: from HOST env var or 0.0.0.0)
        port: Server port for HTTP mode (default: from PORT env var or 8000)

    Returns:
        Configured FastMCP server instance
    """
    global _token_verifier

    # Use module-level defaults if not specified
    if host is None:
        host = HTTP_HOST
    if port is None:
        port = HTTP_PORT

    auth_settings = None
    token_verifier = None

    # Setup Auth0 if requested and available
    if enable_auth and AUTH0_AVAILABLE and AUTH0_DOMAIN and AUTH0_AUDIENCE:
        try:
            _token_verifier = create_auth0_verifier()
            token_verifier = _token_verifier
            auth_settings = AuthSettings(
                issuer_url=AnyHttpUrl(f"https://{AUTH0_DOMAIN}/"),
                resource_server_url=AnyHttpUrl(RESOURCE_SERVER_URL),
                required_scopes=["openid", "profile", "email"],
            )
            logger.info(f"Auth0 authentication enabled for domain: {AUTH0_DOMAIN}")
        except ValueError as e:
            logger.warning(f"Auth0 configuration error: {e}. Running without auth.")
        except Exception as e:
            logger.warning(f"Auth0 setup failed: {e}. Running without auth.")

    # Create the FastMCP server with host/port for HTTP mode
    server = FastMCP(
        name="SigmaPilot Screener",
        instructions=SERVER_INSTRUCTIONS,
        host=host,
        port=port,
        token_verifier=token_verifier,
        auth=auth_settings,
    )

    return server


# Create the default MCP server instance (for stdio mode, tools registered below)
mcp = FastMCP(
    name="SigmaPilot Screener",
    instructions=SERVER_INSTRUCTIONS,
)


# =============================================================================
# MCP Tools - Market Screening
# =============================================================================

@mcp.tool()
def top_gainers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Return top gainers for an exchange and timeframe using bollinger band analysis.
    
    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    # Convert Row objects to dicts properly
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"], 
        "indicators": dict(row["indicators"])
    } for row in rows]


@mcp.tool()
def top_losers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Return top losers for an exchange and timeframe using bollinger band analysis."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    # Reverse sort for losers (lowest change first)
    rows.sort(key=lambda x: x["changePercent"])
    
    # Convert to dict format
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows[:limit]]


@mcp.tool()
def bollinger_scan(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict]:
    """Scan for coins with low Bollinger Band Width (squeeze detection).
    
    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M  
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 100))
    
    rows = _fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    # Convert Row objects to dicts
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows]


@mcp.tool()
def rating_filter(exchange: str = "KUCOIN", timeframe: str = "15m", rating: int = 2, limit: int = 25) -> list[dict]:
    """Filter coins by Bollinger Band rating.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit)
    # Convert Row objects to dicts
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows]

# =============================================================================
# MCP Tools - Single Coin Analysis
# =============================================================================

@mcp.tool()
def coin_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m"
) -> dict:
    """Get detailed analysis for a specific coin on specified exchange and timeframe.
    
    Args:
        symbol: Coin symbol (e.g., "ACEUSDT", "BTCUSDT")
        exchange: Exchange name (BINANCE, KUCOIN, etc.) 
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
    
    Returns:
        Detailed coin analysis with all indicators and metrics
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "15m")
        
        # Format symbol with exchange prefix
        if ":" not in symbol:
            full_symbol = f"{exchange.upper()}:{symbol.upper()}"
        else:
            full_symbol = symbol.upper()
        
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")
        
        try:
            analysis = _throttled_api_call(
                get_multiple_analysis,
                screener=screener,
                interval=timeframe,
                symbols=[full_symbol]
            )

            if full_symbol not in analysis or analysis[full_symbol] is None:
                return {
                    "error": f"No data found for {symbol} on {exchange}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            
            data = analysis[full_symbol]
            indicators = data.indicators
            
            # Calculate all metrics
            metrics = compute_metrics(indicators)
            if not metrics:
                return {
                    "error": f"Could not compute metrics for {symbol}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            
            # Additional technical indicators
            macd = indicators.get("MACD.macd", 0)
            macd_signal = indicators.get("MACD.signal", 0)
            adx = indicators.get("ADX", 0)
            stoch_k = indicators.get("Stoch.K", 0)
            stoch_d = indicators.get("Stoch.D", 0)

            # Volume analysis
            volume = indicators.get("volume", 0)

            # Price levels
            high = indicators.get("high", 0)
            low = indicators.get("low", 0)
            open_price = indicators.get("open", 0)
            close_price = indicators.get("close", 0)

            # =================================================================
            # NEW INDICATORS - Ichimoku Cloud
            # =================================================================
            ichimoku_base = indicators.get("Ichimoku.BLine", 0)  # Kijun-sen (Base Line)
            ichimoku_conversion = indicators.get("Ichimoku.CLine", 0)  # Tenkan-sen (Conversion Line)
            ichimoku_lead_a = indicators.get("Ichimoku.Lead1", 0)  # Senkou Span A (Leading Span A)
            ichimoku_lead_b = indicators.get("Ichimoku.Lead2", 0)  # Senkou Span B (Leading Span B)

            # =================================================================
            # NEW INDICATORS - VWAP
            # =================================================================
            vwap = indicators.get("VWAP", 0)

            # =================================================================
            # NEW INDICATORS - Pivot Points
            # =================================================================
            # Classic Pivot Points
            pivot_classic = indicators.get("Pivot.M.Classic.Middle", 0)
            pivot_classic_r1 = indicators.get("Pivot.M.Classic.R1", 0)
            pivot_classic_r2 = indicators.get("Pivot.M.Classic.R2", 0)
            pivot_classic_r3 = indicators.get("Pivot.M.Classic.R3", 0)
            pivot_classic_s1 = indicators.get("Pivot.M.Classic.S1", 0)
            pivot_classic_s2 = indicators.get("Pivot.M.Classic.S2", 0)
            pivot_classic_s3 = indicators.get("Pivot.M.Classic.S3", 0)

            # Fibonacci Pivot Points
            pivot_fib = indicators.get("Pivot.M.Fibonacci.Middle", 0)
            pivot_fib_r1 = indicators.get("Pivot.M.Fibonacci.R1", 0)
            pivot_fib_r2 = indicators.get("Pivot.M.Fibonacci.R2", 0)
            pivot_fib_r3 = indicators.get("Pivot.M.Fibonacci.R3", 0)
            pivot_fib_s1 = indicators.get("Pivot.M.Fibonacci.S1", 0)
            pivot_fib_s2 = indicators.get("Pivot.M.Fibonacci.S2", 0)
            pivot_fib_s3 = indicators.get("Pivot.M.Fibonacci.S3", 0)

            # Camarilla Pivot Points
            pivot_cam = indicators.get("Pivot.M.Camarilla.Middle", 0)
            pivot_cam_r1 = indicators.get("Pivot.M.Camarilla.R1", 0)
            pivot_cam_r2 = indicators.get("Pivot.M.Camarilla.R2", 0)
            pivot_cam_r3 = indicators.get("Pivot.M.Camarilla.R3", 0)
            pivot_cam_s1 = indicators.get("Pivot.M.Camarilla.S1", 0)
            pivot_cam_s2 = indicators.get("Pivot.M.Camarilla.S2", 0)
            pivot_cam_s3 = indicators.get("Pivot.M.Camarilla.S3", 0)

            # =================================================================
            # NEW INDICATORS - TradingView Recommendations
            # =================================================================
            recommend_all = indicators.get("Recommend.All", 0)  # Overall recommendation
            recommend_ma = indicators.get("Recommend.MA", 0)  # Moving averages recommendation
            recommend_other = indicators.get("Recommend.Other", 0)  # Oscillators recommendation

            # =================================================================
            # NEW INDICATORS - Williams %R
            # =================================================================
            williams_r = indicators.get("W.R", 0)

            # =================================================================
            # NEW INDICATORS - CCI (Commodity Channel Index)
            # =================================================================
            cci = indicators.get("CCI20", 0)

            # =================================================================
            # NEW INDICATORS - Awesome Oscillator
            # =================================================================
            ao = indicators.get("AO", 0)

            # =================================================================
            # NEW INDICATORS - Ultimate Oscillator
            # =================================================================
            uo = indicators.get("UO", 0)

            # =================================================================
            # NEW INDICATORS - Momentum
            # =================================================================
            momentum = indicators.get("Mom", 0)

            # =================================================================
            # NEW INDICATORS - Hull MA and VWMA
            # =================================================================
            hma = indicators.get("HullMA9", 0)
            vwma = indicators.get("VWMA", 0)

            # =================================================================
            # NEW INDICATORS - Parabolic SAR
            # =================================================================
            psar = indicators.get("P.SAR", 0)

            # =================================================================
            # NEW INDICATORS - Additional Moving Averages
            # =================================================================
            sma5 = indicators.get("SMA5", 0)
            sma10 = indicators.get("SMA10", 0)
            sma30 = indicators.get("SMA30", 0)
            sma50 = indicators.get("SMA50", 0)
            sma100 = indicators.get("SMA100", 0)
            sma200 = indicators.get("SMA200", 0)
            ema5 = indicators.get("EMA5", 0)
            ema10 = indicators.get("EMA10", 0)
            ema30 = indicators.get("EMA30", 0)
            ema100 = indicators.get("EMA100", 0)
            
            # =================================================================
            # Helper functions for signal interpretation
            # =================================================================
            def get_recommendation_text(value):
                """Convert TradingView recommendation value to text."""
                if value >= 0.5:
                    return "STRONG_BUY"
                elif value >= 0.1:
                    return "BUY"
                elif value > -0.1:
                    return "NEUTRAL"
                elif value > -0.5:
                    return "SELL"
                else:
                    return "STRONG_SELL"

            def get_williams_r_signal(value):
                """Interpret Williams %R value."""
                if value > -20:
                    return "Overbought"
                elif value < -80:
                    return "Oversold"
                else:
                    return "Neutral"

            def get_cci_signal(value):
                """Interpret CCI value."""
                if value > 100:
                    return "Overbought"
                elif value < -100:
                    return "Oversold"
                else:
                    return "Neutral"

            def get_ichimoku_signal(close, lead_a, lead_b, conversion, base):
                """Determine Ichimoku Cloud signal."""
                if lead_a == 0 or lead_b == 0:
                    return "No Data"
                cloud_top = max(lead_a, lead_b)
                cloud_bottom = min(lead_a, lead_b)
                if close > cloud_top:
                    return "Bullish (Above Cloud)"
                elif close < cloud_bottom:
                    return "Bearish (Below Cloud)"
                else:
                    return "Neutral (Inside Cloud)"

            def get_psar_signal(close, psar_val):
                """Determine Parabolic SAR signal."""
                if psar_val == 0:
                    return "No Data"
                return "Bullish (Price above SAR)" if close > psar_val else "Bearish (Price below SAR)"

            return {
                "symbol": full_symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": "real-time",
                "price_data": {
                    "current_price": metrics['price'],
                    "open": round(open_price, 6) if open_price else None,
                    "high": round(high, 6) if high else None,
                    "low": round(low, 6) if low else None,
                    "close": round(close_price, 6) if close_price else None,
                    "change_percent": metrics['change'],
                    "volume": volume,
                    "vwap": round(vwap, 6) if vwap else None
                },
                "bollinger_analysis": {
                    "rating": metrics['rating'],
                    "signal": metrics['signal'],
                    "bbw": metrics['bbw'],
                    "bb_upper": round(indicators.get("BB.upper", 0), 6),
                    "bb_middle": round(indicators.get("SMA20", 0), 6),
                    "bb_lower": round(indicators.get("BB.lower", 0), 6),
                    "position": "Above Upper" if close_price > indicators.get("BB.upper", 0) else
                               "Below Lower" if close_price < indicators.get("BB.lower", 0) else
                               "Within Bands"
                },
                "ichimoku_cloud": {
                    "conversion_line": round(ichimoku_conversion, 6) if ichimoku_conversion else None,
                    "base_line": round(ichimoku_base, 6) if ichimoku_base else None,
                    "leading_span_a": round(ichimoku_lead_a, 6) if ichimoku_lead_a else None,
                    "leading_span_b": round(ichimoku_lead_b, 6) if ichimoku_lead_b else None,
                    "signal": get_ichimoku_signal(close_price, ichimoku_lead_a, ichimoku_lead_b,
                                                   ichimoku_conversion, ichimoku_base),
                    "tk_cross": "Bullish" if ichimoku_conversion > ichimoku_base else "Bearish" if ichimoku_conversion < ichimoku_base else "Neutral"
                },
                "pivot_points": {
                    "classic": {
                        "pivot": round(pivot_classic, 6) if pivot_classic else None,
                        "r1": round(pivot_classic_r1, 6) if pivot_classic_r1 else None,
                        "r2": round(pivot_classic_r2, 6) if pivot_classic_r2 else None,
                        "r3": round(pivot_classic_r3, 6) if pivot_classic_r3 else None,
                        "s1": round(pivot_classic_s1, 6) if pivot_classic_s1 else None,
                        "s2": round(pivot_classic_s2, 6) if pivot_classic_s2 else None,
                        "s3": round(pivot_classic_s3, 6) if pivot_classic_s3 else None
                    },
                    "fibonacci": {
                        "pivot": round(pivot_fib, 6) if pivot_fib else None,
                        "r1": round(pivot_fib_r1, 6) if pivot_fib_r1 else None,
                        "r2": round(pivot_fib_r2, 6) if pivot_fib_r2 else None,
                        "r3": round(pivot_fib_r3, 6) if pivot_fib_r3 else None,
                        "s1": round(pivot_fib_s1, 6) if pivot_fib_s1 else None,
                        "s2": round(pivot_fib_s2, 6) if pivot_fib_s2 else None,
                        "s3": round(pivot_fib_s3, 6) if pivot_fib_s3 else None
                    },
                    "camarilla": {
                        "pivot": round(pivot_cam, 6) if pivot_cam else None,
                        "r1": round(pivot_cam_r1, 6) if pivot_cam_r1 else None,
                        "r2": round(pivot_cam_r2, 6) if pivot_cam_r2 else None,
                        "r3": round(pivot_cam_r3, 6) if pivot_cam_r3 else None,
                        "s1": round(pivot_cam_s1, 6) if pivot_cam_s1 else None,
                        "s2": round(pivot_cam_s2, 6) if pivot_cam_s2 else None,
                        "s3": round(pivot_cam_s3, 6) if pivot_cam_s3 else None
                    }
                },
                "tradingview_recommendations": {
                    "overall": {
                        "value": round(recommend_all, 3) if recommend_all else 0,
                        "signal": get_recommendation_text(recommend_all or 0)
                    },
                    "moving_averages": {
                        "value": round(recommend_ma, 3) if recommend_ma else 0,
                        "signal": get_recommendation_text(recommend_ma or 0)
                    },
                    "oscillators": {
                        "value": round(recommend_other, 3) if recommend_other else 0,
                        "signal": get_recommendation_text(recommend_other or 0)
                    }
                },
                "oscillators": {
                    "rsi": {
                        "value": round(indicators.get("RSI", 0), 2),
                        "signal": "Overbought" if indicators.get("RSI", 0) > 70 else
                                 "Oversold" if indicators.get("RSI", 0) < 30 else "Neutral"
                    },
                    "williams_r": {
                        "value": round(williams_r, 2) if williams_r else None,
                        "signal": get_williams_r_signal(williams_r or -50)
                    },
                    "cci": {
                        "value": round(cci, 2) if cci else None,
                        "signal": get_cci_signal(cci or 0)
                    },
                    "awesome_oscillator": round(ao, 4) if ao else None,
                    "ultimate_oscillator": {
                        "value": round(uo, 2) if uo else None,
                        "signal": "Overbought" if (uo or 50) > 70 else "Oversold" if (uo or 50) < 30 else "Neutral"
                    },
                    "momentum": round(momentum, 4) if momentum else None,
                    "stochastic": {
                        "k": round(stoch_k, 2),
                        "d": round(stoch_d, 2),
                        "signal": "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
                    }
                },
                "moving_averages": {
                    "simple": {
                        "sma5": round(sma5, 6) if sma5 else None,
                        "sma10": round(sma10, 6) if sma10 else None,
                        "sma20": round(indicators.get("SMA20", 0), 6),
                        "sma30": round(sma30, 6) if sma30 else None,
                        "sma50": round(sma50, 6) if sma50 else None,
                        "sma100": round(sma100, 6) if sma100 else None,
                        "sma200": round(sma200, 6) if sma200 else None
                    },
                    "exponential": {
                        "ema5": round(ema5, 6) if ema5 else None,
                        "ema9": round(indicators.get("EMA9", 0), 6),
                        "ema10": round(ema10, 6) if ema10 else None,
                        "ema21": round(indicators.get("EMA21", 0), 6),
                        "ema30": round(ema30, 6) if ema30 else None,
                        "ema50": round(indicators.get("EMA50", 0), 6),
                        "ema100": round(ema100, 6) if ema100 else None,
                        "ema200": round(indicators.get("EMA200", 0), 6)
                    },
                    "hull_ma": round(hma, 6) if hma else None,
                    "vwma": round(vwma, 6) if vwma else None
                },
                "trend_indicators": {
                    "macd": {
                        "macd": round(macd, 6),
                        "signal": round(macd_signal, 6),
                        "histogram": round(macd - macd_signal, 6),
                        "trend": "Bullish" if macd > macd_signal else "Bearish"
                    },
                    "adx": {
                        "value": round(adx, 2),
                        "trend_strength": "Strong" if adx > 25 else "Weak"
                    },
                    "parabolic_sar": {
                        "value": round(psar, 6) if psar else None,
                        "signal": get_psar_signal(close_price, psar or 0)
                    },
                    "atr": round(indicators.get("ATR", 0), 6)
                },
                "market_sentiment": {
                    "overall_rating": metrics['rating'],
                    "buy_sell_signal": metrics['signal'],
                    "volatility": "High" if metrics['bbw'] > 0.05 else "Medium" if metrics['bbw'] > 0.02 else "Low",
                    "momentum": "Bullish" if metrics['change'] > 0 else "Bearish",
                    "trend_alignment": {
                        "short_term": "Bullish" if close_price > (indicators.get("SMA20", 0) or close_price) else "Bearish",
                        "medium_term": "Bullish" if close_price > (sma50 or close_price) else "Bearish",
                        "long_term": "Bullish" if close_price > (sma200 or close_price) else "Bearish"
                    }
                }
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
    except Exception as e:
        return {
            "error": f"Coin analysis failed: {str(e)}",
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe
        }

# =============================================================================
# MCP Tools - Pattern Detection
# =============================================================================

@mcp.tool()
def candle_pattern_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    mode: str = "consecutive",
    pattern_type: str = "bullish",
    min_change: float = 2.0,
    limit: int = 20
) -> dict:
    """Scan for candle patterns including consecutive candles and advanced patterns.

    Combines consecutive candle detection and advanced pattern analysis into a
    single unified tool. Use the 'mode' parameter to select analysis type.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, BYBIT, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D)
        mode: Analysis mode:
            - "consecutive": Detect consecutive bullish/bearish candles
            - "advanced": Advanced multi-timeframe pattern scoring
        pattern_type: For consecutive mode - "bullish" or "bearish"
        min_change: Minimum price change % for pattern detection (default 2.0)
        limit: Maximum number of results (max 50)

    Returns:
        Dictionary with pattern results including:
        - exchange, timeframe, mode settings
        - total_found: Number of patterns detected
        - data: List of coins with pattern details
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "15m")
        min_change = max(0.5, min(20.0, min_change))
        limit = max(1, min(50, limit))

        # Get symbols for the exchange
        symbols = load_symbols(exchange)
        if not symbols:
            return {
                "error": f"No symbols found for exchange: {exchange}",
                "exchange": exchange,
                "timeframe": timeframe
            }

        # Limit symbols for performance (use configured max)
        max_symbols = min(len(symbols), API_MAX_SYMBOLS_PER_SCAN, limit * 3)
        symbols = symbols[:max_symbols]
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")

        # Process in batches with rate limiting
        all_analysis = {}
        total_batches = (len(symbols) + API_BATCH_SIZE - 1) // API_BATCH_SIZE
        failed_batches = 0
        last_error = None

        for i in range(0, len(symbols), API_BATCH_SIZE):
            batch_symbols = symbols[i:i + API_BATCH_SIZE]

            try:
                batch_analysis = _throttled_api_call(
                    get_multiple_analysis,
                    screener=screener,
                    interval=timeframe,
                    symbols=batch_symbols
                )
                all_analysis.update(batch_analysis)
            except Exception as e:
                failed_batches += 1
                last_error = e
                logger.warning(f"Candle pattern batch {i // API_BATCH_SIZE + 1}/{total_batches} failed: {_classify_api_error(e)}")
                continue

        # If all batches failed, return error
        if failed_batches == total_batches and last_error is not None:
            return {
                "error": _classify_api_error(last_error),
                "exchange": exchange,
                "timeframe": timeframe
            }

        analysis = all_analysis

        if mode == "advanced":
            # Advanced pattern analysis with scoring
            return _scan_advanced_patterns(
                analysis, exchange, timeframe, min_change, limit
            )
        else:
            # Default: Consecutive candle pattern detection
            return _scan_consecutive_patterns(
                analysis, exchange, timeframe, pattern_type, min_change, limit
            )

    except Exception as e:
        return {
            "error": f"Candle pattern scan failed: {str(e)}",
            "exchange": exchange,
            "timeframe": timeframe
        }


def _scan_consecutive_patterns(
    analysis: dict,
    exchange: str,
    timeframe: str,
    pattern_type: str,
    min_change: float,
    limit: int
) -> dict:
    """
    Scan for consecutive bullish/bearish candle patterns.

    Analyzes current candle characteristics along with technical indicators
    to identify symbols showing strong momentum patterns.

    Args:
        analysis: TradingView analysis results dictionary
        exchange: Exchange name
        timeframe: Analysis timeframe
        pattern_type: "bullish" or "bearish"
        min_change: Minimum price change threshold
        limit: Maximum results to return

    Returns:
        Dictionary with pattern results
    """
    pattern_coins = []

    for symbol, data in analysis.items():
        if data is None:
            continue

        try:
            indicators = data.indicators

            # Calculate current candle metrics
            open_price = indicators.get("open")
            close_price = indicators.get("close")
            high_price = indicators.get("high")
            low_price = indicators.get("low")
            volume = indicators.get("volume", 0)

            if not all([open_price, close_price, high_price, low_price]):
                continue

            # Calculate candle metrics
            current_change = ((close_price - open_price) / open_price) * 100
            candle_body = abs(close_price - open_price)
            candle_range = high_price - low_price
            body_to_range_ratio = candle_body / candle_range if candle_range > 0 else 0

            # Get momentum indicators
            rsi = indicators.get("RSI", 50)
            sma20 = indicators.get("SMA20", close_price)
            ema50 = indicators.get("EMA50", close_price)
            price_above_sma = close_price > sma20
            price_above_ema = close_price > ema50

            # Pattern detection logic
            pattern_detected = False
            pattern_strength = 0

            if pattern_type == "bullish":
                conditions = [
                    current_change > min_change,
                    body_to_range_ratio > 0.6,
                    price_above_sma,
                    rsi > 45 and rsi < 80,
                    volume > 1000
                ]
                pattern_strength = sum(conditions)
                pattern_detected = pattern_strength >= 3

            elif pattern_type == "bearish":
                conditions = [
                    current_change < -min_change,
                    body_to_range_ratio > 0.6,
                    not price_above_sma,
                    rsi < 55 and rsi > 20,
                    volume > 1000
                ]
                pattern_strength = sum(conditions)
                pattern_detected = pattern_strength >= 3

            if pattern_detected:
                metrics = compute_metrics(indicators)

                pattern_coins.append({
                    "symbol": symbol,
                    "price": round(close_price, 6),
                    "change_percent": round(current_change, 3),
                    "body_ratio": round(body_to_range_ratio, 3),
                    "pattern_strength": pattern_strength,
                    "volume": volume,
                    "bollinger_rating": metrics.get('rating', 0) if metrics else 0,
                    "rsi": round(rsi, 2),
                    "price_levels": {
                        "open": round(open_price, 6),
                        "high": round(high_price, 6),
                        "low": round(low_price, 6),
                        "close": round(close_price, 6)
                    },
                    "momentum": {
                        "above_sma20": price_above_sma,
                        "above_ema50": price_above_ema,
                        "strong_volume": volume > 5000
                    }
                })

        except Exception:
            continue

    # Sort by pattern strength and change
    if pattern_type == "bullish":
        pattern_coins.sort(key=lambda x: (x['pattern_strength'], x['change_percent']), reverse=True)
    else:
        pattern_coins.sort(key=lambda x: (x['pattern_strength'], -x['change_percent']), reverse=True)

    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "mode": "consecutive",
        "pattern_type": pattern_type,
        "min_change": min_change,
        "total_found": len(pattern_coins),
        "data": pattern_coins[:limit]
    }


def _scan_advanced_patterns(
    analysis: dict,
    exchange: str,
    timeframe: str,
    min_change: float,
    limit: int
) -> dict:
    """
    Advanced pattern analysis with multi-factor scoring.

    Uses candle body ratio, momentum, volume, and trend alignment to
    score pattern strength.

    Args:
        analysis: TradingView analysis results dictionary
        exchange: Exchange name
        timeframe: Analysis timeframe
        min_change: Minimum change threshold for strong momentum
        limit: Maximum results to return

    Returns:
        Dictionary with scored pattern results
    """
    pattern_results = []

    for symbol, data in analysis.items():
        if data is None:
            continue

        try:
            indicators = data.indicators
            pattern_score = _calculate_candle_pattern_score(indicators, 3, min_change)

            if pattern_score['detected']:
                metrics = compute_metrics(indicators)

                pattern_results.append({
                    "symbol": symbol,
                    "pattern_score": pattern_score['score'],
                    "pattern_details": pattern_score['details'],
                    "price": pattern_score['price'],
                    "change_percent": pattern_score['total_change'],
                    "body_ratio": pattern_score['body_ratio'],
                    "volume": pattern_score['volume'],
                    "bollinger_rating": metrics.get('rating', 0) if metrics else 0,
                    "technical": {
                        "rsi": round(indicators.get("RSI", 50), 2),
                        "momentum": "Strong" if abs(pattern_score['total_change']) > min_change else "Moderate",
                        "volume_level": "High" if pattern_score['volume'] > 10000 else "Normal"
                    }
                })

        except Exception:
            continue

    # Sort by pattern score
    pattern_results.sort(key=lambda x: (x['pattern_score'], abs(x['change_percent'])), reverse=True)

    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "mode": "advanced",
        "min_change": min_change,
        "total_found": len(pattern_results),
        "data": pattern_results[:limit]
    }

def _calculate_candle_pattern_score(indicators: dict, pattern_length: int, min_increase: float) -> dict:
    """
    Calculate a pattern detection score based on candle and indicator data.

    Analyzes current candle characteristics (body ratio, price change) along with
    technical indicators (RSI, EMA) to score the strength of a potential pattern.

    Args:
        indicators: Dictionary of TradingView indicator values
        pattern_length: Number of consecutive periods (used for context)
        min_increase: Minimum price change threshold for "strong momentum"

    Returns:
        Dictionary containing:
            - detected: Boolean indicating if pattern meets threshold
            - score: Numeric score (0-7+) based on conditions met
            - details: List of human-readable condition descriptions
            - price: Current close price
            - total_change: Percentage price change
            - body_ratio: Candle body to range ratio
            - volume: Current volume
    """
    try:
        open_price = indicators.get("open", 0)
        close_price = indicators.get("close", 0)
        high_price = indicators.get("high", 0)
        low_price = indicators.get("low", 0)
        volume = indicators.get("volume", 0)
        rsi = indicators.get("RSI", 50)
        
        if not all([open_price, close_price, high_price, low_price]):
            return {"detected": False, "score": 0}
        
        # Current candle analysis
        candle_body = abs(close_price - open_price)
        candle_range = high_price - low_price
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        # Price change
        price_change = ((close_price - open_price) / open_price) * 100
        
        # Pattern scoring
        score = 0
        details = []
        
        # Strong candle body
        if body_ratio > 0.7:
            score += 2
            details.append("Strong candle body")
        elif body_ratio > 0.5:
            score += 1
            details.append("Moderate candle body")
        
        # Significant price movement
        if abs(price_change) >= min_increase:
            score += 2
            details.append(f"Strong momentum ({price_change:.1f}%)")
        elif abs(price_change) >= min_increase / 2:
            score += 1
            details.append(f"Moderate momentum ({price_change:.1f}%)")
        
        # Volume confirmation
        if volume > 5000:
            score += 1
            details.append("Good volume")
        
        # RSI momentum
        if (price_change > 0 and 50 < rsi < 80) or (price_change < 0 and 20 < rsi < 50):
            score += 1
            details.append("RSI momentum aligned")
        
        # Trend consistency (using EMA vs price)
        ema50 = indicators.get("EMA50", close_price)
        if (price_change > 0 and close_price > ema50) or (price_change < 0 and close_price < ema50):
            score += 1
            details.append("Trend alignment")
        
        detected = score >= 3  # Minimum threshold
        
        return {
            "detected": detected,
            "score": score,
            "details": details,
            "price": round(close_price, 6),
            "total_change": round(price_change, 3),
            "body_ratio": round(body_ratio, 3),
            "volume": volume
        }
        
    except Exception as e:
        return {"detected": False, "score": 0, "error": str(e)}

def _fetch_multi_timeframe_patterns(exchange: str, symbols: List[str], base_tf: str, length: int, min_increase: float) -> List[dict]:
    """
    Fetch and analyze multi-timeframe patterns using tradingview-screener.

    Queries TradingView's screener API for OHLC and indicator data, then
    applies pattern scoring to identify symbols with strong patterns.

    Args:
        exchange: Exchange name (e.g., "KUCOIN", "BINANCE")
        symbols: List of symbols to analyze
        base_tf: Base timeframe for analysis (e.g., "15m", "1h")
        length: Pattern length parameter passed to scoring
        min_increase: Minimum increase threshold for pattern detection

    Returns:
        List of dictionaries with pattern scores, sorted by score descending.
        Each entry contains symbol, pattern_score, price, change, etc.
    """
    try:
        from tradingview_screener import Query
        from tradingview_screener.column import Column
        
        # Map timeframe to TradingView format
        tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D"}
        tv_interval = tf_map.get(base_tf, "15")
        
        # Create query for OHLC data
        cols = [
            f"open|{tv_interval}",
            f"close|{tv_interval}", 
            f"high|{tv_interval}",
            f"low|{tv_interval}",
            f"volume|{tv_interval}",
            "RSI"
        ]
        
        q = Query().set_markets("crypto").select(*cols)
        q = q.where(Column("exchange") == exchange.upper())
        q = q.limit(len(symbols))
        
        total, df = q.get_scanner_data()
        
        if df is None or df.empty:
            return []
        
        results = []
        
        for _, row in df.iterrows():
            symbol = row.get("ticker", "")
            
            try:
                open_val = row.get(f"open|{tv_interval}")
                close_val = row.get(f"close|{tv_interval}")
                high_val = row.get(f"high|{tv_interval}")
                low_val = row.get(f"low|{tv_interval}")
                volume_val = row.get(f"volume|{tv_interval}", 0)
                rsi_val = row.get("RSI", 50)
                
                if not all([open_val, close_val, high_val, low_val]):
                    continue
                
                # Calculate pattern metrics
                pattern_score = _calculate_candle_pattern_score({
                    "open": open_val,
                    "close": close_val,
                    "high": high_val,
                    "low": low_val,
                    "volume": volume_val,
                    "RSI": rsi_val
                }, length, min_increase)
                
                if pattern_score['detected']:
                    results.append({
                        "symbol": symbol,
                        "pattern_score": pattern_score['score'],
                        "price": pattern_score['price'],
                        "change": pattern_score['total_change'],
                        "body_ratio": pattern_score['body_ratio'],
                        "volume": volume_val,
                        "rsi": round(rsi_val, 2),
                        "details": pattern_score['details']
                    })
                    
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['pattern_score'], reverse=True)
        
    except Exception as e:
        return []

# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """
    List available exchanges from coinlist directory.

    Scans the coinlist directory for .txt files, each representing an exchange's
    symbol list. Returns a formatted string of available exchanges.

    Returns:
        String listing available exchanges, or fallback static list if directory
        not accessible.
    """
    try:
        import os
        # Get the directory where this module is located
        current_dir = os.path.dirname(__file__)
        coinlist_dir = os.path.join(current_dir, "coinlist")
        
        if os.path.exists(coinlist_dir):
            exchanges = []
            for filename in os.listdir(coinlist_dir):
                if filename.endswith('.txt'):
                    exchange_name = filename[:-4].upper()
                    exchanges.append(exchange_name)
            
            if exchanges:
                return f"Available exchanges: {', '.join(sorted(exchanges))}"
        
        # Fallback to static list
        return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, NASDAQ"
    except Exception:
        return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, NASDAQ"


# =============================================================================
# MCP Tools - Volume Analysis
# =============================================================================

@mcp.tool()
def volume_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    mode: str = "breakout",
    volume_multiplier: float = 2.0,
    price_change_min: float = 3.0,
    rsi_filter: str = "any",
    limit: int = 25
) -> list[dict]:
    """Unified volume scanner combining breakout detection and smart filtering.

    Scans for volume breakouts with optional RSI filtering. Use 'mode' to select
    between basic breakout detection and smart filtering with recommendations.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        mode: Scan mode:
            - "breakout": Basic volume breakout detection
            - "smart": Volume + RSI filtering with trading recommendations
        volume_multiplier: Minimum volume ratio vs average (default 2.0)
        price_change_min: Minimum price change percentage (default 3.0)
        rsi_filter: RSI filter (for smart mode):
            - "oversold": RSI < 30
            - "overbought": RSI > 70
            - "neutral": RSI 30-70
            - "any": No RSI filter
        limit: Number of rows to return (max 50)

    Returns:
        List of coins with volume breakout signals, sorted by volume strength.
        In smart mode, includes trading recommendations.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    volume_multiplier = max(1.5, min(10.0, volume_multiplier))
    price_change_min = max(1.0, min(20.0, price_change_min))
    limit = max(1, min(limit, 50))

    # Get symbols
    symbols = load_symbols(exchange)
    if not symbols:
        return []

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    volume_breakouts = []

    # Process in batches with rate limiting
    max_symbols = min(len(symbols), API_MAX_SYMBOLS_PER_SCAN)
    total_batches = (max_symbols + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    failed_batches = 0
    last_error = None

    for i in range(0, max_symbols, API_BATCH_SIZE):
        batch_symbols = symbols[i:i + API_BATCH_SIZE]

        try:
            analysis = _throttled_api_call(
                get_multiple_analysis,
                screener=screener,
                interval=timeframe,
                symbols=batch_symbols
            )
        except Exception as e:
            failed_batches += 1
            last_error = e
            logger.warning(f"Volume scanner batch {i // API_BATCH_SIZE + 1}/{total_batches} failed: {_classify_api_error(e)}")
            continue

        for symbol, data in analysis.items():
            try:
                if not data or not hasattr(data, 'indicators'):
                    continue

                indicators = data.indicators

                # Get required data
                volume = indicators.get('volume', 0)
                close = indicators.get('close', 0)
                open_price = indicators.get('open', 0)
                sma20_volume = indicators.get('volume.SMA20', 0)

                if not all([volume, close, open_price]) or volume <= 0:
                    continue

                # Calculate price change
                price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0

                # Volume ratio - skip if no historical data
                if not sma20_volume or sma20_volume <= 0:
                    continue
                volume_ratio = volume / sma20_volume

                # Check base conditions
                if abs(price_change) < price_change_min or volume_ratio < volume_multiplier:
                    continue

                # Get additional indicators
                rsi = indicators.get('RSI', 50)
                bb_upper = indicators.get('BB.upper', 0)
                bb_lower = indicators.get('BB.lower', 0)

                # Apply RSI filter in smart mode
                if mode == "smart":
                    if rsi_filter == "oversold" and rsi >= 30:
                        continue
                    elif rsi_filter == "overbought" and rsi <= 70:
                        continue
                    elif rsi_filter == "neutral" and (rsi <= 30 or rsi >= 70):
                        continue

                # Volume strength score (capped at 10x)
                volume_strength = min(10, volume_ratio)

                result = {
                    "symbol": symbol,
                    "change_percent": round(price_change, 2),
                    "volume_ratio": round(volume_ratio, 2),
                    "volume_strength": round(volume_strength, 1),
                    "current_volume": volume,
                    "breakout_type": "bullish" if price_change > 0 else "bearish",
                    "indicators": {
                        "close": close,
                        "rsi": round(rsi, 1),
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower
                    }
                }

                # Add trading recommendation in smart mode
                if mode == "smart":
                    if price_change > 0 and volume_ratio >= 2.0:
                        result["recommendation"] = "STRONG BUY" if rsi < 70 else "OVERBOUGHT - CAUTION"
                    elif price_change < 0 and volume_ratio >= 2.0:
                        result["recommendation"] = "STRONG SELL" if rsi > 30 else "OVERSOLD - OPPORTUNITY?"
                    else:
                        result["recommendation"] = "NEUTRAL"

                volume_breakouts.append(result)

            except Exception:
                continue

    # If all batches failed, raise an error with the last error message
    if failed_batches == total_batches and last_error is not None:
        raise RuntimeError(_classify_api_error(last_error))

    # Sort by volume strength, then price change
    volume_breakouts.sort(key=lambda x: (x["volume_strength"], abs(x["change_percent"])), reverse=True)

    return volume_breakouts[:limit]


@mcp.tool()
def volume_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
	"""Detailed volume confirmation analysis for a specific coin.

	Analyzes volume patterns in relation to price movements to identify
	confirmed breakouts, divergences, and weak signals.

	Args:
		symbol: Coin symbol (e.g., BTCUSDT)
		exchange: Exchange name
		timeframe: Time frame for analysis

	Returns:
		Detailed volume analysis with signals and recommendations
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	timeframe = sanitize_timeframe(timeframe, "15m")

	if not symbol.upper().endswith('USDT'):
		symbol = symbol.upper() + 'USDT'

	screener = EXCHANGE_SCREENER.get(exchange, "crypto")

	try:
		analysis = _throttled_api_call(
			get_multiple_analysis,
			screener=screener,
			interval=timeframe,
			symbols=[symbol]
		)

		if not analysis or symbol not in analysis:
			return {"error": f"No data found for {symbol}"}

		data = analysis[symbol]
		if not data or not hasattr(data, 'indicators'):
			return {"error": f"No indicator data for {symbol}"}

		indicators = data.indicators

		# Get volume data
		volume = indicators.get('volume', 0)
		close = indicators.get('close', 0)
		open_price = indicators.get('open', 0)
		high = indicators.get('high', 0)
		low = indicators.get('low', 0)

		# Calculate price metrics
		price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
		candle_range = ((high - low) / low) * 100 if low > 0 else 0

		# Volume analysis
		sma20_volume = indicators.get('volume.SMA20', 0)
		volume_ratio = volume / sma20_volume if sma20_volume > 0 else 1

		# Technical indicators
		rsi = indicators.get('RSI', 50)
		bb_upper = indicators.get('BB.upper', 0)
		bb_lower = indicators.get('BB.lower', 0)
		bb_middle = (bb_upper + bb_lower) / 2 if bb_upper and bb_lower else close

		# Volume confirmation signals
		signals = []

		# Strong volume + price breakout
		if volume_ratio >= 2.0 and abs(price_change) >= 3.0:
			signals.append(f"STRONG BREAKOUT: {volume_ratio:.1f}x volume + {price_change:.1f}% price")

		# Volume divergence
		if volume_ratio >= 1.5 and abs(price_change) < 1.0:
			signals.append(f"VOLUME DIVERGENCE: High volume ({volume_ratio:.1f}x) but low price movement")

		# Low volume on price move (weak signal)
		if abs(price_change) >= 2.0 and volume_ratio < 0.8:
			signals.append(f"WEAK SIGNAL: Price moved but volume is low ({volume_ratio:.1f}x)")

		# Bollinger Band + Volume confirmation
		if close > bb_upper and volume_ratio >= 1.5:
			signals.append(f"BB BREAKOUT CONFIRMED: Upper band breakout + volume confirmation")
		elif close < bb_lower and volume_ratio >= 1.5:
			signals.append(f"BB SELL CONFIRMED: Lower band breakout + volume confirmation")

		# RSI + Volume analysis
		if rsi > 70 and volume_ratio >= 2.0:
			signals.append(f"OVERBOUGHT + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")
		elif rsi < 30 and volume_ratio >= 2.0:
			signals.append(f"OVERSOLD + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")

		# Overall assessment
		if volume_ratio >= 3.0:
			volume_strength = "VERY STRONG"
		elif volume_ratio >= 2.0:
			volume_strength = "STRONG"
		elif volume_ratio >= 1.5:
			volume_strength = "MEDIUM"
		elif volume_ratio >= 1.0:
			volume_strength = "NORMAL"
		else:
			volume_strength = "WEAK"

		return {
			"symbol": symbol,
			"price_data": {
				"close": close,
				"change_percent": round(price_change, 2),
				"candle_range_percent": round(candle_range, 2)
			},
			"volume_analysis": {
				"current_volume": volume,
				"volume_ratio": round(volume_ratio, 2),
				"volume_strength": volume_strength,
				"average_volume": sma20_volume
			},
			"technical_indicators": {
				"RSI": round(rsi, 1),
				"BB_position": "ABOVE" if close > bb_upper else "BELOW" if close < bb_lower else "WITHIN",
				"BB_upper": bb_upper,
				"BB_lower": bb_lower
			},
			"signals": signals,
			"overall_assessment": {
				"bullish_signals": len([s for s in signals if "BREAKOUT" in s or "OVERSOLD" in s]),
				"bearish_signals": len([s for s in signals if "SELL" in s or "WEAK" in s]),
				"warning_signals": len([s for s in signals if "DIVERGENCE" in s])
			}
		}

	except Exception as e:
		return {"error": f"Analysis failed: {str(e)}"}


# =============================================================================
# MCP Tools - Advanced Indicator Scanners
# =============================================================================

@mcp.tool()
def pivot_points_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    pivot_type: str = "classic",
    position: str = "near_pivot",
    limit: int = 25
) -> list[dict]:
    """Scan for coins near pivot point levels.

    Identifies symbols that are trading near key pivot point levels,
    which often act as support/resistance zones.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        pivot_type: Type of pivot points - "classic", "fibonacci", or "camarilla"
        position: Price position filter:
            - "near_pivot": Price within 1% of pivot
            - "near_support": Price within 1% of S1, S2, or S3
            - "near_resistance": Price within 1% of R1, R2, or R3
            - "any": Any position near levels
        limit: Number of results (max 50)

    Returns:
        List of coins near pivot point levels with analysis
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 50))

    # Get symbols
    symbols = load_symbols(exchange)
    if not symbols:
        return []

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    results = []

    # Map pivot type to indicator keys
    pivot_prefix = {
        "classic": "Pivot.M.Classic",
        "fibonacci": "Pivot.M.Fibonacci",
        "camarilla": "Pivot.M.Camarilla"
    }.get(pivot_type, "Pivot.M.Classic")

    # Process in batches with rate limiting
    max_symbols = min(len(symbols), API_MAX_SYMBOLS_PER_SCAN)
    total_batches = (max_symbols + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    failed_batches = 0
    last_error = None

    for i in range(0, max_symbols, API_BATCH_SIZE):
        batch_symbols = symbols[i:i + API_BATCH_SIZE]

        try:
            analysis = _throttled_api_call(
                get_multiple_analysis,
                screener=screener,
                interval=timeframe,
                symbols=batch_symbols
            )
        except Exception as e:
            failed_batches += 1
            last_error = e
            logger.warning(f"Pivot scanner batch {i // API_BATCH_SIZE + 1}/{total_batches} failed: {_classify_api_error(e)}")
            continue

        for symbol, data in analysis.items():
            try:
                if not data or not hasattr(data, 'indicators'):
                    continue

                indicators = data.indicators
                close = indicators.get('close', 0)
                if not close:
                    continue

                # Get pivot points
                pivot = indicators.get(f"{pivot_prefix}.Middle", 0)
                r1 = indicators.get(f"{pivot_prefix}.R1", 0)
                r2 = indicators.get(f"{pivot_prefix}.R2", 0)
                r3 = indicators.get(f"{pivot_prefix}.R3", 0)
                s1 = indicators.get(f"{pivot_prefix}.S1", 0)
                s2 = indicators.get(f"{pivot_prefix}.S2", 0)
                s3 = indicators.get(f"{pivot_prefix}.S3", 0)

                if not pivot:
                    continue

                # Calculate proximity to levels (within 1%)
                proximity_threshold = 0.01  # 1%

                def is_near(price, level):
                    if not level:
                        return False
                    return abs(price - level) / level <= proximity_threshold

                # Determine position
                near_levels = []
                if is_near(close, pivot):
                    near_levels.append(("Pivot", pivot))
                if is_near(close, s1):
                    near_levels.append(("S1", s1))
                if is_near(close, s2):
                    near_levels.append(("S2", s2))
                if is_near(close, s3):
                    near_levels.append(("S3", s3))
                if is_near(close, r1):
                    near_levels.append(("R1", r1))
                if is_near(close, r2):
                    near_levels.append(("R2", r2))
                if is_near(close, r3):
                    near_levels.append(("R3", r3))

                if not near_levels:
                    continue

                # Apply position filter
                if position == "near_pivot" and not any(l[0] == "Pivot" for l in near_levels):
                    continue
                elif position == "near_support" and not any(l[0].startswith("S") for l in near_levels):
                    continue
                elif position == "near_resistance" and not any(l[0].startswith("R") for l in near_levels):
                    continue

                # Calculate price change
                open_price = indicators.get('open', close)
                change_pct = ((close - open_price) / open_price) * 100 if open_price else 0

                # Determine trading signal based on position
                signal = "NEUTRAL"
                if any(l[0].startswith("S") for l in near_levels) and change_pct > 0:
                    signal = "POTENTIAL_BOUNCE"
                elif any(l[0].startswith("R") for l in near_levels) and change_pct < 0:
                    signal = "POTENTIAL_REJECTION"
                elif any(l[0] == "Pivot" for l in near_levels):
                    signal = "AT_PIVOT"

                results.append({
                    "symbol": symbol,
                    "price": round(close, 6),
                    "change_percent": round(change_pct, 2),
                    "pivot_type": pivot_type,
                    "near_levels": [{"level": l[0], "price": round(l[1], 6)} for l in near_levels],
                    "pivot_levels": {
                        "pivot": round(pivot, 6) if pivot else None,
                        "r1": round(r1, 6) if r1 else None,
                        "r2": round(r2, 6) if r2 else None,
                        "r3": round(r3, 6) if r3 else None,
                        "s1": round(s1, 6) if s1 else None,
                        "s2": round(s2, 6) if s2 else None,
                        "s3": round(s3, 6) if s3 else None
                    },
                    "signal": signal,
                    "rsi": round(indicators.get("RSI", 50), 2)
                })

            except Exception:
                continue

    # If all batches failed, raise an error with the last error message
    if failed_batches == total_batches and last_error is not None:
        raise RuntimeError(_classify_api_error(last_error))

    # Sort by number of near levels (more levels = stronger signal)
    results.sort(key=lambda x: len(x["near_levels"]), reverse=True)

    return results[:limit]


@mcp.tool()
def tradingview_recommendation(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    signal_filter: str = "STRONG_BUY",
    min_strength: float = 0.5,
    limit: int = 25
) -> list[dict]:
    """Scan for coins based on TradingView's technical analysis recommendations.

    Uses TradingView's built-in technical analysis engine which combines
    multiple indicators (MAs, oscillators) to generate Buy/Sell signals.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        signal_filter: Filter by signal type:
            - "STRONG_BUY": Recommendation >= 0.5
            - "BUY": Recommendation >= 0.1
            - "NEUTRAL": Recommendation between -0.1 and 0.1
            - "SELL": Recommendation <= -0.1
            - "STRONG_SELL": Recommendation <= -0.5
            - "any": All signals
        min_strength: Minimum absolute recommendation value (0.0 to 1.0)
        limit: Number of results (max 50)

    Returns:
        List of coins matching recommendation criteria with detailed signals
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    min_strength = max(0.0, min(1.0, min_strength))
    limit = max(1, min(limit, 50))

    # Get symbols
    symbols = load_symbols(exchange)
    if not symbols:
        return []

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    results = []

    def get_signal_text(value):
        """Convert recommendation value to signal text."""
        if value >= 0.5:
            return "STRONG_BUY"
        elif value >= 0.1:
            return "BUY"
        elif value > -0.1:
            return "NEUTRAL"
        elif value > -0.5:
            return "SELL"
        else:
            return "STRONG_SELL"

    # Process in batches with rate limiting
    max_symbols = min(len(symbols), API_MAX_SYMBOLS_PER_SCAN)
    total_batches = (max_symbols + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    failed_batches = 0
    last_error = None

    for i in range(0, max_symbols, API_BATCH_SIZE):
        batch_symbols = symbols[i:i + API_BATCH_SIZE]

        try:
            analysis = _throttled_api_call(
                get_multiple_analysis,
                screener=screener,
                interval=timeframe,
                symbols=batch_symbols
            )
        except Exception as e:
            failed_batches += 1
            last_error = e
            logger.warning(f"TV recommendation batch {i // API_BATCH_SIZE + 1}/{total_batches} failed: {_classify_api_error(e)}")
            continue

        for symbol, data in analysis.items():
            try:
                if not data or not hasattr(data, 'indicators'):
                    continue

                indicators = data.indicators

                # Get recommendation values
                recommend_all = indicators.get("Recommend.All", 0) or 0
                recommend_ma = indicators.get("Recommend.MA", 0) or 0
                recommend_other = indicators.get("Recommend.Other", 0) or 0

                # Get signal type
                signal = get_signal_text(recommend_all)

                # Apply signal filter
                if signal_filter != "any":
                    if signal_filter == "STRONG_BUY" and recommend_all < 0.5:
                        continue
                    elif signal_filter == "BUY" and (recommend_all < 0.1 or recommend_all >= 0.5):
                        continue
                    elif signal_filter == "NEUTRAL" and (recommend_all <= -0.1 or recommend_all >= 0.1):
                        continue
                    elif signal_filter == "SELL" and (recommend_all > -0.1 or recommend_all <= -0.5):
                        continue
                    elif signal_filter == "STRONG_SELL" and recommend_all > -0.5:
                        continue

                # Apply minimum strength filter
                if abs(recommend_all) < min_strength:
                    continue

                # Get price data
                close = indicators.get('close', 0)
                open_price = indicators.get('open', close)
                change_pct = ((close - open_price) / open_price) * 100 if open_price else 0

                results.append({
                    "symbol": symbol,
                    "price": round(close, 6) if close else None,
                    "change_percent": round(change_pct, 2),
                    "recommendations": {
                        "overall": {
                            "value": round(recommend_all, 3),
                            "signal": signal
                        },
                        "moving_averages": {
                            "value": round(recommend_ma, 3),
                            "signal": get_signal_text(recommend_ma)
                        },
                        "oscillators": {
                            "value": round(recommend_other, 3),
                            "signal": get_signal_text(recommend_other)
                        }
                    },
                    "signal_strength": abs(round(recommend_all, 3)),
                    "agreement": "ALIGNED" if (recommend_ma > 0) == (recommend_other > 0) else "DIVERGENT",
                    "technical_data": {
                        "rsi": round(indicators.get("RSI", 50), 2),
                        "macd": round(indicators.get("MACD.macd", 0), 6),
                        "adx": round(indicators.get("ADX", 0), 2)
                    }
                })

            except Exception:
                continue

    # If all batches failed, raise an error with the last error message
    if failed_batches == total_batches and last_error is not None:
        raise RuntimeError(_classify_api_error(last_error))

    # Sort by signal strength (strongest first)
    results.sort(key=lambda x: x["signal_strength"], reverse=True)

    return results[:limit]


# =============================================================================
# Health Check Endpoints (for HTTP mode)
# =============================================================================

def register_health_routes(server: FastMCP) -> None:
    """Register health check routes on the given server instance."""
    if not STARLETTE_AVAILABLE:
        return

    @server.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        """Health check endpoint for deployment platforms."""
        return JSONResponse({
            "status": "healthy",
            "service": "sigmapilot-mcp",
            "version": "2.0.0",
        })

    @server.custom_route("/", methods=["GET"])
    async def root_health(request: Request) -> JSONResponse:
        """Root endpoint returns health status."""
        return JSONResponse({
            "status": "healthy",
            "service": "sigmapilot-mcp",
            "version": "2.0.0",
            "docs": "Use /health for health checks, /mcp for MCP protocol"
        })


def register_tools(server: FastMCP) -> None:
    """Register all MCP tools on the given server instance.

    This is used for HTTP mode where we create a fresh server with auth configuration.
    The tools are defined above with @mcp.tool() decorators for stdio mode.
    """
    # Register all tools - they share implementation with stdio mode
    server.tool()(top_gainers)
    server.tool()(top_losers)
    server.tool()(bollinger_scan)
    server.tool()(rating_filter)
    server.tool()(coin_analysis)
    server.tool()(candle_pattern_scanner)
    server.tool()(volume_scanner)
    server.tool()(volume_analysis)
    server.tool()(pivot_points_scanner)
    server.tool()(tradingview_recommendation)

    # Register resources
    server.resource("exchanges://list")(exchanges_list)


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """
    Main entry point for the SigmaPilot MCP server.

    Parses command line arguments and starts the server in either stdio mode
    (for Claude Desktop) or HTTP mode (for remote access with optional Auth0).

    Command line options:
        transport: "stdio" (default) or "streamable-http"
        --host: Server host for HTTP mode (default: 0.0.0.0)
        --port: Server port for HTTP mode (default: 8000)
        --auth: Enable Auth0 authentication (HTTP mode only)

    Environment Variables:
        AUTH0_DOMAIN: Auth0 tenant domain (e.g., your-tenant.auth0.com)
        AUTH0_AUDIENCE: API identifier from Auth0
        RESOURCE_SERVER_URL: Public URL for OAuth (e.g., https://your-app.railway.app/mcp)
        HOST: Override default host
        PORT: Override default port
    """
    global mcp

    parser = argparse.ArgumentParser(description="SigmaPilot MCP server")
    parser.add_argument(
        "transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport mode (default: stdio)"
    )
    parser.add_argument("--host", default=HTTP_HOST)
    parser.add_argument("--port", type=int, default=HTTP_PORT)
    parser.add_argument("--auth", action="store_true", help="Enable Auth0 authentication")
    args = parser.parse_args()

    # Debug logging if enabled
    if os.environ.get("DEBUG_MCP"):
        import sys
        print(f"[DEBUG_MCP] cwd={os.getcwd()} argv={sys.argv}", file=sys.stderr, flush=True)

    if args.transport == "stdio":
        # Standard I/O mode for Claude Desktop - no auth needed
        logger.info("Starting SigmaPilot MCP in stdio mode")
        mcp.run()
    else:
        # HTTP mode - create a new server with proper host/port configuration
        enable_auth = args.auth or bool(AUTH0_DOMAIN and AUTH0_AUDIENCE)

        # Create server with proper configuration for HTTP mode
        server = create_mcp_server(
            enable_auth=enable_auth,
            host=args.host,
            port=args.port
        )

        # Register all tools on the HTTP server (they're defined as functions below)
        register_tools(server)

        # Register health check routes
        register_health_routes(server)

        if enable_auth and AUTH0_DOMAIN and AUTH0_AUDIENCE:
            logger.info(f"Auth0 enabled - Domain: {AUTH0_DOMAIN}")
        else:
            logger.warning("Running without authentication (development mode)")
            logger.info("Set AUTH0_DOMAIN and AUTH0_AUDIENCE for production")

        logger.info(f"Starting SigmaPilot MCP on {args.host}:{args.port}")
        logger.info(f"Health check: http://{args.host}:{args.port}/health")
        logger.info(f"Environment: PORT={os.environ.get('PORT', 'not set')}, HOST={os.environ.get('HOST', 'not set')}")

        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()

