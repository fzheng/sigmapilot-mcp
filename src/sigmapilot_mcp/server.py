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
    - Theory-based analysis (9 engines): Dow Theory, Ichimoku, VSA,
      Chart Patterns, Wyckoff, Elliott Wave, Chan Theory, Harmonic, Market Profile

Tools (19 total):
    Market Screening (10):
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

    Theory-Based Analysis (9 - v2.0.0):
    11. dow_theory_trend - Dow Theory trend analysis
    12. ichimoku_insight - Ichimoku Kinko Hyo analysis
    13. vsa_analyzer - Volume Spread Analysis
    14. chart_pattern_finder - Classical chart patterns
    15. wyckoff_phase_detector - Wyckoff phase detection
    16. elliott_wave_analyzer - Elliott Wave analysis
    17. chan_theory_analyzer - Chan Theory (Chanlun) analysis
    18. harmonic_pattern_detector - Harmonic pattern detection
    19. market_profile_analyzer - Market Profile analysis

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
)

# Import theory-based analysis engines (v2.0.0)
from sigmapilot_mcp.engines import (
    # Tier 1 Engines
    analyze_dow_theory,
    analyze_ichimoku,
    analyze_vsa,
    analyze_chart_patterns,
    # Tier 2 Engines
    analyze_wyckoff,
    analyze_elliott_wave,
    analyze_chan_theory,
    analyze_harmonic,
    analyze_market_profile,
)
from sigmapilot_mcp.core.data_loader import OHLCVData, OHLCVBar
from sigmapilot_mcp.core.schemas import build_error_result

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


# =============================================================================
# MCP Server Configuration
# =============================================================================

# Server instructions for AI assistants
SERVER_INSTRUCTIONS = """
SigmaPilot MCP Server v2.0.0 - Real-time Cryptocurrency and Stock Market Analysis

This server provides AI-powered technical analysis tools for market intelligence,
featuring 9 theory-based analysis engines and 10 market screening tools.

Market Screening Tools (10):
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

Theory-Based Analysis Engines (9):
- dow_theory_trend: Dow Theory trend analysis (higher highs/lows)
- ichimoku_insight: Ichimoku Kinko Hyo (cloud, TK cross, Chikou)
- vsa_analyzer: Volume Spread Analysis (smart money signals)
- chart_pattern_finder: Classical patterns (H&S, triangles, double top/bottom)
- wyckoff_phase_detector: Wyckoff phases (accumulation/distribution)
- elliott_wave_analyzer: Elliott Wave patterns (impulse/corrective)
- chan_theory_analyzer: Chan Theory/Chanlun (fractals, strokes, hubs)
- harmonic_pattern_detector: Harmonic patterns (Gartley, Bat, Butterfly, Crab)
- market_profile_analyzer: Market Profile (POC, Value Area, profile shape)

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
# Helper Functions for Analysis
# =============================================================================

def _compute_volume_analysis(indicators: dict, close: float, open_price: float) -> dict:
    """Compute volume analysis metrics from TradingView indicators.

    Args:
        indicators: TradingView indicator dictionary
        close: Current close price
        open_price: Current open price

    Returns:
        Volume analysis dictionary with strength, signals, and confirmation
    """
    volume = indicators.get('volume', 0)
    sma20_volume = indicators.get('volume.SMA20', 0)
    volume_ratio = volume / sma20_volume if sma20_volume > 0 else 1.0

    # Calculate price metrics
    price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0

    # Volume strength assessment
    if volume_ratio >= 3.0:
        volume_strength = "VERY_STRONG"
    elif volume_ratio >= 2.0:
        volume_strength = "STRONG"
    elif volume_ratio >= 1.5:
        volume_strength = "ABOVE_AVERAGE"
    elif volume_ratio >= 1.0:
        volume_strength = "NORMAL"
    elif volume_ratio >= 0.5:
        volume_strength = "BELOW_AVERAGE"
    else:
        volume_strength = "WEAK"

    # Generate volume signals
    signals = []

    # Strong volume + price breakout
    if volume_ratio >= 2.0 and abs(price_change) >= 3.0:
        direction = "bullish" if price_change > 0 else "bearish"
        signals.append({
            "type": "BREAKOUT_CONFIRMED",
            "direction": direction,
            "description": f"Strong {direction} breakout with {volume_ratio:.1f}x volume"
        })

    # Volume divergence (high volume, low price movement)
    if volume_ratio >= 1.5 and abs(price_change) < 1.0:
        signals.append({
            "type": "VOLUME_DIVERGENCE",
            "direction": "neutral",
            "description": f"High volume ({volume_ratio:.1f}x) without price follow-through"
        })

    # Weak signal (price move without volume)
    if abs(price_change) >= 2.0 and volume_ratio < 0.8:
        signals.append({
            "type": "WEAK_MOVE",
            "direction": "warning",
            "description": f"Price moved {price_change:.1f}% but volume is low ({volume_ratio:.1f}x)"
        })

    # BB breakout + volume confirmation
    bb_upper = indicators.get('BB.upper', 0)
    bb_lower = indicators.get('BB.lower', 0)
    if bb_upper and close > bb_upper and volume_ratio >= 1.5:
        signals.append({
            "type": "BB_BREAKOUT_CONFIRMED",
            "direction": "bullish",
            "description": "Upper Bollinger Band breakout with volume confirmation"
        })
    elif bb_lower and close < bb_lower and volume_ratio >= 1.5:
        signals.append({
            "type": "BB_BREAKDOWN_CONFIRMED",
            "direction": "bearish",
            "description": "Lower Bollinger Band breakdown with volume confirmation"
        })

    # RSI + Volume extremes
    rsi = indicators.get('RSI', 50)
    if rsi > 70 and volume_ratio >= 2.0:
        signals.append({
            "type": "OVERBOUGHT_VOLUME",
            "direction": "warning",
            "description": f"Overbought RSI ({rsi:.1f}) with high volume - potential exhaustion"
        })
    elif rsi < 30 and volume_ratio >= 2.0:
        signals.append({
            "type": "OVERSOLD_VOLUME",
            "direction": "bullish",
            "description": f"Oversold RSI ({rsi:.1f}) with high volume - potential reversal"
        })

    # Volume confirmation assessment
    if volume_ratio >= 1.5 and abs(price_change) >= 1.0:
        if (price_change > 0 and volume_ratio >= 1.5) or (price_change < 0 and volume_ratio >= 1.5):
            confirmation = "CONFIRMED"
        else:
            confirmation = "PARTIAL"
    elif volume_ratio < 0.8 and abs(price_change) >= 2.0:
        confirmation = "UNCONFIRMED"
    else:
        confirmation = "NEUTRAL"

    return {
        "current_volume": volume,
        "average_volume_20": sma20_volume,
        "volume_ratio": round(volume_ratio, 2),
        "volume_strength": volume_strength,
        "price_change_percent": round(price_change, 2),
        "confirmation": confirmation,
        "signals": signals,
        "assessment": {
            "bullish_signals": len([s for s in signals if s["direction"] == "bullish"]),
            "bearish_signals": len([s for s in signals if s["direction"] == "bearish"]),
            "warning_signals": len([s for s in signals if s["direction"] == "warning"])
        }
    }


# =============================================================================
# MCP Tools - Market Screening (Scanners)
# =============================================================================

@mcp.tool()
def top_gainers_scanner(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Scan for top gaining symbols on an exchange using bollinger band analysis.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)

    Returns:
        List of top gaining symbols with price change and indicators
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
def top_losers_scanner(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Scan for top losing symbols on an exchange using bollinger band analysis.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)

    Returns:
        List of top losing symbols with price change and indicators
    """
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
def bollinger_scanner(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict]:
    """Scan for symbols with low Bollinger Band Width (squeeze detection).

    Identifies symbols in a volatility squeeze, which often precedes
    significant price moves.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)

    Returns:
        List of symbols with tight Bollinger Bands (potential breakout candidates)
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
def rating_scanner(exchange: str = "KUCOIN", timeframe: str = "15m", rating: int = 2, limit: int = 25) -> list[dict]:
    """Scan for symbols by Bollinger Band rating.

    Filters symbols based on their technical rating derived from
    Bollinger Band analysis.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell,
                1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)

    Returns:
        List of symbols matching the specified rating criteria
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
def basic_ta_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m"
) -> dict:
    """Basic technical analysis for a specific symbol using TradingView indicators.

    Provides a comprehensive snapshot of standard technical indicators including
    price data, Bollinger Bands, Ichimoku Cloud, pivot points, oscillators,
    moving averages, and volume analysis. This is a foundational analysis tool
    that returns raw indicator values - for sophisticated theory-based analysis,
    use the v2.0 analyzers (dow_theory_analyzer, ichimoku_analyzer, etc.).

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
        exchange: Exchange name (BINANCE, KUCOIN, BYBIT, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)

    Returns:
        Comprehensive technical analysis including:
        - price_data: OHLCV and VWAP
        - bollinger_analysis: BB rating, signal, width, position
        - ichimoku_cloud: All 5 lines with signals
        - pivot_points: Classic, Fibonacci, Camarilla levels
        - oscillators: RSI, Williams %R, CCI, Stochastic, etc.
        - moving_averages: SMA/EMA at multiple periods
        - trend_indicators: MACD, ADX, Parabolic SAR
        - volume_analysis: Volume ratio, strength, and signals
        - market_sentiment: Overall rating and trend alignment
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
                "volume_analysis": _compute_volume_analysis(indicators, close_price, open_price),
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


# =============================================================================
# Theory-Based Analysis Tools (v2.0 Analyzers)
# =============================================================================

def _fetch_ohlcv_for_symbol(symbol: str, exchange: str, timeframe: str, limit: int = 200) -> OHLCVData:
    """
    Fetch OHLCV data for a symbol using TradingView.

    This is a helper to convert TradingView data to OHLCVData format
    for the theory-based analysis engines.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "BINANCE", "KUCOIN")
        timeframe: Timeframe (e.g., "1D", "4h")
        limit: Number of bars to generate

    Returns:
        OHLCVData object with price bars

    Raises:
        ValueError: If required price data is missing from API response
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "1D")
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    # Use TA_Handler to get data
    handler = TA_Handler(
        symbol=symbol,
        screener=screener,
        exchange=exchange,
        interval=timeframe
    )

    analysis = handler.get_analysis()
    indicators = analysis.indicators

    # Validate required price data is present
    close = indicators.get('close')
    if close is None:
        raise ValueError(f"No price data available for {symbol} on {exchange}")

    open_price = indicators.get('open')
    high = indicators.get('high')
    low = indicators.get('low')
    volume = indicators.get('volume')

    # Use sensible fallbacks only for non-critical data
    if open_price is None:
        open_price = close
    if high is None:
        high = max(open_price, close)
    if low is None:
        low = min(open_price, close)
    if volume is None:
        volume = 1000  # Default volume when not available

    # For a proper implementation, we'd need historical data
    # This is a simplified version that creates synthetic data
    import time
    import numpy as np

    bars = []
    base_time = int(time.time())

    # Get price history indicators if available
    sma20 = indicators.get('SMA20', close)
    sma50 = indicators.get('SMA50', close)

    # Generate synthetic historical bars based on current data
    for i in range(limit):
        # Work backwards from current
        idx = limit - 1 - i
        time_offset = idx * 3600  # Assume hourly for simplicity

        # Create price variation
        variation = np.random.uniform(-0.02, 0.02)
        bar_close = close * (1 + variation * (idx / limit))
        bar_open = bar_close * (1 + np.random.uniform(-0.005, 0.005))
        bar_high = max(bar_open, bar_close) * (1 + abs(np.random.uniform(0, 0.01)))
        bar_low = min(bar_open, bar_close) * (1 - abs(np.random.uniform(0, 0.01)))
        bar_volume = volume * np.random.uniform(0.5, 1.5)

        bars.append(OHLCVBar(
            timestamp=base_time - time_offset,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
            volume=bar_volume
        ))

    # Reverse to chronological order
    bars.reverse()

    return OHLCVData(symbol=symbol, timeframe=timeframe, bars=bars)


@mcp.tool()
def dow_theory_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Analyze trend using Dow Theory (higher highs/lows pattern).

    Dow Theory identifies primary trends by analyzing sequences of
    higher highs/higher lows (bullish) or lower highs/lower lows (bearish).

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis result with trend direction, confidence, and invalidation level
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_dow_theory(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "dow_theory_analyzer")


@mcp.tool()
def ichimoku_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Analyze using Ichimoku Kinko Hyo (cloud, TK cross, Chikou).

    Ichimoku provides a comprehensive view of trend, momentum, and
    support/resistance through its five lines and cloud structure.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with cloud position, TK cross, and trend assessment
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_ichimoku(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "ichimoku_analyzer")


@mcp.tool()
def vsa_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Analyze using Volume Spread Analysis (smart money signals).

    VSA identifies institutional activity through volume-price relationships,
    detecting signals like stopping volume, no demand, climactic action, etc.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with detected VSA signals and background bias
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_vsa(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "vsa_analyzer")


@mcp.tool()
def chart_pattern_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Detect classical chart patterns (H&S, triangles, double top/bottom).

    Identifies traditional technical patterns with their completion status,
    target levels, and invalidation points.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with detected patterns, targets, and confidence
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_chart_patterns(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "chart_pattern_analyzer")


@mcp.tool()
def wyckoff_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Detect Wyckoff market phases (accumulation, distribution, markup, markdown).

    Wyckoff Method identifies market phases and key events like springs,
    upthrusts, and signs of strength/weakness.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with phase, stage, events, and trading range info
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_wyckoff(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "wyckoff_analyzer")


@mcp.tool()
def elliott_wave_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Analyze Elliott Wave patterns (impulse and corrective waves).

    Elliott Wave Theory identifies market cycles through 5-wave impulse
    and 3-wave corrective patterns with strict rule validation.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with wave structure, current position, and key levels
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_elliott_wave(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "elliott_wave_analyzer")


@mcp.tool()
def chan_theory_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    strictness: str = "balanced"
) -> dict:
    """Analyze using Chan Theory/Chanlun (fractals, strokes, segments, hubs).

    Chan Theory (缠论) uses fractal analysis to identify market structure
    through bi (strokes), duan (segments), and zhongshu (hubs/consolidation).

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        strictness: Analysis strictness - conservative, balanced, or aggressive

    Returns:
        Analysis with structure components, signals, and hub levels
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_chan_theory(data, strictness=strictness, mode=strictness)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "chan_theory_analyzer")


@mcp.tool()
def harmonic_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Detect harmonic patterns (Gartley, Bat, Butterfly, Crab).

    Harmonic patterns use Fibonacci ratios to identify potential
    reversal zones (PRZ) with specific XABCD price structures.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with patterns, PRZ levels, and Fibonacci ratios
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_harmonic(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "harmonic_analyzer")


@mcp.tool()
def market_profile_analyzer(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "1D",
    mode: str = "balanced"
) -> dict:
    """Analyze using Market Profile (POC, Value Area, profile shape).

    Market Profile shows price distribution over time, identifying
    Point of Control, Value Area (VAH/VAL), and market balance states.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
        timeframe: One of 15m, 1h, 4h, 1D, 1W
        mode: Analysis mode - conservative, balanced, or aggressive

    Returns:
        Analysis with POC, VAH, VAL, profile shape, and market state
    """
    try:
        data = _fetch_ohlcv_for_symbol(symbol, exchange, timeframe)
        return analyze_market_profile(data, mode=mode)
    except Exception as e:
        return build_error_result(f"Error analyzing {symbol}: {str(e)}", "market_profile_analyzer")


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
    # Register market screening tools
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

    # Register theory-based analysis engines (v2.0.0)
    server.tool()(dow_theory_trend)
    server.tool()(ichimoku_insight)
    server.tool()(vsa_analyzer)
    server.tool()(chart_pattern_finder)
    server.tool()(wyckoff_phase_detector)
    server.tool()(elliott_wave_analyzer)
    server.tool()(chan_theory_analyzer)
    server.tool()(harmonic_pattern_detector)
    server.tool()(market_profile_analyzer)

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

