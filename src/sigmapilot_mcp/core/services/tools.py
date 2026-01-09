"""
Shared MCP Tools Implementation.

This module provides the core implementation of market analysis tools
that are shared between the local (stdio) and remote (HTTP) entry points.

This eliminates code duplication and ensures consistent behavior across
both deployment modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .indicators import compute_metrics
from .coinlist import load_symbols
from ..utils.validators import (
    sanitize_timeframe,
    sanitize_exchange,
    EXCHANGE_SCREENER,
    BBW_HIGH_VOLATILITY,
    BBW_MEDIUM_VOLATILITY,
    ADX_STRONG_TREND,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    DEFAULT_BATCH_SIZE,
    VOLUME_MINIMUM,
)

# Configure module logger
logger = logging.getLogger(__name__)

# TradingView library availability
try:
    from tradingview_ta import get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False


@dataclass
class BatchResult:
    """Result of a batch API operation with success/failure tracking."""
    data: List[Dict[str, Any]] = field(default_factory=list)
    failed_batches: int = 0
    total_batches: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def partial_failure(self) -> bool:
        """Check if some batches failed but others succeeded."""
        return self.failed_batches > 0 and self.failed_batches < self.total_batches

    @property
    def total_failure(self) -> bool:
        """Check if all batches failed."""
        return self.failed_batches > 0 and self.failed_batches == self.total_batches


def fetch_analysis(
    exchange: str,
    timeframe: str,
    limit: int = 50,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BatchResult:
    """
    Fetch analysis data for symbols on an exchange.

    Args:
        exchange: Exchange name (e.g., "kucoin", "binance")
        timeframe: Time interval (e.g., "15m", "4h")
        limit: Maximum number of results
        batch_size: Number of symbols per API batch

    Returns:
        BatchResult containing analysis data and error tracking

    Raises:
        RuntimeError: If tradingview_ta is not available
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is not available. Install with: pip install tradingview-ta")

    symbols = load_symbols(exchange)
    if not symbols:
        logger.warning(f"No symbols found for exchange: {exchange}")
        return BatchResult(errors=[f"No symbols found for exchange: {exchange}"])

    # Get more symbols than needed to account for filtering
    symbols = symbols[:limit * 2]
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    result = BatchResult()
    result.total_batches = 1  # Single batch for simple fetch

    try:
        analysis = get_multiple_analysis(
            screener=screener,
            interval=timeframe,
            symbols=symbols
        )
    except Exception as e:
        error_msg = f"TradingView API error: {str(e)}"
        logger.error(error_msg)
        result.failed_batches = 1
        result.errors.append(error_msg)
        return result

    for key, value in analysis.items():
        if value is None:
            continue

        try:
            indicators = value.indicators
            metrics = compute_metrics(indicators)

            if not metrics or metrics.get('bbw') is None:
                continue

            result.data.append({
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
        except (TypeError, KeyError) as e:
            logger.debug(f"Skipping symbol {key}: {e}")
            continue

    return result


def fetch_trending_analysis(
    exchange: str,
    timeframe: str = "5m",
    filter_type: str = "",
    rating_filter: Optional[int] = None,
    limit: int = 50,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BatchResult:
    """
    Fetch trending coins with technical analysis data.

    Processes symbols in batches and tracks partial failures.

    Args:
        exchange: Exchange name
        timeframe: TradingView interval
        filter_type: Optional filter - use "rating" to enable rating_filter
        rating_filter: When filter_type="rating", only return symbols with this rating
        limit: Maximum number of results
        batch_size: Number of symbols per API batch

    Returns:
        BatchResult containing trending data and error tracking
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is not available. Install with: pip install tradingview-ta")

    symbols = load_symbols(exchange)
    if not symbols:
        logger.warning(f"No symbols found for exchange: {exchange}")
        return BatchResult(errors=[f"No symbols found for exchange: {exchange}"])

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    result = BatchResult()

    # Calculate total batches
    result.total_batches = (len(symbols) + batch_size - 1) // batch_size

    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]

        try:
            analysis = get_multiple_analysis(
                screener=screener,
                interval=timeframe,
                symbols=batch_symbols
            )
        except Exception as e:
            error_msg = f"Batch {i // batch_size + 1} failed: {str(e)}"
            logger.warning(error_msg)
            result.failed_batches += 1
            result.errors.append(error_msg)
            continue

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

                result.data.append({
                    "symbol": key,
                    "changePercent": metrics['change'],
                    "price": metrics['price'],
                    "bbw": metrics['bbw'],
                    "rating": metrics['rating'],
                    "signal": metrics['signal'],
                    "indicators": {
                        "open": metrics.get('open'),
                        "close": metrics.get('price'),
                        "SMA20": indicators.get("SMA20"),
                        "BB_upper": indicators.get("BB.upper"),
                        "BB_lower": indicators.get("BB.lower"),
                        "EMA9": indicators.get("EMA9"),
                        "EMA21": indicators.get("EMA21"),
                        "EMA50": indicators.get("EMA50"),
                        "RSI": indicators.get("RSI"),
                        "ATR": indicators.get("ATR"),
                        "volume": indicators.get("volume"),
                    }
                })

            except (TypeError, ZeroDivisionError, KeyError) as e:
                logger.debug(f"Skipping symbol {key}: {e}")
                continue

    # Sort by change percentage (highest first)
    result.data.sort(key=lambda x: x["changePercent"], reverse=True)

    return result


def get_top_gainers(
    exchange: str = "kucoin",
    timeframe: str = "15m",
    limit: int = 25
) -> Tuple[List[Dict], Optional[str]]:
    """
    Get top gaining assets on an exchange.

    Args:
        exchange: Exchange name
        timeframe: Time interval
        limit: Number of results (max 50)

    Returns:
        Tuple of (results list, warning message if partial failure)
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)
    limit = max(1, min(limit, 50))

    result = fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)

    warning = None
    if result.partial_failure:
        warning = f"Partial data: {result.failed_batches}/{result.total_batches} batches failed"
    elif result.total_failure:
        warning = f"All API requests failed: {'; '.join(result.errors[:3])}"

    # Already sorted by change descending
    return result.data[:limit], warning


def get_top_losers(
    exchange: str = "kucoin",
    timeframe: str = "15m",
    limit: int = 25
) -> Tuple[List[Dict], Optional[str]]:
    """
    Get top losing assets on an exchange.

    Args:
        exchange: Exchange name
        timeframe: Time interval
        limit: Number of results (max 50)

    Returns:
        Tuple of (results list, warning message if partial failure)
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)
    limit = max(1, min(limit, 50))

    result = fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)

    warning = None
    if result.partial_failure:
        warning = f"Partial data: {result.failed_batches}/{result.total_batches} batches failed"
    elif result.total_failure:
        warning = f"All API requests failed: {'; '.join(result.errors[:3])}"

    # Sort by change ascending (losers first)
    result.data.sort(key=lambda x: x["changePercent"])

    return result.data[:limit], warning


def get_bollinger_scan(
    exchange: str = "kucoin",
    timeframe: str = "4h",
    bbw_threshold: float = 0.04,
    limit: int = 25
) -> Tuple[List[Dict], Optional[str]]:
    """
    Scan for assets with Bollinger Band squeeze (low BBW).

    Args:
        exchange: Exchange name
        timeframe: Time interval
        bbw_threshold: Maximum BBW value for squeeze detection
        limit: Number of results (max 50)

    Returns:
        Tuple of (results list, warning message if partial failure)
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 50))

    result = fetch_analysis(exchange, timeframe, limit * 2)

    warning = None
    if result.partial_failure:
        warning = f"Partial data: {result.failed_batches}/{result.total_batches} batches failed"
    elif result.total_failure:
        warning = f"All API requests failed: {'; '.join(result.errors[:3])}"

    # Filter by BBW threshold
    filtered = [r for r in result.data if r["bbw"] and r["bbw"] < bbw_threshold and r["bbw"] > 0]
    filtered.sort(key=lambda x: x["bbw"])

    return filtered[:limit], warning


def get_rating_filter(
    exchange: str = "kucoin",
    timeframe: str = "15m",
    rating: int = 2,
    limit: int = 25
) -> Tuple[List[Dict], Optional[str]]:
    """
    Filter assets by Bollinger Band rating.

    Args:
        exchange: Exchange name
        timeframe: Time interval
        rating: Target rating (-3 to +3)
        limit: Number of results (max 50)

    Returns:
        Tuple of (results list, warning message if partial failure)
    """
    exchange = sanitize_exchange(exchange)
    timeframe = sanitize_timeframe(timeframe)
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))

    result = fetch_trending_analysis(
        exchange,
        timeframe=timeframe,
        filter_type="rating",
        rating_filter=rating,
        limit=limit * 3
    )

    warning = None
    if result.partial_failure:
        warning = f"Partial data: {result.failed_batches}/{result.total_batches} batches failed"
    elif result.total_failure:
        warning = f"All API requests failed: {'; '.join(result.errors[:3])}"

    # Sort by absolute change (most volatile first)
    result.data.sort(key=lambda x: abs(x["changePercent"]), reverse=True)

    return result.data[:limit], warning


def get_coin_analysis(
    symbol: str,
    exchange: str = "kucoin",
    timeframe: str = "15m"
) -> Dict[str, Any]:
    """
    Get detailed technical analysis for a specific symbol.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        exchange: Exchange name
        timeframe: Time interval

    Returns:
        Comprehensive analysis dictionary or error dict
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
        logger.error(f"Analysis failed for {symbol}: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "symbol": symbol,
            "exchange": exchange
        }


def get_exchanges_list() -> Dict[str, Any]:
    """
    List all supported exchanges and their market types.

    Returns:
        Dictionary of exchanges grouped by market type
    """
    exchanges_by_type: Dict[str, List[str]] = {}
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
