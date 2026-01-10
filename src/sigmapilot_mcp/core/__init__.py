"""
SigmaPilot MCP Core Module.

This module provides core infrastructure for the v2.0.0 theory-based analysis engines.

Modules:
- errors: Custom exception hierarchy and error response builders
- timeframes: Multi-timeframe analysis utilities
- sanitize: Input validation and sanitization
- confidence: Confidence formula implementation
- schemas: TypedDict definitions for structured outputs
- data_loader: OHLCV data loading utilities
"""

from .errors import (
    SigmaPilotError,
    DataError,
    ValidationError,
    APIError,
    RateLimitError,
    TimeoutError,
    InsufficientDataError,
    SymbolNotFoundError,
    AnalysisError,
    build_error_response,
    classify_api_error,
)

from .timeframes import (
    TIMEFRAME_HIERARCHY,
    TIMEFRAME_WEIGHTS,
    get_timeframe_weight,
    get_higher_timeframes,
    get_lower_timeframes,
    is_valid_timeframe,
    get_minimum_bars,
)

from .sanitize import (
    sanitize_timeframe,
    sanitize_exchange,
    sanitize_symbol,
    sanitize_limit,
    validate_ohlcv_data,
    ALLOWED_TIMEFRAMES,
    EXCHANGE_SCREENER,
)

from .confidence import (
    ConfidenceFactors,
    calculate_confidence,
    calculate_confidence_simple,
    apply_no_signal_protocol,
    is_signal_valid,
    calculate_multi_engine_bonus,
    SIGNAL_THRESHOLD,
)

from .schemas import (
    Attribution,
    AnalysisResult,
    ATTRIBUTIONS,
    get_attribution,
    build_analysis_result,
    build_error_result,
    build_no_signal_result,
    build_insufficient_data_result,
    validate_analysis_result,
)

from .data_loader import (
    OHLCVBar,
    OHLCVData,
    load_ohlcv_from_csv,
    create_ohlcv_from_arrays,
    validate_minimum_bars,
    ensure_minimum_bars,
)

__all__ = [
    # Errors
    "SigmaPilotError",
    "DataError",
    "ValidationError",
    "APIError",
    "RateLimitError",
    "TimeoutError",
    "InsufficientDataError",
    "SymbolNotFoundError",
    "AnalysisError",
    "build_error_response",
    "classify_api_error",
    # Timeframes
    "TIMEFRAME_HIERARCHY",
    "TIMEFRAME_WEIGHTS",
    "get_timeframe_weight",
    "get_higher_timeframes",
    "get_lower_timeframes",
    "is_valid_timeframe",
    "get_minimum_bars",
    # Sanitize
    "sanitize_timeframe",
    "sanitize_exchange",
    "sanitize_symbol",
    "sanitize_limit",
    "validate_ohlcv_data",
    "ALLOWED_TIMEFRAMES",
    "EXCHANGE_SCREENER",
    # Confidence
    "ConfidenceFactors",
    "calculate_confidence",
    "calculate_confidence_simple",
    "apply_no_signal_protocol",
    "is_signal_valid",
    "calculate_multi_engine_bonus",
    "SIGNAL_THRESHOLD",
    # Schemas
    "Attribution",
    "AnalysisResult",
    "ATTRIBUTIONS",
    "get_attribution",
    "build_analysis_result",
    "build_error_result",
    "build_no_signal_result",
    "build_insufficient_data_result",
    "validate_analysis_result",
    # Data Loader
    "OHLCVBar",
    "OHLCVData",
    "load_ohlcv_from_csv",
    "create_ohlcv_from_arrays",
    "validate_minimum_bars",
    "ensure_minimum_bars",
]
