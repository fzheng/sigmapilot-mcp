"""
Centralized Error Handling for SigmaPilot MCP.

This module provides a custom exception hierarchy and error response builders
for consistent error handling across all theory-based analysis engines.

Key Features:
- Custom exception hierarchy for different error types
- User-friendly error messages for LLM consumers
- Structured error responses matching the AnalysisResult schema
"""

from __future__ import annotations
from typing import Dict, Any


# =============================================================================
# Exception Hierarchy
# =============================================================================

class SigmaPilotError(Exception):
    """Base exception for all SigmaPilot errors."""

    def __init__(self, message: str, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class DataError(SigmaPilotError):
    """Errors related to data fetching, parsing, or validation."""
    pass


class ValidationError(SigmaPilotError):
    """Errors related to input validation."""
    pass


class APIError(SigmaPilotError):
    """Errors related to external API calls."""
    pass


class RateLimitError(APIError):
    """API rate limit exceeded."""

    def __init__(self, message: str = "API rate limit exceeded. Please try again later.",
                 retry_after: int | None = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class TimeoutError(APIError):
    """API request timeout."""

    def __init__(self, message: str = "API request timed out.", timeout_seconds: float | None = None):
        super().__init__(message, {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class InsufficientDataError(DataError):
    """Not enough data bars for analysis."""

    def __init__(self, required: int, available: int, message: str | None = None):
        msg = message or f"Insufficient data: {available} bars available, {required} required."
        super().__init__(msg, {"required": required, "available": available})
        self.required = required
        self.available = available


class SymbolNotFoundError(DataError):
    """Symbol not found or invalid."""

    def __init__(self, symbol: str, message: str | None = None):
        msg = message or f"Symbol '{symbol}' not found or invalid."
        super().__init__(msg, {"symbol": symbol})
        self.symbol = symbol


class AnalysisError(SigmaPilotError):
    """Errors during analysis computation."""
    pass


# =============================================================================
# Error Response Builder
# =============================================================================

def build_error_response(error: Exception, tool_name: str) -> Dict[str, Any]:
    """
    Build a structured error response matching the AnalysisResult schema.

    Args:
        error: The exception that occurred
        tool_name: Name of the tool that encountered the error

    Returns:
        Dictionary with error information in the standard schema format:
        - status: "neutral" (always for errors)
        - confidence: 0 (always for errors)
        - attribution: Tool attribution
        - llm_summary: User-friendly error explanation
        - invalidation: Empty string
        - is_error: True

    Example:
        >>> try:
        ...     raise RateLimitError()
        ... except Exception as e:
        ...     response = build_error_response(e, "dow_theory_trend")
        >>> response["is_error"]
        True
    """
    # Determine user-friendly message based on error type
    if isinstance(error, RateLimitError):
        llm_summary = (
            f"Analysis could not be completed due to API rate limiting. "
            f"The TradingView API has temporarily restricted requests. "
            f"Please wait a moment and try again."
        )
    elif isinstance(error, TimeoutError):
        llm_summary = (
            f"Analysis timed out while fetching data. "
            f"The data provider did not respond in time. "
            f"Please check your connection and try again."
        )
    elif isinstance(error, InsufficientDataError):
        llm_summary = (
            f"Not enough historical data available for analysis. "
            f"Required {error.required} bars but only {error.available} available. "
            f"Try a shorter timeframe or different symbol."
        )
    elif isinstance(error, SymbolNotFoundError):
        llm_summary = (
            f"The symbol '{error.symbol}' was not found. "
            f"Please verify the symbol format (e.g., 'BINANCE:BTCUSDT') "
            f"and check that it's available on the specified exchange."
        )
    elif isinstance(error, ValidationError):
        llm_summary = (
            f"Invalid input parameters: {error.message}. "
            f"Please check your request and try again."
        )
    elif isinstance(error, DataError):
        llm_summary = (
            f"Data error occurred: {error.message}. "
            f"The data may be temporarily unavailable or malformed."
        )
    elif isinstance(error, APIError):
        llm_summary = (
            f"External API error: {error.message}. "
            f"Please try again later."
        )
    elif isinstance(error, SigmaPilotError):
        llm_summary = f"Analysis error: {error.message}"
    else:
        # Generic error handling for unexpected exceptions
        llm_summary = (
            f"An unexpected error occurred during analysis. "
            f"Error type: {type(error).__name__}. "
            f"Please try again or contact support if the issue persists."
        )

    return {
        "status": "neutral",
        "confidence": 0,
        "attribution": {
            "theory": tool_name,
            "author": "SigmaPilot",
            "reference": "Error response"
        },
        "llm_summary": llm_summary,
        "invalidation": "",
        "is_error": True
    }


def classify_api_error(error_message: str) -> SigmaPilotError:
    """
    Classify a raw API error message into the appropriate exception type.

    Args:
        error_message: The raw error message from the API

    Returns:
        Appropriate SigmaPilotError subclass instance

    Example:
        >>> error = classify_api_error("429 Too Many Requests")
        >>> isinstance(error, RateLimitError)
        True
    """
    msg_lower = error_message.lower()

    if any(term in msg_lower for term in ["429", "rate limit", "too many", "throttl"]):
        return RateLimitError(error_message)

    if any(term in msg_lower for term in ["timeout", "timed out", "deadline"]):
        return TimeoutError(error_message)

    if any(term in msg_lower for term in ["not found", "invalid symbol", "unknown symbol"]):
        # Extract symbol if possible
        return SymbolNotFoundError("unknown", error_message)

    if any(term in msg_lower for term in ["connection", "network", "unreachable"]):
        return APIError(f"Network error: {error_message}")

    return APIError(error_message)
