"""
TypedDict Schema Definitions for SigmaPilot MCP.

This module provides structured type definitions for all analysis outputs,
ensuring consistent response formats across all theory-based engines.

Key Features:
- Attribution schema for theory/author credits
- AnalysisResult schema for successful analyses
- ErrorResult schema for error responses
- Builder functions for creating valid responses
- No Signal Protocol integration
"""

from __future__ import annotations
from typing import TypedDict, Literal, Dict, Any

from .confidence import SIGNAL_THRESHOLD, apply_no_signal_protocol, StatusType


# =============================================================================
# Type Definitions
# =============================================================================

class Attribution(TypedDict):
    """
    Attribution information for the analysis source.

    Attributes:
        theory: Name of the theory/methodology (e.g., "Dow Theory", "Ichimoku")
        author: Original author or source attribution
        reference: URL or citation for the methodology
    """
    theory: str
    author: str
    reference: str


class AnalysisResult(TypedDict):
    """
    Standard response schema for all theory-based analysis engines.

    This schema is used for both successful analyses and error responses.
    The is_error field distinguishes between them.

    Attributes:
        status: Market bias ("bullish", "bearish", or "neutral")
        confidence: Confidence score 0-100 (0 for errors)
        attribution: Theory/author attribution
        llm_summary: 2-3 sentence summary for LLM consumption
        invalidation: Condition that would invalidate the analysis
        is_error: True if this is an error response
    """
    status: Literal["bullish", "bearish", "neutral"]
    confidence: int
    attribution: Attribution
    llm_summary: str
    invalidation: str
    is_error: bool


# =============================================================================
# Attribution Constants
# =============================================================================

# Pre-defined attributions for each theory-based engine
ATTRIBUTIONS: Dict[str, Attribution] = {
    "dow_theory_trend": {
        "theory": "Dow Theory",
        "author": "Charles Dow",
        "reference": "https://en.wikipedia.org/wiki/Dow_theory"
    },
    "ichimoku_insight": {
        "theory": "Ichimoku Kinko Hyo",
        "author": "Goichi Hosoda",
        "reference": "https://school.stockcharts.com/doku.php?id=technical_indicators:ichimoku_cloud"
    },
    "vsa_analyzer": {
        "theory": "Volume Spread Analysis",
        "author": "Tom Williams / Richard Wyckoff",
        "reference": "https://en.wikipedia.org/wiki/Volume_spread_analysis"
    },
    "chart_pattern_finder": {
        "theory": "Classical Chart Patterns",
        "author": "Edwards & Magee",
        "reference": "Technical Analysis of Stock Trends"
    },
    "wyckoff_phase_detector": {
        "theory": "Wyckoff Method",
        "author": "Richard D. Wyckoff",
        "reference": "https://school.stockcharts.com/doku.php?id=market_analysis:the_wyckoff_method"
    },
    "elliott_wave_analyzer": {
        "theory": "Elliott Wave Theory",
        "author": "Ralph Nelson Elliott",
        "reference": "https://en.wikipedia.org/wiki/Elliott_wave_principle"
    },
    "chan_theory_analyzer": {
        "theory": "Chan Theory (缠论)",
        "author": "Chan Zhongshu (缠中说禅)",
        "reference": "https://baike.baidu.com/item/缠论"
    },
    "harmonic_pattern_detector": {
        "theory": "Harmonic Trading",
        "author": "Scott Carney",
        "reference": "https://harmonictrader.com/"
    },
    "market_profile_analyzer": {
        "theory": "Market Profile / TPO",
        "author": "J. Peter Steidlmayer",
        "reference": "https://www.cmegroup.com/education/courses/introduction-to-market-profile.html"
    },
}


def get_attribution(tool_name: str) -> Attribution:
    """
    Get the attribution for a given tool name.

    Args:
        tool_name: Name of the analysis tool

    Returns:
        Attribution dictionary for the tool

    Example:
        >>> attr = get_attribution("dow_theory_trend")
        >>> attr["theory"]
        'Dow Theory'
    """
    return ATTRIBUTIONS.get(tool_name, {
        "theory": tool_name,
        "author": "SigmaPilot",
        "reference": ""
    })


# =============================================================================
# Builder Functions
# =============================================================================

def build_analysis_result(
    status: StatusType,
    confidence: int,
    tool_name: str,
    llm_summary: str,
    invalidation: str,
    apply_no_signal: bool = True
) -> AnalysisResult:
    """
    Build a complete AnalysisResult with automatic No Signal Protocol.

    Args:
        status: Detected market bias ("bullish", "bearish", "neutral")
        confidence: Confidence score (0-100)
        tool_name: Name of the analysis tool (for attribution lookup)
        llm_summary: 2-3 sentence summary for LLM consumption
        invalidation: Condition that would invalidate this analysis
        apply_no_signal: Whether to apply No Signal Protocol (default True)

    Returns:
        Complete AnalysisResult dictionary

    Example:
        >>> result = build_analysis_result(
        ...     status="bullish",
        ...     confidence=75,
        ...     tool_name="dow_theory_trend",
        ...     llm_summary="Higher highs and higher lows confirmed.",
        ...     invalidation="Below $40,000"
        ... )
        >>> result["status"]
        'bullish'
    """
    # Apply No Signal Protocol if enabled
    final_confidence = confidence
    final_status = status

    if apply_no_signal:
        final_confidence, final_status = apply_no_signal_protocol(confidence, status)

    return {
        "status": final_status,
        "confidence": final_confidence,
        "attribution": get_attribution(tool_name),
        "llm_summary": llm_summary,
        "invalidation": invalidation,
        "is_error": False
    }


def build_error_result(
    error_message: str,
    tool_name: str
) -> AnalysisResult:
    """
    Build an error response in the AnalysisResult format.

    Error responses always have:
    - status: "neutral"
    - confidence: 0
    - is_error: True

    Args:
        error_message: User-friendly error description
        tool_name: Name of the tool that encountered the error

    Returns:
        AnalysisResult with error information

    Example:
        >>> result = build_error_result(
        ...     "Rate limit exceeded. Please try again later.",
        ...     "ichimoku_insight"
        ... )
        >>> result["is_error"]
        True
        >>> result["confidence"]
        0
    """
    return {
        "status": "neutral",
        "confidence": 0,
        "attribution": get_attribution(tool_name),
        "llm_summary": error_message,
        "invalidation": "",
        "is_error": True
    }


def build_no_signal_result(
    reason: str,
    tool_name: str,
    confidence: int = 0
) -> AnalysisResult:
    """
    Build a "no signal" response when analysis is inconclusive.

    Different from error - analysis completed but no clear signal was found.
    This is a valid outcome, not an error condition.

    Args:
        reason: Explanation of why no signal was generated
        tool_name: Name of the analysis tool
        confidence: Confidence in the "no signal" determination (default 0)

    Returns:
        AnalysisResult with neutral status

    Example:
        >>> result = build_no_signal_result(
        ...     "Market is in consolidation with no clear trend.",
        ...     "dow_theory_trend",
        ...     confidence=45
        ... )
        >>> result["status"]
        'neutral'
    """
    return {
        "status": "neutral",
        "confidence": confidence,
        "attribution": get_attribution(tool_name),
        "llm_summary": reason,
        "invalidation": "",
        "is_error": False
    }


def build_insufficient_data_result(
    tool_name: str,
    required_bars: int,
    available_bars: int
) -> AnalysisResult:
    """
    Build a response for insufficient data scenarios.

    Args:
        tool_name: Name of the analysis tool
        required_bars: Number of bars required for analysis
        available_bars: Number of bars actually available

    Returns:
        AnalysisResult indicating insufficient data

    Example:
        >>> result = build_insufficient_data_result("elliott_wave_analyzer", 200, 50)
        >>> "50" in result["llm_summary"]
        True
    """
    return build_no_signal_result(
        reason=(
            f"Insufficient data for {get_attribution(tool_name)['theory']} analysis. "
            f"Required {required_bars} bars but only {available_bars} available. "
            f"Try a lower timeframe or wait for more data to accumulate."
        ),
        tool_name=tool_name,
        confidence=0
    )


# =============================================================================
# Validation
# =============================================================================

def validate_analysis_result(result: Dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate that a dictionary conforms to the AnalysisResult schema.

    Args:
        result: Dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> result = {"status": "bullish", "confidence": 75, ...}
        >>> is_valid, error = validate_analysis_result(result)
    """
    required_keys = ["status", "confidence", "attribution", "llm_summary", "invalidation", "is_error"]

    for key in required_keys:
        if key not in result:
            return (False, f"Missing required key: {key}")

    # Validate status
    if result["status"] not in ("bullish", "bearish", "neutral"):
        return (False, f"Invalid status: {result['status']}")

    # Validate confidence
    if not isinstance(result["confidence"], int) or result["confidence"] < 0 or result["confidence"] > 100:
        return (False, f"Invalid confidence: {result['confidence']}")

    # Validate attribution
    attribution = result["attribution"]
    if not isinstance(attribution, dict):
        return (False, "Attribution must be a dictionary")
    for attr_key in ["theory", "author", "reference"]:
        if attr_key not in attribution:
            return (False, f"Attribution missing key: {attr_key}")

    # Validate is_error
    if not isinstance(result["is_error"], bool):
        return (False, "is_error must be a boolean")

    # Cross-validation: errors should have confidence 0 and neutral status
    if result["is_error"]:
        if result["confidence"] != 0:
            return (False, "Error results should have confidence 0")
        if result["status"] != "neutral":
            return (False, "Error results should have neutral status")

    return (True, None)
