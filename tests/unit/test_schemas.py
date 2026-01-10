"""
Tests for the schemas module.

Tests cover:
- Attribution structure and constants
- AnalysisResult builder functions
- Error result builder
- No signal result builder
- Result validation
"""

import pytest
from sigmapilot_mcp.core.schemas import (
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


class TestAttribution:
    """Tests for Attribution TypedDict and constants."""

    def test_attributions_exist_for_all_engines(self):
        """Test that all 9 theory-based engines have attributions."""
        expected_engines = [
            "dow_theory_trend",
            "ichimoku_insight",
            "vsa_analyzer",
            "chart_pattern_finder",
            "wyckoff_phase_detector",
            "elliott_wave_analyzer",
            "chan_theory_analyzer",
            "harmonic_pattern_detector",
            "market_profile_analyzer",
        ]
        for engine in expected_engines:
            assert engine in ATTRIBUTIONS, f"Missing attribution for {engine}"

    def test_attribution_structure(self):
        """Test that attributions have required fields."""
        for engine, attr in ATTRIBUTIONS.items():
            assert "theory" in attr, f"{engine} missing 'theory'"
            assert "author" in attr, f"{engine} missing 'author'"
            assert "reference" in attr, f"{engine} missing 'reference'"
            assert isinstance(attr["theory"], str)
            assert isinstance(attr["author"], str)
            assert isinstance(attr["reference"], str)

    def test_get_attribution_known_engine(self):
        """Test getting attribution for known engine."""
        attr = get_attribution("dow_theory_trend")
        assert attr["theory"] == "Dow Theory"
        assert attr["author"] == "Charles Dow"
        assert "wikipedia" in attr["reference"].lower() or "dow" in attr["reference"].lower()

    def test_get_attribution_unknown_engine(self):
        """Test getting attribution for unknown engine."""
        attr = get_attribution("unknown_engine")
        assert attr["theory"] == "unknown_engine"
        assert attr["author"] == "SigmaPilot"
        assert attr["reference"] == ""

    def test_ichimoku_attribution(self):
        """Test Ichimoku attribution details."""
        attr = get_attribution("ichimoku_insight")
        assert "Ichimoku" in attr["theory"]
        assert "Hosoda" in attr["author"]


class TestBuildAnalysisResult:
    """Tests for build_analysis_result function."""

    def test_build_bullish_result(self):
        """Test building a bullish result."""
        result = build_analysis_result(
            status="bullish",
            confidence=75,
            tool_name="dow_theory_trend",
            llm_summary="Higher highs and higher lows confirmed.",
            invalidation="Price below $40,000"
        )

        assert result["status"] == "bullish"
        assert result["confidence"] == 75
        assert result["attribution"]["theory"] == "Dow Theory"
        assert result["llm_summary"] == "Higher highs and higher lows confirmed."
        assert result["invalidation"] == "Price below $40,000"
        assert result["is_error"] is False

    def test_build_bearish_result(self):
        """Test building a bearish result."""
        result = build_analysis_result(
            status="bearish",
            confidence=68,
            tool_name="ichimoku_insight",
            llm_summary="Price below cloud with bearish TK cross.",
            invalidation="Price above Kumo"
        )

        assert result["status"] == "bearish"
        assert result["confidence"] == 68
        assert "Ichimoku" in result["attribution"]["theory"]

    def test_build_neutral_result(self):
        """Test building a neutral result."""
        result = build_analysis_result(
            status="neutral",
            confidence=50,
            tool_name="vsa_analyzer",
            llm_summary="Mixed volume signals, no clear direction.",
            invalidation=""
        )

        assert result["status"] == "neutral"
        assert result["confidence"] == 50

    def test_no_signal_protocol_applied(self):
        """Test that No Signal Protocol is applied by default."""
        result = build_analysis_result(
            status="bullish",
            confidence=55,  # Below 60 threshold
            tool_name="dow_theory_trend",
            llm_summary="Weak bullish signal.",
            invalidation="N/A"
        )

        # Should be forced to neutral
        assert result["status"] == "neutral"
        assert result["confidence"] == 55

    def test_no_signal_protocol_disabled(self):
        """Test that No Signal Protocol can be disabled."""
        result = build_analysis_result(
            status="bullish",
            confidence=55,
            tool_name="dow_theory_trend",
            llm_summary="Weak bullish signal.",
            invalidation="N/A",
            apply_no_signal=False
        )

        # Should preserve original status
        assert result["status"] == "bullish"
        assert result["confidence"] == 55

    def test_confidence_at_threshold_preserves_status(self):
        """Test that confidence at 60 preserves status."""
        result = build_analysis_result(
            status="bullish",
            confidence=60,
            tool_name="dow_theory_trend",
            llm_summary="Threshold bullish signal.",
            invalidation="N/A"
        )

        assert result["status"] == "bullish"
        assert result["confidence"] == 60


class TestBuildErrorResult:
    """Tests for build_error_result function."""

    def test_error_result_structure(self):
        """Test that error result has correct structure."""
        result = build_error_result(
            error_message="Rate limit exceeded.",
            tool_name="dow_theory_trend"
        )

        assert result["status"] == "neutral"
        assert result["confidence"] == 0
        assert result["is_error"] is True
        assert result["llm_summary"] == "Rate limit exceeded."
        assert result["invalidation"] == ""
        assert result["attribution"]["theory"] == "Dow Theory"

    def test_error_result_for_unknown_tool(self):
        """Test error result for unknown tool."""
        result = build_error_result(
            error_message="Something went wrong.",
            tool_name="custom_analyzer"
        )

        assert result["is_error"] is True
        assert result["attribution"]["theory"] == "custom_analyzer"
        assert result["attribution"]["author"] == "SigmaPilot"


class TestBuildNoSignalResult:
    """Tests for build_no_signal_result function."""

    def test_no_signal_result_structure(self):
        """Test that no signal result has correct structure."""
        result = build_no_signal_result(
            reason="Market is in consolidation with no clear trend.",
            tool_name="dow_theory_trend"
        )

        assert result["status"] == "neutral"
        assert result["confidence"] == 0
        assert result["is_error"] is False
        assert "consolidation" in result["llm_summary"]
        assert result["invalidation"] == ""

    def test_no_signal_with_confidence(self):
        """Test no signal result with custom confidence."""
        result = build_no_signal_result(
            reason="Ambiguous pattern detected.",
            tool_name="chart_pattern_finder",
            confidence=45
        )

        assert result["status"] == "neutral"
        assert result["confidence"] == 45
        assert result["is_error"] is False


class TestBuildInsufficientDataResult:
    """Tests for build_insufficient_data_result function."""

    def test_insufficient_data_result(self):
        """Test insufficient data result structure."""
        result = build_insufficient_data_result(
            tool_name="elliott_wave_analyzer",
            required_bars=200,
            available_bars=50
        )

        assert result["status"] == "neutral"
        assert result["confidence"] == 0
        assert result["is_error"] is False
        assert "200" in result["llm_summary"]
        assert "50" in result["llm_summary"]
        assert "Elliott" in result["llm_summary"]


class TestValidateAnalysisResult:
    """Tests for validate_analysis_result function."""

    def test_valid_result(self):
        """Test validation of a valid result."""
        result = build_analysis_result(
            status="bullish",
            confidence=75,
            tool_name="dow_theory_trend",
            llm_summary="Valid result.",
            invalidation="N/A"
        )

        is_valid, error = validate_analysis_result(result)
        assert is_valid is True
        assert error is None

    def test_missing_status(self):
        """Test validation catches missing status."""
        result = {
            "confidence": 75,
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Test",
            "invalidation": "",
            "is_error": False
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False
        assert "status" in error

    def test_invalid_status_value(self):
        """Test validation catches invalid status value."""
        result = {
            "status": "invalid",
            "confidence": 75,
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Test",
            "invalidation": "",
            "is_error": False
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False
        assert "status" in error.lower()

    def test_invalid_confidence_type(self):
        """Test validation catches non-integer confidence."""
        result = {
            "status": "bullish",
            "confidence": "75",  # String instead of int
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Test",
            "invalidation": "",
            "is_error": False
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False
        assert "confidence" in error.lower()

    def test_confidence_out_of_range_high(self):
        """Test validation catches confidence > 100."""
        result = {
            "status": "bullish",
            "confidence": 150,
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Test",
            "invalidation": "",
            "is_error": False
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False

    def test_confidence_out_of_range_low(self):
        """Test validation catches confidence < 0."""
        result = {
            "status": "bullish",
            "confidence": -10,
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Test",
            "invalidation": "",
            "is_error": False
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False

    def test_missing_attribution_field(self):
        """Test validation catches missing attribution field."""
        result = {
            "status": "bullish",
            "confidence": 75,
            "attribution": {"theory": "Test", "author": "Test"},  # Missing reference
            "llm_summary": "Test",
            "invalidation": "",
            "is_error": False
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False
        assert "reference" in error

    def test_error_result_must_have_zero_confidence(self):
        """Test that error results must have confidence 0."""
        result = {
            "status": "neutral",
            "confidence": 50,  # Should be 0 for errors
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Error occurred.",
            "invalidation": "",
            "is_error": True
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False
        assert "confidence" in error.lower() or "0" in error

    def test_error_result_must_be_neutral(self):
        """Test that error results must have neutral status."""
        result = {
            "status": "bullish",  # Should be neutral for errors
            "confidence": 0,
            "attribution": {"theory": "Test", "author": "Test", "reference": ""},
            "llm_summary": "Error occurred.",
            "invalidation": "",
            "is_error": True
        }

        is_valid, error = validate_analysis_result(result)
        assert is_valid is False
        assert "neutral" in error.lower()

    def test_valid_error_result(self):
        """Test validation of a valid error result."""
        result = build_error_result("Error message.", "dow_theory_trend")

        is_valid, error = validate_analysis_result(result)
        assert is_valid is True
        assert error is None
