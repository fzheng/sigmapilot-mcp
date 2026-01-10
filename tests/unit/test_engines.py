"""
Integration tests for all Phase 2 theory-based engines.

Tests verify:
- Each engine produces valid AnalysisResult schema
- Engines handle edge cases gracefully
- Engines work with fixture data
"""

import pytest
import numpy as np
from pathlib import Path

from sigmapilot_mcp.core.data_loader import OHLCVData, OHLCVBar, load_ohlcv_from_csv
from sigmapilot_mcp.engines import (
    analyze_dow_theory,
    analyze_ichimoku,
    analyze_vsa,
    analyze_chart_patterns,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    bars = []
    base_price = 100.0
    timestamps = list(range(1704067200, 1704067200 + 200 * 3600, 3600))

    for i in range(200):
        trend = i * 0.2
        noise = np.sin(i * 0.5) * 3

        open_price = base_price + trend + noise
        high = open_price + 2
        low = open_price - 2
        close = open_price + np.random.uniform(-1, 1)
        volume = 1000000 * (1 + np.sin(i * 0.1) * 0.3)

        bars.append(OHLCVBar(
            timestamp=timestamps[i],
            open=open_price,
            high=max(open_price, close) + 0.5,
            low=min(open_price, close) - 0.5,
            close=close,
            volume=volume
        ))

    return OHLCVData(symbol="TEST:SAMPLE", timeframe="1h", bars=bars)


@pytest.fixture
def minimal_data():
    """Create minimal OHLCV data (edge case)."""
    bars = [
        OHLCVBar(timestamp=1704067200 + i * 3600, open=100, high=101, low=99, close=100, volume=1000)
        for i in range(30)
    ]
    return OHLCVData(symbol="TEST:MINIMAL", timeframe="1h", bars=bars)


@pytest.fixture
def no_volume_data():
    """Create data with zero volume."""
    bars = [
        OHLCVBar(timestamp=1704067200 + i * 3600, open=100+i*0.1, high=101+i*0.1, low=99+i*0.1, close=100+i*0.1, volume=0)
        for i in range(100)
    ]
    return OHLCVData(symbol="TEST:NOVOL", timeframe="1h", bars=bars)


# =============================================================================
# Schema Validation Helper
# =============================================================================

def validate_analysis_result(result):
    """Validate that result matches AnalysisResult schema."""
    assert "status" in result
    assert "confidence" in result
    assert "attribution" in result
    assert "llm_summary" in result
    assert "invalidation" in result
    assert "is_error" in result

    assert result["status"] in ["bullish", "bearish", "neutral"]
    assert isinstance(result["confidence"], int)
    assert 0 <= result["confidence"] <= 100
    assert isinstance(result["attribution"], dict)
    assert "theory" in result["attribution"]
    assert isinstance(result["llm_summary"], str)
    assert isinstance(result["is_error"], bool)


# =============================================================================
# Dow Theory Tests
# =============================================================================

class TestDowTheoryEngine:
    """Tests for Dow Theory engine."""

    def test_returns_valid_schema(self, sample_ohlcv_data):
        """Test that engine returns valid schema."""
        result = analyze_dow_theory(sample_ohlcv_data)
        validate_analysis_result(result)

    def test_handles_minimal_data(self, minimal_data):
        """Test graceful handling of minimal data."""
        result = analyze_dow_theory(minimal_data)
        validate_analysis_result(result)

    def test_attribution_is_dow_theory(self, sample_ohlcv_data):
        """Test that attribution is Dow Theory."""
        result = analyze_dow_theory(sample_ohlcv_data)
        assert result["attribution"]["theory"] == "Dow Theory"

    def test_conservative_mode(self, sample_ohlcv_data):
        """Test conservative mode."""
        result = analyze_dow_theory(sample_ohlcv_data, mode="conservative")
        validate_analysis_result(result)

    def test_aggressive_mode(self, sample_ohlcv_data):
        """Test aggressive mode."""
        result = analyze_dow_theory(sample_ohlcv_data, mode="aggressive")
        validate_analysis_result(result)


# =============================================================================
# Ichimoku Tests
# =============================================================================

class TestIchimokuEngine:
    """Tests for Ichimoku engine."""

    def test_returns_valid_schema(self, sample_ohlcv_data):
        """Test that engine returns valid schema."""
        result = analyze_ichimoku(sample_ohlcv_data)
        validate_analysis_result(result)

    def test_handles_minimal_data(self, minimal_data):
        """Test graceful handling of minimal data."""
        result = analyze_ichimoku(minimal_data)
        validate_analysis_result(result)
        # Minimal data should return insufficient data result
        assert result["status"] == "neutral"

    def test_attribution_is_ichimoku(self, sample_ohlcv_data):
        """Test that attribution is Ichimoku."""
        result = analyze_ichimoku(sample_ohlcv_data)
        assert "Ichimoku" in result["attribution"]["theory"]

    def test_custom_parameters(self, sample_ohlcv_data):
        """Test with custom Ichimoku parameters."""
        result = analyze_ichimoku(
            sample_ohlcv_data,
            tenkan_period=9,
            kijun_period=26,
            senkou_b_period=52
        )
        validate_analysis_result(result)


# =============================================================================
# VSA Tests
# =============================================================================

class TestVSAEngine:
    """Tests for VSA engine."""

    def test_returns_valid_schema(self, sample_ohlcv_data):
        """Test that engine returns valid schema."""
        result = analyze_vsa(sample_ohlcv_data)
        validate_analysis_result(result)

    def test_handles_minimal_data(self, minimal_data):
        """Test graceful handling of minimal data."""
        result = analyze_vsa(minimal_data)
        validate_analysis_result(result)

    def test_handles_no_volume(self, no_volume_data):
        """Test handling of data without volume."""
        result = analyze_vsa(no_volume_data)
        validate_analysis_result(result)
        # Should return neutral when no volume data
        assert result["status"] == "neutral"

    def test_attribution_is_vsa(self, sample_ohlcv_data):
        """Test that attribution is VSA."""
        result = analyze_vsa(sample_ohlcv_data)
        assert "Volume" in result["attribution"]["theory"] or "VSA" in result["attribution"]["theory"]


# =============================================================================
# Chart Pattern Tests
# =============================================================================

class TestChartPatternEngine:
    """Tests for Chart Pattern engine."""

    def test_returns_valid_schema(self, sample_ohlcv_data):
        """Test that engine returns valid schema."""
        result = analyze_chart_patterns(sample_ohlcv_data)
        validate_analysis_result(result)

    def test_handles_minimal_data(self, minimal_data):
        """Test graceful handling of minimal data."""
        result = analyze_chart_patterns(minimal_data)
        validate_analysis_result(result)

    def test_attribution_is_chart_patterns(self, sample_ohlcv_data):
        """Test that attribution is Chart Patterns."""
        result = analyze_chart_patterns(sample_ohlcv_data)
        assert "Pattern" in result["attribution"]["theory"] or "Chart" in result["attribution"]["theory"]

    def test_min_confidence_parameter(self, sample_ohlcv_data):
        """Test min_confidence parameter."""
        result = analyze_chart_patterns(sample_ohlcv_data, min_confidence=0.90)
        validate_analysis_result(result)


# =============================================================================
# Cross-Engine Tests
# =============================================================================

class TestCrossEngineConsistency:
    """Tests for consistency across engines."""

    def test_all_engines_same_data(self, sample_ohlcv_data):
        """Test that all engines can process the same data."""
        dow_result = analyze_dow_theory(sample_ohlcv_data)
        ichimoku_result = analyze_ichimoku(sample_ohlcv_data)
        vsa_result = analyze_vsa(sample_ohlcv_data)
        pattern_result = analyze_chart_patterns(sample_ohlcv_data)

        # All should return valid results
        validate_analysis_result(dow_result)
        validate_analysis_result(ichimoku_result)
        validate_analysis_result(vsa_result)
        validate_analysis_result(pattern_result)

    def test_all_engines_different_attributions(self, sample_ohlcv_data):
        """Test that all engines have different attributions."""
        dow_result = analyze_dow_theory(sample_ohlcv_data)
        ichimoku_result = analyze_ichimoku(sample_ohlcv_data)
        vsa_result = analyze_vsa(sample_ohlcv_data)
        pattern_result = analyze_chart_patterns(sample_ohlcv_data)

        theories = [
            dow_result["attribution"]["theory"],
            ichimoku_result["attribution"]["theory"],
            vsa_result["attribution"]["theory"],
            pattern_result["attribution"]["theory"],
        ]

        # All should be unique
        assert len(set(theories)) == 4


# =============================================================================
# Fixture Integration Tests
# =============================================================================

class TestWithCSVFixtures:
    """Tests using CSV fixture files."""

    @pytest.fixture
    def fixture_dir(self):
        return Path(__file__).parent.parent / "fixtures" / "deterministic_ohlcv"

    def test_btc_uptrend_all_engines(self, fixture_dir):
        """Test all engines with BTC uptrend fixture."""
        filepath = fixture_dir / "btc_1h_uptrend.csv"
        if not filepath.exists():
            pytest.skip("Fixture not found")

        data = load_ohlcv_from_csv(filepath, symbol="TEST:BTC", timeframe="1h")

        # All engines should process without error
        assert analyze_dow_theory(data)["is_error"] is False
        assert analyze_ichimoku(data)["is_error"] is False
        assert analyze_vsa(data)["is_error"] is False
        assert analyze_chart_patterns(data)["is_error"] is False

    def test_eth_downtrend_all_engines(self, fixture_dir):
        """Test all engines with ETH downtrend fixture."""
        filepath = fixture_dir / "eth_4h_downtrend.csv"
        if not filepath.exists():
            pytest.skip("Fixture not found")

        data = load_ohlcv_from_csv(filepath, symbol="TEST:ETH", timeframe="4h")

        # All engines should process without error
        assert analyze_dow_theory(data)["is_error"] is False
        assert analyze_ichimoku(data)["is_error"] is False
        assert analyze_vsa(data)["is_error"] is False
        assert analyze_chart_patterns(data)["is_error"] is False

    def test_sol_range_all_engines(self, fixture_dir):
        """Test all engines with SOL range fixture."""
        filepath = fixture_dir / "sol_1d_range.csv"
        if not filepath.exists():
            pytest.skip("Fixture not found")

        data = load_ohlcv_from_csv(filepath, symbol="TEST:SOL", timeframe="1D")

        # All engines should process without error
        assert analyze_dow_theory(data)["is_error"] is False
        assert analyze_ichimoku(data)["is_error"] is False
        assert analyze_vsa(data)["is_error"] is False
        assert analyze_chart_patterns(data)["is_error"] is False
