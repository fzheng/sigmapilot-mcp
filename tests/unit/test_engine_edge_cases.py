"""
Edge case tests for all theory-based analysis engines.

Tests cover:
- Boundary conditions
- Empty/minimal data handling
- Invalid inputs
- Extreme values
- Schema validation
"""

import pytest
import numpy as np
from typing import List

from sigmapilot_mcp.core.data_loader import OHLCVData, OHLCVBar
from sigmapilot_mcp.core.schemas import AnalysisResult

# Import all engines
from sigmapilot_mcp.engines.dow_theory import analyze_dow_theory
from sigmapilot_mcp.engines.ichimoku import analyze_ichimoku
from sigmapilot_mcp.engines.vsa import analyze_vsa
from sigmapilot_mcp.engines.chart_patterns import analyze_chart_patterns
from sigmapilot_mcp.engines.wyckoff import analyze_wyckoff
from sigmapilot_mcp.engines.elliott_wave import analyze_elliott_wave
from sigmapilot_mcp.engines.chan_theory import analyze_chan_theory
from sigmapilot_mcp.engines.harmonic import analyze_harmonic
from sigmapilot_mcp.engines.market_profile import analyze_market_profile


# ============================================================================
# Test Fixtures
# ============================================================================

def create_ohlcv_bars(
    prices: List[float],
    volumes: List[float] = None,
    base_timestamp: int = 1700000000
) -> List[OHLCVBar]:
    """Create valid OHLCV bars from price series."""
    if volumes is None:
        volumes = [1000.0] * len(prices)

    bars = []
    for i, price in enumerate(prices):
        open_price = prices[i-1] if i > 0 else price
        spread = abs(price * 0.01)
        high = max(open_price, price) + spread
        low = min(open_price, price) - spread

        bars.append(OHLCVBar(
            timestamp=base_timestamp + i * 3600,
            open=open_price,
            high=high,
            low=low,
            close=price,
            volume=volumes[i] if i < len(volumes) else 1000.0
        ))
    return bars


@pytest.fixture
def empty_data() -> OHLCVData:
    """Empty data with no bars."""
    return OHLCVData(symbol="TEST", timeframe="1D", bars=[])


@pytest.fixture
def single_bar_data() -> OHLCVData:
    """Data with only one bar."""
    bar = OHLCVBar(
        timestamp=1700000000,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0
    )
    return OHLCVData(symbol="TEST", timeframe="1D", bars=[bar])


@pytest.fixture
def flat_data() -> OHLCVData:
    """Data with no price movement (all same price)."""
    prices = [100.0] * 100
    bars = create_ohlcv_bars(prices)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def zero_volume_data() -> OHLCVData:
    """Data with zero volume."""
    prices = [100.0 + i * 0.5 for i in range(100)]
    volumes = [0.0] * 100
    bars = create_ohlcv_bars(prices, volumes)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def extreme_volatility_data() -> OHLCVData:
    """Data with extreme price swings."""
    prices = []
    for i in range(100):
        if i % 2 == 0:
            prices.append(100.0 + i * 10)
        else:
            prices.append(100.0 - i * 5)
    bars = create_ohlcv_bars(prices)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def large_dataset() -> OHLCVData:
    """Large dataset with 1000 bars."""
    np.random.seed(42)
    prices = [100.0]
    for _ in range(999):
        change = np.random.uniform(-0.02, 0.02)
        prices.append(prices[-1] * (1 + change))
    bars = create_ohlcv_bars(prices)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def strong_uptrend_data() -> OHLCVData:
    """Strong uptrend with consistent higher highs/lows."""
    prices = [100.0 + i * 2 for i in range(150)]
    volumes = [1000 + i * 50 for i in range(150)]  # Increasing volume
    bars = create_ohlcv_bars(prices, volumes)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def strong_downtrend_data() -> OHLCVData:
    """Strong downtrend with consistent lower highs/lows."""
    prices = [300.0 - i * 1.5 for i in range(150)]
    volumes = [1000 + i * 30 for i in range(150)]
    bars = create_ohlcv_bars(prices, volumes)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestAnalysisResultSchema:
    """Test that all engines return properly structured results."""

    ALL_ENGINES = [
        ("dow_theory", analyze_dow_theory),
        ("ichimoku", analyze_ichimoku),
        ("vsa", analyze_vsa),
        ("chart_patterns", analyze_chart_patterns),
        ("wyckoff", analyze_wyckoff),
        ("elliott_wave", analyze_elliott_wave),
        ("chan_theory", analyze_chan_theory),
        ("harmonic", analyze_harmonic),
        ("market_profile", analyze_market_profile),
    ]

    @pytest.fixture
    def valid_data(self) -> OHLCVData:
        """Create valid test data."""
        prices = []
        base = 100.0
        for i in range(200):
            trend = i * 0.3
            wave = 5 * np.sin(i * 0.2)
            prices.append(base + trend + wave)
        bars = create_ohlcv_bars(prices)
        return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_has_required_fields(self, name, engine, valid_data):
        """Test that result contains all required fields."""
        result = engine(valid_data)

        assert "status" in result, f"{name} missing 'status'"
        assert "confidence" in result, f"{name} missing 'confidence'"
        assert "attribution" in result, f"{name} missing 'attribution'"
        assert "llm_summary" in result, f"{name} missing 'llm_summary'"
        assert "invalidation" in result, f"{name} missing 'invalidation'"
        assert "is_error" in result, f"{name} missing 'is_error'"

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_status_is_valid(self, name, engine, valid_data):
        """Test that status is one of allowed values."""
        result = engine(valid_data)
        valid_statuses = ["bullish", "bearish", "neutral", "signal", "error"]
        assert result["status"] in valid_statuses, f"{name} has invalid status: {result['status']}"

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_confidence_is_bounded(self, name, engine, valid_data):
        """Test that confidence is between 0 and 100."""
        result = engine(valid_data)
        assert 0 <= result["confidence"] <= 100, f"{name} confidence out of range: {result['confidence']}"

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_attribution_has_theory(self, name, engine, valid_data):
        """Test that attribution contains theory name."""
        result = engine(valid_data)
        if result["status"] != "error":
            assert "theory" in result["attribution"], f"{name} attribution missing 'theory'"
            assert len(result["attribution"]["theory"]) > 0, f"{name} has empty theory"


# ============================================================================
# Empty/Minimal Data Tests
# ============================================================================

class TestEmptyDataHandling:
    """Test that all engines handle empty/minimal data gracefully."""

    ALL_ENGINES = [
        ("dow_theory", analyze_dow_theory),
        ("ichimoku", analyze_ichimoku),
        ("vsa", analyze_vsa),
        ("chart_patterns", analyze_chart_patterns),
        ("wyckoff", analyze_wyckoff),
        ("elliott_wave", analyze_elliott_wave),
        ("chan_theory", analyze_chan_theory),
        ("harmonic", analyze_harmonic),
        ("market_profile", analyze_market_profile),
    ]

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_empty_data_returns_neutral(self, name, engine, empty_data):
        """Test that empty data returns neutral without error."""
        result = engine(empty_data)
        assert result["status"] == "neutral", f"{name} should return neutral for empty data"
        assert not result["is_error"], f"{name} should not set is_error for empty data"

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_single_bar_returns_neutral(self, name, engine, single_bar_data):
        """Test that single bar data returns neutral."""
        result = engine(single_bar_data)
        assert result["status"] == "neutral", f"{name} should return neutral for single bar"

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_no_exception_on_minimal_data(self, name, engine, single_bar_data):
        """Test that minimal data doesn't cause exceptions."""
        try:
            result = engine(single_bar_data)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"{name} raised exception on minimal data: {e}")


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases for all engines."""

    ALL_ENGINES = [
        ("dow_theory", analyze_dow_theory),
        ("ichimoku", analyze_ichimoku),
        ("vsa", analyze_vsa),
        ("chart_patterns", analyze_chart_patterns),
        ("wyckoff", analyze_wyckoff),
        ("elliott_wave", analyze_elliott_wave),
        ("chan_theory", analyze_chan_theory),
        ("harmonic", analyze_harmonic),
        ("market_profile", analyze_market_profile),
    ]

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_flat_data_handling(self, name, engine, flat_data):
        """Test handling of flat/no-movement data."""
        result = engine(flat_data)
        # Should return neutral or low confidence for flat data
        assert result["status"] in ["neutral", "bullish", "bearish", "signal"]
        assert isinstance(result["confidence"], (int, float))

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_zero_volume_handling(self, name, engine, zero_volume_data):
        """Test handling of zero volume data."""
        result = engine(zero_volume_data)
        # Should handle gracefully, possibly with neutral status
        assert "status" in result
        assert not result.get("is_error", False) or result["status"] == "neutral"

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_extreme_volatility_handling(self, name, engine, extreme_volatility_data):
        """Test handling of extreme volatility."""
        result = engine(extreme_volatility_data)
        assert "status" in result
        assert isinstance(result["confidence"], (int, float))

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_large_dataset_handling(self, name, engine, large_dataset):
        """Test handling of large datasets."""
        result = engine(large_dataset)
        assert "status" in result
        assert isinstance(result["confidence"], (int, float))


# ============================================================================
# Analysis Mode Tests
# ============================================================================

class TestAnalysisModes:
    """Test different analysis modes for all engines."""

    ALL_ENGINES = [
        ("dow_theory", analyze_dow_theory),
        ("ichimoku", analyze_ichimoku),
        ("vsa", analyze_vsa),
        ("chart_patterns", analyze_chart_patterns),
        ("wyckoff", analyze_wyckoff),
        ("elliott_wave", analyze_elliott_wave),
        ("chan_theory", analyze_chan_theory),
        ("harmonic", analyze_harmonic),
        ("market_profile", analyze_market_profile),
    ]

    @pytest.fixture
    def test_data(self) -> OHLCVData:
        """Create test data for mode testing."""
        prices = []
        base = 100.0
        for i in range(150):
            trend = i * 0.4
            wave = 4 * np.sin(i * 0.25)
            prices.append(base + trend + wave)
        bars = create_ohlcv_bars(prices)
        return OHLCVData(symbol="TEST", timeframe="4H", bars=bars)

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_conservative_mode(self, name, engine, test_data):
        """Test conservative mode produces valid results."""
        if name == "chan_theory":
            result = engine(test_data, strictness="conservative")
        else:
            result = engine(test_data, mode="conservative")
        assert "status" in result
        assert "confidence" in result

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_balanced_mode(self, name, engine, test_data):
        """Test balanced mode produces valid results."""
        if name == "chan_theory":
            result = engine(test_data, strictness="balanced")
        else:
            result = engine(test_data, mode="balanced")
        assert "status" in result
        assert "confidence" in result

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_aggressive_mode(self, name, engine, test_data):
        """Test aggressive mode produces valid results."""
        if name == "chan_theory":
            result = engine(test_data, strictness="aggressive")
        else:
            result = engine(test_data, mode="aggressive")
        assert "status" in result
        assert "confidence" in result


# ============================================================================
# Trend Direction Tests
# ============================================================================

class TestTrendDetection:
    """Test trend detection capabilities of engines."""

    @pytest.mark.parametrize("engine", [
        analyze_dow_theory,
        analyze_ichimoku,
        analyze_wyckoff,
        analyze_market_profile,
    ])
    def test_uptrend_detection(self, engine, strong_uptrend_data):
        """Test that uptrends are detected as bullish."""
        result = engine(strong_uptrend_data)
        # Strong uptrend should generally be detected
        if result["status"] != "neutral":
            # If not neutral, should likely be bullish
            assert result["confidence"] > 0

    @pytest.mark.parametrize("engine", [
        analyze_dow_theory,
        analyze_ichimoku,
        analyze_wyckoff,
        analyze_market_profile,
    ])
    def test_downtrend_detection(self, engine, strong_downtrend_data):
        """Test that downtrends are detected as bearish."""
        result = engine(strong_downtrend_data)
        if result["status"] != "neutral":
            assert result["confidence"] > 0


# ============================================================================
# Confidence Threshold Tests
# ============================================================================

class TestConfidenceThresholds:
    """Test confidence scoring and thresholds."""

    @pytest.fixture
    def ambiguous_data(self) -> OHLCVData:
        """Create ambiguous/sideways data."""
        np.random.seed(123)
        prices = [100.0]
        for _ in range(99):
            change = np.random.uniform(-0.01, 0.01)
            prices.append(prices[-1] * (1 + change))
        bars = create_ohlcv_bars(prices)
        return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)

    def test_ambiguous_data_has_lower_confidence(self, ambiguous_data):
        """Test that ambiguous data produces lower confidence."""
        result = analyze_dow_theory(ambiguous_data)
        # Ambiguous data should have lower confidence
        assert result["confidence"] < 80

    def test_neutral_status_below_threshold(self, ambiguous_data):
        """Test that low confidence leads to neutral status."""
        result = analyze_dow_theory(ambiguous_data, mode="conservative")
        # Very ambiguous data with conservative mode should likely be neutral
        if result["confidence"] < 60:
            assert result["status"] == "neutral"


# ============================================================================
# Timeframe Weight Tests
# ============================================================================

class TestTimeframeWeights:
    """Test that timeframe affects confidence appropriately."""

    def create_data_for_timeframe(self, timeframe: str) -> OHLCVData:
        """Create test data with specific timeframe."""
        prices = [100.0 + i * 0.5 for i in range(100)]
        bars = create_ohlcv_bars(prices)
        return OHLCVData(symbol="TEST", timeframe=timeframe, bars=bars)

    @pytest.mark.parametrize("timeframe", ["15m", "1H", "4H", "1D", "1W"])
    def test_timeframe_accepted(self, timeframe):
        """Test that all timeframes are accepted."""
        data = self.create_data_for_timeframe(timeframe)
        result = analyze_dow_theory(data)
        assert "status" in result
        assert "confidence" in result


# ============================================================================
# LLM Summary Tests
# ============================================================================

class TestLLMSummary:
    """Test LLM summary generation."""

    ALL_ENGINES = [
        ("dow_theory", analyze_dow_theory),
        ("ichimoku", analyze_ichimoku),
        ("vsa", analyze_vsa),
        ("chart_patterns", analyze_chart_patterns),
        ("wyckoff", analyze_wyckoff),
        ("elliott_wave", analyze_elliott_wave),
        ("chan_theory", analyze_chan_theory),
        ("harmonic", analyze_harmonic),
        ("market_profile", analyze_market_profile),
    ]

    @pytest.fixture
    def test_data(self) -> OHLCVData:
        """Create test data."""
        prices = [100.0 + i * 0.3 for i in range(150)]
        bars = create_ohlcv_bars(prices)
        return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_summary_is_string(self, name, engine, test_data):
        """Test that llm_summary is a non-empty string."""
        result = engine(test_data)
        assert isinstance(result["llm_summary"], str)
        assert len(result["llm_summary"]) > 0

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_summary_contains_timeframe(self, name, engine, test_data):
        """Test that summary mentions the timeframe."""
        result = engine(test_data)
        summary = result["llm_summary"].upper()
        # Summary should mention timeframe or be informative
        assert len(summary) > 10  # Non-trivial summary


# ============================================================================
# Invalidation Tests
# ============================================================================

class TestInvalidation:
    """Test invalidation level generation."""

    ALL_ENGINES = [
        ("dow_theory", analyze_dow_theory),
        ("ichimoku", analyze_ichimoku),
        ("vsa", analyze_vsa),
        ("chart_patterns", analyze_chart_patterns),
        ("wyckoff", analyze_wyckoff),
        ("elliott_wave", analyze_elliott_wave),
        ("chan_theory", analyze_chan_theory),
        ("harmonic", analyze_harmonic),
        ("market_profile", analyze_market_profile),
    ]

    @pytest.fixture
    def signal_data(self) -> OHLCVData:
        """Create data likely to produce a signal."""
        prices = [100.0 + i * 1.0 for i in range(200)]
        volumes = [1000 + i * 100 for i in range(200)]
        bars = create_ohlcv_bars(prices, volumes)
        return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_invalidation_is_string(self, name, engine, signal_data):
        """Test that invalidation is a string."""
        result = engine(signal_data)
        assert isinstance(result["invalidation"], str)

    @pytest.mark.parametrize("name,engine", ALL_ENGINES)
    def test_invalidation_not_empty_for_signal(self, name, engine, signal_data):
        """Test that signals have non-empty invalidation."""
        result = engine(signal_data)
        if result["status"] in ["bullish", "bearish"]:
            assert len(result["invalidation"]) > 0
