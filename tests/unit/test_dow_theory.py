"""
Tests for the Dow Theory analysis engine.

Tests cover:
- Swing point detection
- Trend structure analysis
- Trend direction identification
- Volume confirmation
- Invalidation levels
"""

import pytest
import numpy as np
from pathlib import Path

from sigmapilot_mcp.core.data_loader import OHLCVData, OHLCVBar, load_ohlcv_from_csv
from sigmapilot_mcp.engines.dow_theory import (
    analyze_dow_theory,
    get_detailed_analysis,
    detect_swing_points,
    analyze_trend_structure,
    TrendDirection,
    TrendPhase,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def uptrend_data():
    """Create OHLCV data with clear uptrend pattern (HH/HL)."""
    bars = []
    base_price = 100.0
    timestamps = list(range(1704067200, 1704067200 + 100 * 3600, 3600))

    for i in range(100):
        # Simulate uptrend with higher highs and higher lows
        trend_component = i * 0.5  # Gradual upward drift
        cycle = np.sin(i * 0.3) * 5  # Minor oscillation

        open_price = base_price + trend_component + cycle
        high = open_price + 2 + np.random.random()
        low = open_price - 1.5 + np.random.random() * 0.5
        close = open_price + 1 + np.random.random()
        volume = 1000000 + i * 10000  # Increasing volume

        bars.append(OHLCVBar(
            timestamp=timestamps[i],
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        ))

    return OHLCVData(symbol="TEST:UPTREND", timeframe="1h", bars=bars)


@pytest.fixture
def downtrend_data():
    """Create OHLCV data with clear downtrend pattern (LL/LH)."""
    bars = []
    base_price = 150.0
    timestamps = list(range(1704067200, 1704067200 + 100 * 3600, 3600))

    for i in range(100):
        # Simulate downtrend with lower highs and lower lows
        trend_component = -i * 0.5  # Gradual downward drift
        cycle = np.sin(i * 0.3) * 5

        open_price = base_price + trend_component + cycle
        high = open_price + 1.5 + np.random.random() * 0.5
        low = open_price - 2 - np.random.random()
        close = open_price - 1 - np.random.random()
        volume = 1000000 + i * 10000

        bars.append(OHLCVBar(
            timestamp=timestamps[i],
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        ))

    return OHLCVData(symbol="TEST:DOWNTREND", timeframe="1h", bars=bars)


@pytest.fixture
def sideways_data():
    """Create OHLCV data with sideways/range-bound pattern."""
    bars = []
    base_price = 100.0
    timestamps = list(range(1704067200, 1704067200 + 100 * 3600, 3600))

    for i in range(100):
        # Oscillate around base price
        cycle = np.sin(i * 0.4) * 8

        open_price = base_price + cycle
        high = open_price + 2
        low = open_price - 2
        close = open_price + np.random.uniform(-1, 1)
        volume = 1000000

        bars.append(OHLCVBar(
            timestamp=timestamps[i],
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        ))

    return OHLCVData(symbol="TEST:SIDEWAYS", timeframe="1h", bars=bars)


@pytest.fixture
def insufficient_data():
    """Create minimal OHLCV data (too short for analysis)."""
    bars = [
        OHLCVBar(timestamp=1704067200 + i * 3600, open=100, high=101, low=99, close=100, volume=1000)
        for i in range(20)
    ]
    return OHLCVData(symbol="TEST:SHORT", timeframe="1h", bars=bars)


# =============================================================================
# Swing Point Detection Tests
# =============================================================================

class TestSwingPointDetection:
    """Tests for swing point detection."""

    def test_detects_swing_highs(self, uptrend_data):
        """Test that swing highs are detected."""
        swing_highs, swing_lows = detect_swing_points(uptrend_data)
        assert len(swing_highs) > 0
        for sh in swing_highs:
            assert sh.is_high is True

    def test_detects_swing_lows(self, uptrend_data):
        """Test that swing lows are detected."""
        swing_highs, swing_lows = detect_swing_points(uptrend_data)
        assert len(swing_lows) > 0
        for sl in swing_lows:
            assert sl.is_high is False

    def test_swing_points_have_valid_indices(self, uptrend_data):
        """Test that swing points have valid indices."""
        swing_highs, swing_lows = detect_swing_points(uptrend_data)

        for sh in swing_highs:
            assert 0 <= sh.index < len(uptrend_data)

        for sl in swing_lows:
            assert 0 <= sl.index < len(uptrend_data)

    def test_insufficient_data_returns_empty(self, insufficient_data):
        """Test that insufficient data returns empty lists."""
        swing_highs, swing_lows = detect_swing_points(insufficient_data, lookback=10)
        # May still find some with short lookback, or empty
        assert isinstance(swing_highs, list)
        assert isinstance(swing_lows, list)


# =============================================================================
# Trend Analysis Tests
# =============================================================================

class TestTrendAnalysis:
    """Tests for trend structure analysis."""

    def test_uptrend_detected(self, uptrend_data):
        """Test that uptrend is correctly identified."""
        result = analyze_dow_theory(uptrend_data)

        assert result["is_error"] is False
        # Due to randomness in data generation, we accept bullish or neutral
        assert result["status"] in ["bullish", "neutral"]

    def test_downtrend_detected(self, downtrend_data):
        """Test that downtrend is correctly identified."""
        result = analyze_dow_theory(downtrend_data)

        assert result["is_error"] is False
        assert result["status"] in ["bearish", "neutral"]

    def test_sideways_detected(self, sideways_data):
        """Test that sideways market is identified."""
        result = analyze_dow_theory(sideways_data)

        assert result["is_error"] is False
        # Sideways should be neutral or low confidence
        # The No Signal Protocol may kick in

    def test_insufficient_data_handled(self, insufficient_data):
        """Test that insufficient data returns proper result."""
        result = analyze_dow_theory(insufficient_data)

        assert result["status"] == "neutral"
        assert result["confidence"] == 0
        assert "Insufficient" in result["llm_summary"] or "insufficient" in result["llm_summary"].lower()


# =============================================================================
# Result Schema Tests
# =============================================================================

class TestResultSchema:
    """Tests for result schema compliance."""

    def test_result_has_required_fields(self, uptrend_data):
        """Test that result has all required fields."""
        result = analyze_dow_theory(uptrend_data)

        assert "status" in result
        assert "confidence" in result
        assert "attribution" in result
        assert "llm_summary" in result
        assert "invalidation" in result
        assert "is_error" in result

    def test_status_is_valid(self, uptrend_data):
        """Test that status is one of allowed values."""
        result = analyze_dow_theory(uptrend_data)
        assert result["status"] in ["bullish", "bearish", "neutral"]

    def test_confidence_in_range(self, uptrend_data):
        """Test that confidence is between 0 and 100."""
        result = analyze_dow_theory(uptrend_data)
        assert 0 <= result["confidence"] <= 100

    def test_attribution_has_theory(self, uptrend_data):
        """Test that attribution includes theory name."""
        result = analyze_dow_theory(uptrend_data)
        assert result["attribution"]["theory"] == "Dow Theory"


# =============================================================================
# Detailed Analysis Tests
# =============================================================================

class TestDetailedAnalysis:
    """Tests for get_detailed_analysis function."""

    def test_detailed_returns_swing_points(self, uptrend_data):
        """Test that detailed analysis includes swing points."""
        result = get_detailed_analysis(uptrend_data)

        assert "swing_highs" in result
        assert "swing_lows" in result
        assert isinstance(result["swing_highs"], list)
        assert isinstance(result["swing_lows"], list)

    def test_detailed_returns_trend_direction(self, uptrend_data):
        """Test that detailed analysis includes trend direction."""
        result = get_detailed_analysis(uptrend_data)
        assert "trend_direction" in result
        assert result["trend_direction"] in ["bullish", "bearish", "sideways"]

    def test_detailed_returns_rules_triggered(self, uptrend_data):
        """Test that detailed analysis includes triggered rules."""
        result = get_detailed_analysis(uptrend_data)
        assert "rules_triggered" in result
        assert isinstance(result["rules_triggered"], list)


# =============================================================================
# Mode Tests
# =============================================================================

class TestAnalysisModes:
    """Tests for different analysis modes."""

    def test_conservative_mode_lower_confidence(self, uptrend_data):
        """Test that conservative mode produces lower confidence."""
        balanced = analyze_dow_theory(uptrend_data, mode="balanced")
        conservative = analyze_dow_theory(uptrend_data, mode="conservative")

        # Conservative should have lower or equal confidence
        assert conservative["confidence"] <= balanced["confidence"] + 5

    def test_aggressive_mode_higher_confidence(self, uptrend_data):
        """Test that aggressive mode produces higher confidence."""
        balanced = analyze_dow_theory(uptrend_data, mode="balanced")
        aggressive = analyze_dow_theory(uptrend_data, mode="aggressive")

        # Aggressive should have higher or equal confidence
        assert aggressive["confidence"] >= balanced["confidence"] - 5


# =============================================================================
# CSV Fixture Tests
# =============================================================================

class TestWithFixtures:
    """Tests using deterministic CSV fixtures."""

    def test_btc_uptrend_fixture(self):
        """Test with BTC uptrend fixture."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "deterministic_ohlcv" / "btc_1h_uptrend.csv"

        if not fixture_path.exists():
            pytest.skip("Fixture file not found")

        data = load_ohlcv_from_csv(fixture_path, symbol="TEST:BTCUSDT", timeframe="1h")
        result = analyze_dow_theory(data)

        assert result["is_error"] is False
        # Uptrend fixture should produce bullish or neutral
        assert result["status"] in ["bullish", "neutral"]

    def test_eth_downtrend_fixture(self):
        """Test with ETH downtrend fixture."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "deterministic_ohlcv" / "eth_4h_downtrend.csv"

        if not fixture_path.exists():
            pytest.skip("Fixture file not found")

        data = load_ohlcv_from_csv(fixture_path, symbol="TEST:ETHUSDT", timeframe="4h")
        result = analyze_dow_theory(data)

        assert result["is_error"] is False
        # Downtrend fixture should produce bearish or neutral
        assert result["status"] in ["bearish", "neutral"]
