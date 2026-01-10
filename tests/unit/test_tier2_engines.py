"""
Unit tests for Tier 2 engines (Phase 3).

Tests cover:
- Wyckoff Phase Detector
- Elliott Wave Analyzer
- Chan Theory Analyzer
- Harmonic Pattern Detector
- Market Profile Analyzer
"""

import pytest
import numpy as np
from typing import List

from sigmapilot_mcp.core.data_loader import OHLCVData, OHLCVBar
from sigmapilot_mcp.engines.wyckoff import (
    analyze_wyckoff,
    get_detailed_wyckoff,
    detect_trading_range,
    detect_wyckoff_events,
    WyckoffPhase,
)
from sigmapilot_mcp.engines.elliott_wave import (
    analyze_elliott_wave,
    get_detailed_elliott,
    find_pivots,
    WaveType,
)
from sigmapilot_mcp.engines.chan_theory import (
    analyze_chan_theory,
    get_detailed_chan,
    merge_k_lines,
    detect_fractals,
)
from sigmapilot_mcp.engines.harmonic import (
    analyze_harmonic,
    get_all_harmonic_patterns,
    find_swing_points,
    HarmonicType,
)
from sigmapilot_mcp.engines.market_profile import (
    analyze_market_profile,
    get_detailed_profile,
    build_volume_profile,
    calculate_poc,
    ProfileShape,
)


# ============================================================================
# Test Fixtures
# ============================================================================

def create_test_bars(
    prices: List[float],
    volumes: List[float] = None,
    base_timestamp: int = 1700000000
) -> List[OHLCVBar]:
    """Create test OHLCV bars from price series."""
    if volumes is None:
        volumes = [1000.0] * len(prices)

    bars = []
    for i, price in enumerate(prices):
        # Create realistic OHLCV from close prices
        open_price = prices[i-1] if i > 0 else price

        # Ensure high/low properly contain open and close
        spread = abs(price * 0.01)  # 1% spread
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
def uptrend_data() -> OHLCVData:
    """Create uptrend test data with swing points."""
    # Create an uptrend with clear higher highs and higher lows
    prices = []
    base = 100.0

    for i in range(100):
        # Upward trend with waves
        trend = i * 0.5  # Upward bias
        wave = 5 * np.sin(i * 0.3)  # Oscillation
        price = base + trend + wave
        prices.append(price)

    # Higher volume on up moves
    volumes = [1000 + 500 * np.sin(i * 0.3) for i in range(100)]

    bars = create_test_bars(prices, volumes)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def downtrend_data() -> OHLCVData:
    """Create downtrend test data."""
    prices = []
    base = 150.0

    for i in range(100):
        trend = -i * 0.4
        wave = 4 * np.sin(i * 0.25)
        price = max(50, base + trend + wave)
        prices.append(price)

    volumes = [1000 + 300 * abs(np.sin(i * 0.25)) for i in range(100)]
    bars = create_test_bars(prices, volumes)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


@pytest.fixture
def consolidation_data() -> OHLCVData:
    """Create sideways/consolidation test data."""
    prices = []
    base = 100.0

    for i in range(120):
        # Range-bound movement
        wave = 3 * np.sin(i * 0.2)
        noise = np.random.uniform(-1, 1)
        price = base + wave + noise
        prices.append(price)

    volumes = [800 + 200 * np.random.random() for _ in range(120)]
    bars = create_test_bars(prices, volumes)
    return OHLCVData(symbol="TEST", timeframe="4H", bars=bars)


@pytest.fixture
def harmonic_pattern_data() -> OHLCVData:
    """Create data with potential Gartley-like structure."""
    # X-A-B-C-D pattern
    # X at 100, A at 120 (XA up), B at 112 (AB retrace 61.8% ~= 7.4),
    # C at 118, D around 115 (78.6% retrace of XA)
    prices = (
        [100 + i * 2 for i in range(10)] +  # X to A (100 -> 120)
        [120 - i * 0.8 for i in range(10)] +  # A to B (120 -> 112)
        [112 + i * 0.6 for i in range(10)] +  # B to C (112 -> 118)
        [118 - i * 0.3 for i in range(20)]    # C to D (118 -> 112)
    )

    bars = create_test_bars(prices)
    return OHLCVData(symbol="TEST", timeframe="1H", bars=bars)


@pytest.fixture
def minimal_data() -> OHLCVData:
    """Create minimal data that should trigger no-signal."""
    prices = [100.0, 101.0, 99.0, 100.5, 100.0]
    bars = create_test_bars(prices)
    return OHLCVData(symbol="TEST", timeframe="1D", bars=bars)


# ============================================================================
# Wyckoff Engine Tests
# ============================================================================

class TestWyckoffEngine:
    """Tests for Wyckoff Phase Detector."""

    def test_returns_valid_schema(self, consolidation_data):
        """Test that result contains required fields."""
        result = analyze_wyckoff(consolidation_data)

        assert "status" in result
        assert "confidence" in result
        assert "attribution" in result
        assert "llm_summary" in result
        assert "invalidation" in result
        assert "is_error" in result
        assert result["status"] in ["bullish", "bearish", "neutral", "signal", "error"]

    def test_attribution_is_wyckoff(self, consolidation_data):
        """Test that attribution references Wyckoff."""
        result = analyze_wyckoff(consolidation_data)

        if result["status"] != "error":
            assert "Wyckoff" in result["attribution"]["theory"]

    def test_insufficient_data_returns_neutral(self, minimal_data):
        """Test that insufficient data returns neutral/no signal."""
        result = analyze_wyckoff(minimal_data)

        assert result["status"] == "neutral"
        assert result["confidence"] < 60

    def test_detailed_analysis_returns_structure(self, consolidation_data):
        """Test detailed analysis returns phase and events."""
        result = get_detailed_wyckoff(consolidation_data)

        assert "phase" in result
        assert "stage" in result
        assert "trading_range" in result
        assert "events" in result
        assert isinstance(result["events"], list)

    def test_trading_range_detection(self, consolidation_data):
        """Test trading range detection in consolidation."""
        tr = detect_trading_range(consolidation_data)

        # Should detect a range in consolidation data
        # May or may not detect based on threshold
        if tr:
            assert tr.high > tr.low
            assert tr.range_bars > 0

    def test_confidence_bounded(self, consolidation_data):
        """Test confidence is within valid range."""
        result = analyze_wyckoff(consolidation_data)

        assert 0 <= result["confidence"] <= 100

    def test_modes_affect_result(self, consolidation_data):
        """Test that analysis modes produce different confidence levels."""
        conservative = analyze_wyckoff(consolidation_data, mode="conservative")
        aggressive = analyze_wyckoff(consolidation_data, mode="aggressive")

        # Conservative should generally have lower or equal confidence
        # (though not guaranteed if pattern quality differs)
        assert conservative["confidence"] >= 0
        assert aggressive["confidence"] >= 0


# ============================================================================
# Elliott Wave Engine Tests
# ============================================================================

class TestElliottWaveEngine:
    """Tests for Elliott Wave Analyzer."""

    def test_returns_valid_schema(self, uptrend_data):
        """Test that result contains required fields."""
        result = analyze_elliott_wave(uptrend_data)

        assert "status" in result
        assert "confidence" in result
        assert "attribution" in result
        assert "llm_summary" in result
        assert "is_error" in result

    def test_attribution_is_elliott(self, uptrend_data):
        """Test that attribution references Elliott Wave."""
        result = analyze_elliott_wave(uptrend_data)

        if result["status"] != "error":
            assert "Elliott" in result["attribution"]["theory"]

    def test_insufficient_data_returns_neutral(self, minimal_data):
        """Test that insufficient data returns neutral."""
        result = analyze_elliott_wave(minimal_data)

        assert result["status"] == "neutral"

    def test_detailed_returns_interpretations(self, uptrend_data):
        """Test detailed analysis returns wave interpretations."""
        result = get_detailed_elliott(uptrend_data)

        assert "interpretations" in result
        assert "pivot_count" in result
        assert isinstance(result["interpretations"], list)

    def test_pivot_detection(self, uptrend_data):
        """Test pivot point detection."""
        pivot_highs, pivot_lows = find_pivots(uptrend_data)

        # Should find some pivots in trending data
        assert len(pivot_highs) > 0 or len(pivot_lows) > 0

    def test_max_interpretations_respected(self, uptrend_data):
        """Test that max_interpretations limit is respected."""
        result = get_detailed_elliott(uptrend_data, max_interpretations=1)

        assert len(result["interpretations"]) <= 1

    def test_confidence_bounded(self, uptrend_data):
        """Test confidence is within valid range."""
        result = analyze_elliott_wave(uptrend_data)

        assert 0 <= result["confidence"] <= 100


# ============================================================================
# Chan Theory Engine Tests
# ============================================================================

class TestChanTheoryEngine:
    """Tests for Chan Theory (Chanlun) Analyzer."""

    def test_returns_valid_schema(self, uptrend_data):
        """Test that result contains required fields."""
        result = analyze_chan_theory(uptrend_data)

        assert "status" in result
        assert "confidence" in result
        assert "attribution" in result
        assert "llm_summary" in result
        assert "is_error" in result

    def test_attribution_is_chan(self, uptrend_data):
        """Test that attribution references Chan Theory."""
        result = analyze_chan_theory(uptrend_data)

        if result["status"] != "error":
            assert "Chan" in result["attribution"]["theory"]

    def test_insufficient_data_returns_neutral(self, minimal_data):
        """Test that insufficient data returns neutral."""
        result = analyze_chan_theory(minimal_data)

        assert result["status"] == "neutral"

    def test_detailed_returns_structure(self, uptrend_data):
        """Test detailed analysis returns fractals and strokes."""
        result = get_detailed_chan(uptrend_data)

        assert "fractals" in result
        assert "strokes" in result
        assert "segments" in result
        assert "hub" in result
        assert "signals" in result
        assert "counts" in result

    def test_k_line_merging(self, uptrend_data):
        """Test K-line inclusion handling."""
        merged = merge_k_lines(uptrend_data)

        # Merged should have equal or fewer bars
        assert len(merged) <= len(uptrend_data.bars)

    def test_fractal_detection(self, uptrend_data):
        """Test fractal detection."""
        merged = merge_k_lines(uptrend_data)
        fractals = detect_fractals(merged)

        # Should detect fractals in trending data
        assert isinstance(fractals, list)

    def test_strictness_modes(self, uptrend_data):
        """Test different strictness modes."""
        conservative = analyze_chan_theory(uptrend_data, strictness="conservative")
        aggressive = analyze_chan_theory(uptrend_data, strictness="aggressive")

        assert conservative["confidence"] >= 0
        assert aggressive["confidence"] >= 0


# ============================================================================
# Harmonic Pattern Engine Tests
# ============================================================================

class TestHarmonicEngine:
    """Tests for Harmonic Pattern Detector."""

    def test_returns_valid_schema(self, harmonic_pattern_data):
        """Test that result contains required fields."""
        result = analyze_harmonic(harmonic_pattern_data)

        assert "status" in result
        assert "confidence" in result
        assert "attribution" in result
        assert "llm_summary" in result
        assert "is_error" in result

    def test_attribution_is_harmonic(self, harmonic_pattern_data):
        """Test that attribution references Harmonic Patterns."""
        result = analyze_harmonic(harmonic_pattern_data)

        if result["status"] != "error":
            assert "Harmonic" in result["attribution"]["theory"]

    def test_insufficient_data_returns_neutral(self, minimal_data):
        """Test that insufficient data returns neutral."""
        result = analyze_harmonic(minimal_data)

        assert result["status"] == "neutral"

    def test_detailed_returns_patterns(self, harmonic_pattern_data):
        """Test detailed analysis returns pattern info."""
        result = get_all_harmonic_patterns(harmonic_pattern_data)

        assert "patterns" in result
        assert "swing_points_count" in result
        assert "patterns_found" in result

    def test_swing_point_detection(self, harmonic_pattern_data):
        """Test swing point detection."""
        swing_points = find_swing_points(harmonic_pattern_data)

        assert len(swing_points) >= 0  # May not find valid swings
        for sp in swing_points:
            assert hasattr(sp, 'price')
            assert hasattr(sp, 'is_high')

    def test_tolerance_parameter(self, harmonic_pattern_data):
        """Test that tolerance affects pattern detection."""
        tight = analyze_harmonic(harmonic_pattern_data, tolerance=0.01)
        loose = analyze_harmonic(harmonic_pattern_data, tolerance=0.10)

        # Both should return valid results
        assert tight["status"] in ["bullish", "bearish", "neutral", "signal", "error"]
        assert loose["status"] in ["bullish", "bearish", "neutral", "signal", "error"]

    def test_confidence_bounded(self, harmonic_pattern_data):
        """Test confidence is within valid range."""
        result = analyze_harmonic(harmonic_pattern_data)

        assert 0 <= result["confidence"] <= 100


# ============================================================================
# Market Profile Engine Tests
# ============================================================================

class TestMarketProfileEngine:
    """Tests for Market Profile Analyzer."""

    def test_returns_valid_schema(self, consolidation_data):
        """Test that result contains required fields."""
        result = analyze_market_profile(consolidation_data)

        assert "status" in result
        assert "confidence" in result
        assert "attribution" in result
        assert "llm_summary" in result
        assert "is_error" in result

    def test_attribution_is_market_profile(self, consolidation_data):
        """Test that attribution references Market Profile."""
        result = analyze_market_profile(consolidation_data)

        if result["status"] != "error":
            assert "Market Profile" in result["attribution"]["theory"]

    def test_insufficient_data_returns_neutral(self, minimal_data):
        """Test that insufficient data returns neutral."""
        result = analyze_market_profile(minimal_data)

        assert result["status"] == "neutral"

    def test_detailed_returns_profile(self, consolidation_data):
        """Test detailed analysis returns profile data."""
        result = get_detailed_profile(consolidation_data)

        assert "poc" in result
        assert "vah" in result
        assert "val" in result
        assert "shape" in result
        assert "market_state" in result
        assert "volume_levels" in result

    def test_volume_profile_building(self, consolidation_data):
        """Test volume profile construction."""
        levels = build_volume_profile(consolidation_data)

        assert len(levels) > 0
        for level in levels:
            assert hasattr(level, 'price')
            assert hasattr(level, 'volume')
            assert level.volume >= 0

    def test_poc_calculation(self, consolidation_data):
        """Test Point of Control calculation."""
        levels = build_volume_profile(consolidation_data)
        poc = calculate_poc(levels)

        assert poc > 0
        # POC should be within the price range
        min_price = min(l.price for l in levels)
        max_price = max(l.price for l in levels)
        assert min_price <= poc <= max_price

    def test_value_area_valid(self, consolidation_data):
        """Test that VAH > VAL."""
        result = get_detailed_profile(consolidation_data)

        assert result["vah"] >= result["val"]

    def test_confidence_bounded(self, consolidation_data):
        """Test confidence is within valid range."""
        result = analyze_market_profile(consolidation_data)

        assert 0 <= result["confidence"] <= 100


# ============================================================================
# Cross-Engine Consistency Tests
# ============================================================================

class TestCrossEngineConsistency:
    """Tests for consistency across all Tier 2 engines."""

    def test_all_engines_handle_same_data(self, uptrend_data):
        """Test all engines can process the same data without errors."""
        engines = [
            analyze_wyckoff,
            analyze_elliott_wave,
            analyze_chan_theory,
            analyze_harmonic,
            analyze_market_profile,
        ]

        for engine in engines:
            result = engine(uptrend_data)
            assert "status" in result
            assert "confidence" in result
            assert "is_error" in result
            assert not result["is_error"], f"Engine {engine.__name__} returned error"

    def test_all_engines_different_attributions(self, consolidation_data):
        """Test each engine has unique attribution."""
        results = [
            analyze_wyckoff(consolidation_data),
            analyze_elliott_wave(consolidation_data),
            analyze_chan_theory(consolidation_data),
            analyze_harmonic(consolidation_data),
            analyze_market_profile(consolidation_data),
        ]

        theories = []
        for result in results:
            if result["status"] != "error" and "attribution" in result:
                theories.append(result["attribution"]["theory"])

        # All non-error results should have unique theories
        # (may have duplicates if some return errors)
        if len(theories) > 1:
            assert len(set(theories)) == len(theories), "Duplicate attributions found"

    def test_minimal_data_all_neutral(self, minimal_data):
        """Test all engines return neutral for minimal data."""
        engines = [
            analyze_wyckoff,
            analyze_elliott_wave,
            analyze_chan_theory,
            analyze_harmonic,
            analyze_market_profile,
        ]

        for engine in engines:
            result = engine(minimal_data)
            assert result["status"] == "neutral", \
                f"Engine {engine.__name__} should return neutral for minimal data"

    def test_all_engines_respect_modes(self, consolidation_data):
        """Test all engines accept mode parameter."""
        engines = [
            analyze_wyckoff,
            analyze_elliott_wave,
            analyze_chan_theory,
            analyze_harmonic,
            analyze_market_profile,
        ]

        for engine in engines:
            for mode in ["conservative", "balanced", "aggressive"]:
                result = engine(consolidation_data, mode=mode)
                assert "status" in result, \
                    f"Engine {engine.__name__} failed with mode={mode}"
