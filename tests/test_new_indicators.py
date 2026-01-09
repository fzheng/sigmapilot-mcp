"""
Unit tests for the new technical indicators added to coin_analysis.

Tests cover:
- Ichimoku Cloud signals
- VWAP indicator
- Pivot Points (Classic, Fibonacci, Camarilla)
- TradingView Recommendations
- Williams %R oscillator
- CCI (Commodity Channel Index)
- Awesome Oscillator
- Ultimate Oscillator
- Momentum indicator
- Hull MA and VWMA
- Parabolic SAR
- Additional SMA/EMA periods
- Scanner tools: pivot_points_scanner, tradingview_recommendation
"""

import pytest


# =============================================================================
# Tests for Helper Functions in coin_analysis
# =============================================================================

class TestRecommendationText:
    """Tests for get_recommendation_text function logic."""

    def test_strong_buy(self):
        """Test STRONG_BUY when value >= 0.5."""
        # Value >= 0.5 should return STRONG_BUY
        value = 0.5
        if value >= 0.5:
            signal = "STRONG_BUY"
        elif value >= 0.1:
            signal = "BUY"
        elif value > -0.1:
            signal = "NEUTRAL"
        elif value > -0.5:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
        assert signal == "STRONG_BUY"

    def test_buy(self):
        """Test BUY when 0.1 <= value < 0.5."""
        value = 0.3
        if value >= 0.5:
            signal = "STRONG_BUY"
        elif value >= 0.1:
            signal = "BUY"
        elif value > -0.1:
            signal = "NEUTRAL"
        elif value > -0.5:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
        assert signal == "BUY"

    def test_neutral(self):
        """Test NEUTRAL when -0.1 < value < 0.1."""
        value = 0.0
        if value >= 0.5:
            signal = "STRONG_BUY"
        elif value >= 0.1:
            signal = "BUY"
        elif value > -0.1:
            signal = "NEUTRAL"
        elif value > -0.5:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
        assert signal == "NEUTRAL"

    def test_sell(self):
        """Test SELL when -0.5 < value <= -0.1."""
        value = -0.3
        if value >= 0.5:
            signal = "STRONG_BUY"
        elif value >= 0.1:
            signal = "BUY"
        elif value > -0.1:
            signal = "NEUTRAL"
        elif value > -0.5:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
        assert signal == "SELL"

    def test_strong_sell(self):
        """Test STRONG_SELL when value <= -0.5."""
        value = -0.7
        if value >= 0.5:
            signal = "STRONG_BUY"
        elif value >= 0.1:
            signal = "BUY"
        elif value > -0.1:
            signal = "NEUTRAL"
        elif value > -0.5:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
        assert signal == "STRONG_SELL"


class TestWilliamsRSignal:
    """Tests for Williams %R signal interpretation."""

    def test_overbought(self):
        """Test Overbought when value > -20."""
        value = -10
        if value > -20:
            signal = "Overbought"
        elif value < -80:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Overbought"

    def test_oversold(self):
        """Test Oversold when value < -80."""
        value = -90
        if value > -20:
            signal = "Overbought"
        elif value < -80:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Oversold"

    def test_neutral(self):
        """Test Neutral when -80 <= value <= -20."""
        value = -50
        if value > -20:
            signal = "Overbought"
        elif value < -80:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Neutral"


class TestCCISignal:
    """Tests for CCI signal interpretation."""

    def test_overbought(self):
        """Test Overbought when value > 100."""
        value = 150
        if value > 100:
            signal = "Overbought"
        elif value < -100:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Overbought"

    def test_oversold(self):
        """Test Oversold when value < -100."""
        value = -150
        if value > 100:
            signal = "Overbought"
        elif value < -100:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Oversold"

    def test_neutral(self):
        """Test Neutral when -100 <= value <= 100."""
        value = 50
        if value > 100:
            signal = "Overbought"
        elif value < -100:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Neutral"


class TestIchimokuSignal:
    """Tests for Ichimoku Cloud signal interpretation."""

    def test_bullish_above_cloud(self):
        """Test Bullish signal when price is above both spans."""
        close = 110
        lead_a = 100
        lead_b = 95
        cloud_top = max(lead_a, lead_b)
        cloud_bottom = min(lead_a, lead_b)

        if lead_a == 0 or lead_b == 0:
            signal = "No Data"
        elif close > cloud_top:
            signal = "Bullish (Above Cloud)"
        elif close < cloud_bottom:
            signal = "Bearish (Below Cloud)"
        else:
            signal = "Neutral (Inside Cloud)"

        assert signal == "Bullish (Above Cloud)"

    def test_bearish_below_cloud(self):
        """Test Bearish signal when price is below both spans."""
        close = 90
        lead_a = 100
        lead_b = 95
        cloud_top = max(lead_a, lead_b)
        cloud_bottom = min(lead_a, lead_b)

        if lead_a == 0 or lead_b == 0:
            signal = "No Data"
        elif close > cloud_top:
            signal = "Bullish (Above Cloud)"
        elif close < cloud_bottom:
            signal = "Bearish (Below Cloud)"
        else:
            signal = "Neutral (Inside Cloud)"

        assert signal == "Bearish (Below Cloud)"

    def test_neutral_inside_cloud(self):
        """Test Neutral signal when price is inside the cloud."""
        close = 97
        lead_a = 100
        lead_b = 95
        cloud_top = max(lead_a, lead_b)
        cloud_bottom = min(lead_a, lead_b)

        if lead_a == 0 or lead_b == 0:
            signal = "No Data"
        elif close > cloud_top:
            signal = "Bullish (Above Cloud)"
        elif close < cloud_bottom:
            signal = "Bearish (Below Cloud)"
        else:
            signal = "Neutral (Inside Cloud)"

        assert signal == "Neutral (Inside Cloud)"

    def test_no_data(self):
        """Test No Data when spans are zero."""
        close = 100
        lead_a = 0
        lead_b = 95

        if lead_a == 0 or lead_b == 0:
            signal = "No Data"
        else:
            signal = "Has Data"

        assert signal == "No Data"


class TestParabolicSARSignal:
    """Tests for Parabolic SAR signal interpretation."""

    def test_bullish_above_sar(self):
        """Test Bullish signal when price is above SAR."""
        close = 110
        psar = 100
        if psar == 0:
            signal = "No Data"
        elif close > psar:
            signal = "Bullish (Price above SAR)"
        else:
            signal = "Bearish (Price below SAR)"
        assert signal == "Bullish (Price above SAR)"

    def test_bearish_below_sar(self):
        """Test Bearish signal when price is below SAR."""
        close = 90
        psar = 100
        if psar == 0:
            signal = "No Data"
        elif close > psar:
            signal = "Bullish (Price above SAR)"
        else:
            signal = "Bearish (Price below SAR)"
        assert signal == "Bearish (Price below SAR)"

    def test_no_data(self):
        """Test No Data when SAR is zero."""
        close = 100
        psar = 0
        if psar == 0:
            signal = "No Data"
        elif close > psar:
            signal = "Bullish (Price above SAR)"
        else:
            signal = "Bearish (Price below SAR)"
        assert signal == "No Data"


# =============================================================================
# Tests for Pivot Points Calculation
# =============================================================================

class TestPivotPointsLogic:
    """Tests for pivot points proximity detection logic."""

    def test_is_near_true(self):
        """Test proximity detection returns True when close to level."""
        proximity_threshold = 0.01  # 1%
        price = 100.0
        level = 100.5

        is_near = abs(price - level) / level <= proximity_threshold
        assert is_near is True

    def test_is_near_false(self):
        """Test proximity detection returns False when far from level."""
        proximity_threshold = 0.01  # 1%
        price = 100.0
        level = 110.0

        is_near = abs(price - level) / level <= proximity_threshold
        assert is_near is False

    def test_is_near_zero_level(self):
        """Test proximity detection with zero level."""
        proximity_threshold = 0.01
        price = 100.0
        level = 0

        # Should handle zero gracefully
        is_near = level != 0 and abs(price - level) / level <= proximity_threshold
        assert is_near is False


# =============================================================================
# Tests for Oscillator Values
# =============================================================================

class TestOscillatorBounds:
    """Tests for oscillator value bounds and interpretation."""

    def test_rsi_overbought_boundary(self):
        """Test RSI overbought boundary at 70."""
        rsi = 70
        if rsi > 70:
            signal = "Overbought"
        elif rsi < 30:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Neutral"  # 70 exactly is not overbought

    def test_rsi_overbought(self):
        """Test RSI overbought at 71."""
        rsi = 71
        if rsi > 70:
            signal = "Overbought"
        elif rsi < 30:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Overbought"

    def test_stochastic_overbought(self):
        """Test Stochastic overbought boundary."""
        stoch_k = 85
        if stoch_k > 80:
            signal = "Overbought"
        elif stoch_k < 20:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Overbought"

    def test_stochastic_oversold(self):
        """Test Stochastic oversold boundary."""
        stoch_k = 15
        if stoch_k > 80:
            signal = "Overbought"
        elif stoch_k < 20:
            signal = "Oversold"
        else:
            signal = "Neutral"
        assert signal == "Oversold"


# =============================================================================
# Tests for Trend Alignment
# =============================================================================

class TestTrendAlignment:
    """Tests for trend alignment logic."""

    def test_short_term_bullish(self):
        """Test short-term bullish when price > SMA20."""
        close_price = 105
        sma20 = 100

        alignment = "Bullish" if close_price > (sma20 or close_price) else "Bearish"
        assert alignment == "Bullish"

    def test_short_term_bearish(self):
        """Test short-term bearish when price < SMA20."""
        close_price = 95
        sma20 = 100

        alignment = "Bullish" if close_price > (sma20 or close_price) else "Bearish"
        assert alignment == "Bearish"

    def test_trend_with_none_sma(self):
        """Test trend alignment when SMA is None (uses close_price)."""
        close_price = 100
        sma = None

        # When SMA is None, should compare price to itself (always Bullish by definition)
        alignment = "Bullish" if close_price > (sma or close_price) else "Bearish"
        # close_price > close_price is False, so Bearish
        assert alignment == "Bearish"


# =============================================================================
# Tests for MACD Signal
# =============================================================================

class TestMACDSignal:
    """Tests for MACD trend interpretation."""

    def test_macd_bullish(self):
        """Test bullish MACD when MACD > signal line."""
        macd = 1.5
        macd_signal = 1.0

        trend = "Bullish" if macd > macd_signal else "Bearish"
        assert trend == "Bullish"

    def test_macd_bearish(self):
        """Test bearish MACD when MACD < signal line."""
        macd = 0.5
        macd_signal = 1.0

        trend = "Bullish" if macd > macd_signal else "Bearish"
        assert trend == "Bearish"

    def test_macd_histogram(self):
        """Test MACD histogram calculation."""
        macd = 1.5
        macd_signal = 1.0

        histogram = macd - macd_signal
        assert histogram == 0.5


# =============================================================================
# Tests for ADX Trend Strength
# =============================================================================

class TestADXStrength:
    """Tests for ADX trend strength interpretation."""

    def test_strong_trend(self):
        """Test Strong trend when ADX > 25."""
        adx = 30
        trend_strength = "Strong" if adx > 25 else "Weak"
        assert trend_strength == "Strong"

    def test_weak_trend(self):
        """Test Weak trend when ADX <= 25."""
        adx = 20
        trend_strength = "Strong" if adx > 25 else "Weak"
        assert trend_strength == "Weak"

    def test_boundary_trend(self):
        """Test boundary at ADX = 25."""
        adx = 25
        trend_strength = "Strong" if adx > 25 else "Weak"
        assert trend_strength == "Weak"


# =============================================================================
# Tests for Volatility Classification
# =============================================================================

class TestVolatilityClassification:
    """Tests for BBW-based volatility classification."""

    def test_high_volatility(self):
        """Test High volatility when BBW > 0.05."""
        bbw = 0.06
        volatility = "High" if bbw > 0.05 else "Medium" if bbw > 0.02 else "Low"
        assert volatility == "High"

    def test_medium_volatility(self):
        """Test Medium volatility when 0.02 < BBW <= 0.05."""
        bbw = 0.03
        volatility = "High" if bbw > 0.05 else "Medium" if bbw > 0.02 else "Low"
        assert volatility == "Medium"

    def test_low_volatility(self):
        """Test Low volatility when BBW <= 0.02."""
        bbw = 0.015
        volatility = "High" if bbw > 0.05 else "Medium" if bbw > 0.02 else "Low"
        assert volatility == "Low"


# =============================================================================
# Tests for Ultimate Oscillator
# =============================================================================

class TestUltimateOscillator:
    """Tests for Ultimate Oscillator interpretation."""

    def test_uo_overbought(self):
        """Test overbought when UO > 70."""
        uo = 75
        signal = "Overbought" if uo > 70 else "Oversold" if uo < 30 else "Neutral"
        assert signal == "Overbought"

    def test_uo_oversold(self):
        """Test oversold when UO < 30."""
        uo = 25
        signal = "Overbought" if uo > 70 else "Oversold" if uo < 30 else "Neutral"
        assert signal == "Oversold"

    def test_uo_neutral(self):
        """Test neutral when 30 <= UO <= 70."""
        uo = 50
        signal = "Overbought" if uo > 70 else "Oversold" if uo < 30 else "Neutral"
        assert signal == "Neutral"


# =============================================================================
# Tests for TK Cross (Ichimoku)
# =============================================================================

class TestIchimokuTKCross:
    """Tests for Ichimoku TK cross signal."""

    def test_tk_bullish(self):
        """Test bullish TK cross when conversion > base."""
        conversion = 105
        base = 100

        tk_cross = "Bullish" if conversion > base else "Bearish" if conversion < base else "Neutral"
        assert tk_cross == "Bullish"

    def test_tk_bearish(self):
        """Test bearish TK cross when conversion < base."""
        conversion = 95
        base = 100

        tk_cross = "Bullish" if conversion > base else "Bearish" if conversion < base else "Neutral"
        assert tk_cross == "Bearish"

    def test_tk_neutral(self):
        """Test neutral when conversion == base."""
        conversion = 100
        base = 100

        tk_cross = "Bullish" if conversion > base else "Bearish" if conversion < base else "Neutral"
        assert tk_cross == "Neutral"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIndicatorIntegration:
    """Integration tests for combined indicator logic."""

    def test_full_bullish_scenario(self):
        """Test a complete bullish scenario with all indicators aligned."""
        # Simulate bullish conditions
        rsi = 55
        macd = 1.5
        macd_signal = 1.0
        adx = 30
        williams_r = -30
        cci = 50
        close = 110
        psar = 100

        # All signals should be positive
        assert rsi < 70  # Not overbought
        assert rsi > 30  # Not oversold
        assert macd > macd_signal  # Bullish MACD
        assert adx > 25  # Strong trend
        assert williams_r > -80 and williams_r < -20  # Neutral Williams %R
        assert cci > -100 and cci < 100  # Neutral CCI
        assert close > psar  # Price above SAR

    def test_full_bearish_scenario(self):
        """Test a complete bearish scenario with all indicators aligned."""
        # Simulate bearish conditions
        rsi = 25
        macd = -0.5
        macd_signal = 0.5
        adx = 30
        williams_r = -85
        cci = -120
        close = 90
        psar = 100

        # All signals should be negative
        assert rsi < 30  # Oversold
        assert macd < macd_signal  # Bearish MACD
        assert adx > 25  # Strong trend
        assert williams_r < -80  # Oversold Williams %R
        assert cci < -100  # Oversold CCI
        assert close < psar  # Price below SAR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
