"""
SigmaPilot Theory-Based Analysis Engines.

This package contains the 9 theory-based analysis engines for v2.0.0:

Tier 1 (Phase 2):
- dow_theory: Dow Theory trend analysis
- ichimoku: Ichimoku Kinko Hyo analysis
- vsa: Volume Spread Analysis
- chart_patterns: Classical chart pattern detection

Tier 2 (Phase 3):
- wyckoff: Wyckoff accumulation/distribution phases
- elliott_wave: Elliott Wave pattern analysis
- chan_theory: Chan Theory (缠论) fractals and strokes
- harmonic: Harmonic pattern detection
- market_profile: Market Profile / TPO analysis
"""

# Tier 1 Engines (Phase 2)
from .dow_theory import analyze_dow_theory, get_detailed_analysis as get_dow_details
from .ichimoku import analyze_ichimoku, get_detailed_ichimoku
from .vsa import analyze_vsa, get_detailed_vsa
from .chart_patterns import analyze_chart_patterns, get_all_patterns

# Tier 2 Engines (Phase 3)
from .wyckoff import analyze_wyckoff, get_detailed_wyckoff
from .elliott_wave import analyze_elliott_wave, get_detailed_elliott
from .chan_theory import analyze_chan_theory, get_detailed_chan
from .harmonic import analyze_harmonic, get_all_harmonic_patterns
from .market_profile import analyze_market_profile, get_detailed_profile

__all__ = [
    # Tier 1 - Dow Theory
    "analyze_dow_theory",
    "get_dow_details",
    # Tier 1 - Ichimoku
    "analyze_ichimoku",
    "get_detailed_ichimoku",
    # Tier 1 - VSA
    "analyze_vsa",
    "get_detailed_vsa",
    # Tier 1 - Chart Patterns
    "analyze_chart_patterns",
    "get_all_patterns",
    # Tier 2 - Wyckoff
    "analyze_wyckoff",
    "get_detailed_wyckoff",
    # Tier 2 - Elliott Wave
    "analyze_elliott_wave",
    "get_detailed_elliott",
    # Tier 2 - Chan Theory
    "analyze_chan_theory",
    "get_detailed_chan",
    # Tier 2 - Harmonic
    "analyze_harmonic",
    "get_all_harmonic_patterns",
    # Tier 2 - Market Profile
    "analyze_market_profile",
    "get_detailed_profile",
]
