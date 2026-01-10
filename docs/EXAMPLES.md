# SigmaPilot MCP Server v2.0.0 - Usage Examples

This guide provides comprehensive usage examples for all 19 tools available in SigmaPilot MCP Server.

## Table of Contents

1. [Theory-Based Analysis Engines (9 tools)](#theory-based-analysis-engines)
2. [Market Screening Tools (10 tools)](#market-screening-tools)
3. [Tiered Analysis Strategy](#tiered-analysis-strategy)
4. [Output Schema Reference](#output-schema-reference)
5. [Best Practices](#best-practices)

---

## Theory-Based Analysis Engines

All theory-based engines return a standardized `AnalysisResult` with:
- `status`: "bullish", "bearish", or "neutral"
- `confidence`: 0-100 score
- `attribution`: Theory name and rules triggered
- `llm_summary`: Human-readable summary
- `invalidation`: Level or condition that invalidates the signal
- `is_error`: Boolean error flag

### 1. Dow Theory Trend (`dow_theory_trend`)

Analyzes trend using higher highs/higher lows (bullish) or lower highs/lower lows (bearish).

**Example Request:**
```
Analyze BTCUSDT using Dow Theory on the daily timeframe
```

**Tool Call:**
```json
{
  "tool": "dow_theory_trend",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D",
    "mode": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 78,
  "attribution": {
    "theory": "Dow Theory",
    "rules_triggered": ["Higher High confirmed", "Higher Low confirmed"]
  },
  "llm_summary": "Dow Theory on 1D: Bullish trend confirmed with 3 HH/HL sequences.",
  "invalidation": "Break below 42150.00 (last higher low) invalidates bullish structure",
  "is_error": false
}
```

---

### 2. Ichimoku Insight (`ichimoku_insight`)

Comprehensive analysis using Ichimoku Kinko Hyo (cloud, TK cross, Chikou span).

**Example Request:**
```
What does Ichimoku say about ETHUSDT on the 4-hour chart?
```

**Tool Call:**
```json
{
  "tool": "ichimoku_insight",
  "arguments": {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "timeframe": "4H",
    "mode": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 72,
  "attribution": {
    "theory": "Ichimoku Kinko Hyo",
    "rules_triggered": ["Price above cloud", "Bullish TK cross", "Chikou confirming"]
  },
  "llm_summary": "Ichimoku on 4H: Price above cloud with bullish TK cross. Future cloud bullish.",
  "invalidation": "Break below cloud (2890.50) weakens bullish bias",
  "is_error": false
}
```

---

### 3. VSA Analyzer (`vsa_analyzer`)

Volume Spread Analysis for detecting smart money activity.

**Example Request:**
```
Check SOLUSDT for VSA signals on the daily chart
```

**Tool Call:**
```json
{
  "tool": "vsa_analyzer",
  "arguments": {
    "symbol": "SOLUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D",
    "mode": "conservative"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 68,
  "attribution": {
    "theory": "Volume Spread Analysis",
    "rules_triggered": ["Stopping Volume detected", "No Supply bar confirmed"]
  },
  "llm_summary": "VSA on 1D: Stopping Volume with No Supply confirmation. Background bias bullish.",
  "invalidation": "Lower low with high volume invalidates bullish accumulation",
  "is_error": false
}
```

---

### 4. Chart Pattern Finder (`chart_pattern_finder`)

Detects classical chart patterns (Head & Shoulders, triangles, double top/bottom).

**Example Request:**
```
Find chart patterns on BTCUSDT 4H
```

**Tool Call:**
```json
{
  "tool": "chart_pattern_finder",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "4H",
    "mode": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bearish",
  "confidence": 71,
  "attribution": {
    "theory": "Chart Patterns",
    "rules_triggered": ["Double Top identified", "Neckline tested"]
  },
  "llm_summary": "Chart patterns on 4H: Double Top detected. Target: 41500.00.",
  "invalidation": "Invalidation at 44200.00 for double top",
  "is_error": false
}
```

---

### 5. Wyckoff Phase Detector (`wyckoff_phase_detector`)

Identifies Wyckoff market phases (accumulation, distribution, markup, markdown).

**Example Request:**
```
What Wyckoff phase is BTCUSDT in?
```

**Tool Call:**
```json
{
  "tool": "wyckoff_phase_detector",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D",
    "mode": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 65,
  "attribution": {
    "theory": "Wyckoff Method",
    "rules_triggered": ["Accumulation Phase C", "Spring detected"]
  },
  "llm_summary": "Wyckoff Accumulation Phase C detected on 1D. Recent events: Spring, Sign Of Strength.",
  "invalidation": "Break below 39500.00 (range support) invalidates accumulation",
  "is_error": false
}
```

---

### 6. Elliott Wave Analyzer (`elliott_wave_analyzer`)

Analyzes Elliott Wave patterns (impulse and corrective waves).

**Example Request:**
```
What's the Elliott Wave count for ETHUSDT?
```

**Tool Call:**
```json
{
  "tool": "elliott_wave_analyzer",
  "arguments": {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D",
    "mode": "conservative"
  }
}
```

**Sample Response:**
```json
{
  "status": "neutral",
  "confidence": 58,
  "attribution": {
    "theory": "Elliott Wave Theory",
    "rules_triggered": ["Wave 4 in progress"]
  },
  "llm_summary": "Elliott Wave: Bullish Impulse pattern on 1D. Currently in Wave 4. 2 interpretations possible.",
  "invalidation": "Break below Wave 4 at 2650.00 invalidates bullish impulse",
  "is_error": false
}
```

---

### 7. Chan Theory Analyzer (`chan_theory_analyzer`)

Chan Theory (Chanlun) analysis using fractals, strokes, and segments.

**Example Request:**
```
Analyze BTCUSDT using Chan Theory
```

**Tool Call:**
```json
{
  "tool": "chan_theory_analyzer",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "4H",
    "strictness": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 62,
  "attribution": {
    "theory": "Chan Theory (Chanlun)",
    "rules_triggered": ["Upward stroke confirmed", "Hub breakout"]
  },
  "llm_summary": "Chan Theory on 4H: Current trend up. 8 strokes, 2 segments. Hub zone: 41500-43200.",
  "invalidation": "Break below 41500.00 (last stroke low) invalidates bullish structure",
  "is_error": false
}
```

---

### 8. Harmonic Pattern Detector (`harmonic_pattern_detector`)

Detects Fibonacci-based harmonic patterns (Gartley, Bat, Butterfly, Crab).

**Example Request:**
```
Are there any harmonic patterns on BTCUSDT?
```

**Tool Call:**
```json
{
  "tool": "harmonic_pattern_detector",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "4H",
    "mode": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 67,
  "attribution": {
    "theory": "Harmonic Patterns",
    "rules_triggered": ["Gartley pattern complete", "PRZ reached"]
  },
  "llm_summary": "Harmonic Pattern: Bullish Gartley (complete) on 4H. PRZ: 41200-41800.",
  "invalidation": "Break below PRZ (41200.00) invalidates bullish pattern",
  "is_error": false
}
```

---

### 9. Market Profile Analyzer (`market_profile_analyzer`)

Market Profile analysis (POC, Value Area, profile shape).

**Example Request:**
```
What does Market Profile show for BTCUSDT?
```

**Tool Call:**
```json
{
  "tool": "market_profile_analyzer",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D",
    "mode": "balanced"
  }
}
```

**Sample Response:**
```json
{
  "status": "bullish",
  "confidence": 61,
  "attribution": {
    "theory": "Market Profile",
    "rules_triggered": ["Price above value area", "Normal profile shape"]
  },
  "llm_summary": "Market Profile on 1D: Normal profile. POC: 42500, VAH: 43200, VAL: 41800. Bullish imbalance.",
  "invalidation": "Break below POC (42500.00) invalidates bullish bias",
  "is_error": false
}
```

---

## Market Screening Tools

### Top Gainers (`top_gainers`)
```json
{
  "tool": "top_gainers",
  "arguments": {
    "exchange": "BINANCE",
    "timeframe": "4h",
    "limit": 10
  }
}
```

### Top Losers (`top_losers`)
```json
{
  "tool": "top_losers",
  "arguments": {
    "exchange": "KUCOIN",
    "timeframe": "1D",
    "limit": 10
  }
}
```

### Bollinger Band Squeeze (`bollinger_scan`)
```json
{
  "tool": "bollinger_scan",
  "arguments": {
    "exchange": "KUCOIN",
    "timeframe": "4h",
    "bbw_threshold": 0.04,
    "limit": 25
  }
}
```

### Rating Filter (`rating_filter`)
```json
{
  "tool": "rating_filter",
  "arguments": {
    "exchange": "BINANCE",
    "timeframe": "1D",
    "rating": 2,
    "limit": 20
  }
}
```

### Coin Analysis (`coin_analysis`)
```json
{
  "tool": "coin_analysis",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D"
  }
}
```

### Candle Pattern Scanner (`candle_pattern_scanner`)
```json
{
  "tool": "candle_pattern_scanner",
  "arguments": {
    "exchange": "BINANCE",
    "timeframe": "1H",
    "mode": "consecutive",
    "pattern_type": "bullish",
    "limit": 20
  }
}
```

### Volume Scanner (`volume_scanner`)
```json
{
  "tool": "volume_scanner",
  "arguments": {
    "exchange": "BINANCE",
    "timeframe": "4H",
    "mode": "breakout",
    "limit": 25
  }
}
```

### Volume Analysis (`volume_analysis`)
```json
{
  "tool": "volume_analysis",
  "arguments": {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "timeframe": "1D"
  }
}
```

### Pivot Points Scanner (`pivot_points_scanner`)
```json
{
  "tool": "pivot_points_scanner",
  "arguments": {
    "exchange": "BINANCE",
    "timeframe": "1D",
    "pivot_type": "fibonacci",
    "limit": 20
  }
}
```

### TradingView Recommendation (`tradingview_recommendation`)
```json
{
  "tool": "tradingview_recommendation",
  "arguments": {
    "exchange": "BINANCE",
    "timeframe": "1D",
    "signal_filter": "STRONG_BUY",
    "limit": 20
  }
}
```

---

## Tiered Analysis Strategy

For comprehensive analysis, use multiple engines in tiers:

### Tier 1: Quick Assessment (2 tools)
Start with these for rapid trend identification:

1. **`dow_theory_trend`** - Primary trend direction
2. **`ichimoku_insight`** - Holistic trend/momentum snapshot

### Tier 2: Confirmation (2 tools)
Add these for signal confirmation:

3. **`vsa_analyzer`** - Volume confirmation
4. **`chart_pattern_finder`** - Pattern-based targets

### Tier 3: Advanced Analysis (2 tools)
For deeper market structure insights:

5. **`wyckoff_phase_detector`** - Market phase
6. **`market_profile_analyzer`** - Key price levels

### Example: Complete Multi-Engine Analysis

```
Request: "Give me a comprehensive analysis of BTCUSDT"

Workflow:
1. dow_theory_trend(symbol="BTCUSDT", timeframe="1D") -> Primary trend
2. ichimoku_insight(symbol="BTCUSDT", timeframe="1D") -> Confirmation
3. vsa_analyzer(symbol="BTCUSDT", timeframe="1D") -> Volume context
4. chart_pattern_finder(symbol="BTCUSDT", timeframe="4H") -> Patterns
```

**Interpreting Multi-Engine Results:**

| Agreement | Interpretation | Action |
|-----------|----------------|--------|
| All bullish | High conviction | Consider long position |
| All bearish | High conviction | Consider short/exit |
| Mixed signals | Low conviction | Wait for clarity |
| All neutral | No clear signal | Stay sidelined |

---

## Output Schema Reference

### AnalysisResult Schema

```typescript
interface AnalysisResult {
  status: "bullish" | "bearish" | "neutral" | "error";
  confidence: number;  // 0-100
  attribution: {
    theory: string;
    rules_triggered?: string[];
  };
  llm_summary: string;
  invalidation: string;
  is_error: boolean;
}
```

### Confidence Interpretation

| Range | Meaning | Recommended Action |
|-------|---------|-------------------|
| 80-100 | High conviction | Strong signal, standard position |
| 70-79 | Moderate | Valid signal, normal sizing |
| 60-69 | Developing | Possible signal, reduced size |
| <60 | No signal | Status = "neutral", avoid trading |

### Analysis Modes

Each engine supports three modes:

| Mode | Behavior | Best For |
|------|----------|----------|
| `conservative` | Strict rules, fewer signals | Risk-averse traders |
| `balanced` | Standard parameters | Most users (default) |
| `aggressive` | Relaxed rules, more signals | Active traders |

---

## Best Practices

### 1. Timeframe Selection

| Timeframe | Best For | Signal Reliability |
|-----------|----------|-------------------|
| 1W, 1D | Position trading | Highest |
| 4H | Swing trading | High |
| 1H | Day trading | Medium |
| 15m | Scalping | Lower |

### 2. Multi-Engine Confirmation

Always use at least 2 engines for confirmation:

```
Good:
- Dow Theory bullish + Ichimoku bullish = Strong signal
- Wyckoff accumulation + VSA stopping volume = Confirmed accumulation

Weak:
- Only one engine showing signal
- Engines showing conflicting signals
```

### 3. Invalidation Levels

Every signal comes with an invalidation level. Use these for:
- Stop-loss placement
- Risk management
- Signal validation

### 4. Handling Neutral Status

When confidence < 60, engines return `"status": "neutral"`:

- **Do not force a trade** when multiple engines are neutral
- Wait for clearer signals to develop
- Use market screening tools to find better opportunities

---

## Limitations

### Data Limitations
- Market Profile uses volume-at-price approximation (true TPO requires tick data)
- Elliott Wave analysis is inherently subjective
- Chan Theory implementation is simplified

### Volume Quality
- VSA and Wyckoff engines require meaningful volume data
- Crypto exchange volume may vary in reliability
- Volume confidence is reduced when data quality is uncertain

### Rate Limits
- TradingView API has rate limits
- Scanner tools process in batches with delays
- For bulk analysis, expect some processing time
