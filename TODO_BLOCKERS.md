# Technical Blockers & Deferred Improvements

This document tracks technical blockers identified during v2.0.0 implementation.
Each item includes the desired approach vs. current approach, and resolution path.

---

## BLOCKER-001: Chan Theory Library Dependency

**Status**: Deferred (Fallback Implemented)

**Desired Approach**:
- Use established Chan Theory library (e.g., `czsc` or `chan.py`)
- Full Chanlun implementation with:
  - Proper fractal (分型) detection with merge rules
  - Stroke (笔) construction with strict inclusion processing
  - Segment (线段) identification
  - Hub/Zhongshu (中枢) detection
  - Buy/Sell point signals (1st, 2nd, 3rd types)
  - Divergence detection (MACD-based)

**Current Approach (Fallback)**:
- Minimal custom implementation:
  - Basic fractal detection (top/bottom)
  - Simplified stroke construction
  - Basic segment trend identification
- No hub detection in initial version
- No buy/sell point classification
- Divergence detection deferred

**Resolution Path**:
1. Evaluate `czsc` library compatibility with our data format
2. If compatible, add as optional dependency
3. Implement adapter layer in `data_loader.py`
4. Gradually replace minimal implementation with full library

**References**:
- https://github.com/waditu/czsc
- Chan Theory original texts

---

## BLOCKER-002: Harmonic Pattern Detection

**Status**: Deferred (Partial Implementation)

**Desired Approach**:
- Full harmonic pattern suite:
  - Gartley (0.618 XA retracement)
  - Bat (0.886 XA retracement)
  - Butterfly (1.27 XA extension)
  - Crab (1.618 XA extension)
  - Shark, Cypher, Deep Crab variants
- Precise Fibonacci ratio validation with configurable tolerance
- Potential Reversal Zone (PRZ) calculation
- Pattern completion percentage tracking

**Current Approach (Fallback)**:
- Start with Gartley pattern only
- Strict Fibonacci tolerances (default 3%)
- Basic PRZ calculation
- Add other patterns incrementally after validation

**Resolution Path**:
1. Implement Gartley with comprehensive test coverage
2. Add Bat pattern (shares XAB structure)
3. Add Butterfly and Crab (extension patterns)
4. Consider external library if maintenance burden grows

**References**:
- Scott Carney's "Harmonic Trading" methodology
- Fibonacci ratio tables for each pattern

---

## BLOCKER-003: Market Profile / TPO Data Requirements

**Status**: Deferred (Approximation Implemented)

**Desired Approach**:
- True Time Price Opportunity (TPO) calculation:
  - Tick-level or minute-level data
  - Time-at-price tracking (30-min brackets typical)
  - Letter-based TPO assignment
  - Initial Balance (IB) calculation
  - Value Area computation from time distribution
- Volume Profile with actual volume-at-price data

**Current Approach (Fallback)**:
- Volume-at-price approximation using OHLCV:
  - Distribute candle volume across price range
  - Weight by proximity to close (or use TWAP assumption)
- POC, VAH, VAL from volume distribution
- Profile shape analysis (P, b, D shapes)
- Document limitations clearly in output

**Resolution Path**:
1. Identify data providers with volume-at-price data
2. Add optional `@mcp:quant-data-provider` integration
3. When tick data available, implement true TPO
4. Keep approximation as fallback for OHLCV-only sources

**Data Requirements for True Implementation**:
- Tick data with timestamp and volume
- OR aggregated volume-at-price from exchange
- OR minute-bar data for time-based approximation

---

## BLOCKER-004: Volume Data Quality for VSA/Wyckoff

**Status**: Mitigated (Confidence Penalty Applied)

**Desired Approach**:
- Reliable volume data from primary exchange
- Volume normalized across venues
- Distinction between real volume and wash trading
- Tick-level volume for spread analysis

**Current Approach (Mitigation)**:
- Use TradingView-reported volume (aggregated)
- Apply `V_conf = 0.85` (neutral) when volume quality uncertain
- Apply `V_conf = 0.70` when volume data missing/zero
- Document venue-specific concerns in analysis output

**Resolution Path**:
1. For crypto: prefer CEX volume over DEX
2. Add venue quality indicators to data_loader
3. Consider volume ratio vs. SMA as quality proxy
4. For stocks: volume generally reliable, use `V_conf = 1.0`

---

## BLOCKER-005: External MCP Servers Not Available

**Status**: Deferred (Fixtures Used)

**Desired Approach**:
- `@mcp:quant-data-provider` for live OHLCV data
- `@mcp:ta-reference` for cross-validation of:
  - Ichimoku calculations
  - ATR values
  - Swing point detection

**Current Approach (Fallback)**:
- Deterministic fixtures in `tests/fixtures/deterministic_ohlcv/`
- Pre-computed expected values for validation
- Integration tests marked with `@pytest.mark.api` for future

**Resolution Path**:
1. Define MCP server interface contracts
2. Create mock MCP servers for testing
3. When real servers available, enable integration tests
4. Keep fixture-based tests for CI/CD reliability

---

## BLOCKER-006: Elliott Wave Subjectivity

**Status**: Acknowledged (Conservative Default)

**Desired Approach**:
- Multiple valid wave counts with probability ranking
- Degree labeling (Grand Supercycle to Subminuette)
- Rule validation (Wave 2 never retraces 100% of Wave 1, etc.)
- Guideline scoring (alternation, Fibonacci relationships)
- Real-time wave tracking with invalidation

**Current Approach (Conservative)**:
- Strict rule enforcement (reject violations immediately)
- Return `status: neutral` for ambiguous counts
- Maximum 2 interpretations returned
- Focus on impulse pattern detection (5-wave)
- Corrective patterns simplified (ABC only)
- Default to neutral when confidence < 60

**Resolution Path**:
1. Start with clear impulse structures
2. Add corrective pattern variants (zigzag, flat, triangle)
3. Implement degree analysis for multi-timeframe
4. Consider ML-based wave counting (research phase)

---

## BLOCKER-007: Real-Time Data Latency

**Status**: Acknowledged (Best-Effort)

**Desired Approach**:
- WebSocket streaming for live updates
- Sub-second latency for pattern detection
- Real-time invalidation alerts

**Current Approach**:
- Poll-based via TradingView API
- Analysis on request (not streaming)
- Invalidation levels provided for manual monitoring

**Resolution Path**:
1. Current approach sufficient for analysis use case
2. If real-time needed, add WebSocket data source
3. Consider separate streaming service for alerts

---

## BLOCKER-008: Pattern Detection Accuracy Validation

**Status**: Deferred (Manual Validation Required)

**Desired Approach**:
- Backtested pattern recognition accuracy
- Statistical validation of confidence scores
- Comparison against commercial tools

**Current Approach**:
- Unit tests with synthetic patterns
- Manual validation on historical data
- Conservative confidence to prefer false negatives

**Resolution Path**:
1. Build backtesting framework
2. Collect labeled pattern dataset
3. Measure precision/recall per pattern type
4. Adjust confidence formula coefficients

---

## Summary Table

| ID | Blocker | Severity | Status | ETA for Resolution |
|----|---------|----------|--------|-------------------|
| 001 | Chan Theory Library | Medium | Fallback | Post-v2.0.0 |
| 002 | Harmonic Patterns | Medium | Partial | Incremental |
| 003 | Market Profile Data | Medium | Approx | When data available |
| 004 | Volume Quality | Low | Mitigated | Ongoing |
| 005 | External MCPs | Low | Fixtures | When servers ready |
| 006 | Elliott Subjectivity | Low | Conservative | Research |
| 007 | Real-Time Latency | Low | Best-Effort | If needed |
| 008 | Pattern Validation | Medium | Manual | Post-v2.0.0 |

---

## Notes for Future Contributors

1. **Do not remove fallback implementations** - they provide baseline functionality
2. **Confidence penalties are intentional** - prefer false negatives
3. **Test fixtures are source of truth** - update when improving accuracy
4. **Document limitations in tool output** - transparency for LLM consumers
