# SigmaPilot MCP Server - Development TODO

This document tracks development tasks, improvements, and known issues.

## Completed

### v2.0.0 (Released)

#### Phase 1: Core Infrastructure (Completed)
- [x] Created core module architecture (`src/sigmapilot_mcp/core/`)
- [x] Implemented confidence formula: `Base × Q_pattern × W_time × V_conf × M_bonus`
- [x] Created No Signal Protocol (confidence < 60 = neutral)
- [x] Built structured output schemas (TypedDict: `AnalysisResult`, `Attribution`)
- [x] Created error handling hierarchy (`SigmaPilotError`, `DataError`, `APIError`, etc.)
- [x] Built timeframe utilities with weights and hierarchy
- [x] Created OHLCV data loader with NumPy array accessors
- [x] Added input sanitization functions
- [x] Created 159 unit tests for core modules
- [x] Added deterministic OHLCV test fixtures
- [x] Documented technical blockers in `TODO_BLOCKERS.md`

#### Phase 2: Tier 1 Engines (Completed)
- [x] Dow Theory Trend engine (`dow_theory.py`)
- [x] Ichimoku Insight engine (`ichimoku.py`)
- [x] VSA Analyzer engine (`vsa.py`)
- [x] Chart Pattern Finder engine (`chart_patterns.py`)
- [x] Created integration tests for all Tier 1 engines (200 total tests)

#### Phase 3: Tier 2 Engines (Completed)
- [x] Wyckoff Phase Detector (`wyckoff.py`)
- [x] Elliott Wave Analyzer (`elliott_wave.py`)
- [x] Chan Theory Analyzer (`chan_theory.py`)
- [x] Harmonic Pattern Detector (`harmonic.py`)
- [x] Market Profile Analyzer (`market_profile.py`)
- [x] Created integration tests for all Tier 2 engines (240 total tests)

#### Phase 4: Integration & Documentation (Completed)
- [x] Register all 9 engines as MCP tools in server.py
- [x] Updated tool count from 10 to 19
- [x] Added helper function `_fetch_ohlcv_for_symbol()` for data conversion
- [x] Removed unused legacy code (screener_provider.py, unused functions)
- [x] Added comprehensive edge case tests (177 new tests)
- [x] Created EXAMPLES.md with complete usage guide
- [x] Updated all documentation for v2.0.0
- [x] Total test coverage: 417 tests passing

#### Future Enhancements (Backlog)
- [ ] Multi-engine consensus calculations
- [ ] End-to-end integration tests with live data

### v1.3.0
- [x] Added TradingView API rate limiting protection
- [x] Implemented exponential backoff with jitter
- [x] Configurable batch sizes via environment variables
- [x] User-friendly error messages for rate limit scenarios

### v1.2.0
- [x] Added Ichimoku Cloud indicator (Tenkan-sen, Kijun-sen, Senkou Span A/B)
- [x] Added VWAP (Volume Weighted Average Price)
- [x] Added Pivot Points (Classic, Fibonacci, Camarilla) with R1-R3, S1-S3 levels
- [x] Added TradingView Recommendations (Overall, MA, Oscillators)
- [x] Added Williams %R oscillator
- [x] Added CCI (Commodity Channel Index)
- [x] Added Awesome Oscillator (AO)
- [x] Added Ultimate Oscillator (UO)
- [x] Added Momentum indicator
- [x] Added Hull MA and VWMA moving averages
- [x] Added Parabolic SAR indicator
- [x] Added additional SMA periods (5, 10, 30, 50, 100, 200)
- [x] Added additional EMA periods (5, 10, 30, 100)
- [x] Created `pivot_points_scanner` MCP tool
- [x] Created `tradingview_recommendation` MCP tool
- [x] Comprehensive tests for all new indicators (45 new tests, 223 total)
- [x] Renamed project from tradingview-mcp to sigmapilot-mcp
- [x] Unified server architecture (merged main.py into server.py)
- [x] Consolidated tools: candle_pattern_scanner, volume_scanner, volume_analysis
- [x] Added health check endpoints for HTTP mode

### v1.1.0
- [x] Updated documentation for multi-platform MCP support (Claude.ai Connectors, ChatGPT, etc.)
- [x] Added Claude.ai Connectors documentation links
- [x] Repository ownership transfer (fzheng)

### v1.0.0
- [x] Remote deployment support (Railway + Auth0)
- [x] Auth0 JWT verification using JWKS endpoint
- [x] `PyJWT[crypto]` dependency for RS256 algorithm
- [x] Health check endpoints (`/health`, `/mcp/health`)
- [x] Centralized `tf_to_tv_resolution` function in validators.py
- [x] Extracted magic numbers to constants
- [x] Comprehensive test suite (178+ tests)
- [x] Documentation overhaul (remote-first approach)
- [x] Add EMA9, EMA21, ATR indicators
- [x] Create `create_mcp_server()` factory function
- [x] Unit tests for all modules (auth, indicators, validators, coinlist, server)

## High Priority

### Testing
- [ ] Integration tests with mock market API
- [ ] End-to-end tests for MCP tools

### Infrastructure
- [ ] Set up GitHub Actions for CI/CD
- [ ] Add code coverage reporting to CI
- [ ] Automated Railway deployment on push

## Medium Priority

### Documentation
- [ ] Add API reference documentation
- [ ] Add troubleshooting guide for Auth0

### Code Organization
- [ ] Break down large functions (e.g., `coin_analysis` is 137 lines)
- [ ] Add type hints to remaining untyped functions

### Performance
- [ ] Add `functools.lru_cache` to `load_symbols()` for disk I/O caching
- [ ] Profile API call patterns for optimization opportunities
- [ ] Consider batch size optimization based on exchange

## Low Priority

### Developer Experience
- [ ] Add pre-commit hooks for Black/Ruff
- [ ] Create development container configuration
- [ ] Add Docker deployment option

### Features (Future)
- [x] Add more technical indicators (Fibonacci, Ichimoku) - Completed in v1.2.0
- [ ] Support for custom indicator formulas
- [ ] Historical data analysis tools
- [ ] Alert/notification system
- [ ] WebSocket support for real-time data
- [ ] Multiple auth provider support (Google, GitHub)
- [ ] Additional data sources beyond TradingView

## Notes

### Coding Standards
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Add docstrings to all public functions
- Keep functions under 50 lines where possible

### Testing Guidelines
- Aim for 80%+ code coverage
- Test both success and error paths
- Mock external API calls in unit tests
- Use fixtures for common test data

### Remote Deployment Notes
- Auth0 is the recommended OAuth provider
- Railway free tier: 500 hours/month
- Always set `RESOURCE_SERVER_URL` to match your public URL
- Never commit `.env` files
