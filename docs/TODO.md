# SigmaPilot MCP Server - Development TODO

This document tracks development tasks, improvements, and known issues.

## Completed

### v1.2.0 (Current)
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
