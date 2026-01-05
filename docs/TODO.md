# TradingView MCP Server - Development TODO

This document tracks development tasks, improvements, and known issues.

## Completed

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
- [ ] Integration tests with mock TradingView API
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
- [ ] Consider separating MCP tools into a dedicated module
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
- [ ] Add more technical indicators (Fibonacci, Ichimoku)
- [ ] Support for custom indicator formulas
- [ ] Historical data analysis tools
- [ ] Alert/notification system
- [ ] WebSocket support for real-time data
- [ ] Multiple auth provider support (Google, GitHub)

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
