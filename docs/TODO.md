# TradingView MCP Server v1.0.0 - Development TODO

This document tracks development tasks, improvements, and known issues.

## Completed in v1.0.0

- [x] Remote deployment support (Railway + Auth0)
- [x] Auth0 JWT verification using JWKS endpoint
- [x] Centralized `tf_to_tv_resolution` function in validators.py
- [x] Extracted magic numbers to constants
- [x] Comprehensive test suite (183+ tests)
- [x] Documentation overhaul (remote-first approach)
- [x] Add EMA9, EMA21, ATR indicators
- [x] Create comprehensive test suite
- [x] Reorganize documentation structure
- [x] Create `create_mcp_server()` factory function
- [x] Unit tests for `auth.py` JWT verifier

## High Priority

### Testing

- [x] Unit tests for `indicators.py`
- [x] Unit tests for `validators.py`
- [x] Unit tests for `coinlist.py`
- [x] Unit tests for server helper functions
- [x] Unit tests for `auth.py` JWT verifier
- [x] Tests for `main.py` remote server entry point
- [ ] Integration tests with mock TradingView API
- [ ] End-to-end tests for MCP tools

### Infrastructure

- [ ] Set up GitHub Actions for CI/CD
- [ ] Add code coverage reporting to CI
- [ ] Automated Railway deployment on push

## Medium Priority

### Documentation

- [x] Document remote deployment process
- [x] Update installation guide for v1.0.0
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

### Error Handling Pattern

```python
# Recommended error response format
def some_tool(...) -> dict:
    try:
        # ... processing ...
        return {"data": result, "success": True}
    except SpecificException as e:
        return {
            "error": f"Descriptive message: {str(e)}",
            "error_type": "SpecificException",
            "context": {...}
        }
```

### Remote Deployment Notes

- Auth0 is the recommended OAuth provider
- Railway free tier: 500 hours/month
- Always set `RESOURCE_SERVER_URL` to match your public URL
- Never commit `.env` files
