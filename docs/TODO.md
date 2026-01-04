# TradingView MCP Server - Development TODO

This document tracks development tasks, improvements, and known issues.

## High Priority

### Code Quality

- [ ] **Centralize `_tf_to_tv_resolution` function**
  - Location: Currently duplicated in `server.py:79-82` and `screener_provider.py:5-20`
  - Action: Move to `validators.py` and import from there
  - Impact: Reduces code duplication, single source of truth

- [ ] **Standardize error responses**
  - Current: Some tools return `{}`, others `{"error": "..."}`
  - Action: All tools should return `{"error": "message", ...context}` on failure
  - Impact: Consistent API behavior for consumers

- [ ] **Extract magic numbers to constants**
  - `batch_size = 200` (server.py:165)
  - `limit * 2` multipliers (server.py:96)
  - Pattern thresholds: `0.6`, `0.7` (server.py:610, 845)
  - Volume thresholds: `1000`, `5000`, `10000`
  - Action: Create constants module or add to validators.py

### Testing

- [x] Unit tests for `indicators.py`
- [x] Unit tests for `validators.py`
- [x] Unit tests for `coinlist.py`
- [x] Unit tests for server helper functions
- [ ] Integration tests with mock TradingView API
- [ ] End-to-end tests for MCP tools

## Medium Priority

### Documentation

- [ ] Add docstrings to all helper functions in `server.py`
- [ ] Document the Bollinger Band rating algorithm
- [ ] Add inline comments explaining complex logic
- [ ] Create API reference documentation

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
- [ ] Set up GitHub Actions for CI/CD
- [ ] Add code coverage reporting
- [ ] Create development container configuration

### Features (Future)

- [ ] Add more technical indicators (Fibonacci, Ichimoku)
- [ ] Support for custom indicator formulas
- [ ] Historical data analysis tools
- [ ] Alert/notification system
- [ ] WebSocket support for real-time data

## Completed

- [x] Add EMA9, EMA21, ATR indicators
- [x] Create comprehensive test suite
- [x] Reorganize documentation structure
- [x] Create TODO documentation

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
