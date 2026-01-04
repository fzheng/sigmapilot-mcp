# TradingView MCP Server - Makefile
# Common development commands

.PHONY: test test-verbose test-cov lint format typecheck clean install dev help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Install with dev dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make test-verbose - Run tests with verbose output"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make lint        - Run linter (ruff)"
	@echo "  make format      - Format code (ruff)"
	@echo "  make typecheck   - Run type checker (mypy)"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make run         - Run the MCP server"

# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --dev

# Run all tests
test:
	uv run pytest tests/ -v

# Run tests with verbose output
test-verbose:
	uv run pytest tests/ -v --tb=long

# Run tests with coverage
test-cov:
	uv run pytest tests/ -v --cov=src/tradingview_mcp --cov-report=term-missing

# Run linter
lint:
	uv run ruff check src/ tests/

# Format code
format:
	uv run ruff format src/ tests/

# Run type checker
typecheck:
	uv run mypy src/

# Clean build artifacts
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	rm -rf .coverage
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run the MCP server
run:
	uv run python src/tradingview_mcp/server.py
