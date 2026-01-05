#!/usr/bin/env python3
"""
Manual API Test Script for TradingView MCP Server.

This script provides a quick way to test MCP tools outside of the Claude Desktop
environment. It runs each tool via subprocess to simulate isolated execution.

Usage:
    uv run python test_api.py

Tests Included:
    1. top_gainers - Find top performing assets on KuCoin
    2. bollinger_scan - Detect Bollinger Band squeeze patterns
    3. volume_breakout_scanner - Find volume + price breakouts

Note:
    This is a manual testing utility, not part of the automated test suite.
    For automated tests, use: make test
"""

import sys
import subprocess
import json
from pathlib import Path


def test_tool_via_subprocess(tool_name: str, **kwargs):
    """
    Test an MCP tool by running it in an isolated subprocess.

    This approach ensures each tool is tested in a clean environment,
    similar to how Claude Desktop would invoke it.

    Args:
        tool_name: Name of the tool function to test (e.g., "top_gainers")
        **kwargs: Arguments to pass to the tool

    Returns:
        subprocess.CompletedProcess: Result of the subprocess execution

    Example:
        >>> test_tool_via_subprocess("top_gainers", exchange="KUCOIN", limit=5)
    """
    try:
        # Create a simple test script that imports and runs the tool
        script_content = f'''
import sys
import json
sys.path.insert(0, "src")

from tradingview_mcp.server import {tool_name}

try:
    result = {tool_name}(**{kwargs})
    print("SUCCESS:")
    print(json.dumps(result, indent=2, default=str))
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
        
        # Write to temp file
        test_file = Path("temp_test.py")
        test_file.write_text(script_content)
        
        # Run with uv
        result = subprocess.run(
            ["uv", "run", "python", "temp_test.py"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        print(f"\n{'='*50}")
        print(f"Testing: {tool_name}({kwargs})")
        print(f"{'='*50}")
        
        if result.returncode == 0:
            print("STDOUT:", result.stdout)
        else:
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
        
        # Cleanup
        test_file.unlink(missing_ok=True)
        
        return result
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None

def main():
    """
    Run manual API tests for core MCP tools.

    Executes a series of tests against KuCoin exchange data to verify
    that the main screening and analysis tools are working correctly.
    """
    print("TradingView MCP Server API Test")
    print("=" * 60)

    # Test 1: Top gainers - verifies basic market screening
    test_tool_via_subprocess("top_gainers",
                           exchange="KUCOIN",
                           timeframe="15m",
                           limit=5)

    # Test 2: Bollinger scan - verifies technical analysis with BBW filtering
    test_tool_via_subprocess("bollinger_scan",
                           exchange="KUCOIN",
                           timeframe="4h",
                           bbw_threshold=0.04,
                           limit=5)

    # Test 3: Volume breakout - verifies volume-based pattern detection
    test_tool_via_subprocess("volume_breakout_scanner",
                           exchange="KUCOIN",
                           timeframe="15m",
                           volume_multiplier=2.0,
                           limit=5)

    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
