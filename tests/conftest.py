"""Pytest configuration — ensures src/ is on sys.path for all tests."""

import os
import sys

# Add src/ to path so tests can import ggTrader packages
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
