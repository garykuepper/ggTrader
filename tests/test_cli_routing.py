"""Integration tests for the ggt.py CLI routing."""

import unittest
import subprocess
import sys
from pathlib import Path

class TestCliRouting(unittest.TestCase):
    def setUp(self):
        self.ggt_path = Path(__file__).parent.parent / "ggt.py"
        
    def test_help_routing(self):
        """Ensure the CLI and all subparsers can be invoked without import errors."""
        result = subprocess.run(
            [sys.executable, str(self.ggt_path), "--help"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("research", result.stdout)
        self.assertIn("backtest", result.stdout)
        self.assertIn("production", result.stdout)
        self.assertIn("trade", result.stdout)

if __name__ == "__main__":
    unittest.main()
