"""Run `ggt lab --strategy sma_cross` with a scratch dir added to the strategies package path."""
import sys
sys.path.insert(0, "/home/flynn/ggTrader-vbt12/src")
import ggTrader.lab.strategies as pkg
pkg.__path__.append(sys.argv[1])
from ggTrader.lab.strategies import registry
registry.discover()
print("registered:", sorted(registry.REGISTRY))
sys.argv = ["ggt.py", "lab", *sys.argv[2:]]
from ggTrader.cli.main import main
main()
