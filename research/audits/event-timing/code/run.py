#!/usr/bin/env python3
"""Run the registered read-only event-timing evidence audit."""

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from event_timing_audit import main

if __name__ == "__main__":
    main()
