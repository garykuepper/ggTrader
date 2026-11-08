#!/usr/bin/env python3
"""Reproduce cached insider acceptance-source counts offline."""

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from event_acceptance_diagnostic import main

if __name__ == "__main__":
    main()
