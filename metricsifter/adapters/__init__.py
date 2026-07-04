"""Adapters that convert third-party monitoring payloads into MetricSifter input.

Adapters are pure data transforms: they never perform I/O (no HTTP clients, no
network). You fetch the payload however you like and hand the parsed structure
to the adapter.
"""

from metricsifter.adapters import prometheus

__all__ = ["prometheus"]
