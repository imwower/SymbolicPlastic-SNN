"""Monitoring utilities (DEPRECATED shim).

New work should use `symbolicplastic_snn.*` packages. This namespace is kept
temporarily for compatibility.
"""

from .stability import StabilityMetrics, StabilityMonitor

__all__ = ["StabilityMetrics", "StabilityMonitor"]
