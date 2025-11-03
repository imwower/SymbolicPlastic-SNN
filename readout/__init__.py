"""Readout policies (DEPRECATED shim).

Prefer `symbolicplastic_snn.readout.readout` for the new counting/latency readout.
This module remains to support legacy tests and examples.
"""

from .policies import PerceptronReadout, WinnerTakeAll

__all__ = ["PerceptronReadout", "WinnerTakeAll"]
