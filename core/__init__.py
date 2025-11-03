"""Core modules (DEPRECATED shim).

This top-level package remains for compatibility only. Please migrate imports to
`symbolicplastic_snn.core.*`. APIs are thin wrappers around the new package.
"""

from .simulator import (
    DelayBuffer,
    EventDrivenLIF,
    NeuronState,
    StepResult,
    SynapseCSR,
    SynapseTopology,
)

__all__ = [
    "DelayBuffer",
    "EventDrivenLIF",
    "NeuronState",
    "StepResult",
    "SynapseCSR",
    "SynapseTopology",
]
