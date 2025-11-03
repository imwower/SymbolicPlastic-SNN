"""Topology utilities (DEPRECATED shim).

Use `symbolicplastic_snn.conn` and `symbolicplastic_snn.conn.generator` for new APIs.
This namespace is kept temporarily for compatibility.
"""

from .generator import (
    SmallWorldSampler,
    TopologyParams,
    build_small_world_topology,
    distance_metric,
)

__all__ = [
    "SmallWorldSampler",
    "TopologyParams",
    "build_small_world_topology",
    "distance_metric",
]
