"""Topology utilities for SymbolicPlastic-SNN."""

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
