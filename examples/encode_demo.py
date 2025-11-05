from __future__ import annotations

"""Demonstrate Poisson and LatencyRank encoders (no plotting).

Writes `poisson_spikes.npy` and `latency_spikes.npy` to the current directory.
"""

import numpy as np

from symbolicplastic_snn.encode import PoissonRateEncoder, LatencyRankEncoder
from symbolicplastic_snn.core.prng import SeedSpace, FloatRng


def main() -> None:
    N, T = 128, 100

    # Build deterministic inputs
    rng = FloatRng(1234)
    intensities = rng.random(N, dtype=np.float32)

    # Poisson rate spikes (constant rate per neuron derived from intensities)
    rates = np.clip(intensities * 0.3, 0.0, 1.0)
    pois = PoissonRateEncoder(rate=rates, T=T)
    spikes_poisson = pois.encode(SeedSpace(0x2024), trial=0)
    np.save("poisson_spikes.npy", spikes_poisson)

    # Latency rank spikes in a window
    lat = LatencyRankEncoder(window=16)
    spikes_latency = lat.encode(intensities, SeedSpace(0x2024), trial=0)
    np.save("latency_spikes.npy", spikes_latency)

    print("Saved poisson_spikes.npy and latency_spikes.npy")


if __name__ == "__main__":
    main()

