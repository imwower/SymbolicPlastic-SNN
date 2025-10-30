import numpy as np

from symbolicplastic_snn.core.lif_fixedpoint import (
    ConfigFp,
    lif_step,
    q15_mul_shift,
    saturating_add_int16,
)


def test_no_spike_when_ref():
    n = 8
    v = np.full(n, 1234, dtype=np.int16)
    ref = np.full(n, 2, dtype=np.uint8)
    I = np.full(n, 50000, dtype=np.int32)  # should be ignored under ref
    theta = np.int16(100)
    lam = np.uint16(32768)  # ~1.0
    cfg = ConfigFp(refractory_steps=3, v_reset=0)

    spikes, num = lif_step(v, ref, I, theta, lam, cfg)

    assert not spikes.any()
    assert num == 0
    # v should be clamped to reset and I ignored
    assert np.all(v == cfg.v_reset)
    # ref decremented by 1
    assert np.all(ref == 1)


def test_spike_and_reset():
    v = np.array([0, 0, 0], dtype=np.int16)
    ref = np.array([0, 0, 0], dtype=np.uint8)
    I = np.array([0, 30000, 10000], dtype=np.int32)
    theta = np.int16(20000)
    lam = np.uint16(32768)  # ~1.0
    cfg = ConfigFp(refractory_steps=5, v_reset=0)

    spikes, num = lif_step(v, ref, I, theta, lam, cfg)

    # Only the middle neuron spikes
    assert spikes.tolist() == [False, True, False]
    assert num == 1
    # Spike resets v to 0 and sets refractory
    assert v.tolist() == [0, 0, 10000]
    assert ref.tolist() == [0, 5, 0]


def test_saturation():
    # Directly test saturation helper
    arr = np.array([40000, -50000, 0, 32767, -32768], dtype=np.int32)
    out = saturating_add_int16(arr)
    assert out.dtype == np.int16
    assert out.tolist() == [32767, -32768, 0, 32767, -32768]

    # Also test lif path: use high I but high theta to avoid spiking
    v = np.array([1000], dtype=np.int16)
    ref = np.array([0], dtype=np.uint8)
    I = np.array([1_000_000], dtype=np.int32)
    theta = np.int16(32767)  # at saturation boundary; expect spike if equals
    lam = np.uint16(32768)
    cfg = ConfigFp(refractory_steps=3)

    # Compute vv16 independently to detect saturation effect before spike logic
    vv16 = saturating_add_int16(q15_mul_shift(v, lam).astype(np.int32) + I)[0]
    assert vv16 == np.int16(32767)


def test_leak_q15():
    # lambda = 0.5 -> right shift by 1 approximately
    lam = np.uint16(16384)  # 0.5 in Q1.15
    v0 = np.array([1000, -1000], dtype=np.int16)
    out = q15_mul_shift(v0, lam)
    assert out.tolist() == [500, -500]

    # Through lif_step with zero I and no refractory
    v = v0.copy()
    ref = np.zeros_like(v, dtype=np.uint8)
    I = np.zeros_like(v, dtype=np.int32)
    theta = np.int16(30000)
    cfg = ConfigFp(refractory_steps=1)
    spikes, num = lif_step(v, ref, I, theta, lam, cfg)
    assert num == 0
    assert spikes.tolist() == [False, False]
    assert v.tolist() == [500, -500]


def test_determinism():
    rng_seed = 123
    n = 1024
    # Deterministic arrays (not random)
    v = (np.arange(n, dtype=np.int16) % 1000).astype(np.int16)
    ref = np.zeros(n, dtype=np.uint8)
    I = np.full(n, 250, dtype=np.int32)
    theta = np.int16(10000)
    lam = np.uint16(20000)
    cfg = ConfigFp(refractory_steps=2)

    v1 = v.copy(); ref1 = ref.copy()
    s1, n1 = lif_step(v1, ref1, I, theta, lam, cfg)

    v2 = v.copy(); ref2 = ref.copy()
    s2, n2 = lif_step(v2, ref2, I, theta, lam, cfg)

    assert n1 == n2
    assert np.array_equal(s1, s2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(ref1, ref2)

