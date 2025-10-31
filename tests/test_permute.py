import numpy as np

from symbolicplastic_snn.conn.permute import permute_first_m


def test_no_duplicates_and_range():
    n, m = 100, 20
    key = np.uint64(1234567890123456789)
    out = permute_first_m(n, m, key)
    assert out.dtype == np.int32
    assert out.shape[0] == m
    assert np.all(out >= 0) and np.all(out < n)
    # No duplicates
    assert len(np.unique(out)) == m


def test_determinism_different_keys():
    n, m = 50, 15
    key1 = np.uint64(0xDEADBEEFCAFEBABE)
    key2 = np.uint64(0xDEADBEEFCAFEBABE)
    key3 = np.uint64(0x1234567890ABCDEF)

    out1 = permute_first_m(n, m, key1)
    out2 = permute_first_m(n, m, key2)
    out3 = permute_first_m(n, m, key3)

    # Same key -> identical
    assert np.array_equal(out1, out2)
    # Different key -> very likely different
    assert not np.array_equal(out1, out3)


def test_prefix_property():
    n = 200
    m1, m2 = 25, 80
    key = np.uint64(0xA5A5A5A5A5A5A5A5)
    out1 = permute_first_m(n, m1, key)
    out2 = permute_first_m(n, m2, key)
    assert np.array_equal(out1, out2[:m1])
