import numpy as np

from symbolicplastic_snn.conn.permute import permute_first_m


def test_no_duplicates():
    n, m = 100, 20
    key = np.uint64(1234567890123456789)
    out = permute_first_m(n, m, key)
    assert out.dtype == np.int32
    assert out.shape[0] == m
    assert np.all(out >= 0) and np.all(out < n)
    # No duplicates
    assert len(np.unique(out)) == m


def test_determinism():
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

