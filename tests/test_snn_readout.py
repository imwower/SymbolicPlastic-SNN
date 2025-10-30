import numpy as np

from symbolicplastic_snn.readout.readout import Readout, ReadoutConfig


def test_counting_window_slide():
    cfg = ReadoutConfig(window=3, early_exit=False)
    rd = Readout(cfg)
    # Two channels on 4 neurons: A-> [0,1], B-> [2,3]
    rd.add_channel("A", np.array([0, 1], dtype=np.int32))
    rd.add_channel("B", np.array([2, 3], dtype=np.int32))

    # Step 1: only A fires once
    rd.step(np.array([True, False, False, False], dtype=bool))  # A=1, B=0
    # Step 2: only B fires once
    rd.step(np.array([False, False, True, False], dtype=bool))  # A=1, B=1
    # Step 3: both fire once
    rd.step(np.array([True, False, True, False], dtype=bool))   # A=2, B=2
    out = rd.emit()
    assert out["scores"]["A"] == 2
    assert out["scores"]["B"] == 2

    # Step 4: only B fires; window slides, dropping step1
    rd.step(np.array([False, False, False, True], dtype=bool))  # window over steps 2..4
    out2 = rd.emit()
    # In window [2,3,4]: A counts = 0 + 1 + 0 = 1, B = 1 + 1 + 1 = 3
    assert out2["scores"]["A"] == 1
    assert out2["scores"]["B"] == 3


def test_earliest_time_wins():
    cfg = ReadoutConfig(window=10, early_exit=True)
    rd = Readout(cfg)
    rd.add_channel("X", np.array([0, 1], dtype=np.int32))
    rd.add_channel("Y", np.array([2, 3], dtype=np.int32))

    # Step 1: X fires -> latch to X at latency 1
    rd.step(np.array([True, False, False, False], dtype=bool))
    # Step 2: Y fires heavily but should not override earliest latch
    rd.step(np.array([False, False, True, True], dtype=bool))
    out = rd.emit()
    assert out["label"] == "X"
    assert out["latency"] == 1


def test_early_exit_trigger():
    # Configure high ratio to make exiting easier to trigger
    cfg = ReadoutConfig(window=20, early_exit=True, max_future_gain_ratio=0.5)
    rd = Readout(cfg)
    rd.add_channel("C0", np.array([0, 1], dtype=np.int32))
    rd.add_channel("C1", np.array([2, 3], dtype=np.int32))

    # Build up a lead for C0
    # 5 steps: C0 gets 2 spikes/step, C1 gets 0-1
    steps = [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, False, True, False],
        [True, True, False, False],
    ]
    for s in steps:
        rd.step(np.array(s, dtype=bool))

    # Now C0 should lead; small remaining budget should allow early exit
    # Estimate: average per step to readout ~ ~2 for C0 + small for C1; delta should exceed bound
    assert rd.can_early_exit(budget_remaining=3)
