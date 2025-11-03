import numpy as np

from symbolicplastic_snn.readout.readout import Readout, ReadoutConfig


def test_window_counting_and_slide():
    cfg = ReadoutConfig(window=3, early_exit=False)
    rd = Readout(cfg)
    rd.add_channel("A", np.array([0, 1], dtype=np.int32))
    rd.add_channel("B", np.array([2, 3], dtype=np.int32))

    # Step 1: A fires once
    rd.step(np.array([True, False, False, False], dtype=bool))
    # Step 2: B fires once
    rd.step(np.array([False, False, True, False], dtype=bool))
    # Step 3: both fire once
    rd.step(np.array([True, False, True, False], dtype=bool))
    out = rd.emit()
    assert out["scores"]["A"] == 2
    assert out["scores"]["B"] == 2

    # Step 4: only B fires; drop step 1 from window
    rd.step(np.array([False, False, False, True], dtype=bool))
    out2 = rd.emit()
    assert out2["scores"]["A"] == 1  # steps 2..4: A = 0+1+0
    assert out2["scores"]["B"] == 3  # steps 2..4: B = 1+1+1


def test_earliest_time_wins():
    cfg = ReadoutConfig(window=10, early_exit=True)
    rd = Readout(cfg)
    rd.add_channel("X", np.array([0, 1], dtype=np.int32))
    rd.add_channel("Y", np.array([2, 3], dtype=np.int32))

    # Step 1: X fires -> latch at t=1
    rd.step(np.array([True, False, False, False], dtype=bool))
    # Step 2: Y fires heavily but should not override earliest latch
    rd.step(np.array([False, False, True, True], dtype=bool))
    out = rd.emit()
    assert out["label"] == "X"
    assert out["latency"] == 1


def test_early_exit_triggered_by_lead_margin():
    # Use higher gain ratio to make exiting easier
    cfg = ReadoutConfig(window=20, early_exit=True, max_future_gain_ratio=0.5)
    rd = Readout(cfg)
    rd.add_channel("C0", np.array([0, 1], dtype=np.int32))
    rd.add_channel("C1", np.array([2, 3], dtype=np.int32))

    # Build a lead for C0
    steps = [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, False, True, False],
        [True, True, False, False],
    ]
    for s in steps:
        rd.step(np.array(s, dtype=bool))

    assert rd.can_early_exit(budget_remaining=3)

