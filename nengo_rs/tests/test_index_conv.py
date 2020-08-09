from nengo.builder.signal import Signal
from nengo_rs.index_conv import (
    offset_to_multiindex,
    slices_from_signal,
    strides_to_steps,
)
import numpy as np


def test_offset_to_multiindex():
    assert offset_to_multiindex(3, (3,)) == (1,)
    assert offset_to_multiindex(10, (8, 4, 1)) == (1, 0, 2)
    assert offset_to_multiindex(215, (60, 12, 6, 1)) == (3, 2, 1, 5)


def test_strides_to_steps():
    assert strides_to_steps((1,), (1,)) == (1,)
    assert strides_to_steps((480, 192, 192, 24), (480, 96, 48, 8)) == (1, 2, 4, 3)


def test_slices_from_signal():
    base = Signal(initial_value=np.zeros((20, 20, 20)))
    assert slices_from_signal(base[:, 2:10:3, ::4]) == (
        slice(0, 20, 1),
        slice(2, 11, 3),
        slice(0, 20, 4),
    )
    assert slices_from_signal(base[4:15, :5, 4:]) == (
        slice(4, 15, 1),
        slice(0, 5, 1),
        slice(4, 20, 1),
    )
