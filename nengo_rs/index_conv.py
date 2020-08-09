def offset_to_multiindex(offset, base_strides):
    multiindex = []
    remainder = offset
    for divisor in base_strides:
        multiindex.append(remainder // divisor)
        remainder = remainder % divisor
    return tuple(multiindex)


def strides_to_steps(strides, base_strides):
    return tuple(
        stride // base_stride for stride, base_stride in zip(strides, base_strides)
    )


def slices_from_signal(signal):
    return tuple(
        slice(start, min(start + step * size, base_size), step)
        for start, size, step, base_size in zip(
            offset_to_multiindex(signal.elemoffset, signal.base.elemstrides),
            signal.shape,
            strides_to_steps(signal.elemstrides, signal.base.elemstrides),
            signal.base.shape,
        )
    )
