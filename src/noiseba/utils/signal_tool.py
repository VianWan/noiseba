import numpy as np

def apply_edge_taper(data, taper_ratio=0.1, window_fn=np.hamming):
    """ Apply edge taper to 1, 2, 3D array."""
    if not (0 < taper_ratio < 0.5):
        raise ValueError("taper_ratio in (0, 0.5)")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be numpy.ndarray")

    T = data.shape[-1]
    win_len = int(T * taper_ratio)
    if win_len % 2 != 0:
        win_len += 1  # even length

    if win_len >= T:
        raise ValueError("Window length must be less than data length.")

    taper = window_fn(win_len)
    half = win_len // 2
    left_win = taper[:half]
    right_win = taper[half:]

    shape = [1] * data.ndim
    shape[-1] = half

    data = data.copy()
    data[..., :half] *= left_win.reshape(shape)
    data[..., -half:] *= right_win.reshape(shape)

    return data
