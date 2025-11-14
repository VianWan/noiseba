import numpy as np
from typing import List

from noiseba.inversion.model import Curve


def load_dispersion_curves(file_path: str, wave_type: str = "rayleigh") -> List[Curve]:
    freq, vel, mode = np.loadtxt(file_path).T
    modes = np.unique(mode).astype(int)

    curves: List[Curve] = []
    for m in modes:
        mask = mode == m
        curves.append(Curve(freq=freq[mask], velocity=vel[mask] * 1e-3, wave_type=wave_type, mode=int(m)))
    return curves
