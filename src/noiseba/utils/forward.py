import numpy as np

from disba import PhaseDispersion
from typing import Tuple

def forward_disp(model: np.ndarray, period: np.ndarray, mode: int = 0, wave_type: str = 'rayleigh' ) -> Tuple[np.ndarray, np.ndarray]:
    pd = PhaseDispersion(*model.T)
    forward_period, forward_vel = pd(period, mode=mode, wave=wave_type)[:2]

    return forward_period, forward_vel
