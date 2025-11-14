"""
Forward modeling utilities for surface wave dispersion curves.

This module provides functions for computing synthetic dispersion curves
from velocity models using the disba library.
"""

import numpy as np
from disba import PhaseDispersion
from typing import Tuple


def forward_disp(model: np.ndarray, period: np.ndarray, mode: int = 0, wave_type: str = 'rayleigh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward modeling of surface wave dispersion curves.
    
    Computes synthetic dispersion curves for a given velocity model
    using the disba library.
    
    Args:
        model: Model matrix with columns [thickness, vp, vs, rho]
        period: Array of periods to compute dispersion for (seconds)
        mode: Mode number (default: 0)
        wave_type: Wave type (default: 'rayleigh')
        
    Returns:
        Tuple of (periods, velocities) arrays
        
    Raises:
        ValueError: If model format is invalid
        RuntimeError: If forward modeling fails
    """
    # Validate model input
    if model.ndim != 2 or model.shape[1] != 4:
        raise ValueError("Model must be 2D array with 4 columns [thickness, vp, vs, rho]")
    
    if len(model) == 0:
        raise ValueError("Model cannot be empty")
    
    try:
        pd = PhaseDispersion(*model.T)
        forward_period, forward_vel = pd(period, mode=mode, wave=wave_type)[:2]
        return forward_period, forward_vel
    except Exception as e:
        raise RuntimeError(f"Forward modeling failed: {e}")
