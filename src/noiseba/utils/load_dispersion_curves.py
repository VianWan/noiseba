import numpy as np
from typing import List, Optional

from noiseba.inversion.model import Curve


def load_dispersion_curves(file_path: str, wave_type: str = "rayleigh", x: Optional[float] = None, y: Optional[float] = None) -> List[Curve]:
    """
    Load dispersion curves from a file.
    
    The file should contain at least 3 columns: frequency, velocity, and mode.
    Optionally, it can also contain x and y coordinates as additional columns.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing dispersion data.
        Expected format: freq, vel, mode, [x, y, ...]
    wave_type : str
        Type of wave (e.g., 'rayleigh', 'love')
    x, y : float, optional
        Coordinates for the dispersion curve. If provided, all curves from this file
        will have the same x,y location. If not provided, the function will try to 
        load coordinates from the file if present (5+ columns).
        
    Returns
    -------
    List[Curve]
        List of Curve objects with frequency, velocity, mode and optional location information
    """
    data = np.loadtxt(file_path)
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Check if the file has enough columns for x, y coordinates
    if data.shape[1] >= 5:  # freq, vel, mode, x, y
        freq, vel, mode = data[:, 0], data[:, 1], data[:, 2]
        # Use coordinates from file only if not provided as parameters
        if x is None and y is None:
            # Use the median or mean of coordinates in the file if available
            file_x, file_y = data[:, 3], data[:, 4]
            # For a single dispersion curve, use the representative coordinate (median)
            x = np.median(file_x)
            y = np.median(file_y)
    elif data.shape[1] >= 3:  # freq, vel, mode (no coordinates in file)
        freq, vel, mode = data[:, 0], data[:, 1], data[:, 2]
        # Use provided coordinates or remain None
    else:
        raise ValueError(f"File must contain at least 3 columns (freq, vel, mode), but got {data.shape[1]}")
    
    modes = np.unique(mode).astype(int)

    curves: List[Curve] = []
    for m in modes:
        mask = mode == m
        curves.append(Curve(
            freq=freq[mask], 
            velocity=vel[mask] * 1e-3, 
            wave_type=wave_type, 
            mode=int(m),
            x=x,
            y=y
        ))
    return curves