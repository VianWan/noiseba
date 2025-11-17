"""
Utility functions for assigning location information to dispersion curves.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from noiseba.inversion.model import Curve


def assign_location_to_curve(
    curve: Curve, 
    station_a_x: float, 
    station_a_y: float, 
    station_b_x: float, 
    station_b_y: float
) -> Curve:
    """
    Assign location information to a dispersion curve based on the two stations 
    that were used to generate the CCF from which the curve was extracted.
    
    For a cross-correlation between two stations, the dispersion curve typically 
    represents the structure along the path between the two stations. The location
    can be defined as the midpoint between the two stations.
    
    Parameters
    ----------
    curve : Curve
        The dispersion curve to which location will be assigned
    station_a_x, station_a_y : float
        Coordinates of the first station (source)
    station_b_x, station_b_y : float
        Coordinates of the second station (receiver)
    
    Returns
    -------
    Curve
        The curve with assigned location information
    """
    # Calculate the midpoint between the two stations
    mid_x = (station_a_x + station_b_x) / 2
    mid_y = (station_a_y + station_b_y) / 2
    
    curve.x = mid_x
    curve.y = mid_y
    
    return curve

def calculate_effective_location(
    ccf_index: pd.DataFrame,
    method: str = "median_midpoint",
    target_point: Optional[Tuple[float, float]] = None,
    profile_line: Optional[List[Tuple[float, float]]] = None
) -> Tuple[float, float]:
    """
    Calculate the representative central location for a set of station pairs.
    
    Used in ambient noise studies when multiple inter-station paths constrain
    the same 1D velocity model (linear arrays, local 1D inversion, volcano/reservoir monitoring).
    
    Parameters
    ----------
    ccf_index : pd.DataFrame
        Must contain columns: 'x_a', 'y_a', 'x_b', 'y_b'
    method : str
        'mean_midpoint'     : arithmetic mean of all midpoints
        'median_midpoint'   : median of all midpoints (recommended, robust to outliers)
        'array_center'      : mean of all station coordinates (not just midpoints)
    target_point : tuple, optional
        Manually specified location (highest priority, e.g. volcano vent, well head)
    profile_line : list of two tuples, optional
        [[x1,y1], [x2,y2]] → project all midpoints onto this line and take mean position
    
    Returns
    -------
    (x, y) : tuple[float, float]
        Single representative location assigned to all dispersion curves
    """
    # Handle manual override first
    if target_point is not None:
        return target_point

    # Ensure we're working with numpy arrays for mathematical operations
    x_a = np.asarray(ccf_index['x_a'])
    y_a = np.asarray(ccf_index['y_a'])
    x_b = np.asarray(ccf_index['x_b'])
    y_b = np.asarray(ccf_index['y_b'])

    # Handle projection onto profile line if specified
    if profile_line is not None:
        if len(profile_line) != 2:
            raise ValueError("profile_line must contain exactly two points")
            
        p1 = np.array(profile_line[0])
        p2 = np.array(profile_line[1])
        vec = p2 - p1
        L = np.linalg.norm(vec)
        
        if L == 0:
            return tuple(p1.tolist())
            
        unit = vec / L
        
        # Calculate all midpoints
        midpoints = np.column_stack([
            (x_a + x_b) / 2,
            (y_a + y_b) / 2
        ])
        
        # Project midpoints onto the line
        proj_lengths = np.dot(midpoints - p1, unit)
        proj_lengths = np.clip(proj_lengths, 0, L)
        central_point = p1 + np.mean(proj_lengths) * unit
        return tuple(central_point.tolist())

    # Default: statistical center of midpoints or stations
    if method == "median_midpoint":
        x = np.median((x_a + x_b) / 2)
        y = np.median((y_a + y_b) / 2)
    elif method == "mean_midpoint":
        x = np.mean((x_a + x_b) / 2)
        y = np.mean((y_a + y_b) / 2)
    elif method == "array_center":
        all_x = np.hstack([x_a, x_b])
        all_y = np.hstack([y_a, y_b])
        x, y = np.mean(all_x), np.mean(all_y)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return float(x), float(y)


def assign_locations_from_ccf_index(
    curves: List['Curve'],
    ccf_index: pd.DataFrame,
    method: str = "median_midpoint",
    target_point: Optional[Tuple[float, float]] = None,
    profile_line: Optional[List[Tuple[float, float]]] = None
) -> List['Curve']:

    x, y = calculate_effective_location(ccf_index, method=method, 
                                        target_point=target_point, profile_line=profile_line)
    
    for curve in curves:
        curve.x = x
        curve.y = y
        
    return curves