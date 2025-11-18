"""
Module for interpolating 1D velocity structures into 2D profiles.

This module provides functions to combine multiple 1D velocity models
into a 2D cross-section using pygmt for interpolation.
"""
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import griddata

import pygmt

def extract_1d_inversion_models(
    inversion_results: Dict,
    x_coords: Optional[List[float]] = None,
    unit: str = "km"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract best-fitting 1D Vs models from all sites.
    
    Returns scattered points ready for 2D interpolation:
        x_scattered, z_scattered (layer bottoms), vs_scattered
    
    Parameters
    ----------
    unit : {"km", "m"}
        Output unit for distances and depths.
    """
    sites = list(inversion_results.keys())

    # --- Extract x coordinates if not provided ---
    if x_coords is None:
        x_coords = []
        for site in sites:
            obs_curves = inversion_results[site].result.get(site, {}).get("obs_curves", [])
            if obs_curves:
                site_x = [getattr(c, 'x', 0.0) for c in obs_curves if getattr(c, 'x', None) is not None]
                x_coords.append(np.nanmedian(site_x) if site_x else 0.0)
            else:
                x_coords.append(0.0)

    x_scattered, z_scattered, vs_scattered = [], [], []

    for i, site in enumerate(sites):
        best_pos = inversion_results[site].best_position(site)
        if best_pos is None:
            raise ValueError(f"No best position for site '{site}'")

        linear_pos = np.exp(best_pos)
        costfunc = inversion_results[site].costfunction

        if (hasattr(costfunc, '_mode') and costfunc._mode == "gradient"
            and hasattr(costfunc, '_fixed_thickness')):
            vs_values = linear_pos
            thicknesses = costfunc._fixed_thickness
        else:
            n = len(linear_pos) // 2
            thicknesses = linear_pos[:n]
            vs_values = linear_pos[n:]

        cum_depth = np.cumsum(thicknesses)  # depth to bottom of each layer
        x_site = x_coords[i]

        for z_bot, vs in zip(cum_depth, vs_values):
            x_scattered.append(x_site)
            z_scattered.append(z_bot)
            vs_scattered.append(vs)

    x_scattered = np.array(x_scattered)
    z_scattered = np.array(z_scattered)
    vs_scattered = np.array(vs_scattered)

    # Unit conversion
    factor = 1000.0 if unit == "m" else 1.0
    return x_scattered * factor, z_scattered * factor, vs_scattered



def interpolate_profile(
    x_scattered: np.ndarray,
    z_scattered: np.ndarray,
    vs_scattered: np.ndarray,
    grid_spacing_x: Union[float, str] = 1.0,      # km or m
    grid_spacing_z: Union[float, str] = 0.5,      # usually finer vertically
    depth_range: Optional[Tuple[float, float]] = None,
    x_padding: float = 5.0,        # extra km/m on left & right
    z_extra_bottom: float = 10.0,  # extra space below deepest point
    method: str = "pygmt_surface",
    tension: float = 0.35,
    gaussian_sigma: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a smooth 2D Vs profile from scattered 1D models.
    
    Best-practice defaults:
      • Always starts at z = 0 (surface)
      • Slightly extends beyond data on all sides
      • Independent horizontal and vertical spacing
    
    Parameters
    ----------
    tension_regions : list of ( (x1,x2,z1,z2), tension_value )
        Apply different tension in different rectangles (e.g., stronger smoothing at depth)
    gaussian_sigma : float or None
        Additional light Gaussian smoothing after interpolation (in grid cells)
    
    Returns
    -------
    x_grid, z_grid, vs_grid   (2D arrays, z increasing downward)
    """
    # --- Define intelligent grid limits ---
    x_min, x_max = x_scattered.min(), x_scattered.max()
    z_max_data = z_scattered.max()                   

    x_min -= x_padding
    x_max += x_padding

    if depth_range:
        z_min, z_max = depth_range
    else:
        z_min = 0.0
        z_max = z_max_data + z_extra_bottom
    # Round to nice multiples (ensures GMT compatibility)
    def _nice_grid(val_min, val_max, spacing):
        start = np.floor(val_min / spacing) * spacing
        end   = np.ceil(val_max / spacing) * spacing
        return start, end + spacing*1e-6  # tiny epsilon for GMT happiness

    x_min, x_max = _nice_grid(x_min, x_max, float(grid_spacing_x))
    z_min, z_max = _nice_grid(z_min, z_max, float(grid_spacing_z))

    region = [x_min, x_max, z_min, z_max]

    # --- Pure PyGMT surface (recommended) ---
    if method == "pygmt_surface":
        # Base interpolation with global tension
        grid = pygmt.surface(
            x=x_scattered,
            y=z_scattered,
            z=vs_scattered,
            spacing=f"{grid_spacing_x}/{grid_spacing_z}",
            region=region,
            tension=tension,
            verbose="error",
        )
        vs_grid = grid.values

    else:  # fallback scipy
        x_g = np.arange(x_min, x_max + grid_spacing_x, grid_spacing_x)
        z_g = np.arange(z_min, z_max + grid_spacing_z, grid_spacing_z)
        xg, zg = np.meshgrid(x_g, z_g)
        vs_grid = griddata(
            (x_scattered, z_scattered), vs_scattered,
            (xg, zg), method='linear'
        )

    # Optional very light Gaussian post-smoothing (often looks nicer)
    if gaussian_sigma and method == "pygmt_surface":
        vs_grid = gaussian_filter(vs_grid, sigma=gaussian_sigma)

    # Final grids
    x_grid = np.linspace(x_min, x_max, vs_grid.shape[1])
    z_grid = np.linspace(z_min, z_max, vs_grid.shape[0])
    x_grid, z_grid = np.meshgrid(x_grid, z_grid)

    return x_grid, z_grid, vs_grid



def create_2d_profile_plot(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    vs_grid: np.ndarray,
    title: str = "2D Shear-Wave Velocity Profile",
    vs_range: Optional[Tuple[float, float]] = None,
    cmap: str = 'rainbow',
    ax: Optional[Axes] = None,
    add_colorbar: bool = True,
) -> Axes:
    """
    Plot 2-D shear-velocity profile.

    Parameters
    ----------
    x_grid, z_grid, vs_grid : 2-D array
        Coordinates and velocity values (from meshgrid).
    title : str
        Plot title.
    vs_range : (vmin, vmax), optional
        Color-bar limits; default = 5–95 percentile of vs_grid.
    cmap : str
        Matplotlib colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; if None, a new figure is created.
    add_colorbar : bool
        Attach color-bar.

    Returns
    -------
    ax : plt.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        return_fig = True
    else:
        return_fig = False

    if vs_range is None:
        vmin, vmax = np.nanpercentile(vs_grid, [5, 95])
    else:
        vmin, vmax = vs_range

    pm = ax.pcolormesh(
        x_grid, z_grid, vs_grid,
        shading='auto',
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax.set(xlabel='Distance (km)', ylabel='Depth (km)', title=title)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, ls='--', lw=0.4, c='gray', alpha=0.5)

    if add_colorbar:
        cbar = plt.colorbar(pm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Vs (km/s)')

    if return_fig:
        fig.tight_layout()

    return ax




def trans_2d_profile(
    inversion_results: Dict,
    x_coords: Optional[List[float]] = None,
    unit: str = "km",
    grid_spacing_x: Union[float, str] = 1.0,
    grid_spacing_z: Union[float, str] = 0.5,
    depth_range: Optional[Tuple[float, float]] = None,
    x_padding: float = 5.0,
    z_extra_bottom: float = 10.0,
    tension: float = 0.35,
    gaussian_sigma: Optional[float] = None,
    vs_range: Optional[Tuple[float, float]] = None,
    cmap: str = 'rainbow',
    ax: Optional[Axes] = None,
    add_colorbar: bool = True,
) -> Axes:
    """
    One-stop function: extract 1-D models → interpolate → plot 2-D Vs profile.

    All parameters share the same definitions as those in
    extract_1d_inversion_models, interpolate_profile and create_2d_profile_plot.
    """
    # 1. extract
    x_s, z_s, vs_s = extract_1d_inversion_models(
        inversion_results, x_coords=x_coords, unit=unit
    )

    # 2. interpolate
    x_grid, z_grid, vs_grid = interpolate_profile(
        x_s, z_s, vs_s,
        grid_spacing_x=grid_spacing_x,
        grid_spacing_z=grid_spacing_z,
        depth_range=depth_range,
        x_padding=x_padding,
        z_extra_bottom=z_extra_bottom,
        tension=tension,
        gaussian_sigma=gaussian_sigma,
    )

    # 3. plot
    ax = create_2d_profile_plot(
        x_grid, z_grid, vs_grid,
        vs_range=vs_range,
        cmap=cmap,
        ax=ax,
        add_colorbar=add_colorbar,
    )
    return ax

