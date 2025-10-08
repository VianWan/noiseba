from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from disba import PhaseDispersion, PhaseSensitivity


def plot_sensitivity(
    model,
    freq_min: Union[int, float],
    freq_max: Union[int, float],
    num_freq: int = 101,
    mode: int = 0,
    wave: str = "rayleigh",
    parameter: str = "velocity_s",
    plot_type: str = "freq",
    ax=None,
):
    """
    Plot depth-dependent sensitivity kernels for surface-wave phase velocity.

    Parameters
    ----------
    model : ndarray
        2-D array with columns [thickness, velocity_p, velocity_s, density].
    freq_min, freq_max : float
        Minimum/maximum frequency (Hz) for the kernel calculation.
    num_freq : int, optional
        Number of frequency samples. Default is 101.
    mode : int, optional
        Mode number (0 = fundamental). Default is 0.
    wave : {'rayleigh', 'love'}, optional
        Wave type. Default is 'rayleigh'.
    parameter : str, optional
        Model parameter to which sensitivity is computed.
        Default is 'velocity_s'.
    plot_type : {'freq', 'period'}, optional
        X-axis variable. Default is 'freq'.
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If None, a new figure is created.

    Returns
    -------
    depth : ndarray
        1-D depth array (km) corresponding to kernel rows.
    """
    if freq_min <= 0:
        raise ValueError("freq_min must be positive and non-zero.")

    # Frequency / period vectors
    freq = np.linspace(freq_min, freq_max, num_freq)
    period = 1.0 / freq  # s

    # Phase-dispersion calculation
    pd = PhaseDispersion(*model.T, algorithm="dunkin")
    cpr = pd(period, mode=mode, wave=wave)  

    # Prepare x-axis data
    if plot_type == "freq":
        x_centres = np.flipud(1.0 / cpr[0])  # Hz
        velocity = np.flipud(cpr[1])
    elif plot_type == "period":
        x_centres = cpr[0]
        velocity = cpr[1]
    else:
        raise ValueError("plot_type must be 'freq' or 'period'.")
    x_edges = _build_edges(x_centres)

    # Sensitivity kernels
    ps = PhaseSensitivity(*model.T)
    kernels = [ps(t, mode=mode, wave=wave, parameter=parameter) for t in period]
    sensitivity = np.column_stack([k.kernel for k in kernels])  # (n_depth, n_period)
    depth = kernels[0].depth

    # Extend depth to mimic half-space
    depths = np.zeros(depth.size + 1)
    depths[:-1] = depth
    depths[-1] = depth[-1] * 1.2

    # Colour map: white -> pink -> red
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(1, 1, 1), (1, 0.8, 0.8), (1, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0)],
    )

    # Create figure if no axes supplied
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    # Flip matrix for frequency axis
    if plot_type == "freq":
        sensitivity = np.fliplr(sensitivity)

    # Main pcolormesh plot
    mesh = ax.pcolormesh(x_edges, depths, sensitivity, cmap=cmap)

    # Overlay phase-velocity curve
    ax_vel = ax.twinx()
    ax_vel.plot(
        x_centres,
        velocity,
        color="#AF59CE9E",
        lw=2,
        marker="o",
        markerfacecolor="#A25893",
        markeredgecolor="white",
    )

    # Cosmetics
    ax.set_xlim(x_edges.min() * 0.9, x_edges.max() * 1.0)
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=14)

    xlabel = "Frequency (Hz)" if plot_type == "freq" else "Period (s)"
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Depth (km)", fontsize=20)

    ax_vel.set_ylabel("Phase Velocity (km/s)", fontsize=18)
    ax_vel.tick_params(axis="y", labelsize=14)

    # Colour bar
    cbar = fig.colorbar(mesh, ax=ax_vel, pad=0.12)
    cbar.set_label("Sensitivity", fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    plt.show()
    return depth


def _build_edges(centres: np.ndarray) -> np.ndarray:
    """
    Convert array of cell centres to array of cell edges.

    Parameters
    ----------
    centres : ndarray
        Sorted or unsorted 1-D array of centre positions.

    Returns
    -------
    edges : ndarray
        1-D array with length len(centres) + 1.
    """
    centres = np.asarray(centres)
    centres_sorted = np.sort(centres)
    deltas = np.diff(centres_sorted)

    if deltas.size == 0:  # single point
        return np.array(
            [centres_sorted[0] - 0.5, centres_sorted[0] + 0.5], dtype=centres.dtype
        )

    edges = np.empty(centres.size + 1, dtype=centres.dtype)
    edges[1:-1] = centres_sorted[:-1] + deltas / 2.0
    edges[0] = centres_sorted[0] - deltas[0] / 2.0
    edges[-1] = centres_sorted[-1] + deltas[-1] / 2.0
    return edges