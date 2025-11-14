import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from obspy.core import read

from .stack_utils import batch_hilbert_mkl
from .stream_to_array import stream_to_array

def stack_ccf(
    ccf2: np.ndarray,
    method: str = "linear",
    nu: float = 2.0,
) -> np.ndarray:
    """
    Stack cross-correlation functions (CCFs) using linear or phase-weighted stack.

    Parameters
    ----------
    ccf_data : np.ndarray
        Array of shape (n_segments, n_lags) or (n_lags,).
    method : {"linear", "pws"}, optional
        Stacking method. "linear" for simple average;
        "pws" for phase-weighted stack, by default "linear".
    nu : float, optional
        Exponent for phase weighting (PWS). Must be > 0, usually between 1 to 4, by default 2.0.

    Returns
    -------
    np.ndarray
        Stacked CCF of shape (n_lags,).

    Raises
    ------
    ValueError
        If an unknown method is provided or `nu` is not positive.
    """
    if nu <= 0:
        raise ValueError("Parameter 'nu' must be positive for PWS.")

    arr = np.atleast_2d(ccf2)

    method = method.lower()
    ccf_linear = np.nanmean(arr, axis=0)

    if method == "linear":
        return ccf_linear

    if method == "pws":
        analytic = batch_hilbert_mkl(arr, axis=1)
        phase_vectors = analytic / (np.abs(analytic) + 1e-12)
        coherence = np.abs(phase_vectors.mean(axis=0)) ** nu
        return ccf_linear * coherence

    raise ValueError("Unknown method. Use 'linear' or 'pws'.")

def plot_ccf(
    ccf,
    distance,
    dt,
    time_window=1,
    vmin=None,
    vmax=None,
    axes=None,
    plot_kwargs=None,
):
    """
    Plot stacked cross-correlation functions (CCFs) sorted by interstation distance.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation matrix (N × T).
    distance : np.ndarray
        Interstation distances for each CCF trace.
    dt : float
        Sampling interval in seconds.
    time_window : float, optional
        Time range around zero lag to display.
    vmin, vmax : float, optional
        Reference velocities (m/s) for overlay lines.
    axes : matplotlib.axes.Axes, optional
        Target axes for plotting.
    plot_kwargs : dict, optional
        Additional arguments for `Axes.plot`.
    """
    uniq_dist = np.unique(distance)
    ccfs_array = np.zeros((uniq_dist.size, ccf.shape[1]))

    for i, dist in enumerate(uniq_dist):
        ccfs_array[i] = stack_ccf(ccf[distance == dist], method="pws", nu=2.0)

    ccfs_array = ensure_odd_length(ccfs_array)
    ccfs_array /= np.maximum(np.max(ccfs_array, axis=1, keepdims=True), 1e-8)

    half_len = ccfs_array.shape[1] // 2
    time_axis = np.arange(-half_len, half_len + 1) * dt
    mask = (time_axis >= -time_window) & (time_axis <= time_window)
    trimmed_ccfs = ccfs_array[:, mask]
    trimmed_time = time_axis[mask]

    # Velocity reference lines
    t_pos = np.arange(0, time_window, dt)
    vmin_line = vmin * t_pos if vmin else None
    vmax_line = vmax * t_pos if vmax else None

    if axes is None:
        _, axes = plt.subplots(figsize=(10, 7))
    if plot_kwargs is None:
        plot_kwargs = {}

    for i in range(trimmed_ccfs.shape[0]):
        axes.plot(trimmed_time, trimmed_ccfs[i] + uniq_dist[i], color="k", **plot_kwargs)

    if vmin_line is not None:
        axes.plot(t_pos, vmin_line, color="r", ls="--", lw=1.5, label=f"Vmin: {vmin} m/s")
        axes.plot(-t_pos, vmin_line, color="r", ls="--", lw=1.5)

    if vmax_line is not None:
        axes.plot(t_pos, vmax_line, color="b", ls="--", lw=1.5, label=f"Vmax: {vmax} m/s")
        axes.plot(-t_pos, vmax_line, color="b", ls="--", lw=1.5)

    axes.set(
             xlim=(-time_window, time_window),
             ylim=(uniq_dist.min() * 0.5, uniq_dist.max() * 1.05)
             )
    axes.set_xlabel("Lag time (s)", fontsize=18)
    axes.set_ylabel("Interstation distance (m)", fontsize=18)


    # Add legend with styling
    axes.legend(
        fontsize=15,
        markerscale=1.5,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.8,
    )

    axes.tick_params(labelsize=18)
    plt.tight_layout()


def plot_ccf_dir(
    dirpath: Path,
    time_window=1,
    vmin=None,
    vmax=None,
):
    """
    Load SAC CCF files from directory and plot stacked cross-correlations.

    Parameters
    ----------
    dirpath : Path
        Directory containing SAC files matching 'CCF*.sac'.
    time_window : float, optional
        Time range around zero lag to display.
    vmin, vmax : float, optional
        Reference velocities (m/s) for overlay lines.
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise ValueError("Invalid directory path.")
    
    st = read(str(dirpath.joinpath("CCF*.sac")))
    data, _ = stream_to_array(st)
    distance = np.array([tr.stats.sac.dist for tr in st])
    dt = st[0].stats.delta
    plot_ccf(data, distance, dt, time_window, vmin, vmax)




def ensure_odd_length(data) -> np.array:  # type: ignore
    """
    Ensures the input data has an odd length.
    For 1D data, removes the last element if length is even.
    For 2D data, removes the last column if number of columns is even.

    Parameters:
    data (array-like): 1D or 2D array-like structure

    Returns:
    numpy.ndarray: Modified array with odd length/columns
    """
    # Convert to numpy array for easier manipulation
    data = np.array(data)

    # Check if data is 1D
    if data.ndim == 1:
        # If length is even, remove last element
        if len(data) % 2 == 0 and len(data) > 0:
            return data[:-1]
        else:
            return data

    # Check if data is 2D
    elif data.ndim == 2:
        rows, cols = data.shape
        # If number of columns is even, remove last column
        if cols % 2 == 0 and cols > 0:
            return data[:, :-1]
        else:
            return data

    # For other dimensions, return as is
    else:
        NotImplementedError("Input data must be 1D or 2D")
