import matplotlib.pyplot as plt
import numpy as np

from noiseba.preprocessing import stack_ccf


def plot_ccf(
    ccf_dict,
    ccf_distance,
    dt,
    time_window=1,
    vmin=None,
    vmax=None,
    axes=None,
    plot_kwargs=None,
):
    """
    Plot cross-correlation functions (CCFs) with distance sorting and optional velocity lines.

    Parameters:
    -----------
    ccf_dict : dict
        Dictionary containing CCF data for each station pair
    ccf_distance : dict
        Dictionary containing inter-station distances
    dt : float
        Sampling interval (seconds)
    time_window : float, optional
        Time range to display around zero lag (default: 1 second)
    vmin : float, optional
        Minimum velocity for reference line (m/s)
    vmax : float, optional
        Maximum velocity for reference line (m/s)
    axes : matplotlib.axes.Axes, optional
        Axes object to plot on (default: creates new figure)
    plot_kwargs : dict, optional
        Additional keyword arguments for plotting CCFs
    """

    # Extract and stack CCF data
    stacked_ccfs = []
    distances = []

    for station_pair, ccf_data in ccf_dict.items():
        # Stack CCF data using linear method
        stacked_data = stack_ccf(ccf_data, method="pws", nu=2.0)
        distance = ccf_distance[station_pair][0]

        stacked_ccfs.append(stacked_data)
        distances.append(distance)

    # Convert to numpy arrays and sort by distance
    ccfs_array = np.r_[stacked_ccfs]
    distance_array = np.array(distances)

    # Sort CCFs based on inter-station distances
    sorted_indices = np.argsort(distance_array)
    ccfs_array = ccfs_array[sorted_indices]
    distance_array = distance_array[sorted_indices]

    # Ensure odd length for symmetric time axis
    ccfs_array = ensure_odd_length(ccfs_array)

    # Create symmetric time axis
    half_length = ccfs_array.shape[1] // 2
    time_axis = np.arange(-half_length, half_length + 1) * dt

    # Trim data to only show requested time window
    time_mask = (time_axis >= -time_window) & (time_axis <= time_window)
    trimmed_ccfs = ccfs_array[:, time_mask]
    trimmed_time = time_axis[time_mask]

    # Generate time vectors for velocity lines
    positive_time = np.arange(0, time_window, dt)

    # Calculate velocity lines if parameters provided
    vmin_line = vmin * positive_time if vmin is not None else None
    vmax_line = vmax * positive_time if vmax is not None else None

    # Create figure and axes if not provided
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 7))

    # Set default plot parameters if not provided
    if plot_kwargs is None:
        plot_kwargs = {}

    # Plot each CCF with distance-based vertical positioning
    scaling_factor = 5
    for i in range(trimmed_ccfs.shape[0]):
        axes.plot(
            trimmed_time,
            trimmed_ccfs[i] * scaling_factor + distance_array[i],
            color="k",
            **plot_kwargs,
        )

    # Add velocity reference lines if provided
    if vmin_line is not None:
        axes.plot(
            positive_time,
            vmin_line,
            color="r",
            ls="--",
            lw=1.5,
            label=f"Vmin: {vmin} m/s",
        )
        axes.plot(
            -positive_time,
            vmin_line,
            color="r",
            ls="--",
            lw=1.5,
            # label=f"Vmin: {vmin} m/s",
        )

    if vmax_line is not None:
        axes.plot(
            positive_time,
            vmax_line,
            color="b",
            ls="--",
            lw=1.5,
            label=f"Vmax: {vmax} m/s",
        )
        axes.plot(
            -positive_time,
            vmax_line,
            color="b",
            ls="--",
            lw=1.5,
            # label=f"Vmax: {vmax} m/s",
        )

    # Configure plot appearance
    axes.set_xlabel("Lag time (s)", fontsize=24)
    axes.set_ylabel("Interstation distance (m)", fontsize=24)
    axes.set_xlim(-time_window, time_window)
    axes.set_ylim(distance_array.min() * 0.5, distance_array.max() * 1.05)
    axes.grid(ls=":", lw=1.5, color="#AAAAAA")

    # Add legend with styling
    axes.legend(
        fontsize=12,
        markerscale=1.5,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.8,
    )

    axes.tick_params(labelsize=18)
    plt.tight_layout()


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
