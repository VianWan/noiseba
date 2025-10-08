import numpy as np

def plot_phase_velocity_curves(x_values, curves, axes=None, unit='km', x_axis='PER', stride=1, plot_kwargs=None):
    r"""
    Plot phase velocity curves against period or frequency.
    
    Parameters:
    -----------
    curves (np.ndarray): 
        2D array of phase velocity curves. Each row corresponds to a curve.
    x_values (np.ndarray): 
        1D array of period or frequency values.
    axes (matplotlib.axes.Axes, optional): 
        Axes to plot on. If None, a new figure and axes will be created.
    unit (str): 
        Unit of measurement for phase velocity. Default is 'km'. Use 'm' for meters.
    x_axis (str): 
        Type of x-axis. Can be 'PRE' (default) or 'FREQ'.
    stride (int): 
        Step size for selecting curves to plot. Default is 1 (plot all curves).
    plot_kwargs (dict, optional): 
        Dictionary of keyword arguments for customizing the plot (e.g., line style, color, thickness).
                                 Default is None, which uses default settings.
    """
    # Lazy load matplotlib.pyplot
    global plt
    if 'plt' not in globals():
        import matplotlib.pyplot as plt

    # Create a new figure and axes if not provided
    if axes is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        ax = axes

    # Convert phase velocity to meters if unit is 'm'
    if unit == 'm':
        curves = curves * 1000

    # Ensure curves is a 2D array
    curves = np.atleast_2d(curves)

    # Default plot settings
    default_kwargs = {'ls': '-', 'lw': 1, 'alpha': 1.0}
    if plot_kwargs is not None:
        default_kwargs.update(plot_kwargs)

    # Plot curves with stride
    for i in range(0, len(curves), stride):
        ax.plot(x_values, curves[i], **default_kwargs)

    # Set x-axis label based on x_axis parameter
    if x_axis == 'FREQ':
        ax.set_xlabel('Frequency (Hz)', fontsize=16)
    elif x_axis == 'PER':
        ax.set_xlabel('Period (s)', fontsize=16)
    else:
        raise ValueError("x_axis must be 'FREQ or PER (e.g., frequency' or 'period'.")

    # Set y-axis label
    ax.set_ylabel(f'Phase velocity ({unit}/s)', fontsize=16)

    # Set tick label size
    ax.tick_params(axis='both', labelsize=14)

    return ax