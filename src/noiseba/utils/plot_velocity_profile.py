import numpy as np
# import matplotlib.pyplot as plt

def plot_velocity_profile(velocities, thickness, axes=None, unit='km', stride=1, plot_kwargs=None):
    r"""
    Plot a velocity profile against depth using step function.
    
    Parameters:
    -----------
    velocities (list or np.ndarray):
        List or array of velocities for each layer.
    thickness (list or np.ndarray): 
        List or array containing the top and bottom depths for each layer.
    axes (matplotlib.axes.Axes, optional): 
        Axes to plot on. If None, a new figure and axes will be created.
    unit (str): 
        Unit of measurement for velocity and depth. Default is 'km'. Use 'm' for meters.
    stride (int): 
        Step size for selecting models to plot. Default is 1 (plot all models).
    plot_kwargs (dict, optional): 
        Dictionary of keyword arguments for customizing the plot (e.g., line color, thickness).
                                 Default is None, which uses default settings.
    """
    # Lazy load matplotlib.pyplot
    global plt
    if 'plt' not in globals():
        import matplotlib.pyplot as plt

    # Create a new figure and axes if not provided
    if axes is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = axes

    # Convert velocities and thickness to numpy arrays
    velocities = np.array(velocities)
    thickness = np.array(thickness)

    # Convert velocities and thickness to meters if unit is 'm'
    if unit == 'm':
        velocities = velocities * 1000
        thickness = thickness * 1000

    # Ensure velocities and thickness are 2D arrays
    velocities = np.atleast_2d(velocities)
    thickness = np.atleast_2d(thickness)

    # Default plot settings
    default_kwargs = {'color': '#8174A0', 'ls': '-', 'lw': 3, 'alpha': 1.0}
    if plot_kwargs is not None:
        default_kwargs.update(plot_kwargs)

    # Iterate over each model with stride
    for i in range(0, len(velocities), stride):
        model_velocities = velocities[i]
        model_thickness = thickness[i]

        # Calculate cumulative depths to use for plotting
        depth_intervals = np.cumsum(np.r_[0, model_thickness[:-1], 2 * model_thickness[-2]])

        # Plot Velocity vs Depth for the current model
        for ind in range(len(model_velocities)):
            # Only add label for the first segment
            if ind == 0 and 'label' in default_kwargs:
                ax.plot([model_velocities[ind], model_velocities[ind]],
                        [depth_intervals[ind], depth_intervals[ind+1]],
                        **default_kwargs)
            else:
                # Avoid adding label for other segments
                ax.plot([model_velocities[ind], model_velocities[ind]],
                        [depth_intervals[ind], depth_intervals[ind+1]],
                        **{**default_kwargs, 'label': '_nolegend_'})

            if ind < len(model_velocities) - 1:
                ax.plot([model_velocities[ind], model_velocities[ind + 1]],
                        [depth_intervals[ind+1], depth_intervals[ind+1]],
                        **{**default_kwargs, 'label': '_nolegend_'})

    # Set plot labels and title
    ax.set_xscale('linear')
    ax.set_xlabel(f'Vs ({unit}/s)', fontsize=28)
    ax.set_ylabel(f'Depth ({unit})', fontsize=28)
    ax.set_title('Velocity Profiles', fontsize=28)
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=28)

    # Set margins
    margin = 0.1
    ax.margins(x=margin,)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)
    
    return ax