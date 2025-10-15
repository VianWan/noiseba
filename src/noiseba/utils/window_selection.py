"""
Window selection utilities for noise-based seismic analysis.

This module provides functions for selecting optimal time windows in cross-correlation
functions (CCFs) based on signal-to-noise ratio (SNR) analysis and energy distribution.
The main functionality includes:

- Energy window extraction for signal and noise components
- SNR calculation and optimization
- Time segment selection based on asymmetry ratios
- CCF stacking with SNR-based weighting

Date: 2025
"""

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from noiseba.preprocessing import stack_ccf


def ensure_odd_length(data: Union[np.ndarray, list]) -> np.ndarray:
    """
    Ensure input data has odd length for symmetric processing.
    
    For 1D arrays: removes the last element if length is even
    For 2D arrays: removes the last column if number of columns is even
    This ensures proper centering for time-domain operations.
    
    Parameters
    ----------
    data : array-like
        Input data (1D or 2D array)
        
    Returns
    -------
    np.ndarray
        Data with odd length/dimensions
        
    Raises
    ------
    NotImplementedError
        If input data is not 1D or 2D
    """
    data = np.asarray(data)
    
    if data.ndim == 1:
        # For 1D data, ensure odd number of samples
        if len(data) % 2 == 0 and len(data) > 0:
            return data[:-1]
        return data
    
    elif data.ndim == 2:
        # For 2D data, ensure odd number of columns (time samples)
        rows, cols = data.shape
        if cols % 2 == 0 and cols > 0:
            return data[:, :-1]
        return data
    
    else:
        raise NotImplementedError("Input data must be 1D or 2D")


def _create_time_windows(
    ccf: np.ndarray, 
    dt: float, 
    dist: float, 
    vmin: float, 
    vmax: float,
    window_type: str = "signal"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time windows for signal or noise extraction.
    
    Internal helper function that creates time windows based on velocity bounds
    and applies appropriate windowing functions.
    
    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation function data
    dt : float
        Time sampling interval (seconds)
    dist : float
        Inter-station distance (meters)
    vmin, vmax : float
        Minimum and maximum velocity bounds (m/s)
    window_type : str
        Type of window: "signal" or "noise"
        
    Returns
    -------
    window_neg, window_pos : np.ndarray
        Negative and positive time windows
    times : np.ndarray
        Time vector corresponding to the windows
    """
    # Ensure odd length for proper centering
    ccf = np.asarray(ccf, dtype=float)
    ccf = ensure_odd_length(ccf)
    
    # Determine array dimensions and create time vector
    if ccf.ndim == 1:
        N = ccf.shape[0] // 2
    elif ccf.ndim == 2:
        N = ccf.shape[1] // 2
    else:
        raise ValueError("Input array must be 1D or 2D")
    
    times = np.arange(-N, N + 1) * dt
    
    # Calculate time bounds based on velocity
    if window_type == "signal":
        tmin = dist / vmax
        tmax = dist / vmin
        # Signal windows: between tmin and tmax
        idx_pos = (times >= tmin) & (times <= tmax)
        idx_neg = (times <= -tmin) & (times >= -tmax)
    else:  # noise
        tmax = dist / vmin
        # Noise windows: outside of tmax
        idx_pos = times >= tmax
        idx_neg = times <= -tmax
    
    # Initialize windows
    window_pos = np.zeros_like(times)
    window_neg = np.zeros_like(times)
    
    # Apply windowing functions
    if np.any(idx_pos):
        if window_type == "signal":
            window_pos[idx_pos] = np.hanning(np.sum(idx_pos))
        else:
            window_pos[idx_pos] = np.hanning(np.sum(idx_pos))
    
    if np.any(idx_neg):
        if window_type == "signal":
            window_neg[idx_neg] = np.hanning(np.sum(idx_neg))
        else:
            window_neg[idx_neg] = np.hanning(np.sum(idx_neg))
    
    return window_neg, window_pos, times


def energy_window(
    ccf: np.ndarray, 
    dt: float, 
    dist: float, 
    vmin: float, 
    vmax: float, 
    window_type: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signal energy within velocity-defined time windows.
    
    Creates time windows based on surface wave velocity bounds and extracts
    the corresponding signal components from the cross-correlation function.
    
    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation function data (1D or 2D array)
    dt : float
        Time sampling interval (seconds)
    dist : float
        Inter-station distance (meters)
    vmin, vmax : float
        Minimum and maximum velocity bounds for surface waves (m/s)
    window_type : str
        Window function type: "hann" (default) or "rectangle"
        
    Returns
    -------
    energy_neg, energy_pos : np.ndarray
        Windowed signal components for negative and positive time lags
    """
    # Create signal windows
    window_neg, window_pos, _ = _create_time_windows(ccf, dt, dist, vmin, vmax, "signal")
    
    # Apply windowing based on type
    if window_type == "rectangle":
        window_neg = (window_neg > 0).astype(float)
        window_pos = (window_pos > 0).astype(float)
    
    # Extract windowed signals
    energy_neg = np.zeros_like(ccf)
    energy_pos = np.zeros_like(ccf)
    
    if np.any(window_neg > 0):
        energy_neg = window_neg * ccf
    if np.any(window_pos > 0):
        energy_pos = window_pos * ccf
    
    return energy_neg, energy_pos


def noise_window(
    ccf: np.ndarray, 
    dt: float, 
    dist: float, 
    vmin: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract noise components outside signal windows.
    
    Creates time windows outside the surface wave arrival times to capture
    noise components for SNR calculation.
    
    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation function data (1D or 2D array)
    dt : float
        Time sampling interval (seconds)
    dist : float
        Inter-station distance (meters)
    vmin : float
        Minimum velocity bound (m/s) - defines noise window start
        
    Returns
    -------
    noise_neg, noise_pos : np.ndarray
        Windowed noise components for negative and positive time lags
    """
    # Create noise windows
    window_neg, window_pos, _ = _create_time_windows(ccf, dt, dist, vmin, vmin, "noise")
    
    # Extract windowed noise
    noise_neg = np.zeros_like(ccf)
    noise_pos = np.zeros_like(ccf)
    
    if np.any(window_neg > 0):
        noise_neg = window_neg * ccf
    if np.any(window_pos > 0):
        noise_pos = window_pos * ccf
    
    return noise_neg, noise_pos


def snr(signal: np.ndarray, noise: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate Signal-to-Noise Ratio (SNR) for 1D or 2D data.
    
    Computes the power ratio between signal and noise components.
    For 2D arrays, calculates SNR along each row (time segment).
    
    Parameters
    ----------
    signal : array-like
        Signal data (1D or 2D array)
    noise : array-like
        Noise data (1D or 2D array, same shape as signal)
        
    Returns
    -------
    float or np.ndarray
        SNR value(s). For 1D input returns scalar, for 2D input returns array
        with SNR for each time segment
        
    Raises
    ------
    ValueError
        If input arrays are not 1D or 2D
    """
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    
    # Validate input dimensions
    if signal.shape != noise.shape:
        raise ValueError("Signal and noise arrays must have the same shape")
    
    if signal.ndim == 1:
        # 1D case: single SNR value
        signal_power = np.sum(signal**2)
        noise_power = np.sum(noise**2)
        if noise_power == 0:
            return np.inf
        return signal_power / noise_power
    
    elif signal.ndim == 2:
        # 2D case: SNR for each time segment (row)
        signal_power = np.sum(signal**2, axis=1)
        noise_power = np.sum(noise**2, axis=1)
        # Avoid division by zero
        mask = noise_power > 0
        snr_values = np.full(signal_power.shape, np.inf)
        snr_values[mask] = signal_power[mask] / noise_power[mask]
        return snr_values
    
    else:
        raise ValueError("Input arrays must be 1D or 2D")


def calculate_snr_ratios(
    ccf_dict: Dict[str, np.ndarray], 
    ccf_distance: Dict[str, Tuple], 
    dt: float, 
    vmin: float = 200, 
    vmax: float = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate average SNR ratios for negative and positive time windows across all CCFs.
    
    Computes weighted average SNR values for all station pairs, with distance-based
    weighting to account for amplitude decay with distance.
    
    Parameters
    ----------
    ccf_dict : dict
        Dictionary containing CCF data for each station pair
        Keys: station pair names (e.g., "STA1_STA2")
        Values: CCF data arrays (n_segments, n_lags)
    ccf_distance : dict
        Dictionary containing distances for each station pair
        Values: (distance, x1, y1, x2, y2) tuples
    dt : float
        Time sampling interval (seconds)
    vmin, vmax : float
        Minimum and maximum velocity bounds for windowing (m/s)
        
    Returns
    -------
    snr_neg_ave, snr_pos_ave, acr : np.ndarray
        Average SNR for negative/positive windows and their asymmetry ratio (ACR)
        ACR = snr_neg_ave / snr_pos_ave
    """
    # Validate input dictionaries
    if not ccf_dict or not ccf_distance:
        raise ValueError("Input dictionaries cannot be empty")
    
    # Get dimensions from first CCF
    first_key = next(iter(ccf_dict))
    if first_key not in ccf_distance:
        raise ValueError("Station pair missing from distance dictionary")
    
    m, n = ccf_dict[first_key].shape
    
    # Initialize output arrays
    snr_neg_ave = np.zeros(m)
    snr_pos_ave = np.zeros(m)
    
    # Calculate SNR for each CCF with distance weighting
    for ccf_name, ccf_data in ccf_dict.items():
        if ccf_name not in ccf_distance:
            continue
            
        dist = ccf_distance[ccf_name][0]
        
        # Extract signal and noise windows
        signal_neg, signal_pos = energy_window(ccf_data, dt, dist, vmin, vmax)
        noise_neg, noise_pos = noise_window(ccf_data, dt, dist, vmin)
        
        # Calculate SNR with distance weighting (amplitude decay correction)
        snr_neg = snr(signal_neg, noise_neg)
        snr_pos = snr(signal_pos, noise_pos)
        
        # Apply distance weighting (1/sqrt(dist) for amplitude decay)
        distance_weight = 1.0 / np.sqrt(max(dist, 1e-6))  # Avoid division by zero
        snr_neg_weight = snr_neg * distance_weight
        snr_pos_weight = snr_pos * distance_weight
        
        # Accumulate weighted SNRs
        snr_neg_ave += snr_neg_weight
        snr_pos_ave += snr_pos_weight
    
    # Average across all CCFs
    num_ccfs = len(ccf_dict)
    if num_ccfs > 0:
        snr_neg_ave /= num_ccfs
        snr_pos_ave /= num_ccfs
    
    # Calculate asymmetry ratio (ACR)
    # Avoid division by zero
    acr = np.full_like(snr_neg_ave, np.inf)
    mask = snr_pos_ave > 0
    acr[mask] = snr_neg_ave[mask] / snr_pos_ave[mask]
    
    return snr_neg_ave, snr_pos_ave, acr


def select_optimal_segments(
    acr: np.ndarray,
    snr_neg: Optional[np.ndarray] = None,
    snr_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select optimal time segments based on SNR asymmetry ratio (ACR).
    
    Segments are classified as negative-dominant (ACR > 1) or positive-dominant (ACR < 1).
    Within each category, segments are ranked by quality score.
    
    Parameters
    ----------
    acr : np.ndarray
        Asymmetry ratio array (negative SNR / positive SNR)
    snr_neg, snr_pos : np.ndarray, optional
        SNR arrays for negative and positive windows (if provided, used for ranking)
        
    Returns
    -------
    ind_opt_ccf_neg, ind_opt_ccf_pos : np.ndarray
        Indices of optimal segments for negative and positive dominant components,
        sorted by quality (best first)
    """
    # Separate negative and positive dominant segments
    ind_acr_neg = np.where(acr > 1)[0]  # indices of segments for acausal CCF
    ind_acr_pos = np.where(acr < 1)[0]  # indices of segments for causal CCF
    
    # Calculate quality scores and sort segments
    if snr_neg is not None and snr_pos is not None:
        # Use combined SNR and ACR score
        score_neg = snr_neg[ind_acr_neg] * acr[ind_acr_neg]
        score_pos = (1.0 / acr[ind_acr_pos]) * snr_pos[ind_acr_pos]
        
        # Sort by score (descending order, best first)
        ind_neg = np.argsort(score_neg)[::-1]
        ind_pos = np.argsort(score_pos)[::-1]
    else:
        # Use ACR only for ranking
        ind_neg = np.argsort(acr[ind_acr_neg])[::-1]
        ind_pos = np.argsort(1.0 / acr[ind_acr_pos])[::-1]
    
    # Return sorted indices
    ind_opt_ccf_neg = ind_acr_neg[ind_neg]
    ind_opt_ccf_pos = ind_acr_pos[ind_pos]
    
    return ind_opt_ccf_neg, ind_opt_ccf_pos


def calculate_segment_snr(
    ccf_dict: Dict[str, np.ndarray],
    ccf_distance: Dict[str, Tuple],
    selected_indices: np.ndarray,
    window_type: str,
    dt: float,
    vmin: float,
    vmax: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate SNR for selected time segments and prepare data for stacking.
    
    For each selected time segment, calculates the cumulative SNR across all
    station pairs and prepares the segment data for optimal stacking.
    
    Parameters
    ----------
    ccf_dict : dict
        Dictionary containing CCF data for each station pair
    ccf_distance : dict
        Dictionary containing distances for each station pair
    selected_indices : np.ndarray
        Indices of selected time segments
    window_type : str
        Window type: 'neg' for negative window, 'pos' for positive window
    dt, vmin, vmax : float
        Processing parameters (time step, velocity bounds)
        
    Returns
    -------
    snr_values : np.ndarray
        SNR values for each selected segment
    segment_data : dict
        Dictionary containing selected segment data for each station pair
    """
    # Validate inputs
    if window_type not in ['neg', 'pos']:
        raise ValueError("window_type must be 'neg' or 'pos'")
    
    if not ccf_dict or not ccf_distance:
        raise ValueError("Input dictionaries cannot be empty")
    
    # Get dimensions from first CCF
    first_key = next(iter(ccf_dict))
    _, n = ccf_dict[first_key].shape
    
    # Initialize output arrays
    n_segments = len(selected_indices)
    snr_values = np.zeros(n_segments)
    segment_data = {
        key: np.zeros((n_segments, n)) for key in ccf_dict.keys()
    }
    
    # Process each segment efficiently

    for seg_idx, time_idx in enumerate(selected_indices):
        total_snr = 0.0
        stack_data = np.zeros(n)
        
        # Accumulate data from all station pairs
        for ccf_name, ccf_data in ccf_dict.items():
            if ccf_name not in ccf_distance:
                continue
                
            dist = ccf_distance[ccf_name][0]
            segment_ccf = ccf_data[time_idx, :]
            
            # Accumulate for stacking
            stack_data += segment_ccf
            segment_data[ccf_name][seg_idx, :] = segment_ccf
            # stack_data = np.mean(segment_data[ccf_name], axis=0)
            
        # Calculate SNR for this segment using stacked data
        if np.any(np.abs(stack_data) > 0):  # Avoid processing zero data
            dist_list = [ccf_distance[key][0] for key in ccf_dict.keys() if key in ccf_distance]
            if dist_list:
                avg_dist = float(np.mean(dist_list))
                
                # Extract signal and noise based on window type
                if window_type == "neg":
                    signal, _ = energy_window(stack_data, dt, avg_dist, vmin, vmax)
                    noise, _ = noise_window(stack_data, dt, avg_dist, vmin)
                else:  # pos
                    _, signal = energy_window(stack_data, dt, avg_dist, vmin, vmax)
                    _, noise = noise_window(stack_data, dt, avg_dist, vmin)
                
                # Calculate segment SNR with distance weighting
                segment_snr = snr(signal, noise) / np.sqrt(max(avg_dist, 1e-6))
                total_snr = segment_snr
                snr_values[seg_idx] = total_snr

        
        # Accumulate data from all station pairs
    # stack_data = np.zeros(n)
    # for seg_idx, time_idx in enumerate(selected_indices):
    #     total_snr = 0.0
    #     for ccf_name, ccf_data in ccf_dict.items():
    #         if ccf_name not in ccf_distance:
    #             continue
                
    #         dist = ccf_distance[ccf_name][0]
    #         segment_ccf = ccf_data[time_idx, :]
            
    #         # Accumulate for stacking
    #         stack_data += segment_ccf
    #         segment_data[ccf_name][seg_idx, :] = segment_ccf
    #         stack_data = np.mean(segment_data[ccf_name], axis=0)
        
    #         # Extract signal and noise based on window type
    #         if window_type == "neg":
    #             signal, _ = energy_window(stack_data, dt, dist, vmin, vmax)
    #             noise, _ = noise_window(stack_data, dt, dist, vmin)
    #         else:  # pos
    #             _, signal = energy_window(stack_data, dt, dist, vmin, vmax)
    #             _, noise = noise_window(stack_data, dt, dist, vmin)
            
    #         # Calculate segment SNR with distance weighting
    #         segment_snr = snr(signal, noise) / np.sqrt(max(dist, 1e-6))
    #         total_snr += segment_snr
        
    #     snr_values[seg_idx] = total_snr
    
    return snr_values, segment_data


def stack_selected_segments(
    segment_data: Dict[str, np.ndarray], 
    max_idx: int, 
    window: np.ndarray, 
    stack_method: str = "pws"
) -> Dict[str, np.ndarray]:
    """
    Stack selected segments using specified method and apply windowing.
    
    Stacks the best segments (up to max_idx) using linear or phase-weighted
    stacking, then applies the specified time window.
    
    Parameters
    ----------
    segment_data : dict
        Dictionary containing segment data for each station pair
        Shape: (n_segments, n_lags)
    max_idx : int
        Index of best segment (inclusive) - stack segments 0 to max_idx
    window : np.ndarray
        Time window to apply (same length as CCF)
    stack_method : str
        Stacking method: 'pws' for phase-weighted stacking, 'linear' for simple average
        
    Returns
    -------
    stacked_data : dict
        Dictionary containing stacked and windowed data for each station pair
    """
    # Validate inputs
    if max_idx < 0:
        raise ValueError("max_idx must be non-negative")
    
    if stack_method not in ['linear', 'pws']:
        raise ValueError("stack_method must be 'linear' or 'pws'")
    
    stacked_data = {}
    
    for key, data in segment_data.items():
        # Validate array dimensions
        if data.ndim != 2:
            raise ValueError(f"Segment data for {key} must be 2D array")
        
        n_segments, n_lags = data.shape
        
        # Ensure max_idx is within bounds
        valid_max_idx = min(max_idx, n_segments - 1)
        if valid_max_idx < 0:
            # Handle case with no valid segments
            stacked_data[key] = np.zeros(n_lags)
            continue
        
        # Select segments up to and including the best one
        selected_data = data[:valid_max_idx + 1, :]
        
        # Apply windowing to each segment
        windowed_data = selected_data * window
        
        # Stack using specified method
        stacked = stack_ccf(windowed_data, stack_method)
        
        # Normalize to unit maximum amplitude
        max_amp = np.max(np.abs(stacked))
        if max_amp > 0:
            stacked_data[key] = stacked / max_amp
        else:
            stacked_data[key] = stacked
    
    return stacked_data


def snr_optimal_select(
    ccf_dict: Dict[str, np.ndarray],
    ccf_distance: Dict[str, Tuple],
    dt: float,
    vmin: float = 200,
    vmax: float = 500,
    signal_truncation: bool = True,
    plot_results: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Main function to select optimal time windows for CCF stacking based on SNR analysis.
    
    Implements a multi-step algorithm:
    1. Calculate SNR ratios for all time segments
    2. Select optimal segments based on asymmetry ratio
    3. Calculate segment-specific SNR values
    4. Stack best segments for final CCF
    
    Parameters
    ----------
    ccf_dict : dict
        Dictionary containing CCF data for each station pair
    ccf_distance : dict
        Dictionary containing distances for each station pair
    dt : float
        Time sampling interval (seconds)
    vmin, vmax : float
        Velocity bounds for surface wave windowing (m/s)
    signal_truncation : bool
        Whether to truncate final CCF to signal windows only
    plot_results : bool
        Whether to plot SNR selection results
        
    Returns
    -------
    ccf_select : dict
        Final selected and stacked CCFs
    sc_neg_select_stack : dict
        Stacked negative window data
    sc_pos_select_stack : dict
        Stacked positive window data
    snr_neg_new : np.ndarray
        SNR values for negative window segments
    snr_pos_new : np.ndarray
        SNR values for positive window segments
    """
    # Step 1: Calculate SNR ratios across all segments
    snr_neg_ave, snr_pos_ave, acr = calculate_snr_ratios(
        ccf_dict, ccf_distance, dt, vmin, vmax
    )
    
    # Step 2: Select optimal segments based on asymmetry ratio
    ind_acr_neg, ind_acr_pos = select_optimal_segments(
        acr, snr_neg_ave, snr_pos_ave
    )
    
    # Step 3: Calculate segment-specific SNR values
    snr_neg_new, sc_neg = calculate_segment_snr(
        ccf_dict, ccf_distance, ind_acr_neg, "neg", dt, vmin, vmax
    )
    snr_pos_new, sc_pos = calculate_segment_snr(
        ccf_dict, ccf_distance, ind_acr_pos, "pos", dt, vmin, vmax
    )
    
    # Step 4: Find best segments (highest SNR)
    if len(snr_neg_new) > 0:
        snr_neg_new_max = int(np.argmax(snr_neg_new))
    else:
        snr_neg_new_max = 0
    
    if len(snr_pos_new) > 0:
        snr_pos_new_max = int(np.argmax(snr_pos_new))
    else:
        snr_pos_new_max = 0
    
    # Step 5: Create time windows for positive/negative components
    first_key = next(iter(ccf_dict))
    _, n = ccf_dict[first_key].shape
    neg_window = np.zeros(n)
    pos_window = np.zeros(n)
    N = n // 2
    t = np.arange(-N, N + 1) * dt
    neg_window[t < 0] = 1.0
    pos_window[t >= 0] = 1.0
    
    # Step 6: Stack selected segments
    sc_neg_select_stack = stack_selected_segments(
        sc_neg, snr_neg_new_max, neg_window, stack_method="pws"
    )
    sc_pos_select_stack = stack_selected_segments(
        sc_pos, snr_pos_new_max, pos_window, stack_method="pws"
    )
    
    # Step 7: Combine negative and positive components
    ccf_select = {}
    for key in sc_neg_select_stack.keys():
        if key in sc_pos_select_stack:
            # Normalize components
            ccf_neg = sc_neg_select_stack[key]
            ccf_pos = sc_pos_select_stack[key]
            
            max_neg = np.max(np.abs(ccf_neg))
            max_pos = np.max(np.abs(ccf_pos))
            
            if max_neg > 0:
                ccf_neg = ccf_neg / max_neg
            if max_pos > 0:
                ccf_pos = ccf_pos / max_pos
            
            # Combine components
            ccf_select[key] = ccf_neg + ccf_pos
    
    # Optional: Truncate to signal windows only
    if signal_truncation:
        for key, data in ccf_select.items():
            if key in ccf_distance:
                dist = ccf_distance[key][0]
                neg, pos = energy_window(data, dt, dist, vmin, vmax)
                ccf_select[key] = neg + pos
    
    # Plot results if requested
    if plot_results:
        try:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot negative window SNR
            if len(snr_neg_new) > 0:
                ax[0].plot(snr_neg_new, "k", linewidth=1.5, label="SNR-")
                ax[0].axvline(snr_neg_new_max, color="b", linestyle="--", 
                             label=f"Best segment: {snr_neg_new_max}")
                ax[0].set_xlabel("Time Segment Index")
                ax[0].set_ylabel("Negative Window SNR")
                ax[0].set_title("Negative Time Window Selection")
                ax[0].legend()
                ax[0].grid(True, alpha=0.3)
            
            # Plot positive window SNR
            if len(snr_pos_new) > 0:
                ax[1].plot(snr_pos_new, "k", linewidth=1.5, label="SNR+")
                ax[1].axvline(snr_pos_new_max, color="b", linestyle="--",
                             label=f"Best segment: {snr_pos_new_max}")
                ax[1].set_xlabel("Time Segment Index")
                ax[1].set_ylabel("Positive Window SNR")
                ax[1].set_title("Positive Time Window Selection")
                ax[1].legend()
                ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Warning: Could not plot results: {e}")
    
    return (
        ccf_select,
        sc_neg_select_stack,
        sc_pos_select_stack,
        snr_neg_new,
        snr_pos_new,
    )


def _energy_symmetry_select(ccf: np.ndarray, dt: float, dist: float, vmin: float, vmax: float, pct: int = 20) -> np.ndarray:
    """
    Select segments based on energy symmetry between positive and negative windows.
    
    Internal helper function that removes segments with extreme energy asymmetry.
    
    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation function data (2D array: n_segments, n_lags)
    dt : float
        Time sampling interval (seconds)
    dist : float
        Inter-station distance (meters)
    vmin, vmax : float
        Velocity bounds for windowing (m/s)
    pct : int
        Percentile threshold for symmetry selection (default: 20)
        
    Returns
    -------
    np.ndarray
        Filtered CCF data with symmetric energy distribution
    """
    # Extract signal windows
    ccf_neg, ccf_pos = energy_window(ccf, dt, dist, vmin, vmax)
    
    # Remove mean to focus on signal variations
    ccf_neg -= np.mean(ccf_neg, axis=1, keepdims=True)
    ccf_pos -= np.mean(ccf_pos, axis=1, keepdims=True)
    
    # Calculate energy in each window
    E_neg = np.sum(ccf_neg**2, axis=1)
    E_pos = np.sum(ccf_pos**2, axis=1)
    
    # Calculate energy ratio (log scale for symmetry)
    energy_ratio = np.log(E_pos / np.maximum(E_neg, 1e-10))  # Avoid division by zero
    
    # Select segments with balanced energy (within percentile threshold)
    cut = np.percentile(np.abs(energy_ratio), pct)
    mask = np.abs(energy_ratio) < cut
    
    return ccf[mask]


def energy_symmetry_select(
    ccf_dict: Dict[str, np.ndarray],
    ccf_dist: Dict[str, Tuple],
    dt: float,
    vmin: float,
    vmax: float,
    pct: int = 20,
    stack_method: str = "pws",
    signal_truncation: bool = True,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Select CCF segments based on energy symmetry for all station pairs.
    
    Applies energy symmetry selection to remove segments with extreme
    positive/negative energy imbalance, then stacks the remaining segments
    with optional normalization and signal truncation.
    
    Parameters
    ----------
    ccf_dict : dict
        Dictionary containing CCF data for each station pair
    ccf_dist : dict
        Dictionary containing distances for each station pair
    dt : float
        Time sampling interval (seconds)
    vmin, vmax : float
        Velocity bounds for windowing (m/s)
    pct : int
        Percentile threshold for symmetry selection (default: 20)
    stack_method : str
        Stacking method: 'pws' for phase-weighted stacking, 'linear' for simple average
    signal_truncation : bool
        Whether to truncate final CCF to signal windows only
    normalize : bool
        Whether to normalize the final stacked CCF to unit maximum amplitude
        
    Returns
    -------
    dict
        Dictionary containing filtered, stacked, and processed CCF data for each station pair
    """
    # Validate inputs
    if not ccf_dict or not ccf_dist:
        raise ValueError("Input dictionaries cannot be empty")
    
    if stack_method not in ['linear', 'pws']:
        raise ValueError("stack_method must be 'linear' or 'pws'")
    
    ccf_select = {}
    
    for key, ccf in ccf_dict.items():
        if key not in ccf_dist:
            # Skip if distance info missing
            continue
            
        # Step 1: Apply energy symmetry selection
        filtered_ccf = _energy_symmetry_select(ccf, dt, ccf_dist[key][0], vmin, vmax, pct)
        
        # Step 2: Stack the filtered segments
        if filtered_ccf.size > 0:  # Check if any segments remain after filtering
            stacked_ccf = stack_ccf(filtered_ccf, method=stack_method)
            
            # Step 3: Apply optional normalization
            if normalize:
                max_amp = np.max(np.abs(stacked_ccf))
                if max_amp > 0:
                    stacked_ccf = stacked_ccf / max_amp
            
            # Step 4: Apply optional signal truncation
            if signal_truncation:
                dist = ccf_dist[key][0]
                neg, pos = energy_window(stacked_ccf, dt, dist, vmin, vmax)
                stacked_ccf = neg + pos
                
            ccf_select[key] = stacked_ccf
        else:
            # If no segments remain after filtering, create zero array
            _, n_lags = ccf.shape
            ccf_select[key] = np.zeros(n_lags)
    
    return ccf_select
