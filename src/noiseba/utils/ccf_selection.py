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

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

from .stack_utils import stack_pws_numba, stack_linear, batch_hilbert_mkl
from .plot_ccf import stack_ccf

MAXWORKERS = 8


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



def _create_signal_windows(
    ccf: np.ndarray, dt: float, dist: Union[float, np.ndarray],
    vmin: float, vmax: float, window: str = "signal", window_type: str = "hann",
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
    window: str
        Ccf part: support "signal" or "noise"
    window_type : str
        Type of window: "hann" or "rectangle"

    Returns
    -------
    window_neg, window_pos : np.ndarray
        Negative and positive time windows
    times : np.ndarray
        Time vector corresponding to the windows
    """
    ccf = np.asarray(ccf, dtype=float)
    dist = np.atleast_1d(dist)

    if ccf.ndim == 1:
        ccf = ccf[np.newaxis, :]
        is_1d_input = True
    else:
        is_1d_input = False

    if ccf.ndim != 2:
        raise ValueError("ccf must be 1D or 2D")

    n_pairs, n_lags = ccf.shape
    assert dist.size in (1, n_pairs), "dist size must be 1 or match n_pairs"
    if dist.size == 1:
        dist = np.full(n_pairs, dist[0])

    N = n_lags // 2
    lag_times = np.arange(-N, N + 1) * dt  # (n_lags,)
    abs_lag = np.abs(lag_times)

    tmin = dist[:, np.newaxis] / vmax  # (n_pairs, 1)
    tmax = dist[:, np.newaxis] / vmin  # (n_pairs, 1)

    match window:
        case "signal":
            mask_neg = (lag_times < 0) & (abs_lag >= tmin) & (abs_lag <= tmax)
            mask_pos = (lag_times > 0) & (abs_lag >= tmin) & (abs_lag <= tmax)
        case "noise":
            mask_neg = (lag_times < 0) & (abs_lag >= tmax)
            mask_pos = (lag_times > 0) & (abs_lag >= tmax)
        case _:
            raise ValueError("Invalid window")

    win_neg = np.zeros_like(ccf, dtype=np.float32)
    win_pos = np.zeros_like(ccf, dtype=np.float32)

    if window_type == "rectangle":
        win_neg = mask_neg.astype(np.float32)
        win_pos = mask_pos.astype(np.float32)

    elif window_type == "hann":
        for i in range(n_pairs):
            len_neg = int(mask_neg[i].sum())
            len_pos = int(mask_pos[i].sum())
            # if len_neg > 0:
            win_neg[i, mask_neg[i]] = np.hanning(len_neg)
            # if len_pos > 0:
            win_pos[i, mask_pos[i]] = np.hanning(len_pos)
    else:
        raise ValueError("Invalid window_type")

    if is_1d_input:
        win_neg = win_neg[0]
        win_pos = win_pos[0]

    return win_neg, win_pos, lag_times


def energy_window(
    ccf: np.ndarray, dt: float, dist: Union[float, np.ndarray],
    vmin: float, vmax: float, window_type: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signal energy within velocity-defined time windows.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation function (1D: single pair, 2D: (n_pairs, n_lags))
    dt : float
        Sampling interval (s)
    dist : float or np.ndarray
        Inter-station distance(s): scalar (1D/2D) or array of length n_pairs (2D only)
    vmin, vmax : float
        Velocity bounds (m/s)
    window_type : str
        "hann" or "rectangle"

    Returns
    -------
    energy_neg, energy_pos : np.ndarray
        Windowed signal (same shape as input ccf)
    """
    ccf = np.asarray(ccf, dtype=float)
    dist = np.atleast_1d(dist)

    if ccf.ndim == 1:
        ccf_2d = ccf[np.newaxis, :]
        is_1d = True
    else:
        ccf_2d = ccf
        is_1d = False

    n_pairs, n_lags = ccf_2d.shape

    if dist.size == 1:
        dist = np.full(n_pairs, dist[0])
    elif dist.size != n_pairs:
        raise ValueError(f"dist must be scalar or length {n_pairs}, got {dist.size}")

    win_neg, win_pos, _ = _create_signal_windows(
        ccf_2d, dt, dist, vmin, vmax, window="signal", window_type=window_type
    )

    energy_neg = win_neg * ccf_2d
    energy_pos = win_pos * ccf_2d

    if is_1d:
        energy_neg = energy_neg[0]
        energy_pos = energy_pos[0]

    return energy_neg, energy_pos


def noise_window(ccf: np.ndarray, dt: float, dist: float, vmin: float) -> Tuple[np.ndarray, np.ndarray]:
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
    window_neg, window_pos, _ = _create_signal_windows(ccf, dt, dist, vmin, vmin, "noise")

    # Extract windowed noise
    noise_neg = np.zeros_like(ccf)
    noise_pos = np.zeros_like(ccf)

    if np.any(window_neg > 0):
        noise_neg = window_neg * ccf
    if np.any(window_pos > 0):
        noise_pos = window_pos * ccf

    return noise_neg, noise_pos

@njit(cache=True, parallel=True)
def snr_numba(signal: np.ndarray, noise: np.ndarray) -> Union[float, np.ndarray]:
    signal = np.asarray(signal)
    noise = np.asarray(noise)

    # Validate input dimensions
    if signal.shape != noise.shape:
        raise ValueError("Signal and noise arrays must have the same shape")

    if signal.ndim == 1:
        # 1D case: single SNR value
        signal_power = np.sqrt(np.mean(signal**2))
        noise_power  = np.sqrt(np.mean(noise**2))
        return np.array([signal_power / max(noise_power, 1e-6)])

    elif signal.ndim == 2:
        n_rows = signal.shape[0]
        snr_values = np.zeros(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            sig_row = signal[i]
            noi_row = noise[i]
            signal_power = np.sqrt(np.mean(sig_row**2))
            noise_power  = np.sqrt(np.mean(noi_row**2))
            if noise_power > 0:
                snr_values[i] = signal_power / noise_power
            else:
                snr_values[i] = 0.0
        return snr_values

    else:
        raise ValueError("Input arrays must be 1D or 2D")


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
        signal_power = np.sqrt(np.mean(signal**2))
        noise_power  = np.sqrt(np.mean(noise**2))
        return signal_power / np.maximum(noise_power, 1e-6)

    elif signal.ndim == 2:
        # 2D case: SNR for each time segment (row)
        signal_power = np.sqrt(np.mean(signal**2, axis=1))
        noise_power  = np.sqrt(np.mean(noise**2, axis=1))
        # Avoid division by zero
        mask = noise_power > 0
        snr_values = np.zeros_like(noise_power)
        snr_values[mask] = signal_power[mask] / noise_power[mask]
        return snr_values

    else:
        raise ValueError("Input arrays must be 1D or 2D")

def calculate_snr_ratios(
    ccf: np.ndarray, dist: np.ndarray, dt: float, vmin: float = 100, vmax: float = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate average SNR ratios for negative and positive time windows.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation data at given station pair.
        - 2D array: (n_pairs, n_lags) for single pair
        - 3D array: (n_pairs, n_segments, n_lags) for multiple pairs
    dist : float or np.ndarray
        Inter-station distance(s).
        - array of shape (n_pairs,) for 2/3D input
    dt : float
        Time sampling interval (seconds)
    vmin, vmax : float
        Minimum and maximum velocity bounds for windowing (m/s)

    Returns
    -------
    snr_neg_ave, snr_pos_ave, acr : np.ndarray
        Average SNR for negative/positive windows and their asymmetry ratio (ACR).
        For 2D input: arrays of shape (n_segments,)
        For 3D input: arrays of shape (n_pairs, n_segments)
    """
    if ccf.ndim == 2:
        ccf = ccf[:, None, :]
    n_pairs, n_segments, n_lags = ccf.shape

    dist = np.asarray(dist, dtype=float)
    if dist.shape[0] != n_pairs:
        raise ValueError("dist length must match n_pairs for 3D input")

    snr_neg_array = np.zeros((n_pairs, n_segments))
    snr_pos_array = np.zeros((n_pairs, n_segments))

    for i in range(n_pairs):
        signal_neg, signal_pos = energy_window(ccf[i], dt, dist[i], vmin, vmax)
        noise_neg, noise_pos = noise_window(ccf[i], dt, dist[i], vmin)

        snr_neg = snr(signal_neg, noise_neg)
        snr_pos = snr(signal_pos, noise_pos)

        weight = 1.0 / np.sqrt(max(dist[i], 1e-6))
        snr_neg_w = snr_neg * weight
        snr_pos_w = snr_pos * weight
        snr_neg_array[i], snr_pos_array[i] = snr_neg_w, snr_pos_w

    snr_neg_ave = np.sum(snr_neg_array, axis=0)
    snr_pos_ave = np.sum(snr_pos_array, axis=0)
    acr = snr_neg_ave / np.maximum(snr_pos_ave, 1e-6)

    return snr_neg_ave, snr_pos_ave, acr



def score_neg_log(acr, snr_neg, snr_pos, eps=1e-6, beta=0.5, gamma=0.5):
    """snr_pos too small → prevent false high"""
    return np.log(acr + eps) + beta * np.log(snr_neg + eps) - gamma / (snr_pos + eps)


def score_pos_log(acr, snr_neg, snr_pos, eps=1e-6, beta=0.3, gamma=0.7):
    """snr_neg too small → prevent false lower"""
    return np.log(1.0 / (acr + eps)) + beta * np.log(snr_pos + eps) - gamma / (snr_neg + eps)


def score_neg_rank(acr, snr_neg, snr_pos):
    """Overall ranking: Avoid differences in numerical ranges"""
    r_acr = np.argsort(np.argsort(acr))
    r_neg = np.argsort(np.argsort(snr_neg))
    r_pos = np.argsort(np.argsort(snr_pos))
    return r_acr + r_neg + (r_pos.max() - r_pos)


def score_pos_rank(acr, snr_neg, snr_pos):
    """Overall ranking: Avoid differences in numerical ranges"""
    r_acr = np.argsort(np.argsort(1.0 / acr))
    r_pos = np.argsort(np.argsort(snr_pos))
    r_neg = np.argsort(np.argsort(snr_neg))
    return r_acr + r_pos + (r_neg.max() - r_neg)


def select_optimal_segments(
    acr: np.ndarray,
    snr_neg: Optional[np.ndarray] = None,
    snr_pos: Optional[np.ndarray] = None,
    method: str = "log",
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
    method : str
        Method for calculating quality scores:
        - "log": log(ACR) + beta * log(SNR_neg) - gamma / SNR_pos
        - "rank": rank(ACR) + rank(SNR_neg) + (rank(SNR_pos).max() - rank(SNR_pos))
        - "acr": only acr

    Returns
    -------
    ind_opt_ccf_neg, ind_opt_ccf_pos : np.ndarray
        Indices of optimal segments for negative and positive dominant components,
        sorted by quality (best first)
    """
    # Separate negative and positive dominant segments
    ind_acr_neg = np.where(acr >= 1)[0]  # indices of segments for acausal CCF
    ind_acr_pos = np.where(acr < 1)[0]  # indices of segments for causal CCF

    # Calculate quality scores and sort segments
    acr_neg = acr[ind_acr_neg]
    acr_pos = acr[ind_acr_pos]

    if snr_neg is not None and snr_pos is not None:
        snr_neg_neg = snr_neg[ind_acr_neg] 
        snr_pos_neg = snr_pos[ind_acr_neg] 

        snr_neg_pos = snr_neg[ind_acr_pos] 
        snr_pos_pos = snr_pos[ind_acr_pos] 

        match method:
            case "log":
                score_neg = score_neg_log(acr_neg, snr_neg_neg, snr_pos_neg)
                score_pos = score_pos_log(acr_pos, snr_neg_pos, snr_pos_pos)
                ind_neg = np.argsort(score_neg)[::-1]
                ind_pos = np.argsort(score_pos)[::-1]
            case "rank":
                score_neg = score_neg_rank(acr_neg, snr_neg_neg, snr_pos_neg)
                score_pos = score_pos_rank(acr_pos, snr_neg_pos, snr_pos_pos)
                ind_neg = np.argsort(score_neg)[::-1]
                ind_pos = np.argsort(score_pos)[::-1]
            case "acr":
                ind_neg = np.argsort(acr[ind_acr_neg])[::-1]
                ind_pos = np.argsort(1.0 / acr[ind_acr_pos])[::-1]
            case _:
                raise ValueError(f"Invalid method: {method}")
    else:
        print(f'{method} needs snr_neg and snr_pos. Switching to acr.')
        ind_neg = np.argsort(acr[ind_acr_neg])[::-1]
        ind_pos = np.argsort(1.0 / acr[ind_acr_pos])[::-1]

    # Return sorted indices
    ind_opt_ccf_neg = ind_acr_neg[ind_neg]
    ind_opt_ccf_pos = ind_acr_pos[ind_pos]

    return ind_opt_ccf_neg, ind_opt_ccf_pos

# -------------------------------------------------
# cumsum_snr is too slow when using pws stacking.
# -------------------------------------------------

def cumsum_snr(
    ccf: np.ndarray,
    dist: Union[float, np.ndarray],
    selected_indices: np.ndarray,
    window: str,
    dt: float,
    vmin: float,
    vmax: float,
    stack_method: str = "pws",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative SNR for selected time segments and return
    the SNR curve plus the stacked CCF corresponding to the maximum SNR.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation data.
        - 3D array: (n_pairs, n_segments, n_lags)
    dist : float or np.ndarray
        Inter-station distance(s).
        - array of shape (n_pairs,) for 3D input
    selected_indices : np.ndarray
        Indices of selected time segments
    window: str
        'neg' for negative window, 'pos' for positive window
    dt, vmin, vmax : float
        Processing parameters (time step, velocity bounds)
    stack_method : str
        Stacking method: 'pws' or 'linear'

    Returns
    -------
    snr_values : np.ndarray
        SNR values for each cumulative stack
    best_stack : np.ndarray
        Stacked CCF corresponding to the maximum SNR
    """
    # Validate window type
    if window not in ("neg", "pos"):
        raise ValueError("window must be 'neg' or 'pos'")

    # Validate stacking method
    if stack_method not in ("pws", "linear"):
        raise ValueError("stack_method must be 'pws' or 'linear'")

    # Select stacking function
    stack_func = stack_pws_numba if stack_method == "pws" else stack_linear

    # Ensure ccf is 3D
    if ccf.ndim == 2:
        ccf = ccf[np.newaxis, :, :]

    n_pairs, n_segments, n_lags = ccf.shape
    dist = np.asarray(dist)
    if dist.shape[0] != n_pairs:
        raise ValueError("dist length must match n_pairs for 3D input")

    n_sel = len(selected_indices)
    snr_pos_array = np.empty(n_sel, dtype=np.float32)
    snr_neg_array = np.empty(n_sel, dtype=np.float32)

    # Pre-select all required segments once
    ccf_sel = ccf[:, selected_indices, :]   # (n_pairs, n_sel, n_lags)

    best_snr = -np.inf
    best_k = -1

    for k in range(n_sel):
        # Use continuous slice to avoid repeated fancy indexing
        subset = ccf_sel[:, :k+1, :]        # view, not copy
        stacked = stack_func(subset)

        # Compute SNR for current stack
        snr_neg, snr_pos, _ = calculate_snr_ratios(stacked, dist, dt, vmin, vmax)
        snr_neg_array[k] = snr_neg
        snr_pos_array[k] = snr_pos

        # Track best SNR
        current_snr = snr_pos if window == "pos" else snr_neg
        if current_snr > best_snr:
            best_snr = current_snr
            best_k = k

    # Final stack corresponding to maximum SNR
    best_stack = stack_pws_numba(ccf_sel[:, :best_k+1, :])
    snr_values = snr_pos_array if window == "pos" else snr_neg_array

    return snr_values, best_stack

# -------------------------------------------------
# The following functions are accelerated by numba
# -------------------------------------------------
def cumsum_snr_pws(
    ccf: np.ndarray,
    dist: np.ndarray,
    selected_indices: np.ndarray,
    window: str,
    dt: float,
    vmin: float,
    vmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incremental cumulative SNR curve + final Phase-Weighted Stack (PWS).

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation data of shape (n_pairs, n_segments, n_lags).
    dist : np.ndarray
        Inter-station distances of shape (n_pairs,).
    selected_indices : np.ndarray
        Indices of selected segments to include in stacking.
    window : str
        'pos' for positive window, 'neg' for negative window.
    dt : float
        Sampling interval.
    vmin, vmax : float
        Velocity bounds for window construction.

    Returns
    -------
    snr_curve : np.ndarray
        Array of cumulative SNR values for each incremental stack.
    best_pws : np.ndarray
        PWS stack corresponding to the maximum SNR.
    """
    # -------- 1. Input validation --------
    if ccf.ndim != 3:
        raise ValueError("ccf must be 3D: (n_pairs, n_segments, n_lags)")
    if window not in ("pos", "neg"):
        raise ValueError("window must be 'pos' or 'neg'")

    n_pairs, n_segments, n_lags = ccf.shape
    dist = np.asarray(dist, dtype=np.float32)
    if dist.shape[0] != n_pairs:
        raise ValueError("dist length must match n_pairs")

    ccf = ccf.astype(np.float32)

    # -------- 2. Precompute signal/noise masks --------
    signal_masks = np.zeros((n_pairs, n_lags), dtype=bool)
    noise_masks  = np.zeros((n_pairs, n_lags), dtype=bool)

    dummy = np.zeros(n_lags, dtype=np.float32)
    is_pos = window == "pos"

    for i in range(n_pairs):
        wn, wp, _ = _create_signal_windows(dummy, dt, dist[i], vmin, vmax, "signal", 'rectangle')
        nn, np_, _ = _create_signal_windows(dummy, dt, dist[i], vmin, vmin, "noise", 'rectangle')

        if is_pos:
            signal_masks[i] = wp > 0
            noise_masks[i]  = np_ > 0
        else:
            signal_masks[i] = wn > 0
            noise_masks[i]  = nn > 0

    # -------- 3. Initialize accumulators --------
    sum_ccf       = np.zeros((n_pairs, n_lags), dtype=np.float32)
    sum_unit_re   = np.zeros((n_pairs, n_lags), dtype=np.float32)
    sum_unit_im   = np.zeros((n_pairs, n_lags), dtype=np.float32)
    count         = 0

    snr_weighted_sum = np.zeros(n_pairs, dtype=np.float32)  # cumulative weighted SNR per pair
    snr_curve = []

    # Pre-select all required segments once (contiguous slice)
    selected_ccf = np.ascontiguousarray(ccf[:, selected_indices, :])
    n_sel = len(selected_indices)

    best_snr = -np.inf
    best_idx = 0

    # -------- 4. Incremental stacking loop --------
    for idx in range(n_sel):
        x = selected_ccf[:, idx, :]  # (n_pairs, n_lags)

        # Hilbert transform + unit complex vectors
        a = batch_hilbert_mkl(x)  # (n_pairs, n_lags), complex64
        mag = np.abs(a)
        u = np.where(mag > 1e-12, a / mag, 0j)

        # Update accumulators
        sum_ccf     += x
        sum_unit_re += np.real(u)
        sum_unit_im += np.imag(u)
        count += 1

        # Current PWS
        S = sum_ccf / count
        C = np.sqrt(sum_unit_re**2 + sum_unit_im**2) / count
        P = np.power(C, np.float32(2.0)) * S  # power=2.0

        # Update SNR selectively
        update_snr_selective(P, signal_masks, noise_masks, dist, snr_weighted_sum)

        # Average SNR across pairs
        current_snr = np.sum(snr_weighted_sum) / count
        snr_curve.append(current_snr)

        # Track best stack
        if current_snr > best_snr:
            best_snr = current_snr
            best_idx = idx

    best_pws = stack_pws_numba(selected_ccf[:, :best_idx+1, :], power=2.0)
    return np.array(snr_curve, dtype=np.float32), best_pws


def cumsum_snr_linear(
    ccf: np.ndarray,
    dist: np.ndarray,
    selected_indices: np.ndarray,
    window: str,
    dt: float,
    vmin: float,
    vmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incremental cumulative SNR curve + final linear stack.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation data of shape (n_pairs, n_segments, n_lags).
    dist : np.ndarray
        Inter-station distances of shape (n_pairs,).
    selected_indices : np.ndarray
        Indices of selected segments to include in stacking.
    window : str
        'pos' for positive window, 'neg' for negative window.
    dt : float
        Sampling interval.
    vmin, vmax : float
        Velocity bounds for window construction.

    Returns
    -------
    snr_curve : np.ndarray
        Array of cumulative SNR values for each incremental stack.
    best_stack : np.ndarray
        Linear stack corresponding to the maximum SNR.
    """
    # -------- 1. Input validation --------
    if ccf.ndim != 3:
        raise ValueError("ccf must be 3D: (n_pairs, n_segments, n_lags)")
    if window not in ("pos", "neg"):
        raise ValueError("window must be 'pos' or 'neg'")

    n_pairs, n_segments, n_lags = ccf.shape
    dist = np.asarray(dist, dtype=np.float32)
    if dist.shape[0] != n_pairs:
        raise ValueError("dist length must match n_pairs")

    ccf = ccf.astype(np.float32)

    # -------- 2. Precompute signal/noise masks --------
    signal_masks = np.zeros((n_pairs, n_lags), dtype=bool)
    noise_masks  = np.zeros((n_pairs, n_lags), dtype=bool)

    dummy = np.zeros(n_lags, dtype=np.float32)
    is_pos = window == "pos"

    for i in range(n_pairs):
        wn, wp, _ = _create_signal_windows(dummy, dt, dist[i], vmin, vmax, "signal")
        nn, np_, _ = _create_signal_windows(dummy, dt, dist[i], vmin, vmin, "noise")

        if is_pos:
            signal_masks[i] = wp > 0
            noise_masks[i]  = np_ > 0
        else:
            signal_masks[i] = wn > 0
            noise_masks[i]  = nn > 0

    # -------- 3. Initialize accumulators --------
    sum_ccf       = np.zeros((n_pairs, n_lags), dtype=np.float32)
    count         = 0

    snr_weighted_sum = np.zeros(n_pairs, dtype=np.float32)  # cumulative weighted SNR per pair
    snr_curve = []

    # Pre-select all required segments once (contiguous slice)
    selected_ccf = np.ascontiguousarray(ccf[:, selected_indices, :])
    n_sel = len(selected_indices)

    best_snr = -np.inf
    best_idx = 0

    # -------- 4. Incremental stacking loop --------
    for idx in range(n_sel):
        x = selected_ccf[:, idx, :]  # (n_pairs, n_lags)

        # Incremental update
        sum_ccf += x
        count += 1
        linear_stack = sum_ccf / count

        # Update SNR selectively
        update_snr_selective(linear_stack, signal_masks, noise_masks, dist, snr_weighted_sum)

        # Average SNR across pairs
        current_snr = np.sum(snr_weighted_sum) / count
        snr_curve.append(current_snr)

        # Track best stack index
        if current_snr > best_snr:
            best_snr = current_snr
            best_idx = idx

    # -------- 5. Final best stack --------
    best_subset = selected_ccf[:, :best_idx+1, :]  # (n_pairs, best_k+1, n_lags)
    best_stack = stack_pws_numba(best_subset)

    return np.array(snr_curve, dtype=np.float32), best_stack

@njit(fastmath=True, parallel=True)
def update_snr_selective(
    pws: np.ndarray,
    signal_masks: np.ndarray,
    noise_masks: np.ndarray,
    dist: np.ndarray,
    snr_sum: np.ndarray
):
    n_pairs, n_lags = pws.shape
    for i in prange(n_pairs):
        sig = pws[i][signal_masks[i]]
        noi = pws[i][noise_masks[i]]

        rms_sig = np.sqrt(np.mean(sig**2)) if len(sig) > 0 else 0.0
        rms_noi = np.sqrt(np.mean(noi**2)) if len(noi) > 0 else 1e-6

        snr = rms_sig / rms_noi
        weight = 1.0 / np.sqrt(max(dist[i], 1e-6))
        snr_sum[i] += snr * weight

# -------------------------------------------
# Accelerated Cumsum SNR
# -------------------------------------------
def cumsum_snr_numba(
    ccf: np.ndarray,
    dist: np.ndarray,
    selected_indices: np.ndarray,
    window: str,
    dt: float,
    vmin: float,
    vmax: float,
    method: str = 'pws'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incremental cumulative SNR curve + final linear stack.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation data of shape (n_pairs, n_segments, n_lags).
    dist : np.ndarray
        Inter-station distances of shape (n_pairs,).
    selected_indices : np.ndarray
        Indices of selected segments to include in stacking.
    window : str
        'pos' for positive window, 'neg' for negative window.
    dt : float
        Sampling interval.
    vmin, vmax : float
        Velocity bounds for window construction.
    method : str
        Stacking method: 'pws' for PWS, 'linear' for linear.

    Returns
    -------
    snr_curve : np.ndarray
        Array of cumulative SNR values for each incremental stack.
    best_stack : np.ndarray
        Linear stack corresponding to the maximum SNR.
    """
    
    try:
        match method:
            case "pws":
                return cumsum_snr_pws(
                    ccf, dist, selected_indices, window, dt, vmin, vmax
                )
            case "linear":
                return cumsum_snr_linear(
                    ccf, dist, selected_indices, window, dt, vmin, vmax
                )
            case _:
                raise ValueError("Invalid method")
    except Exception as e:
        raise ValueError(f"Error in cumsum_snr_numba: {e}")


# ------------------------------
# Will be removed
# ------------------------------
def stack_selected_segments(
    segment_data: Dict[str, np.ndarray], max_idx: int, window: np.ndarray, stack_method: str = "pws"
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

    if stack_method not in ["linear", "pws"]:
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
        selected_data = data[: valid_max_idx + 1, :]

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
    ccf: np.ndarray,
    dist: np.ndarray,
    dt: float,
    vmin: float,
    vmax: float,
    select_method: str = "log",
    stack_method: str = "pws",
    signal_truncation: bool = True,
):
    """
    Select optimal segments based on SNR ratios and perform stacking.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlation data of shape (n_pairs, n_segments, n_lags).
    dist : np.ndarray
        Inter-station distances of shape (n_pairs,).
    dt : float
        Sampling interval.
    vmin, vmax : float
        Velocity bounds for window construction.
    select_method : str
        Method for selecting optimal segments ('log', 'rank', etc.).
    stack_method : str
        Stacking method ('pws' or 'linear').
    signal_truncation : bool
        If True, truncate to signal windows only.

    Returns
    -------
    snr_curves : dict
        Dictionary containing SNR curves for negative and positive windows.
    opt_ccf : np.ndarray
        Final optimal stacked CCF (truncated or full).
    """
    # Step 1: Calculate SNR ratios across all segments
    snr_neg_ave, snr_pos_ave, acr = calculate_snr_ratios(ccf, dist, dt, vmin, vmax)

    # Step 2: Select optimal segments based on SNR ratios
    ind_acr_neg, ind_acr_pos = select_optimal_segments(acr, snr_neg_ave, snr_pos_ave, select_method)

    # Step 3: Stack selected segments
    snr_neg_curve, opt_ccf_neg = cumsum_snr_numba(ccf, dist, ind_acr_neg, "neg", dt, vmin, vmax, stack_method)
    snr_pos_curve, opt_ccf_pos = cumsum_snr_numba(ccf, dist, ind_acr_pos, "pos", dt, vmin, vmax, stack_method)

    # Optional: Truncate to signal windows only
    N = opt_ccf_neg.shape[1] // 2
    lag_times = np.arange(-N, N + 1) * dt
    if signal_truncation:
        for i in range(opt_ccf_neg.shape[0]):
            wn, wp, _ = _create_signal_windows(lag_times, dt, dist[i], vmin, vmax, "signal", 'hann')
            opt_ccf_neg[i] *= wn
            opt_ccf_pos[i] *= wp

        opt_ccf = opt_ccf_neg + opt_ccf_pos
    else:
        opt_ccf = np.c_[opt_ccf_neg[:, :N], opt_ccf_pos[:, N:]]

    return snr_neg_curve, snr_pos_curve, opt_ccf
        
def _energy_symmetry_select(
    ccf: np.ndarray, dt: float, dist: float, vmin: float, vmax: float, pct: int = 20, window_type: str = "hann"
) -> np.ndarray:
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
    ccf_neg, ccf_pos = energy_window(ccf, dt, dist, vmin, vmax, window_type)

    # Remove mean
    ccf_neg -= np.mean(ccf_neg, axis=1, keepdims=True)
    ccf_pos -= np.mean(ccf_pos, axis=1, keepdims=True)

    # Calculate energy in each window
    E_neg = np.sum(ccf_neg**2, axis=1)
    E_pos = np.sum(ccf_pos**2, axis=1)

    # Calculate energy ratio (log scale for symmetry)
    energy_ratio = np.log(np.maximum(E_pos, 1e-10) / np.maximum(E_neg, 1e-10))  # Avoid division by zero

    # Select segments with balanced energy (within percentile threshold)
    cut = np.percentile(np.abs(energy_ratio), pct)
    mask = np.abs(energy_ratio) < cut

    return ccf[mask]


def process_pair(ccf_i, dist_i, dt, vmin, vmax, pct, stack_method, normalize, signal_truncation, window_type: str = "hann"):
    """Window_selection, stack and truncate"""
    filtered_ccf = _energy_symmetry_select(ccf_i, dt, dist_i, vmin, vmax, pct, window_type)

    if filtered_ccf.size > 0:
        stacked_ccf = stack_ccf(filtered_ccf, method=stack_method)

        if normalize:
            max_amp = np.max(np.abs(stacked_ccf))
            if max_amp > 0:
                stacked_ccf /= max_amp

        if signal_truncation:
            neg, pos = energy_window(stacked_ccf, dt, dist_i, vmin, vmax)
            stacked_ccf = neg + pos
    else:
        _, n_lags = ccf_i.shape
        stacked_ccf = np.zeros(n_lags)

    return stacked_ccf


def energy_symmetry_select(
    ccf,
    dist,
    dt: float,
    vmin: float,
    vmax: float,
    pct: int = 20,
    stack_method: str = "pws",
    signal_truncation: bool = True,
    normalize: bool = False,
    window_type: str = "hann",
):
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
    window_type : str
        Window function type: "hann" (default) or "rectangle"

    Returns
    -------
    np.ndarray
        Filtered, stacked, and processed CCF data for each station pair
    """

    if ccf.ndim == 2:
        dist_val = float(np.atleast_1d(dist)[0])
        return process_pair(ccf, dist_val, dt, vmin, vmax, pct, stack_method, normalize, signal_truncation, window_type)

    elif ccf.ndim == 3:
        n_pairs = ccf.shape[0]
        dist = np.asarray(dist)
        if dist.shape[0] != n_pairs:
            raise ValueError("dist length must match n_pairs for 3D input")

        with ThreadPoolExecutor(max_workers=MAXWORKERS) as ex:
            results = list(
                ex.map(
                    lambda args: process_pair(*args),
                    [
                        (ccf[i], dist[i], dt, vmin, vmax, pct, stack_method, normalize, signal_truncation, window_type)
                        for i in range(n_pairs)
                    ],
                )
            )
        return np.stack(results, axis=0)

    else:
        raise ValueError("ccf must be 2D or 3D array")
