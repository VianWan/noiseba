"""
Cross-Correlation Function (CCF) Processing Module

This module provides utilities for seismic data processing and cross-correlation
analysis. It includes functionality for loading SAC files, preprocessing seismic
data, computing cross-correlations, and saving results in various formats.

Key features:
- SAC file loading and stream management
- Sliding window data segmentation
- Spectral whitening and filtering
- Cross-correlation computation (with memory-aware threading)
- Result normalization and stacking
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Iterable

import numpy as np
import psutil
from obspy import read, Stream
from scipy.fft import rfft, irfft
from scipy.ndimage import uniform_filter1d, convolve1d
import mkl

from concurrent.futures import ThreadPoolExecutor
from noiseba.utils import apply_edge_taper

# Thread configuration
os.environ["OMP_NUM_THREADS"] = "1"
MKL_THREADS = 36
mkl.set_num_threads(MKL_THREADS)

# Memory management
TOTAL_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
MEMORY_THRESHOLD_GB = 0.8 * TOTAL_MEMORY_GB  # 80% of total system memory


def _ordered_paths(directory: Path, order_file: str, component: str) -> list[Path]:
    """Get SAC file paths in specified order from an order file."""

    order_txt = directory.joinpath(order_file)
    if not order_txt.is_file():
        raise FileNotFoundError(f"Order file {order_txt} does not exist.")

    idx_list = np.loadtxt(order_txt)[:, 0].astype(int)
    return [directory / f"{idx}.{component}.sac" for idx in idx_list]


def _glob_paths(directory: Path, component: str) -> list[Path]:
    return list(directory.glob(f"*.{component}.sac"))


def load_stream(
    sac_dir: str | Path,
    component: str = "Z",
    order_file: Optional[str] = None,
) -> Stream:
    """
    Load sac files from a directory.
    """
    directory = Path(sac_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory.")

    if order_file:
        sac_paths: Iterable[Path] = _ordered_paths(directory, order_file, component)
    else:
        sac_paths = _glob_paths(directory, component)

    stream = Stream()

    for sta in sac_paths:
        try:
            stream.extend(read(str(sta), headonly=False))
        except Exception as e:
            print(f"{sta} is not a valid sac file.")
            print(e)

    return stream

def sliding_window_indices_2d(data_len: int, win_len: int, step: int) -> np.ndarray:
    """
    2-D version: return (n_win, 2) indices [[start, end), ...]
    """
    if win_len > data_len:
        return np.empty((0, 2), dtype=np.int32)
    starts = np.arange(0, data_len - win_len + 1, step, dtype=np.int32)
    ends = starts + win_len
    return np.column_stack((starts, ends))


def sliding_window_2d_to_3d(data: np.ndarray, win_len: int, step: int) -> np.ndarray:
    """
    Input:  2-D array (n_sta, n_samp)
    Output: 3-D array (n_sta, n_win, win_len)  zero-copy view
    """
    n_sta, n_samp = data.shape
    idx = sliding_window_indices_2d(n_samp, win_len, step)  # (n_win, 2)
    n_win = idx.shape[0]

    # stride: jump one sample along last axis
    new_strides = (data.strides[0], step * data.strides[1], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=(n_sta, n_win, win_len), strides=new_strides, writeable=False)


def apply_taper(data: np.ndarray, taper_ratio: float = 0.2) -> np.ndarray:
    """row-wise taper, 2-D/3-D real array"""
    out = data - data.mean(axis=-1, keepdims=True, dtype=np.float32)
    return apply_edge_taper(out, taper_ratio)


def compute_fft(data, n_fft: Optional[int] = None):
    """Compute FFT of data along the last axis."""
    return rfft(data, axis=-1, n=n_fft)


def whiten_spectrum(S, smooth_width=20, window="rect", axis=-1, eps=1e-12):
    """
    Whiten spectrum.
    Parameters:
    S: (N, W, F) complex array
    smooth_width: smoothing width
    window: smoothing window
    axis: axis along which to smooth
    """
    amp = np.abs(S)
    if window == "rect":
        amp = uniform_filter1d(amp, size=smooth_width, axis=axis)
    elif window == "gaussian":
        r = int(3 * smooth_width / 6 + .5)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / (smooth_width / 6)) ** 2)
        k /= k.sum()
        amp = convolve1d(amp, k, axis=-1, mode="reflect")
    elif window == "ones":       
        k = np.ones(smooth_width)
        k /= k.sum()
        amp = np.apply_along_axis(lambda x: np.convolve(x, k, mode='same'), axis=-1, arr=amp)
    else:
        raise ValueError(f"unknown window: {window}")

    np.clip(amp, eps, None, out=amp)
    return (S / amp).astype(np.complex64)


def ccf_broadcast(S):
    """
    Compute cross-spectral density (CCF) using broadcasting.

    Parameters:
    S: (N, W, F) complex array

    Returns:
    idx: (M, 2) array of indices (i, j) for upper triangle
    C: (M, W, F) complex array, where M = N*(N-1)/2
    """
    N, W, F = S.shape
    M = N * (N - 1) // 2
    C = S[:, None] * S.conj()[None, :]  # (N, N, W, F)
    idx = np.triu_indices(N, k=1)
    C = C[idx]  # (M, W, F)

    return idx, C


def ccf_threaded(S, tile_W=500, max_workers=48):
    """
    Compute cross-spectral density (CCF) using ThreadPoolExecutor.

    Parameters:
    S: (N, W, F) complex array
    tile_W: int, tile size for parallel processing
    max_workers: int, maximum number of threads

    Returns:
    idx: (M, 2) array of indices (i, j) for upper triangle
    C: (M, W, F) complex array, where M = N*(N-1)/2
    """
    N, W, F = S.shape
    M = N * (N - 1) // 2
    idx = np.triu_indices(N, k=1)
    C = np.empty((M, W, F), dtype=S.dtype)

    def tile_job(args):
        i, j, tile = args
        return i, j, S[i, tile] * S[j, tile].conj()

    for w0 in range(0, W, tile_W):
        w1 = min(w0 + tile_W, W)
        tile = slice(w0, w1)
        jobs = [(i, j, tile) for i in range(N) for j in range(i + 1, N)]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(tile_job, jobs)
            for i, j, c in results:
                k = np.where((idx[0] == i) & (idx[1] == j))[0][0]
                C[k, w0:w1] = c

    return idx, C


def ccf(S, tile_W=80, max_workers=40):
    """
    Compute cross-spectral density (CCF) with automatic method selection.

    Parameters:
    A: (N, W, F) complex array
    tile_W: int, tile size for parallel processing (used in threaded version)
    max_workers: int, maximum number of threads (used in threaded version)

    Returns:
    idx: (M, 2) array of indices (i, j) for upper triangle
    C: (M, W, F) complex array, where M = N*(N-1)/2
    """
    N, W, F = S.shape
    M = N * (N - 1) // 2
    peak_memory = M * W * F * 16 / (1024**3)  # Peak memory usage in GB

    try:
        if peak_memory > MEMORY_THRESHOLD_GB:
            raise MemoryError("Memory usage exceeds threshold")
        idx, C = ccf_broadcast(S)
    except MemoryError:
        print(
            f"Memory threshold exceeded. Switching to threaded version. "
            f"(Peak memory: {peak_memory:.2f} GB, Threshold: {MEMORY_THRESHOLD_GB:.2f} GB)"
        )
        idx, C = ccf_threaded(S, tile_W=tile_W, max_workers=max_workers)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    return idx, C


def ifft_real(C, axis=-1, n_fft=None):
    """
    Inverse real FFT with optional trimming.

    Args:
        C: Complex spectrum array (..., n_freqs)
        axis: Axis along which to perform IFFT
        n_fft: Optional FFT length

    Returns:
        Real signal array
    """
    if n_fft is None:
        return irfft(C, axis=axis, n=n_fft)
    else:
        segment_length = C.shape[axis]
        result = irfft(C, axis=axis, n=n_fft)
        # Trim to correct length
        return result[:, :, :segment_length] # type: ignore


def ifft_real_shift(C, axis=-1, n_fft=None):
    """C should be A * B.conj(), or should trim the last column to the correct length"""
    return np.fft.fftshift(irfft(C, axis=axis, n=n_fft), axes=axis)  # type: ignore # lag time = 0 in the middle


def ifft_to_lags(C, max_lag, axis=-1, n_fft=None):
    """
    Convert cross-spectral density (CCF) to lags.

    Parameters:
    C: (M, W, F) complex array
    max_lag: maximum lag to return, point.
    axis: axis along which to perform IFFT
    n_fft: optional FFT length

    Returns:
    (M, 2 * max_lag + 1) array of lags
    """
    center = C.shape[axis] // 2
    return ifft_real_shift(C, axis=axis, n_fft=n_fft)[:, :, center - max_lag : center + max_lag + 1]


def normalize_ccf(ccf, method="max", axis=-1, eps=1e-12):
    """
    Normalize CCF array.

    Parameters:
    ccf: 3D array (M, W, F) - cross-correlation functions in frequency domain
    method: str - normalization method ("rms", "max", or "l1")
    """
    if method == "rms":
        norm = np.sqrt(np.mean(ccf**2, axis=axis, keepdims=True))
    elif method == "max":
        norm = np.max(np.abs(ccf), axis=axis, keepdims=True)
    elif method == "l1":
        norm = np.sum(np.abs(ccf), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    norm = np.maximum(norm, eps)
    return ccf / norm



