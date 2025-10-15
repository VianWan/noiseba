from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy.fft import fft, ifft
from obspy.core import AttribDict, Trace
from scipy.signal import convolve, correlate, correlation_lags, hilbert
from scipy.signal.windows import hann
from tqdm import tqdm


def read_coord(file_path):
    """Load station coordinate"""
    stations = {}
    with open(file_path, "r") as fin:
        for line in fin.readlines():
            tmp = line.strip().split()
            if len(tmp) >= 3:
                stations[tmp[2]] = [float(tmp[0]), float(tmp[1])]
            elif len(tmp) == 2:
                stations[tmp[1]] = [float(tmp[0]), 0]
    return stations


def process_trace(data, operator, p=0.05):
    data = np.atleast_2d(data)  # at least 2d
    n_traces, n_samples = data.shape
    sn = int(n_samples * p)
    hn = 2 * sn
    h = hann(hn)

    d = data - np.mean(data, axis=1, keepdims=True)
    d[:, :sn] *= h[:sn][None, :]  # taper
    d[:, -sn:] *= h[-sn:][None, :]

    # FFT
    fd = fft(d, axis=1)
    abs_fd = np.abs(fd)

    # spectral whitening
    smoothed = np.array(
        [convolve(abs_fd[i], operator, mode="same") for i in range(n_traces)]
    )

    fd /= smoothed
    fd[np.isnan(fd)] = 0

    return ifft(fd).real


def stream_data(stream) -> Tuple[np.ndarray, List[str]]:
    """Convert stream to np.ndarray and trim data according to the  dataset's minimal and maximum time."""
    stream.detrend("demean")
    stream.detrend("linear")
    # stream.taper(max_percentage=0.05, type='cosine')
    begin_time = np.max([i.stats.starttime for i in stream])
    end_time = np.min([i.stats.endtime for i in stream])
    stream.trim(begin_time, end_time)
    data = np.array([tr.data for tr in stream])
    station = ["45" + tr.stats.station for tr in stream]

    return data, station


def sliding_window_indices(data_len, win_len, step):
    """return sliding window indices"""
    starts = list(range(0, data_len - win_len + 1, step))
    ends = [s + win_len for s in starts]
    return list(zip(starts, ends))


def get_ccf(
    stream,
    operator,
    taper,
    win_len: int,
    step: int,
    coord,
    cor_time_begin: float,
    cor_time_end: float,
):
    """
    Compute inter-station cross-correlation functions (CCF).

    Parameters
    ----------
    stream : obspy.Stream
        Waveform data container.
    operator : str or callable
        Pre-processing operator applied to each window.
    taper : str or callable
        Taper function applied to each window.
    win_len : int
        Window length in samples.
    step : int
        Step size in samples for sliding window.
    coord : dict
        Mapping from station code to (x, y) coordinates.
    cor_time_begin : float
        Start lag time for CCF truncation (seconds).
    cor_time_end : float
        End lag time for CCF truncation (seconds).

    Returns
    -------
    ccf_dict : defaultdict
        {station_pair: list of CCF matrices (n_windows, n_lag)}.
    ccf_distance : defaultdict
        {station_pair: list of (distance, x1, y1, x2, y2)}.
    """
    dt = round(stream[0].stats.delta, 4)
    data, stations = stream_data(stream)
    n_sta, n_pts = data.shape
    window_idx = sliding_window_indices(n_pts, win_len, step)
    lag_times = correlation_lags(win_len, win_len, mode="full") * dt
    idx_begin = int(np.where(np.isclose(lag_times, cor_time_begin))[0])
    idx_end = int(np.where(np.isclose(lag_times, cor_time_end))[0]) + 1
    used_lag_times = lag_times[idx_begin:idx_end]

    # allocate memory
    ccf_tem = np.zeros((len(window_idx), len(used_lag_times)))

    ccf_dict = dict()
    ccf_dist = dict()
    # compute ccf
    for i in range(n_sta - 1):
        sta1 = stations[i]
        x1, y1 = coord[sta1]
        for j in range(i + 1, n_sta):
            sta2 = stations[j]
            x2, y2 = coord[sta2]
            distance = np.hypot(x1 - x2, y1 - y2)

            for win_row, (start, end) in enumerate(window_idx):
                tr1 = process_trace(data[i, start:end], operator, taper).ravel()
                tr2 = process_trace(data[j, start:end], operator, taper).ravel()

                ccf_full = correlate(tr1, tr2, mode="full", method="fft")
                ccf_tem[win_row, :] = ccf_full[idx_begin:idx_end]

            pair = f"{sta1}_{sta2}"

            ccf_dict[pair] = ccf_tem.copy()
            ccf_dist[pair] = (distance, x1, y1, x2, y2)

    return ccf_dict, ccf_dist


def stack_ccf(
    ccf_data: np.ndarray,
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

    arr = np.atleast_2d(ccf_data)

    method = method.lower()
    ccf_linear = np.nanmean(arr, axis=0)

    if method == "linear":
        return ccf_linear

    if method == "pws":
        analytic = hilbert(arr, axis=1)
        phases = np.angle(analytic)
        coherence = np.abs(np.exp(1j * phases).mean(axis=0))
        weights = coherence**nu
        return ccf_linear * weights

    raise ValueError("Unknown method. Use 'linear' or 'pws'.")


# Save CCF
def write_ccf(
    ccf_dict,
    ccf_distance,
    output_dir,
    freq_min,
    freq_max,
    dt,
    start_time,
    method="pws",
    nu=2.0,
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for ccf_name, ccf_data in ccf_dict.items():
        dist, c1_x, c1_y, c2_x, c2_y = ccf_distance[ccf_name]
        cor_data = stack_ccf(ccf_data, method=method, nu=nu)
        tr = Trace(data=cor_data)
        tr.stats.delta = dt
        tr.stats.sac = AttribDict()
        tr.stats.sac.b = start_time
        tr.stats.sac.dist = dist
        tr.stats.sac.evlo = c1_x
        tr.stats.sac.evla = c1_y
        tr.stats.sac.stlo = c2_x
        tr.stats.sac.stla = c2_y
        tr.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
        tr.write(output_dir.joinpath(f"COR_{ccf_name}.sac").as_posix(), format="SAC")


# parallel save CCF
def _write_ccf(
    ccf_name,
    ccf_data,
    ccf_distance,
    output_dir,
    freq_min,
    freq_max,
    dt,
    start_time,
    method="pws",
    nu=2.0,
):
    dist, c1_x, c1_y, c2_x, c2_y = ccf_distance[ccf_name]
    cor_data = stack_ccf(ccf_data, method=method, nu=nu)
    tr = Trace(data=cor_data)
    tr.stats.delta = dt
    tr.stats.sac = AttribDict()
    tr.stats.sac.b = start_time
    tr.stats.sac.dist = dist
    tr.stats.sac.evlo = c1_x
    tr.stats.sac.evla = c1_y
    tr.stats.sac.stlo = c2_x
    tr.stats.sac.stla = c2_y
    tr.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
    tr.write(output_dir.joinpath(f"COR_{ccf_name}.sac").as_posix(), format="SAC")


def write_ccf_parallel(
    ccf_dict,
    ccf_distance,
    output_dir,
    freq_min,
    freq_max,
    dt,
    start_time,
    method="pws",
    nu=2.0,
    n_jobs=4,
):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_write_ccf)(
            ccf_name,
            ccf_data,
            ccf_distance,
            output_dir,
            freq_min,
            freq_max,
            dt,
            start_time,
            method,
            nu,
        )
        for ccf_name, ccf_data in tqdm(ccf_dict.items(), desc="Writing CCFs...")
    )


def compute_ccf_pair(
    idx,
    idy,
    data,
    station,
    coord,
    window_idx,
    operator,
    taper,
    cor_time_begin_idx,
    cor_time_end_idx,
    used_lag_time,
):
    c1_station = station[idx]
    c2_station = station[idy]
    c1_x, c1_y = coord[c1_station]
    c2_x, c2_y = coord[c2_station]
    distance = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    ccf_tem = np.zeros((len(window_idx), len(used_lag_time)))

    for tem_row, (ind1, ind2) in enumerate(window_idx):
        c1_data = data[idx][ind1:ind2]
        c2_data = data[idy][ind1:ind2]
        c1_data = process_trace(c1_data, operator, taper).flatten()
        c2_data = process_trace(c2_data, operator, taper).flatten()

        ccf_tem[tem_row, :] = correlate(c1_data, c2_data, mode="full", method="fft")[
            cor_time_begin_idx:cor_time_end_idx
        ]

    key = f"{c1_station}_{c2_station}"
    return key, ccf_tem.copy(), (distance, c1_x, c1_y, c2_x, c2_y)


def get_ccf_parallel(
    stream,
    operator,
    taper,
    win_len,
    step,
    coord,
    cor_time_begin,
    cor_time_end,
    n_jobs=4,
):
    dt = np.around(stream[0].stats.delta, decimals=3)
    data, station = stream_data(stream)
    row, col = data.shape

    window_idx = sliding_window_indices(col, win_len, step)
    lag_time = correlation_lags(win_len, win_len, mode="full") * dt
    cor_time_begin_idx = np.where(lag_time == cor_time_begin)[0][0]
    cor_time_end_idx = np.where(lag_time == cor_time_end)[0][0] + 1
    used_lag_time = lag_time[cor_time_begin_idx:cor_time_end_idx]

    tasks = [(idx, idy) for idx in range(row - 1) for idy in range(idx + 1, row)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_ccf_pair)(
            idx,
            idy,
            data,
            station,
            coord,
            window_idx,
            operator,
            taper,
            cor_time_begin_idx,
            cor_time_end_idx,
            used_lag_time,
        )
        for idx, idy in tqdm(tasks, desc="Computing CCFs...")
    )

    ccf_dict = dict()
    ccf_distance = dict()

    for result in results:
        if result is not None:
            key, ccf_data, dist_info = result

            ccf_dict[key] = ccf_data
            ccf_distance[key] = dist_info

    return ccf_dict, ccf_distance
