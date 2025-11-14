from tkinter import W
from typing import Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from obspy import read
from scipy.ndimage import gaussian_filter

from noiseba.utils import stream_to_array, apply_edge_taper


def _maps(
    data: np.ndarray,
    dist: np.ndarray,
    freq_min: float,
    freq_max: float,
    vel_min: float,
    vel_max: float,
    num_vel: int,
    # time_window: float,
    dt: float,
    part: str = 'right',
    ax = None,
    theta = 0.
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute and plot dispersion spectrum using phase-shift method.

    Args:
        data (np.ndarray): Cross-correlation matrix (station pairs × time samples).
        dist (np.ndarray): Interstation distances (meters).
        vel_min (float): Minimum phase velocity (m/s).
        vel_max (float): Maximum phase velocity (m/s).
        num_vel (int): Number of velocity samples.
        dt (float): Sampling interval (seconds).
        part (str): 'right', 'left', or 'both' side of CCF to use.
        ax (matplotlib.axes.Axes): Axes to plot on.
        theta (float): Angle of plane wavefront (degrees).
    """
    n_tr, n_samp = data.shape

    # 2.1 Extract causal / acausal part
    if part == "right":
        ccf = data[:, n_samp // 2 :]
    elif part == "left":
        ccf = np.fliplr(data[:, : n_samp // 2])
    elif part == "both":
        left = np.fliplr(data[:, : n_samp // 2])
        right = data[:, n_samp // 2 :]
        ccf = 0.5 * (left + right) if n_samp % 2 == 0 else np.empty_like(right)
        if n_samp % 2:
            ccf[:, 0] = right[:, 0]
            ccf[:, 1:] = 0.5 * (left + right[:, 1:])
    else:
        raise ValueError("part must be 'right', 'left', or 'both'")

    # 2.2 Distance stacking
    sort_idx = np.argsort(dist)
    dist_sorted = dist[sort_idx]
    ccf = ccf[sort_idx]

    # 2.3 Windowing + phase correction
    # ccf = ccf * win

    uniq_dist, inv = np.unique(dist_sorted, return_inverse=True)
    stacked = np.empty((uniq_dist.size, ccf.shape[1]))
    for i, d in enumerate(uniq_dist):
        stacked[i] = ccf[inv == i].mean(axis=0)

    # 2.4 Frequency domain
    D = np.fft.rfft(stacked, axis=1)
    # D *= phase_corr
    freq = np.fft.rfftfreq(stacked.shape[1], dt)

    # Frequency band mask
    mask = (freq >= freq_min * 0.9) & (freq <= freq_max * 1.1)
    f = freq[mask]
    D = D[:, mask]

    # 2.5 Velocity scan
    v = np.linspace(vel_min, vel_max, num_vel)
    spectrum = np.empty((v.size, f.size))

    for idx in range(f.size):
        phase = np.exp(1j * 2 * np.pi * f[idx] * np.cos(theta) * uniq_dist[None, :] / v[:, None])
        spectrum[:, idx] = np.real(np.sum(phase * D[:, idx], axis=1))


    # 2.6 Post-processing & display
    spectrum[spectrum <= 0] = 0
    spectrum = spectrum / np.maximum(spectrum.max(axis=0, keepdims=True), 1e-8)
    spectrum = gaussian_filter(spectrum**2 / spectrum.max(), sigma=3)
    spectrum /= np.max(spectrum)

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 6))
    else:
        fig = ax.figure
    pcm = ax.pcolormesh(f, v / 1e3, spectrum, cmap="Spectral_r", vmin=0, vmax=1)
    ax.plot(f, f * uniq_dist[-1] / 1e3, "w--", lw=2, label=r"$\lambda$")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(v[0] / 1e3, v[-1] / 1e3)
    ax.set_xlabel("Frequency (Hz)", fontsize=24)
    ax.set_ylabel("Phase Velocity (km/s)", fontsize=24)
    ax.set_title("Dispersion Spectrum", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=20, loc="upper right", frameon=True,
              facecolor="white", edgecolor="black", framealpha=0.8)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Normalised Spectrum", size=24)
    cbar.ax.tick_params(labelsize=20)
    fig.tight_layout()
    plt.show()
    return f, v, spectrum


def prepare_maps_data(st):
    dist = np.array([tr.stats.sac.dist for tr in st])
    ind = np.argsort(dist)
    data, _ = stream_to_array(st)

    data = data[ind]
    dist = dist[ind]

    valid_rows = ~np.all(data == 0, axis=1)
    data = data[valid_rows]
    dist = dist[valid_rows]

    uniq_dist = np.unique(dist)
    I = np.ones((len(uniq_dist), data.shape[1]))
    for idx, udist in enumerate(uniq_dist):
        I[idx] = np.mean(data[dist == udist], axis=0)

    dt = np.around(st[0].stats.delta, 4)
    
    I = apply_edge_taper(I, taper_ratio=0.1)
    return I, uniq_dist, dt


def maps_from_dir(dirpath: Union[str, Path], 
                  freq_min: Optional[float] = None, 
                  freq_max: Optional[float] = None, 
                  vel_min: float = 100., 
                  vel_max: float = 500., 
                  num_vel: int = 101,
                  part: str = 'right'):
    """
    Compute and plot dispersion spectrum using phase-shift (frequency domain) method from SAC files.

    Parameters
    ----------
    dirpath : str or Path
        Directory containing cross-correlation SAC files (pattern: 'COR*.sac').
    freq_min, freq_max : float, optional
        Frequency bounds for scanning (Hz).
    vel_min, vel_max : float
        Minimum and maximum phase velocity (m/s).
    num_vel : int
        Number of velocity samples.
    part : str
        Part of the CCF to use for dispersion calculation ('left', 'right' or 'both').

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    st = read(Path(dirpath).joinpath('CCF*.sac'))
    data, uniq_dist, dt = prepare_maps_data(st)

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate // 2

    f, v, spectrum = _maps(data, uniq_dist, freq_min, freq_max, vel_min, vel_max, num_vel, dt=dt, part=part)

def maps_from_stream(st, 
                     freq_min: Optional[float] = None, 
                     freq_max: Optional[float] = None, 
                     vel_min: float = 100., 
                     vel_max: float = 500., 
                     num_vel: int = 101,
                     part: str = 'right'):
    """
    Compute and plot dispersion spectrum using phase-shift (frequency domain) method from an ObsPy Stream.

    Parameters
    ----------
    st : obspy.Stream
        Stream object containing cross-correlation traces.
    freq_min, freq_max : float, optional
        Frequency bounds for scanning (Hz).
    vel_min, vel_max : float
        Minimum and maximum phase velocity (m/s).
    num_vel : int
        Number of velocity samples.
    part : str
        Part of the CCF to use for dispersion calculation ('left', 'right' or 'both').

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    data, uniq_dist, dt = prepare_maps_data(st)

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate // 2

    _maps(data, uniq_dist, freq_min, freq_max, vel_min, vel_max, num_vel, dt, part=part)


if __name__ == "__main__":

    dirpath = Path('/media/wdp/disk4/git/noiseba1/examples/CCF')

    maps_from_dir(dirpath, freq_min=0.1, freq_max=45, vel_min=100, vel_max=500, num_vel=101, part='left')