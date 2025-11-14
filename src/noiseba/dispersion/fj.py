from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import ccfj

from obspy import read
from pathlib import Path
from scipy.ndimage import gaussian_filter
from typing import Union, Optional, Tuple

from noiseba.utils import stream_to_array


def prepare_ccf_data(st, part: str = "right"):
    """
    Extract, sort, stack and window CCFs from Stream.

    Returns
    -------
    ccf : np.ndarray
        Stacked correlations, shape (n_dist, n_lag).
    uniq_dist : np.ndarray
        Sorted unique distances (m).
    dt : float
        Sampling interval (s).
    """
    dt = np.around(st[0].stats.delta, 4)
    dist = np.array([tr.stats.sac.dist for tr in st])
    idx = np.argsort(dist)
    data, _ = stream_to_array(st)
    data = data[idx]
    dist = dist[idx]

    # remove dead traces
    alive = ~np.all(data == 0, axis=1)
    data = data[alive]
    dist = dist[alive]

    # stack per distance
    uniq_dist, inv = np.unique(dist, return_inverse=True)
    stacked = np.empty((uniq_dist.size, data.shape[1]), dtype=data.dtype)
    for i, d in enumerate(uniq_dist):
        stacked[i] = data[inv == i].mean(axis=0)

    # select causal / acausal / both
    n = stacked.shape[1]
    if part == "right":
        ccf = stacked[:, n // 2 :]
    elif part == "left":
        ccf = np.fliplr(stacked[:, : n // 2 + 1])
    elif part == "both":
        left  = np.fliplr(stacked[:, : n // 2])
        right = stacked[:, n // 2 :]
        if n % 2 == 0:
            ccf = 0.5 * (left + right)
        else:
            ccf = np.empty_like(right)
            ccf[:, 0]  = right[:, 0]
            ccf[:, 1:] = 0.5 * (left + right[:, 1:])
    else:
        raise ValueError('part must be "right", "left" or "both"')
    

    return ccf, uniq_dist, dt


def fj_spectrum(
    ccf: np.ndarray,
    dist: np.ndarray,
    dt: float,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    vel_min: float = 100.0,
    vel_max: float = 500.0,
    num_vel: int = 101,
    njobs: int = 1,
    ax: Optional[plt.Axes] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute and plot dispersion spectrum via the frequency-bessel (fj) method.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-correlations, shape (n_dist, n_lag).
    dist : np.ndarray
        Inter-station distances (m).
    dt : float
        Sampling interval (s).
    freq_min, freq_max : float, optional
        Frequency bounds (Hz).  Defaults to full rfft range.
    vel_min, vel_max : float
        Minimum / maximum phase velocity (m/s).
    num_vel : int
        Number of velocity samples.
    njobs : int
        Parallel jobs for ccfj.
    ax : matplotlib.axes.Axes, optional
        Axes for plotting.  Created if None.

    Returns
    -------
    f : np.ndarray
        Frequency axis (Hz).
    v : np.ndarray
        Velocity axis (m/s).
    image : np.ndarray
        Normalised dispersion image, shape (num_vel, len(f)).
    """
    offset_length = dist.max() - dist.min()

    nt = ccf.shape[1]
    freqs = np.fft.rfftfreq(nt, dt)
    freq_min = freq_min if freq_min is not None else freqs[0]
    freq_max = freq_max if freq_max is not None else freqs[-1]
    mask = (freqs >= freq_min) & (freqs <= freq_max)  # type: ignore
    rfreq = freqs[mask]

    v = np.linspace(vel_min, vel_max, num_vel)
    U = np.fft.rfft(ccf, axis=1)[:, mask]

    image = np.real(
        ccfj.fj_noise(
        U, dist, v, rfreq, fstride=1, itype=1, func=1, num=njobs
    ))

    image[image <= 0] = 0
    image[np.isnan(image)] = 1
    image /= np.maximum(image.max(axis=0, keepdims=True), 1e-8)
    image **= 2
    image = gaussian_filter(image, sigma=3)
    image /= image.max()

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 6))
    else:
        fig = ax.figure

    pcm = ax.pcolormesh(
        rfreq, v / 1e3, image, cmap="Spectral_r", vmin=0, vmax=1, rasterized=True
    )
    ax.plot(rfreq, rfreq * offset_length / 1e3, "w--", lw=2, label=r"$\lambda$")
    ax.set_xlim(rfreq[0], rfreq[-1])
    ax.set_ylim(v[0] / 1e3, v[-1] / 1e3)
    ax.set_xlabel("Frequency (Hz)", fontsize=24)
    ax.set_ylabel("Phase Velocity (km/s)", fontsize=24)
    ax.set_title("Dispersion Spectrum", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(
        fontsize=20,
        markerscale=1.5,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.8,
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Normalised Spectrum", fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.show()

    return rfreq, v, image


def fj_from_stream(
    st,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    vel_min: float = 100.0,
    vel_max: float = 500.0,
    num_vel: int = 101,
    part: str = "right",
    njobs: int = 1,
    ax: Optional[plt.Axes] = None,
):
    """
    Compute and plot dispersion spectrum using FJ from an Obspy Stream object.

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
    njobs : int
        Parallel jobs for ccfj.

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    ccf, uniq_dist, dt = prepare_ccf_data(st, part)
    return fj_spectrum(
        ccf, uniq_dist, dt, freq_min, freq_max, vel_min, vel_max, num_vel, njobs, ax
    )


def fj_from_dir(
    dirpath: Union[str, Path],
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    vel_min: float = 100.0,
    vel_max: float = 500.0,
    num_vel: int = 101,
    part: str = "right",
    njobs: int = 1,
    ax: Optional[plt.Axes] = None,
):
    """
    Compute and plot dispersion spectrum using FJ from SAC files.

    Parameters
    ----------
    dirpath : str or Path
        Directory containing cross-correlation SAC files (pattern: 'CCF*.sac').
    freq_min, freq_max : float, optional
        Frequency bounds for scanning (Hz).
    vel_min, vel_max : float
        Minimum and maximum phase velocity (m/s).
    num_vel : int
        Number of velocity samples.
    part : str
        Part of the CCF to use for dispersion calculation ('left', 'right' or 'both').
    njobs : int
        Parallel jobs for ccfj.

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    st = read(Path(dirpath).joinpath("CCF*.sac"))
    return fj_from_stream(
        st, freq_min, freq_max, vel_min, vel_max, num_vel, part, njobs, ax
    )


if __name__ == "__main__":
    dirpath = Path("/media/wdp/disk4/git/noiseba1/examples/CCF")
    fj_from_dir(dirpath, freq_min=0.01, freq_max=45,
                vel_min=50, vel_max=500, num_vel=101, part="left", njobs=10)