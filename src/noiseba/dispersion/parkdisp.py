from typing import Union, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from pathlib import Path
from obspy import read, Stream

from noiseba.utils import stream_to_array


# offset start from where is not import, means (0,2,4..) has the same result as (-4,-2,0,...) offset, dx is important


def parkdispersion(
    data: np.ndarray,
    offset: np.ndarray,
    dt: float,
    cmin: float,
    cmax: float,
    nc: int,
    fmin: float,
    fmax: float,
    ax: Optional[plt.Axes] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Active-source dispersion spectrum (Park et al. 1998).

    Parameters
    ----------
    data : np.ndarray
        Shot gathers, shape (nx, nt).
    offset : np.ndarray
        Source-receiver offsets (m).
    dt : float
        Time sampling (s).
    cmin, cmax : float
        Min / max phase-velocity (m/s).
    nc : int
        Number of velocity samples.
    fmin, fmax : float
        Frequency band (Hz).
    ax : matplotlib.axes.Axes, optional
        Axes for plotting.  Created internally if None.

    Returns
    -------
    f : np.ndarray
        Frequency axis (Hz).
    c : np.ndarray
        Velocity axis (km/s).
    spectrum : np.ndarray
        Normalised dispersion image, shape (nc, len(f)).
    """
    nt = data.shape[1]
    f = np.fft.rfftfreq(nt, dt)
    mask = (f >= fmin * 0.9) & (f <= fmax * 1.1)
    f = f[mask]

    # ---- velocity axis ----
    c = np.linspace(cmin, cmax, nc)
    offset_length = offset.max() - offset.min()

    # ---- frequency-domain data ----
    U = np.fft.rfft(data, axis=1)[:, mask]  # (nx, nf)

    spectrum = np.empty((nc, f.size), dtype=np.float32)
    for i, freq in enumerate(f):
        k = 2 * np.pi * freq / c  # wavenumber (dc,)
        phase = np.exp(1j * np.outer(k, offset))  # (dc, nx)
        spectrum[:, i] = np.abs(phase @ U[:, i])  # stack

    # ---- post-processing ----
    spectrum[spectrum <= 0] = 0
    spectrum /= np.maximum(spectrum.max(axis=0, keepdims=True), 1e-8)
    spectrum **= 2
    spectrum = gaussian_filter(spectrum, sigma=3)
    spectrum /= spectrum.max()

    # ---- plotting ----
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 6))
    else:
        fig = ax.figure

    im = ax.pcolormesh(f, c / 1e3, spectrum, cmap="Spectral_r", vmin=0, vmax=1, rasterized=True)
    ax.plot(f, f * offset_length / 1e3, "w--", lw=2, label=r"$\lambda$")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(c[0] / 1e3, c[-1] / 1e3)
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
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalised Spectrum", fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.show()

    return f, c, spectrum


def prepare_ccf_data(st: Stream, part: str = "right"):
    dt = np.around(st[0].stats.delta, 4)
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

    num_samples = I.shape[1]
    if part == "right":
        ccf = I[:, num_samples // 2 :]
    elif part == "left":
        ccf = np.fliplr(I[:, : num_samples // 2 + 1])
    elif part == "both":
        ccf_left = np.fliplr(I[:, : num_samples // 2])
        ccf_right = I[:, num_samples // 2 :]
        if num_samples % 2 == 0:
            ccf = 0.5 * (ccf_left + ccf_right)
        else:
            ccf = np.empty_like(ccf_right)
            ccf[:, 0] = ccf_right[:, 0]
            ccf[:, 1:] = 0.5 * (ccf_left + ccf_right[:, 1:])
    else:
        raise ValueError('part must be "right", "left" or "both"')


    return ccf, uniq_dist, dt


def park_from_dir(
    dirpath: Union[str, Path],
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    vel_min: float = 100.0,
    vel_max: float = 500.0,
    num_vel: int = 101,
    part: str = "right",
):
    """
    Compute and plot dispersion spectrum using phase-shift (time domain) method from SAC files.

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

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    st = read(Path(dirpath).joinpath("CCF*.sac"))
    ccf, uniq_dist, dt = prepare_ccf_data(st, part)

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate / 2

    f, c, spectrum = parkdispersion(ccf, uniq_dist, dt, vel_min, vel_max, num_vel, freq_min, freq_max)
    return f, c, spectrum


def park_from_stream(
    st: Stream,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    vel_min: float = 100.0,
    vel_max: float = 500.0,
    num_vel: int = 101,
    part: str = "left",
):
    """
    Compute and plot dispersion spectrum using phase-shift (time domain) method from an ObsPy Stream.

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
    ccf, uniq_dist, dt = prepare_ccf_data(st, part)

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate / 2  # type: ignore

    f, c, spectrum = parkdispersion(ccf, uniq_dist, dt, vel_min, vel_max, num_vel, freq_min, freq_max)
    return f, c, spectrum


if __name__ == "__main__":
    dirpath = Path("/media/wdp/disk4/git/noiseba1/examples/CCF")

    park_from_dir(dirpath, freq_min=0.1, freq_max=45, vel_min=50, vel_max=500, num_vel=101, part="left")
