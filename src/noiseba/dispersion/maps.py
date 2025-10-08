
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from obspy import read
from scipy.ndimage import gaussian_filter

from noiseba.utils.stream_data import stream_data

def _maps(
    data: np.ndarray,
    dist: np.ndarray,
    freq_min: float,
    freq_max: float,
    vel_min: float,
    vel_max: float,
    num_vel: int,
    time_window: float,
    dt: float,
    part: str = 'right'
) -> None:
    """
    Compute and plot dispersion spectrum using phase-shift method.

    Args:
        data (np.ndarray): Cross-correlation matrix (station pairs × time samples).
        dist (np.ndarray): Interstation distances (meters).
        vel_min (float): Minimum phase velocity (m/s).
        vel_max (float): Maximum phase velocity (m/s).
        num_vel (int): Number of velocity samples.
        time_window (float): Time window to extract (seconds).
        dt (float): Sampling interval (seconds).
        part (str): 'right', 'left', or 'both' side of CCF to use.
    """
    num_traces, num_samples = data.shape
    window_idx = int(time_window / dt)

    if part == 'right':
        ccf = data[:, num_samples // 2:]
    elif part == 'left':
        ccf = np.fliplr(data[:, :num_samples // 2])
    elif part == 'both':
        ccf_left = np.fliplr(data[:, :num_samples // 2])
        ccf_right = data[:, num_samples // 2:]
        ccf = np.empty_like(ccf_right)

        if num_samples % 2 == 0:
           ccf = 0.5 * (ccf_left + ccf_right)

        else:
            ccf[:, 0] = ccf_right[:, 0]
            ccf[:, 1:] = 0.5 * (ccf_left + ccf_right[:, 1:])
   
    else:
        raise ValueError("part must be 'right', 'left', or 'both'")

    ccf = ccf[:, :window_idx+1]
    # ccf /= np.max(abs(ccf), axis=1, keepdims=True)

    # Sort by distance; actually doesn't impact the result
    ind = np.argsort(dist)
    dist = dist[ind]
    ccf = ccf[ind]

    # Stack traces with same distance
    uniq_dist = np.unique(dist)
    I = np.ones((len(uniq_dist), ccf.shape[1]))
    for ind, udist in enumerate(uniq_dist):
        I[ind] = np.mean(ccf[udist == dist], axis=0)
    
    ccf = I.copy()
    D = np.fft.rfft(ccf, axis=1)
    freq = np.fft.rfftfreq(ccf.shape[1], dt)

    # set freq domain
    freq_min = max(0, freq_min)
    freq_max = min(freq.max(), freq_max)
    f_ind = np.where((freq >= freq_min * 0.9 ) & (freq <= freq_max * 1.1))[0]
    f = freq[f_ind]
    D = D[:, f_ind]

    v = np.linspace(vel_min, vel_max, num_vel)
    spectrum = np.zeros((len(v), len(f)))

    # Compute the dispersion matrix using phase matching
    for idx in range(len(f)):
        for idy in range(len(v)):
            term = np.exp(f[idx] * 1j * 2 * np.pi * uniq_dist / v[idy]) * D[:, idx]
            spectrum[idy, idx] = np.sum(term).real
    
    # enchance the spectrum resolution
    spectrum[spectrum <= 0] = 0
    spectrum[np.isnan(spectrum)] = 1
    spectrum /= np.maximum(spectrum.max(axis=0, keepdims=True), 1e-8)
    spectrum = spectrum**2
    spectrum = gaussian_filter(spectrum, sigma=3)
    

    # Plotting 
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    cax = ax.pcolormesh(f, v / 1e3, spectrum, cmap='Spectral_r', vmin=0, vmax=1.)
    ax.plot(f, f * dist[-1] / 1e3, 'w--', lw=2, label=r"$\lambda$")
    ax.set_xlim((f.min(), f.max()))
    ax.set_ylim((v.min() / 1e3, v.max() / 1e3))

    ax.set_xlabel('Frequency (Hz)', fontsize=24)
    ax.set_ylabel('Phase Velocity (km/s)', fontsize=24)
    ax.set_title('Dispersion Spectrum', fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    ax.legend(
        fontsize=20,          # 字体大小
        markerscale=1.5,      # 图标大小
        loc='upper right',    # 图例位置
        frameon=True,         # 显示边框
        facecolor='white',    # 背景颜色
        edgecolor='black',    # 边框颜色
        framealpha=0.8        # 背景透明度
    )

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Normalized Spectrum', size=24)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.show()


def maps(dirpath: Union[str, Path], 
         freq_min: Optional[float] = None, 
         freq_max: Optional[float] = None, 
         vel_min : float = 100, 
         vel_max: float = 500, 
         num_vel: int = 101, 
         part: str = 'right'):
    """
    `Ambient noise` Compute and plot dispersion spectrum using phase-shift (frequency domain) method.

    Args:
        dirpath (`Path | Str`): 
            path stored the cross-correlation data
        freq_min (`float`):
            Minimum frequency to scan.
        freq_max (`float`):
            Maximum frequency to scan.
        vel_min (`float`):
            Minimum phase velocity to scan.
        vel_max (`float`):
            Maximum phase velocity to scan.
        num_vel (`int`):
            Number of phase velocity samples.
        part (`str`):
            'right', 'left', or 'both' side of CCF to use.
    """
    dirpath = Path(dirpath).joinpath('COR*.sac')

    st = read(dirpath)
    dist = np.array([i.stats.sac.dist for i in st])
    ind = np.argsort(dist)
    data, _ = stream_data(st)
    data = data[ind]
    dist = dist[ind]
    
    uniq_dist = np.unique(dist)
    I = np.ones((len(uniq_dist), data.shape[1]))
    for ind, udist in enumerate(uniq_dist):
        I[ind] = np.mean(data[udist == dist], axis=0)
    
    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate / 2

    _maps(I, uniq_dist, freq_min, freq_max, vel_min, vel_max, num_vel, 10, st[0].stats.delta, part)



if __name__ == "__main__":
    from noiseba.preprocessing.get_ccf import stream_data

    st = read('/media/wdp/disk4/site1_line1/CCF_gate_ZZ/*.sac')
    dist = np.array([i.stats.sac.dist for i in st])
    dt = st[0].stats.delta
    time_window = 15 # seconds
    vmin = 100       # minimal velocity
    vmax = 500       # maximal velocity
    num_vel = 301    # velocity sampling number
    data, _ = stream_data(st)
    
    _maps(data, dist, 0, 40, vmin, vmax, num_vel, time_window, dt, part='both')