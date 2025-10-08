from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import ccfj

from obspy import read
from pathlib import Path    
from scipy.ndimage import gaussian_filter
from typing import Union

def fj_spectra(dir_path: Union[str, Path], config: dict):
    """
    Calculate dispersion spectrum using fj method.

    Args:
        dirpath (Path | Str): path stored the cross-correlation data
        config (Dict): define window size, filter frequenct range, etc.
    """
    dir_path = Path(dir_path)
    files = sorted([file.absolute() for file in dir_path.glob('*.sac')])

    if not files:
        print('No files found')
        exit()

    freq_min = config.get('freq_min', 1)
    freq_max = config.get('freq_max', 40)
    time_window = config.get('time_window', 15)   # window size in seconds
    t1 = -time_window
    t2 = time_window
 
    phase_vel_min = config.get('vel_min', 100)     # minimal velocity to scan (m/s)
    phase_vel_max = config.get('vel_max', 400)     # maximal velocity to scan (m/s)
    num_vel = config.get('num_vel', 301)                 # samping number
    core = config.get('njobs', 1)                         # number of cores to use
    distances = []
    fft_real_parts = []
    sample_trace = None
   
    for sac_file in files:
        tr = read(sac_file)[0] 
        evla = tr.stats.sac.evla # x and y coordinates of tow cross-correlation stations
        evlo = tr.stats.sac.evlo
        stla = tr.stats.sac.stla
        stlo = tr.stats.sac.stlo
        
        # distance
        dist = np.sqrt((evlo - stlo) ** 2 + (evla - stla) ** 2)
        if dist < 1e-6:
            continue

        #  FFT
        b = tr.stats.sac.b
        dt = tr.stats.delta
        n1 = int((t1 - b) / dt)
        n2 = int((t2 - b) / dt)
        segment = tr.data[n1:n2 + 1]
        segment = np.fft.fftshift(segment)
        fft_result = np.fft.fft(segment)

        fft_real = fft_result.real[:len(fft_result) // 2]

        distances.append(dist)
        fft_real_parts.append(fft_real)
        sample_trace = segment  # 保存一个样本用于后续长度参考

    
    distances = np.array(distances)
    fft_real_parts = np.array(fft_real_parts)

    # ascending by distance
    sorted_indices = np.argsort(distances)
    distances = distances[sorted_indices]
    fft_real_parts = fft_real_parts[sorted_indices]

    if sample_trace is not None:
        nd = len(sample_trace)
        
    freqs = np.arange(nd) / dt / (nd - 1)
    fn1 = int(freq_min * nd * dt)
    fn2 = int(freq_max * nd * dt)


    phase_velocities = np.linspace(phase_vel_min, phase_vel_max, num_vel)
    image = ccfj.fj_noise(
        fft_real_parts[:, fn1:fn2 + 1],
        distances,
        phase_velocities,
        freqs[fn1:fn2 + 1],
        fstride=1,
        itype=1,
        func=1,
        num=core, # used cpu cores
    )
    
    # Normalize 
    image[image <= 0] = 0
    image[np.isnan(image)] = 1
    image /= np.maximum(image.max(axis=1, keepdims=True), 1e-8)
    image = image ** 2
    freqs = freqs[fn1:fn2 + 1]

    # Apply Gaussian smoothing to the normalized dispersion spectrum
    sigma_value = 5  # Adjust this value to reduce smoothing
    smoothed_disp_spectrum = gaussian_filter(image, sigma=sigma_value)
    normalized_smoothed_disp_spectrum = smoothed_disp_spectrum / np.max(smoothed_disp_spectrum)
    norm_min = np.min(normalized_smoothed_disp_spectrum)
    norm_max = np.max(normalized_smoothed_disp_spectrum)

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    cax = ax.pcolormesh(freqs, phase_velocities / 1e3, normalized_smoothed_disp_spectrum, 
                        cmap='Spectral_r', vmin=norm_min, vmax=norm_max,  rasterized=True)
    ax.plot(freqs, freqs * distances[-1] / 1e3, 'w--', lw=2, label=r"$\lambda$")
    ax.set_xlim((freqs.min(), freqs.max()))
    ax.set_ylim((phase_velocities.min() / 1e3, phase_velocities.max() / 1e3))


    # Plot theoretical dispersion curves
    colors = ['w', '#E73879', '#7E1891', '#E195AB', '#F26B0F', '#FCC737']#497D74

    ax.set_xlabel('Frequency (Hz)', fontsize=24)
    ax.set_ylabel('Phase Velocity (km/s)', fontsize=24)
    ax.set_title('Dispersion Spectrum', fontsize=24)
    ax.legend(
        fontsize=20,          # 字体大小
        markerscale=1.5,      # 图标大小
        loc='upper right',    # 图例位置
        frameon=True,         # 显示边框
        facecolor='white',    # 背景颜色
        edgecolor='black',    # 边框颜色
        framealpha=0.8        # 背景透明度
    )
    ax.tick_params(axis='both', labelsize=24)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Normalized Spectrum', size=24)
    plt.show()
    np.savez('reef.npz', f=freqs, v=phase_velocities, f_v=normalized_smoothed_disp_spectrum)


if __name__ == "__main__":
    dir_path = Path(r'/media/wdp/disk4/site1_line1/CCF_gate_NN') # NCF dir path
    config = {
        'freq_min': 1,
        'freq_max': 45,
        'time_window': 15,
        'vel_min': 100,
        'vel_max': 500,
        'num_vel': 301,
        'njobs': 36,
    }
 
    fj_spectra(dir_path, config)