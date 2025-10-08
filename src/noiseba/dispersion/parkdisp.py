from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from pathlib import Path
from obspy import read

from noiseba.utils import stream_data


# offset start from where is not import, means (0,2,4..) has the same result as (-4,-2,0,...) offset, dx is important

def parkdispersion(data, offset, dt, cmin, cmax, dc, fmin, fmax):
    """Dispersion panel
    
    `Acitve source` Calculate dispersion curves using the method of
    Park et al. 1998
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `(nx, nt)`
    offset : :obj:`np.ndarray`
        distance from source to receiver
    dt : :obj:`float`
        Time sampling
    cmin : :obj:`float`
        Minimum velocity
    cmax : :obj:`float`
        Maximum velocity
    dc : :obj:`float`
        Velocity sampling
    fmax : :obj:`float`
        Maximum frequency
        
    Returns
    -------
    f : :obj:`numpy.ndarray`
        Frequency axis
    c : :obj:`numpy.ndarray`
        Velocity axis`
    disp : :obj:`numpy.ndarray`
        Dispersion panel of size `nc x nf`
    """
    nr, nt = data.shape

    f = np.fft.fftfreq(nt, dt)[:nt//2]
    df = f[1] - f[0]
    fmax = min(fmax, f[-1])
    fmin = max(fmin, f[0])
    f_ind = np.where((f >= fmin * 0.9 ) & (f <= fmax * 1.1))[0]
    f = f[f_ind]

    c = np.linspace(cmin, cmax, dc)  # set phase velocity range
    array_length = np.abs(offset.max() - offset.min())
    x = offset

    # spectrum
    U = np.fft.fft(data, axis=1)[:, :nt//2]
    U = U[:, f_ind]

    # Dispersion panel
    disp = np.zeros((len(c), len(f)))
    for fi in range(f.size):
        for ci in range(len(c)):
            k = 2.0*np.pi*f[fi]/(c[ci])
            disp[ci, fi] = np.abs(
                np.dot(np.exp(1.0j*k*x), U[:, fi]/np.abs(U[:, fi])))
    
    image = disp.copy()
    image[image <= 0] = 0
    image[np.isnan(image)] = 1
    image /= np.maximum(image.max(axis=0, keepdims=True), 1e-8)
    image = image ** 2

    # # Apply Gaussian smoothing to the normalized dispersion spectrum
    sigma_value = 3  # Adjust this value to reduce smoothing
    smoothed_disp_spectrum = gaussian_filter(image, sigma=sigma_value)
    normalized_smoothed_disp_spectrum = smoothed_disp_spectrum / np.max(smoothed_disp_spectrum)
    norm_min = np.min(normalized_smoothed_disp_spectrum)
    norm_max = np.max(normalized_smoothed_disp_spectrum)

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    cax = ax.pcolormesh(f, c / 1e3, normalized_smoothed_disp_spectrum, 
                        cmap='Spectral_r', vmin=norm_min, vmax=norm_max,  rasterized=True)
    ax.plot(f, f * array_length / 1e3, 'w--', lw=2, label=r"$\lambda$")
    ax.set_xlim((f.min(), f.max()))
    ax.set_ylim((c.min() / 1e3, c.max() / 1e3))

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
    fig.savefig('Park.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # return f, c, disp


def park(dirpath: Union[str, Path], 
         freq_min: Optional[float] = None, 
         freq_max: Optional[float] = None, 
         vel_min : float = 100., 
         vel_max: float = 500, 
         num_vel: int = 101, 
         part: str = 'right'):
    """
    `Ambient noise` Compute and plot dispersion spectrum using phase-shift (time domain) method.

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

    dt = np.around(st[0].stats.delta, 4)
    dist = np.array([i.stats.sac.dist for i in st])
    ind = np.argsort(dist)
    data, _ = stream_data(st)

    data = data[ind]
    dist = dist[ind]
    
    uniq_dist = np.unique(dist)
    I = np.ones((len(uniq_dist), data.shape[1]))
    for ind, udist in enumerate(uniq_dist):
        I[ind] = np.mean(data[udist == dist], axis=0)
    
    # choise NCF part to calculate
    num_samples = I.shape[1]
    if part == 'right':
        ccf = I[:, num_samples // 2:]
    
    elif part == 'left':
        ccf = np.fliplr(I[:, :num_samples // 2 +1])
   
    elif part == 'both':
        ccf_left = np.fliplr(I[:, :num_samples // 2])
        ccf_right = I[:, num_samples // 2:]
        ccf = np.empty_like(ccf_right)

        if num_samples % 2 == 0:
           ccf = 0.5 * (ccf_left + ccf_right)

        else:
            ccf[:, 0] = ccf_right[:, 0]
            ccf[:, 1:] = 0.5 * (ccf_left + ccf_right[:, 1:])
   
    else:
        raise ValueError('part must be "right", "left" or "both"')
    
    time_window = 10 # only need the part of NCF that include surface wave
    window_ind = int(time_window / dt)
    ccf = ccf[:, :window_ind+1]

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate / 2
    
    parkdispersion(ccf, uniq_dist, dt, vel_min, vel_max, num_vel, freq_min, freq_max)



if __name__ == "__main__":

    dirpath = Path('/media/wdp/disk4/site1_line1/CCF_gate_EE')

    park(dirpath, freq_min=0.01, freq_max=45, vel_min=100, vel_max=500, num_vel=101, part='both')