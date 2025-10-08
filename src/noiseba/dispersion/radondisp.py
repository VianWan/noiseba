from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.fft import next_fast_len
from scipy.interpolate import interp1d
from scipy.sparse.linalg import cg
from scipy.ndimage import gaussian_filter
from obspy import read

from typing import Union, Optional
from pathlib import Path
from joblib import Parallel, delayed

from noiseba.utils import stream_data
from noiseba.optimization import cg_weight
from noiseba.optimization import irls_cg

def _radon(
        data: np.ndarray,
        p: np.ndarray,
        dt: float,
        dist: np.ndarray,
        f_min: float,
        f_max: float,
        num_vel: int = 101,
        max_iter: int = 50,
        tol: float = 1e-4,
        method: str = 'CG_weight',
        reg_lambda: float = 0.01,
        norm: float = 2
):
    """
    Radon transform from t-d domina to τ-p domain
        D(x, f) = L · U(p, f)

    Args:
        data (`n_trace, n_t`):
            Input seismic gather in t–x domain. Each row corresponds to one trace recorded at the offset given by dist.
        p (`n_p`):
            Slowness vector in s/m (or s/ft). Monotonically increasing.
        dt (`float`):
            Temporal sampling interval in **seconds**.
        dist (`n_trace,`):
            Source-receiver offset (or absolute position) in **metres** (or feet). Must be aligned with the row order of ``data``.
        f_min (`float`):
            Lower frequency bound in **Hz** for inversion window.
        f_max (`float`):
            Upper frequency bound in **Hz** for inversion window.
        max_iter (`int`):
            Maximum CG iterations per frequency slice. Rarely needs >100.
        tol (`float`):
            Convergence tolerance on the residual ℓ₂ norm.
        method (`{'CG', 'CG_weight', 'L1', 'L2'}`):
            Inversion algorithm.
        reg_lambda (`float`):
            Regularization parameter for 'L1' or 'L2' methods. Ignored by CG variants.
        norm (`float`):
            ℓⁿ exponent used for diagonal weighting when
            ``method='CG_weight'``. Default is 2.
    
    """
    n_trace, point = data.shape
    n = next_fast_len(point)

    Dfft = np.fft.rfft(data, n=n, axis=1)
    freq = np.fft.rfftfreq(n=n, d=dt)
    
    f_ind = np.where((freq >= f_min * 0.9 ) & (freq <= f_max * 1.1))[0]
    freq = freq[f_ind]
    Dfft = Dfft[:, f_ind]
  
    # survey line length 1D
    array_length = np.abs(dist.max() - dist.min())

    # allocate space
    M = np.empty((len(p), len(freq)), dtype=np.complex64) # size slowness * freq
    dist_dot_slow = np.outer(dist, p)

    # m0
    m0 = np.zeros(len(p), dtype=np.complex64)
    
    results = Parallel(n_jobs=-1)(
        delayed(solve_one_frequency)(
            method, f, i, dist_dot_slow, Dfft, m0, reg_lambda, max_iter, tol, norm
        ) for i, f in enumerate(tqdm(freq, desc="Solving frequencies"))
    )

    for i, m in enumerate(results):
        M[:, i] = m


    # for ind, f in enumerate(tqdm(freq,  desc="Processing frequencies")):
        # L = np.exp(-1j * 2 * np.pi * f * dist_dot_slow)
        # d = Dfft[:, ind]     # all tracea at same frequency
        
        # if method == 'CG_weight': # too slow
        #     m = cg_weight(L, d, x0=m0, reg_lambda=1, max_iter=max_iter, tol=tol, norm=norm)
        
        # elif method == 'adj':
        #     m = L.conj().T @ d
        
        # elif method == 'CG':
        #     m, _ = cg(L.conj().T @ L, L.conj().T @ d, x0=m0, maxiter=max_iter, rtol=tol)
        
        # elif method == 'L1':
        #     m = irls_cg(L, d, reg_lambda=reg_lambda, p=1, maxiter=max_iter, tol=tol)
        
        # elif method == 'L2':
        #     m = irls_cg(L, d, reg_lambda=reg_lambda, p=2, maxiter=max_iter, tol=tol)
        
        # else:
        #     raise ValueError('Only support Adj, CG, CG_weight, L1 and L2 methods')
        
        # M[:, ind] = m
        
    # mapping to frequency-velocity
    f, phase_vel, image = freqslow2freqvel(freq, p, np.abs(M), num_vel)
    image[image <= 0] = 0
    image[np.isnan(image)] = 1
    image /= np.maximum(image.max(axis=0, keepdims=True), 1e-6)
    image = image ** 2

    # # Apply Gaussian smoothing to the normalized dispersion spectrum
    sigma_value = 3  # Adjust this value to reduce smoothing
    smoothed_disp_spectrum = gaussian_filter(image, sigma=sigma_value)
    normalized_smoothed_disp_spectrum = smoothed_disp_spectrum / np.max(smoothed_disp_spectrum)
    norm_min = np.min(normalized_smoothed_disp_spectrum)
    norm_max = np.max(normalized_smoothed_disp_spectrum)


    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    cax = ax.pcolormesh(f, phase_vel / 1e3, normalized_smoothed_disp_spectrum, 
                        cmap='Spectral_r', vmin=norm_min, vmax=norm_max,  rasterized=True)
    ax.plot(f, f * array_length / 1e3, 'w--', lw=2, label=r"$\lambda$")
    ax.set_xlim((f.min(), f.max()))
    ax.set_ylim((phase_vel.min() / 1e3, phase_vel.max() / 1e3))

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
    fig.savefig('radon.png', dpi=300, bbox_inches='tight')
    
    # # return phase_vel, M


def freqslow2freqvel(f, p, data, num_vel: int = 101):
    """Slowness-frequency to velocity-frequency

    Args:
        f (numpy.ndarray): frequency
        p (numpy.ndarray): slowness
        data (numpy.ndarray): f-p data
    """
    
    I = np.zeros_like(data, dtype=np.float64)
    vel = 1 / p
    new_vel = np.linspace(vel.min(), vel.max(), num_vel)

    for ind in range(len(f)):
        d = data[:, ind]
        func = interp1d(vel, d, kind='cubic')
        I[:, ind] = func(new_vel)
    
    return f, new_vel, I

def solve_one_frequency(
    method: str,
    frequency: float,
    index: int,
    dist_dot_slow: np.ndarray,
    data_fft: np.ndarray,
    m0: np.ndarray,
    reg_lambda: float,
    max_iter: int,
    tol: float,
    norm: float
) -> np.ndarray:
    L = np.exp(-1j * 2 * np.pi * frequency * dist_dot_slow)
    d = data_fft[:, index]

    match method:
        case "L1":
            return irls_cg(L, d, reg_lambda=reg_lambda, p=1, maxiter=max_iter, tol=tol)
        case "L2":
            return irls_cg(L, d, reg_lambda=reg_lambda, p=2, maxiter=max_iter, tol=tol)
        case "CG_weight":
            return cg_weight(L, d, x0=m0, reg_lambda=reg_lambda, max_iter=max_iter, tol=tol, norm=norm)
        case "CG":
            return cg(L.conj().T @ L, L.conj().T @ d, x0=m0, maxiter=max_iter, rtol=tol)[0]
        case "adj":
            return L.conj().T @ d
        case _:
            raise ValueError(f"Unsupported method: {method}")

def radon(dirpath: Union[str, Path], 
         freq_min: Optional[float] = None, 
         freq_max: Optional[float] = None, 
         vel_min : float = 100., 
         vel_max: float = 500, 
         num_vel: int = 101, 
         part: str = 'right',
         maxiter: int = 100,
         tol: float = 1e-1, 
         method='CG_weight', 
         reg_lambda=1e-2, 
         norm=2
         ):
    """
    Compute and plot the dispersion spectrum.

    Parameters
    ----------
    dirpath : str or pathlib.Path
        Directory that contains the cross-correlation functions files.
    freq_min : float, optional
        Lower frequency bound for the scan (Hz).  
        If *None*, the lowest reliable frequency is used.
    freq_max : float, optional
        Upper frequency bound for the scan (Hz).  
        If *None*, the highest reliable frequency is used.
    vel_min : float, default 100
        Minimum phase-velocity value to scan (m/s).
    vel_max : float, default 500
        Maximum phase-velocity value to scan (m/s).
    num_vel : int, default 101
        Number of phase-velocity samples between *vel_min* and *vel_max*.
    part : {'right', 'left', 'both'}, default 'right'
        Which side(s) of the cross-correlation function are used.
    maxiter : int, default 100
        Maximum iterations for the solver.
    tol : float, default 0.1
        Convergence tolerance.
    method : str, default 'CG_weight'
        Inversion algorithm: 'CG_weight', 'CG', 'LSQR', ...
    reg_lambda : float, default 1e-2
        Regularisation weight λ (trade-off parameter).
    norm : int, default 2
        L-norm order used in the regularisation term.

    Returns
    -------
    spectrum : numpy.ndarray
        2-D dispersion spectrum (velocity × frequency).
    vels : numpy.ndarray
        Velocity vector (m/s).
    freqs : numpy.ndarray
        Frequency vector (Hz).
            
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
    
    # slowness
    p = np.linspace(1/vel_max, 1/vel_min, num_vel)

    # choise NCF casual or acasual part to calculate
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
    
    time_window = 2 # only need the part of NCF that include surface wave
    window_ind = int(time_window / dt)
    ccf = ccf[:, :window_ind+1]

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate // 2
    
    _radon(ccf, p, dt, uniq_dist, freq_min, freq_max, num_vel=num_vel, max_iter=maxiter, tol=tol, method=method, reg_lambda=reg_lambda, norm=norm)