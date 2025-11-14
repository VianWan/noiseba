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



from noiseba.utils import stream_to_array
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
        norm: float = 2,
        ax: Optional[plt.Axes] = None
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
        
    # mapping to frequency-velocity
    f, phase_vel, spectrum = freqslow2freqvel(freq, p, np.abs(M), num_vel)

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

    cax = ax.pcolormesh(f, phase_vel / 1e3, spectrum, 
                        cmap='Spectral_r', vmin=0, vmax=1, rasterized=True)
    ax.plot(f, f * array_length / 1e3, 'w--', lw=2, label=r"$\lambda$")
    ax.set_xlim((f.min(), f.max()))
    ax.set_ylim((phase_vel.min() / 1e3, phase_vel.max() / 1e3))

    ax.set_xlabel('Frequency (Hz)', fontsize=24)
    ax.set_ylabel('Phase Velocity (km/s)', fontsize=24)
    ax.set_title('Dispersion Spectrum', fontsize=24)
    ax.legend(
        fontsize=20,         
        markerscale=1.5,     
        loc='upper right',   
        frameon=True,        
        facecolor='white',   
        edgecolor='black',   
        framealpha=0.8        
    )
    ax.tick_params(axis='both', labelsize=24)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Normalized Spectrum', size=24)
    plt.tight_layout()
    plt.show()
    return f, phase_vel, 
    


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

def prepare_ccf_data(st, part: str = 'right', time_window: float = 2.0):
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
    if part == 'right':
        ccf = I[:, num_samples // 2:]
    elif part == 'left':
        ccf = np.fliplr(I[:, :num_samples // 2 + 1])
    elif part == 'both':
        ccf_left = np.fliplr(I[:, :num_samples // 2])
        ccf_right = I[:, num_samples // 2:]
        if num_samples % 2 == 0:
            ccf = 0.5 * (ccf_left + ccf_right)
        else:
            ccf = np.empty_like(ccf_right)
            ccf[:, 0] = ccf_right[:, 0]
            ccf[:, 1:] = 0.5 * (ccf_left + ccf_right[:, 1:])
    else:
        raise ValueError('part must be "right", "left" or "both"')

    
    window_ind = int(time_window / dt)
    ccf = ccf[:, :window_ind + 1]

    return ccf, uniq_dist, dt

def radon_from_dir(dirpath: Union[str, Path], 
                   freq_min: Optional[float] = None, 
                   freq_max: Optional[float] = None, 
                   vel_min: float = 100., 
                   vel_max: float = 500., 
                   num_vel: int = 101, 
                   part: str = 'right',
                   maxiter: int = 100,
                   tol: float = 1e-1, 
                   method: str = 'CG_weight', 
                   reg_lambda: float = 1e-2, 
                   norm: int = 2):
    """
    Compute and plot dispersion spectrum using Radon transform from SAC files in a directory.

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
    part : {'right', 'left', 'both'}
        Which side(s) of the CCF to use.
    maxiter : int
        Maximum iterations for inversion.
    tol : float
        Convergence tolerance.
    method : str
        Inversion method ('CG_weight', 'CG', 'LSQR', etc.).
    reg_lambda : float
        Regularization weight λ.
    norm : int
        L-norm order for regularization.

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    st = read(Path(dirpath).joinpath('CCF*.sac'))
    ccf, uniq_dist, dt = prepare_ccf_data(st, part)

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate // 2
    p = np.linspace(1 / vel_max, 1 / vel_min, num_vel)

    return _radon(ccf, p, dt, uniq_dist, freq_min, freq_max,
           num_vel=num_vel, max_iter=maxiter, tol=tol,
           method=method, reg_lambda=reg_lambda, norm=norm)
    

def radon_from_stream(st, 
                      freq_min: Optional[float] = None, 
                      freq_max: Optional[float] = None, 
                      vel_min: float = 100., 
                      vel_max: float = 500., 
                      num_vel: int = 101, 
                      part: str = 'right',
                      maxiter: int = 100,
                      tol: float = 1e-1, 
                      method: str = 'CG_weight', 
                      reg_lambda: float = 1e-2, 
                      norm: int = 2):
    """
    Compute and plot dispersion spectrum using Radon transform from an ObsPy Stream.

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
    part : {'right', 'left', 'both'}
        Which side(s) of the CCF to use.
    maxiter : int
        Maximum iterations for inversion.
    tol : float
        Convergence tolerance.
    method : str
        Inversion method ('CG_weight', 'CG', 'L1', etc.).
    reg_lambda : float
        Regularization weight λ.
    norm : int
        L-norm order for regularization.

    Returns
    -------
    None
        Displays or saves the dispersion spectrum.
    """
    ccf, uniq_dist, dt = prepare_ccf_data(st, part)

    freq_min = freq_min if freq_min is not None else 0
    freq_max = freq_max if freq_max is not None else st[0].stats.sampling_rate // 2
    p = np.linspace(1 / vel_max, 1 / vel_min, num_vel)

    return _radon(ccf, p, dt, uniq_dist, freq_min, freq_max,
           num_vel=num_vel, max_iter=maxiter, tol=tol,
           method=method, reg_lambda=reg_lambda, norm=norm)


if __name__ == "__main__":
    dirpath = Path("/media/wdp/disk4/git/noiseba1/examples/CCF")

    radon_from_dir(dirpath, freq_min=0.1, freq_max=45, vel_min=50, vel_max=500,
                    num_vel=101, part="left", method='CG', reg_lambda=10, maxiter=300, tol=1)