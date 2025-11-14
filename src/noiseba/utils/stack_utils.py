import numpy as np
from numba import njit, prange
import mkl_fft  
import mkl

MKL_THREADS = 36
mkl.set_num_threads(MKL_THREADS)

# =====================
# Linear stack
# =====================
def stack_linear(ccf3: np.ndarray) -> np.ndarray:
    """
    Linear stack across segments.
    Parameters
    ----------
    ccf3 : np.ndarray
        Shape (n_pairs, n_segment, n_lags).
    Returns
    -------
    np.ndarray
        Linear stack result with segment dimension averaged out.
    """
    ccf3 = np.asarray(ccf3, dtype=np.float32)
    return ccf3.mean(axis=1)


# =====================
# PWS stack slow

# =====================

def stack_pws(ccf3: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Phase-Weighted Stack (PWS).
    
    Parameters
    ----------
    ccf3 : np.ndarray
        Input array of shape (n_pairs, n_segment, n_lags).
    power : float
        Exponent for phase weight (default=2.0).
    
    Returns
    -------
    np.ndarray
        PWS stacked result of shape (W, F).
    """
    analytic = batch_hilbert_mkl(ccf3, axis=-1)  # (nseg, W, F)

    phase_vectors = analytic / (np.abs(analytic) + 1e-12)

    coherence = np.abs(phase_vectors.mean(axis=1)) ** power  # (W, F)

    ccf = stack_linear(ccf3)  # (W, F)

    return coherence * ccf


# =====================
# PWS stack Numba

# =====================

# 1. Batch Hilbert Transform (MKL-FFT version)
def batch_hilbert_mkl(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Perform batch Hilbert transform using MKL-FFT.
    Input:
        x: array of shape (n_pairs, n_segment, n_lags), float32
    Output:
        analytic signal of same shape, complex64
    """
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[axis]
    X = mkl_fft.fft(x, axis=axis)

    h = np.zeros(N, dtype=np.float32)
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    idx = [np.newaxis] * x.ndim
    idx[axis] = slice(None)
    H = h[tuple(idx)]

    analytic = mkl_fft.ifft(X * H, axis=axis)
    return analytic.astype(np.complex64)


# 2. Numba-parallel PWS core
@njit(fastmath=True, parallel=True, cache=True)
def _pws_core(
    analytic_real: np.ndarray,
    analytic_imag: np.ndarray,
    ccf_real: np.ndarray,
    power: np.float32,
    n_pairs: int,
    n_segment: int,
    n_lags: int
) -> np.ndarray:
    """
    Compute phase-weighted stack (PWS) across segments for each pair.
    Inputs:
        analytic_real, analytic_imag: real/imag parts of analytic signal
        ccf_real: original real-valued CCFs
        power: exponent for coherence weighting
        n_pairs, n_segment, n_lags: dimensions
    Output:
        stacked CCFs of shape (n_pairs, n_lags)
    """
    result = np.zeros((n_pairs, n_lags), dtype=np.float32)
    
    # Parallel loop over pairs
    for p in prange(n_pairs):
        # Serial loop over lags
        for lag in range(n_lags):
            re_sum = 0.0
            im_sum = 0.0
            lin_sum = 0.0
            valid_count = 0
            
            # Loop over segments (stacking dimension)
            for s in range(n_segment):
                # Extract analytic signal at this pair/segment/lag
                re = analytic_real[p, s, lag]
                im = analytic_imag[p, s, lag]
                mag2 = re*re + im*im
                
                # Normalize phase contribution if magnitude is valid
                if mag2 > 1e-12:
                    inv_mag = 1.0 / np.sqrt(mag2)
                    re_sum += re * inv_mag
                    im_sum += im * inv_mag
                    valid_count += 1
                
                # Always accumulate linear stack of raw CCF
                lin_sum += ccf_real[p, s, lag]
            
            # Compute coherence (phase consistency across segments)
            if valid_count > 0:
                coherence = np.sqrt(re_sum*re_sum + im_sum*im_sum) / valid_count
            else:
                coherence = 0.0
            
            # Raise coherence to given power for weighting
            weight = coherence ** power
            
            # Final output: weighted linear stack
            result[p, lag] = weight * (lin_sum / n_segment)
    
    return result


# 3. Main function (high-performance PWS)
def stack_pws_numba(ccf3: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Perform phase-weighted stack (PWS) on 3D CCF array.
    Input:
        ccf3: array of shape (n_pairs, n_segment, n_lags), float32
    Output:
        stacked CCFs of shape (n_pairs, n_lags)
    """
    ccf3 = np.asarray(ccf3, dtype=np.float32)
    n_pairs, n_segment, n_lags = ccf3.shape
    
    # Step 1: Compute analytic signal via batch Hilbert transform
    analytic = batch_hilbert_mkl(ccf3, axis=-1)
    
    # Step 2: Separate real and imaginary parts
    analytic_real = np.real(analytic)
    analytic_imag = np.imag(analytic)
    
    # Step 3: Apply Numba-parallel PWS core
    return _pws_core(
        analytic_real, analytic_imag,
        ccf3, np.float32(power),
        n_pairs, n_segment, n_lags
    )
