"""
A python based package for high-frequency seismic ambient noise surface wave imaging.
"""

__author__       = "wdp"
__email__        = "wangdp77@163.com"
__version__      = "0.1.0"
__license__      = "MIT"
__url__          = "https://github.com/yourname/mypkg"

from .utils import (
    stream_to_array,
    plot_phase_velocity_curves,
    plot_sensitivity,
    plot_velocity_profile,
    plot_ccf,
    snr_optimal_select,
    energy_symmetry_select,
    read_stations,
    build_ccf_index,
    front_k_pairs,
    CCFIndex,
    stack_pws_numba,
)
from .preprocessing import (
    load_stream,
    sliding_window_2d_to_3d,
    apply_taper,
    ccf,
    compute_fft,
    ifft_real_shift,
    whiten_spectrum,
    batch_process,
)
from .dispersion import fj_from_dir, park_from_dir, radon_from_dir
from .inversion import (
    create_2d_profile_plot,
    trans_2d_profile,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Utils functions
    "stream_to_array",
    "plot_phase_velocity_curves",
    "plot_sensitivity",
    "plot_velocity_profile",
    "plot_ccf",
    "snr_optimal_select",
    "energy_symmetry_select",
    "read_stations",
    "build_ccf_index",
    "front_k_pairs",
    "CCFIndex",
    "stack_pws_numba",
    # Preprocessing functions
    "load_stream",
    "sliding_window_2d_to_3d",
    "apply_taper",
    "ccf",
    "compute_fft",
    "ifft_real_shift",
    "whiten_spectrum",
    "batch_process",
    # Dispersion functions
    "fj_from_dir",
    "park_from_dir",
    "radon_from_dir",
    # Inversion functions
    "interpolate_1d_to_2d_profile",
    "create_2d_profile_plot",
    "extract_1d_profiles_at_locations",
]

