"""
A package for seismic ambient noise data processing.
"""

__author__       = "wdp"
__email__        = "wangdp77@163.com"
__version__      = "0.1.0"
__license__      = "MIT"
__url__          = "https://github.com/yourname/mypkg"

from .preprocessing import preprocess
from .utils import stream_data, plot_phase_velocity_curves, plot_sensitivity, plot_velocity_profile


__all__ = [
    "__version__",
    "__author__",
    "__email__",
]

