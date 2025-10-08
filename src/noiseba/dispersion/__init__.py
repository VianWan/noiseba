"""
For now, the package supports three algorithms to compute the dispersion spectrum:

1. Phase-shift (Park et al., 1998)  applicable in either the time or the frequency domain.
2. High-resolution Radon transform (Luo et al., 2015).
3. Frequency-Bessel (FJ) transform (Wang et al., 2019).

Note: The FJ method requires prior installation of CC-FJpy: pip install git+https://github.com/ColinLii/CC-FJpy.

In practice, we recommend the following two-step strategy:

1. Run the phase-shift method first to obtain a quick, coarse dispersion image.
2. Use the Radon method to refine the spectrum and pick dispersion curves with higher confidence.

If highest lateral resolution is required, the FJ method is the best single option, provided the extra dependency is acceptable.
"""


from .fjdisp import fj_spectra
from .parkdisp import park, parkdispersion
from .maps import maps
from .radondisp import radon

__all__ = [
    'fj_spectra',
    'park',
    'maps',
    'radon'
]