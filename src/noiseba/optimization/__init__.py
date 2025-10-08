"""The following solver is used to calculate the dispersion spectrum and solve the inversion problem"""

from .CDO import CDO
from .PSO import PSO
from .nelder import NelderMeadOptimizer
from .cg_weight import cg_weight
from .irls import irls_cg

__all__ = [
    "CDO",
    "PSO",
    "NelderMeadOptimizer",
    "cg_weight",
    "irls_cg"
]