from __future__ import annotations
import numpy as np

from disba import DispersionError, PhaseDispersion
from .model import Curve
from typing import List, Optional

from noiseba.utils import rhoesi, vs2vp


class CostFunction:
    r"""
    Multi-mode dispersion cost function defined as:

        f = (1/M) ∑_{m=1}^{M} ω_m (1/N_m) ∑_{n=1}^{N_m} (C_{mn}^{o} - C_{mn}^{s})^2 + α ||ΔV_s||²

    where:
        C_{mn}^{o} : observed dispersion curve
        C_{mn}^{s} : synthetic dispersion curve
        ω_m        : weight of mode m
        N_m        : number of frequencies in mode m
        ΔV_s       : second derivative of shear velocity profile
        aplha          : regularization parameter
    """

    def __init__(
        self,
        curves: List[Curve],
        alpha: float = 1e-5,
        weights: Optional[List[float]] = None,
    ):
        self.curves = curves
        self.alpha = alpha
        self.weights = weights if weights is not None else np.ones(len(curves))

    # so ugly, sepearte the forward modeling into its own function later.
    def __call__(self, model: np.ndarray, nlayers: Optional[int] = None) -> float:

        nlayers = len(model) // 2 if nlayers is None else nlayers
        thick = model[:nlayers]
        vs = model[nlayers:]
        vp = vs2vp(vs)            # defalut by assume VP/VS ratio is 4
        rho = rhoesi(vp)          # Constant density, 

        # Construct model matrix for dispersion calculation
        curr_model = np.c_[thick, vp, vs, rho]
        pd = PhaseDispersion(*curr_model.T)

        # define para
        misfit = 0.0
        M = len(self.curves)

        for ind, curve in enumerate(self.curves):
            obs_period = curve.period
            obs_vel = curve._pvelocity

            try:
                forward_period, forward_vel = pd(obs_period, 
                                                 mode=curve.mode, 
                                                 wave=curve.wave_type)[:2]
            except DispersionError:
                return 1e8  # Penalize invalid models

            # Match observed periods with forward periods
            mask = np.isin(obs_period, forward_period)
            if not np.any(mask):
                continue

            obs_vel = obs_vel[mask]
            N = len(obs_vel)
            weight = self.weights[ind]

            # Compute weighted mean squared error
            misfit += weight * np.sum(np.square(obs_vel - forward_vel)) / N

        misfit /= M

        # Add smoothness regularization if alpha > 0
        if self.alpha > 0:
            dx = np.cumsum(thick)
            misfit += self.alpha * self.smoothness(vs, dx)

        return misfit
    

    def smoothness(self, vs: np.ndarray, dx: Optional[np.ndarray] = None) -> float:
        """
        Compute second-order spatial derivative of shear velocity profile.

        Parameters:
            vs : shear velocity array
            dx : model depth array (optional)

        Returns:
            Smoothness penalty term
        """
        vs = np.array(vs) * 1e3  # Convert to m/s

        if dx is not None:
            assert vs.size == dx.size, "Vs and dx must be the same size"
            dx = dx * 1e3  # Convert to meters
            first_derivative = np.gradient(vs, dx)
            second_derivative = np.gradient(first_derivative, dx)
        else:
            second_derivative = np.gradient(np.gradient(vs))

        return np.sum(np.square(second_derivative))
