from __future__ import annotations

import numpy as np
from typing import List, Optional

from disba import DispersionError, PhaseDispersion, PhaseSensitivity
from .model import Curve
from noiseba.utils import rho_from_vp, vpvsv


class CostFunction:
    """
    A class of Cost function for dispersion curves inversion.
    where objective funciton is defined as:

    f = data_misfit + λ * model_regularization

    Supports global optimization (e.g., PSO, CDO) recovering shear velocity (vs) and
    layer thickness, and gradient-based optimization (e.g., LBFGS) recovering only
    shear velocity with fixed thickness.
    """

    def __init__(
        self,
        curves: List[Curve],
        nlayers: int,
        mode: str = "global",
        lamb: float = 1.0,
        alpha_s: float = 1e-3,
        alpha_z: float = 1e-3,
        alpha_zz: float = 0.0,
        alpha_h: float = 1.0,
        weights: Optional[List[float]] = None,
        vs_ref: Optional[np.ndarray] = None,
        vs_weight: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None,
    ):
        if mode not in {"global", "gradient"}:
            raise ValueError("mode must be 'global' or 'gradient'")

        self.curves = curves
        self._nlayers = nlayers
        self._mode = mode
        self.lamb = lamb
        self.alpha_s = alpha_s
        self.alpha_z = alpha_z
        self.alpha_zz = alpha_zz
        self.alpha_h = alpha_h
        self.weights = np.asarray(weights if weights is not None else np.ones(len(curves)))
        self.vs_ref = vs_ref
        self.vs_weight = vs_weight
        self.uncertainty = uncertainty

        # Fixed thickness for gradient mode - will be set from model
        self._fixed_thickness: Optional[np.ndarray] = None

        # cache
        self._clear_cache()

        # Cache variables
        self._chi_factor: float = np.inf
        self._model_norm: Optional[float] = None
        self._data_misfit: Optional[float] = None
        self._model_penalty: Optional[float] = None
        self._total_misfit: Optional[float] = None

    # ------------------- Public Properties -------------------

    @property
    def data_misfit(self) -> float:
        if self._data_misfit is None:
            raise ValueError("Call the instance with a model before accessing data_misfit.")
        return self._data_misfit

    @property
    def model_penalty(self) -> float:
        if self._model_penalty is None:
            raise ValueError("Call the instance with a model before accessing model_penalty.")
        return self._model_penalty

    @property
    def total_misfit(self) -> float:
        if self._total_misfit is None:
            raise ValueError("Call the instance with a model before accessing total_misfit.")
        return self._total_misfit

    @property
    def chi_factor(self) -> float:
        return self._chi_factor

    @property
    def model_norm(self):
        return self._model_norm

    def set_fixed_thickness(self, fixed_thickness: np.ndarray):
        """Set fixed thickness values for gradient mode"""
        if self._mode != "gradient":
            raise ValueError("fixed_thickness can only be set for gradient mode")
        self._fixed_thickness = np.asarray(fixed_thickness)

    def _unpack(self, x):
        if self._mode == "global":
            nl = self._nlayers
            if 2 * nl != len(x):
                raise ValueError("❌ The figure of nlayers you set is not equal to the model layers.")
            return x[:nl], x[nl:]
        if self._mode == "gradient":
            nl = self._nlayers
            if nl != len(x):
                raise ValueError("❌ The figure of nlayers you set is not equal to the model layers.")
            if self._fixed_thickness is None:
                raise ValueError("fixed_thickness must be set for gradient mode")
            return self._fixed_thickness, x

    def _clear_cache(self):
        self._data_misfit = self._model_penalty = self._total_misfit = None

    # ------------------- Core Interface -------------------

    def __call__(self, model: np.ndarray) -> float:
        thick_log, vs_log = self._unpack(model)  # type: ignore
        thick = np.exp(thick_log)
        vs = np.exp(vs_log)
        vp = vpvsv(vs, mode="s2p", lith="sandstone")
        rho = rho_from_vp(vp, constant=1.8, method="constant")  # only suit for near-surface
        model_matrix = np.c_[thick, vp, vs, rho]

        # Misfit computation
        data_misfit = self._compute_data_misfit(model_matrix)
        model_penalty = self._compute_model_penalty(vs_log, thick_log)
        total_misfit = data_misfit + self.lamb * model_penalty

        self._data_misfit = data_misfit
        self._model_penalty = model_penalty
        self._total_misfit = total_misfit

        return total_misfit

    def gradient(self, model: np.ndarray):
        try:
            thick_log, vs_log = self._unpack(model)
            thick = np.exp(thick_log)
            vs = np.exp(vs_log)
            vp = vpvsv(vs, mode="s2p", lith="sandstone")
            rho = rho_from_vp(vp, constant=1.8, method="constant")
            model_matrix = np.c_[thick, vp, vs, rho]

            # ------------------- Data Misfit Gradient -------------------
            data_grad = self._compute_data_misfit_gradient(model_matrix)

            # ------------------- Model Penalty Gradient -------------------
            penalty_grad = self._compute_model_penalty_gradient(vs_log, thick_log)

            total = data_grad + self.lamb * penalty_grad

            if total.shape != model.shape:
                raise ValueError("Gradient shape mismatch.")
            if not np.all(np.isfinite(total)):
                raise ValueError("Non-finite total gradient.")

            return total

        except Exception as e:
            print(f"[Analytical gradient failed: {e}]  ->  fallback to numerical.")
            return None

    # ------------------- Data Misfit -------------------

    def _compute_data_misfit(self, model_matrix) -> float:
        misfit = 0.0
        M = len(self.curves)
        pd = PhaseDispersion(*model_matrix.T)

        for ind, curve in enumerate(self.curves):
            obs_period = curve.period
            obs_vel = curve._pvelocity

            try:
                forward_period, forward_vel = pd(obs_period, mode=curve.mode, wave=curve.wave_type)[:2]
            except DispersionError:
                return 1e8

            mask = np.isin(obs_period, forward_period)
            if not np.any(mask):
                continue

            obs_vel = obs_vel[mask]
            N = len(obs_vel)
            weight = self.weights[ind]

            sigma = self.uncertainty[ind] if self.uncertainty is not None else 0.02 * obs_vel

            # Log-domain residuals (relative misfit)
            sigma_log = sigma / obs_vel
            obs_log = np.log(obs_vel)
            fwd_log = np.log(forward_vel)
            chi = np.sum(((obs_log - fwd_log) / sigma_log) ** 2) / N

            misfit += weight * chi

        self._chi_factor = misfit / np.sum(self.weights)
        return misfit / M

    def _compute_data_misfit_gradient(self, model_matrix) -> np.ndarray:  # result in log
        """Gradient of data misfit"""

        M = len(self.curves)
        nlayers = self._nlayers
        gradient = np.zeros(nlayers)
        vs = model_matrix[:, 2]
        pd = PhaseDispersion(*model_matrix.T)
        ps = PhaseSensitivity(*model_matrix.T)

        for ind, curve in enumerate(self.curves):
            obs_period = curve.period
            obs_vel = curve._pvelocity

            try:
                forward_period, forward_vel = pd(obs_period, mode=curve.mode, wave=curve.wave_type)[:2]
                sensitivity_kernels = [
                    ps(period, mode=curve.mode, wave=curve.wave_type, parameter="velocity_s") for period in forward_period
                ]
                jacobian = np.column_stack([kernel.kernel for kernel in sensitivity_kernels])  # (nlayers, periods)
                jacobian_log = vs[:, None] * jacobian / forward_vel  # ∂logcv/∂logvs
            except DispersionError:
                return np.full(nlayers, np.nan) # caution

            mask = np.isin(obs_period, forward_period)
            if not np.any(mask):
                continue

            obs_vel = obs_vel[mask]
            N = len(obs_vel)
            weight = self.weights[ind]
            sigma = self.uncertainty[ind] if self.uncertainty is not None else 0.02 * obs_vel
            sigma_log = sigma / obs_vel
            obs_log = np.log(obs_vel)
            fwd_log = np.log(forward_vel)
            residual = (obs_log - fwd_log) / 1**2
            gradient += weight * (jacobian_log @ residual) / N
        return gradient / M
        

    # ------------------- Model Regularization -------------------
    def _compute_model_penalty(self, vs_log: np.ndarray, thick_log: np.ndarray) -> float:
        """
        Compute the total model penalty by combining all regularization terms.
        """
        penalty = 0.0
        nlayers = self._nlayers
        thick = np.exp(thick_log)
        depth = np.cumsum(thick)
        depth /= np.max(depth)

        # Reference model (log domain)
        vs_ref_log = self.vs_ref if self.vs_ref is not None else np.zeros(nlayers)
        diff = vs_log - vs_ref_log

        self.vs_weight = self.vs_weight if self.vs_weight is not None else np.ones(nlayers)
        self._model_norm = np.sum(diff**2)

        # Add reference model penalty
        penalty += self._compute_reference_penalty(vs_log, vs_ref_log)

        # Add first derivative (smoothness) penalty
        if self.alpha_z > 0 and nlayers > 1:
            penalty += self._compute_smoothness_penalty(vs_log, depth)

        # Add second derivative (curvature) penalty
        if self.alpha_zz > 0 and nlayers > 2:
            penalty += self._compute_curvature_penalty(vs_log, depth)

        # Add thickness smoothness penalty
        if self.alpha_h > 0 and nlayers > 1 and self._mode == "global":
            penalty += self._compute_thickness_penalty(thick_log)

        return penalty

    def _compute_reference_penalty(self, vs_log: np.ndarray, vs_ref_log: np.ndarray) -> float:
        """
        Compute reference model penalty (L2 norm weighted).
        """
        diff = vs_log - vs_ref_log
        return self.alpha_s * np.sum((self.vs_weight * diff) ** 2)

    def _compute_smoothness_penalty(self, vs_log: np.ndarray, depth: np.ndarray) -> float:
        """
        Compute first derivative smoothness penalty.
        """
        dz = np.maximum(np.diff(depth), 1e-5)
        grad = np.diff(vs_log) / dz
        return self.alpha_z * np.sum(grad**2)

    def _compute_curvature_penalty(self, vs_log: np.ndarray, depth: np.ndarray) -> float:
        """
        Compute second derivative curvature penalty.
        """
        nlayers = len(vs_log)
        if nlayers <= 2 or self.alpha_zz == 0:
            return 0.0

        # Avoid division by zero
        epsilon = 1e-12
        total_curv = 0.0

        for i in range(nlayers - 2):
            d1 = max(depth[i + 1] - depth[i], epsilon)
            d2 = max(depth[i + 2] - depth[i + 1], epsilon)
            denominator = max(depth[i + 2] - depth[i], epsilon)

            # Calculate curvature with safe divisions
            diff1 = (vs_log[i + 1] - vs_log[i]) / d1
            diff2 = (vs_log[i + 2] - vs_log[i + 1]) / d2
            curvature = 2.0 / denominator * (diff2 - diff1)

            total_curv += curvature**2

        return self.alpha_zz * total_curv

    def _compute_thickness_penalty(self, thick_log: np.ndarray) -> float:
        """
        Compute thickness smoothness penalty.
        """
        grad_h = np.diff(thick_log)
        return self.alpha_h * np.sum(grad_h**2)

    # ------------------- Model Gradient Regularization -------------------
    def _compute_model_penalty_gradient(self, vs_log: np.ndarray, thick_log: np.ndarray) -> np.ndarray:
        """
        Compute gradient of the model penalty with respect to vs_log parameters.
        """
        nlayers = self._nlayers
        thick = np.exp(thick_log)
        depth = np.cumsum(thick)
        depth /= depth[-1] + 1e-10

        # Initialize gradient vector
        gradient = np.zeros(nlayers)

        # Reference model gradient
        vs_ref_log = self.vs_ref if self.vs_ref is not None else np.zeros(nlayers)
        gradient += self._compute_reference_penalty_gradient(vs_log, vs_ref_log)

        # First derivative (smoothness) gradient
        if self.alpha_z > 0 and nlayers > 1:
            gradient += self._compute_smoothness_penalty_gradient(vs_log, depth)

        # Second derivative (curvature) gradient
        if self.alpha_zz > 0 and nlayers > 2:
            gradient += self._compute_curvature_penalty_gradient(vs_log, depth)

        return gradient

    def _compute_reference_penalty_gradient(self, vs_log: np.ndarray, vs_ref_log: np.ndarray) -> np.ndarray:
        """
        Compute gradient of reference model penalty.
        """
        self.vs_weight = self.vs_weight if self.vs_weight is not None else np.ones_like(vs_log)
        diff = vs_log - vs_ref_log
        return 2 * self.alpha_s * self.vs_weight**2 * diff

    def _compute_smoothness_penalty_gradient(self, vs_log, depth) -> np.ndarray:
        dz = np.maximum(np.diff(depth), 1e-5)
        g = np.zeros_like(vs_log)

        g[1:-1] = 2 * self.alpha_z * ((vs_log[1:-1] - vs_log[:-2]) / dz[:-1] ** 2 - (vs_log[2:] - vs_log[1:-1]) / dz[1:] ** 2)

        g[0] = -2 * self.alpha_z * (vs_log[1] - vs_log[0]) / dz[0] ** 2
        g[-1] = 2 * self.alpha_z * (vs_log[-1] - vs_log[-2]) / dz[-1] ** 2

        return g

    def _compute_curvature_penalty_gradient(self, vs_log: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Compute gradient of second derivative curvature penalty.
        """
        nlayers = len(vs_log)
        g = np.zeros(nlayers)

        for i in range(nlayers - 2):
            denominator = depth[i + 2] - depth[i]
            d1 = depth[i + 1] - depth[i]
            d2 = depth[i + 2] - depth[i + 1]

            factor = 2 / denominator
            diff1 = (vs_log[i + 1] - vs_log[i]) / d1
            diff2 = (vs_log[i + 2] - vs_log[i + 1]) / d2
            curvature = factor * (diff2 - diff1)

            # Derivatives with respect to each of the three points
            dcurv_dvi = factor * (-1 / d1 - 1 / d2)  # wrt vs_log[i]
            dcurv_dvi1 = factor * (1 / d1 + 1 / d2)  # wrt vs_log[i+1]
            dcurv_dvi2 = factor * (-1 / d2)  # wrt vs_log[i+2]

            # Gradient contribution (2 * alpha_zz * curvature * dcurvature/dv)
            g[i] += 2 * self.alpha_zz * curvature * dcurv_dvi
            g[i + 1] += 2 * self.alpha_zz * curvature * dcurv_dvi1
            g[i + 2] += 2 * self.alpha_zz * curvature * dcurv_dvi2

        return g

    def __repr__(self):
        if self._total_misfit is None:
            return "<CostFunction: model not yet evaluated>"
        lines = []
        lines.append("=" * 60)
        title = "INVERSION COST FUNCTION BREAKDOWN"
        lines.append(title.center(60))
        lines.append("-" * 60)
        lines.append(f"{'Component':<25} | {'Model Regularization':<30}")
        lines.append("-" * 60)
        lines.append(f"{'λ (lambda)':<25} | {self.lamb:<30.4f}")
        lines.append(f"{'α_s (prior weight)':<25} | {self.alpha_s:<30.4e}")
        lines.append(f"{'α_z (smoothness)':<25} | {self.alpha_z:<30.4e}")
        lines.append(f"{'α_zz (curvature)':<25} | {self.alpha_zz:<30.4e}")
        lines.append(f"{'α_h (thickness smooth)':<25} | {self.alpha_h:<30.4e}")
        lines.append("-" * 60)
        lines.append(f"{'Data Misfit':<25} | {self._data_misfit:<30.4f}")
        lines.append(f"{'Model Penalty':<25} | {self._model_penalty:<30.4f}")
        lines.append(f"{'Chi Factor':<25} | {self._chi_factor:<30.4f}")
        lines.append(f"{'Model Norm':<25} | {self._model_norm:<30.4f}")
        lines.append("=" * 60)
        lines.append(f"{'Total Misfit':<25} | {self._total_misfit:<30.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)
