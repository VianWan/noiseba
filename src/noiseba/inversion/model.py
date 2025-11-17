from __future__ import annotations

import numpy as np
from typing import Union, List, Tuple
from noiseba.utils import rho_from_vp, vpvsv


class Model:
    """
    Model class for handling layer parameter bounds.
    Supports both 3-column (gradient mode) and 4-column (global mode) formats.

    3-column format: [fixed_thickness, vs_min, vs_max] - for gradient optimization
    4-column format: [thickness_min, thickness_max, vs_min, vs_max] - for global optimization
    """

    def __init__(self, layer_bounds: Union[np.ndarray, List[Tuple]]):
        layer_bounds = np.asarray(layer_bounds)

        if layer_bounds is not None:
            n_cols = layer_bounds.shape[1]

            if n_cols == 3:
                # 3-column format: [fixed_thickness, vs_min, vs_max] - gradient mode
                fixed_thickness = layer_bounds[:, 0]
                self.thickness_bounds = np.column_stack([fixed_thickness, fixed_thickness])  # min = max = fixed
                self.vs_bounds = layer_bounds[:, 1:3]
                self.dimension = self.vs_bounds.shape[0]  # only vs parameters
                self._mode = "gradient"

            elif n_cols == 4:
                # 4-column format: [thickness_min, thickness_max, vs_min, vs_max] - global mode
                self.thickness_bounds = layer_bounds[:, :2]
                self.vs_bounds = layer_bounds[:, 2:4]
                self.dimension = self.thickness_bounds.shape[0] + self.vs_bounds.shape[0]
                self._mode = "global"

            else:
                raise ValueError(f"layer_bounds must have 3 or 4 columns, got {n_cols}")
        else:
            self.thickness_bounds = None
            self.vs_bounds = None
            self.dimension = None
            self._mode = None

    @property
    def lower_bounds(self):
        """lower bounds about [thickness, vs]"""
        if self.thickness_bounds is None or self.vs_bounds is None:
            return None

        if self._mode == "gradient":
            # gradient mode: only vs bounds
            return self.vs_bounds[:, 0]
        else:
            # global mode: thickness + vs bounds
            return np.r_[self.thickness_bounds[:, 0], self.vs_bounds[:, 0]]

    @property
    def upper_bounds(self):
        """upper bounds about [thickness, vs]"""
        if self.thickness_bounds is None or self.vs_bounds is None:
            return None

        if self._mode == "gradient":
            # gradient mode: only vs bounds
            return self.vs_bounds[:, 1]
        else:
            # global mode: thickness + vs bounds
            return np.r_[self.thickness_bounds[:, 1], self.vs_bounds[:, 1]]

    @property
    def mode(self):
        """Get the optimization mode: 'gradient' or 'global'"""
        return self._mode

    @property
    def fixed_thickness(self):
        """Get fixed thickness values for gradient mode"""
        if self._mode != "gradient" or self.thickness_bounds is None:
            return None
        return self.thickness_bounds[:, 0]  # min and max are the same in gradient mode

    @classmethod
    def model(
        cls,
        param_vector: Union[np.ndarray, List[float]],
        vp: Union[np.ndarray, None] = None,
        rho: Union[np.ndarray, None] = None,
        mode: str = "global",
        fixed_thickness: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Create model matrix from parameter vector.

        Args:
            param_vector: Model parameters [thickness, vs] for global mode or [vs] for gradient mode
            vp: P-wave velocity (optional)
            rho: Density (optional)
            mode: Optimization mode - 'global' or 'gradient'
            fixed_thickness: Fixed thickness values for gradient mode
        """
        param_vector = np.asarray(param_vector)

        if mode == "gradient":
            # gradient mode: param_vector contains only vs values
            if fixed_thickness is None:
                raise ValueError("fixed_thickness required for gradient mode")
            vs = param_vector
            thickness = np.asarray(fixed_thickness)
        else:
            # global mode: param_vector contains [thickness, vs]
            nlayers = len(param_vector) // 2
            thickness = param_vector[:nlayers]
            vs = param_vector[nlayers:]

        vp = vpvsv(vs, mode="s2p", lith="sandstone") if vp is None else vp
        rho = rho_from_vp(vp, constant=1.8, method="constant") if rho is None else rho

        return np.c_[thickness, vp, vs, rho]

    def add(self, new_bounds: Union[np.ndarray, List[Tuple]]):
        """Add new layer bounds to existing model"""
        new_bounds = np.asarray(new_bounds)
        n_cols = new_bounds.shape[1]

        # Ensure consistent format
        if n_cols == 3 and self._mode == "gradient":
            new_thickness_bounds = np.column_stack([new_bounds[:, 0], new_bounds[:, 0]])
            new_vs_bounds = new_bounds[:, 1:3]
        elif n_cols == 4 and self._mode == "global":
            new_thickness_bounds = new_bounds[:, :2]
            new_vs_bounds = new_bounds[:, 2:4]
        else:
            raise ValueError(f"Inconsistent bounds format: model mode is {self._mode} but new bounds have {n_cols} columns")

        if self.thickness_bounds is None:
            self.thickness_bounds = new_thickness_bounds
            self.vs_bounds = new_vs_bounds
        else:
            # Ensure both existing and new bounds are valid for vstack
            if self.thickness_bounds is not None and self.vs_bounds is not None:
                self.thickness_bounds = np.vstack((self.thickness_bounds, new_thickness_bounds))
                self.vs_bounds = np.vstack((self.vs_bounds, new_vs_bounds))
            else:
                self.thickness_bounds = new_thickness_bounds
                self.vs_bounds = new_vs_bounds

        # Update dimension
        if self._mode == "gradient":
            self.dimension = self.vs_bounds.shape[0]
        else:
            self.dimension = self.thickness_bounds.shape[0] + self.vs_bounds.shape[0]


class Curve:
    """
    Store dispersion curve, mode, wave type, and location information.
    """

    def __init__(self, freq, velocity, wave_type, mode=0, x=None, y=None):
        self.freq = freq
        self.period = np.flipud(1 / self.freq)
        self.velocity = velocity
        self._pvelocity = np.flipud(velocity)
        self.mode = mode
        self.wave_type = wave_type
        self.x = x  # x coordinate of the representative location
        self.y = y  # y coordinate of the representative location
