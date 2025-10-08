from __future__ import annotations

import numpy as np
from typing import Union, List, Tuple
from noiseba.utils import rhoesi, vs2vp

class Model:
    """
    Model class for handling layer parameter bounds.
    """

    def __init__(self, layer_bounds: Union[np.ndarray, List[Tuple]]):
        layer_bounds = np.asarray(layer_bounds)

        if layer_bounds is not None:
            self.thickness_bounds = layer_bounds[:, :2]
            self.vs_bounds = layer_bounds[:, 2:]
            self.dimension = self.thickness_bounds.shape[0] + self.vs_bounds.shape[0]
        else:
            self.thickness_bounds = None
            self.vs_bounds = None
            self.dimension = None


    @property
    def lower_bounds(self):
        """lower bounds about [thickness, vs]"""
        if self.thickness_bounds is None or self.vs_bounds is None:
            return None
        return np.r_[self.thickness_bounds[:, 0], self.vs_bounds[:, 0]]
    
    @property
    def upper_bounds(self):
        """upper bounds about [thickness, vs]"""
        if self.thickness_bounds is None or self.vs_bounds is None:
            return None
        return np.r_[self.thickness_bounds[:, 1], self.vs_bounds[:, 1]]

    @classmethod
    def model(cls, 
              param_vector: Union[np.ndarray, List[float]],
              vp: Union[np.ndarray, None] = None,
              rho: Union[np.ndarray, None] = None
              ) -> np.ndarray:
        
        param_vector = np.asarray(param_vector)
        nlayers = len(param_vector) // 2
        thickness = param_vector[:nlayers]
        vs = param_vector[nlayers:]
        vp = vs2vp(vs) if vp is None else vp
        rho = rhoesi(vp) if rho is None else rho
        # rho = 1.741 * vp**0.25
        
        return np.c_[thickness, vp, vs, rho]

    def add(self, new_bounds: Union[np.ndarray, List[Tuple]]):
        new_bounds = np.asarray(new_bounds)
        new_thickness_bounds = new_bounds[:, :2]
        new_vs_bounds = new_bounds[:, 2:]

        if self.thickness_bounds is None:
            self.thickness_bounds = new_thickness_bounds
            self.vs_bounds = new_vs_bounds
        else:
            self.thickness_bounds = np.vstack((self.thickness_bounds, new_thickness_bounds))
            self.vs_bounds = np.vstack((self.vs_bounds, new_vs_bounds))


class Curve:
    """
    Store dispersion curve, mode and wave type.
    """
    def __init__(self, freq, velocity, wave_type, mode=0):
        self.freq = freq
        self.period = np.flipud(1 / self.freq)
        self.velocity = velocity 
        self._pvelocity = np.flipud(velocity)
        self.mode = mode
        self.wave_type = wave_type


    

        

 

