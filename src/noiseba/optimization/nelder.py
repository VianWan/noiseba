import numpy as np

from scipy.optimize import minimize, Bounds
from typing import Union, Callable

class NelderMeadOptimizer:
    r"""
    A wrapper class for the Nelder-Mead optimization algorithm using SciPy.
    """
    def __init__(self, 
                objective_function,
                lower_bound: Union[np.ndarray, float] = 0., 
                upper_bound: Union[np.ndarray, float] = 1., 
                max_iterations: int = 200,
                ):
        
      self.fobj = objective_function
      self.lb = np.asarray(lower_bound)
      self.ub = np.asarray(upper_bound)
      self.max_iter = max_iterations
      self.res = None
      self._traj_cache = None

    def _ensure_traj(self):
        if self._traj_cache is None:
            if self.res is None:
                raise RuntimeError("Call .optimize() first.")
            allvecs = np.asarray(self.res.allvecs)
            fitness = np.array([self.fobj(p) for p in allvecs])
            ind = np.argsort(fitness) 
            self._traj_cache = {
                "fitness": fitness[ind][::-1],
                "positions": allvecs[ind].copy()[::-1],
            }
        return self._traj_cache

    @property
    def position_history(self):
        return self._ensure_traj()["positions"]
    @property
    def convergence_curve(self):
        return self._ensure_traj()["fitness"]
    @property
    def best_fitness(self):
        return self.res.fun if self.res is not None else None
    @property
    def best_position(self):
        return self.res.x if self.res is not None else None

    def optimize(self, x0=None):
        x0 = self.ub if x0 is None else x0
        bd = Bounds(self.lb, self.ub, keep_feasible=False)
        self.res = minimize(
            self.fobj, 
            x0, 
            method='nelder-mead', 
            options={'maxiter': self.max_iter, 'adaptive': True, 'xatol': 1e-8, 'return_all': True, 'disp': False}, 
            bounds=bd)
        self._traj_cache = None

    def positions_below(self, threshold: float):
        fit = self.convergence_curve
        pos = self.position_history
        mask = fit < threshold
        return pos[mask]
     
      
    

