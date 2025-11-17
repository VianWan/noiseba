"""
Surface wave dispersion curve inversion module.

This module provides the main inversion class for surface wave dispersion curve
inversion using various optimization algorithms.
"""

import numpy as np
import inspect
from collections import defaultdict
from typing import List, Dict, Optional, Any

from .model import Model, Curve
from noiseba.utils.forward import forward_disp


class Inversion:
    """
    Main inversion class for surface wave dispersion curve optimization.

    This class manages the inversion process using various optimization algorithms
    and provides methods to access inversion results and statistics.

    Attributes:
        costfunction: Cost function instance for evaluating model fitness
        optimizer_cls: Optimizer class to use for inversion
        directives: Dictionary of optimization parameters
        result: Dictionary storing inversion results
        optimizer: Dictionary storing optimizer instances
    """

    def __init__(self, costfunction: Any, optimizer: Any, directives: Dict[str, Any]) -> None:
        """
        Initialize the inversion instance.

        Args:
            costfunction: Cost function instance for evaluating model fitness
            optimizer: Optimizer class to use for inversion
            directives: Dictionary of optimization parameters

        Raises:
            ValueError: If costfunction or optimizer is None
        """
        if costfunction is None:
            raise ValueError("Cost function cannot be None")
        if optimizer is None:
            raise ValueError("Optimizer cannot be None")

        self.costfunction = costfunction
        self.optimizer_cls = optimizer
        self.directives = directives
        self.result: Dict[str, Dict[str, Any]] = {}
        self.optimizer: Dict[str, Any] = {}

    def best_position(self, site: str = "site0") -> Optional[np.ndarray]:
        """
        Return best model parameters for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Best model parameters as numpy array, or None if not found
        """
        return self.result.get(site, {}).get("best_position")

    def best_misfit(self, site: str = "site0") -> Optional[float]:
        """
        Return best fitness value for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Best fitness value, or None if not found
        """
        return self.result.get(site, {}).get("best_fitness")

    def chi_factor(self, site: str = "site0") -> Optional[float]:
        """
        Return chi factor for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Chi factor value, or None if not found
        """
        return self.result.get(site, {}).get("best_chi_factor")

    def model_norm(self, site: str = "site0") -> Optional[float]:
        """
        Return model norm for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Model norm value, or None if not found
        """
        return self.result.get(site, {}).get("best_model_norm")

    def chi_factor_history(self, site: str = "site0") -> np.ndarray:
        """
        Return chi factor history for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Chi factor history as numpy array
        """
        curves = self.result.get(site, {}).get("chi_factor_history")
        if curves is None:
            return np.array([])
        return curves

    def model_norm_history(self, site: str = "site0") -> np.ndarray:
        """
        Return model norm history for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Model norm history as numpy array
        """
        curves = self.result.get(site, {}).get("model_norm_history")
        if curves is None:
            return np.array([])
        return curves

    def total_misfit_history(self, site: str = "site0") -> np.ndarray:
        """
        Return total misfit history for a given site.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Total misfit history as numpy array
        """
        curves = self.result.get(site, {}).get("total_misfit_history")
        if curves is None:
            return np.array([])
        return curves

    def best_curves(self, site: str = "site0") -> Dict[str, List[Curve]]:
        """
        Generate predicted dispersion curves from best model parameters.

        Args:
            site: Site identifier (default: "site0")

        Returns:
            Dictionary of Curve objects grouped by mode

        Raises:
            ValueError: If no best position is available for the site
        """
        curves = self.result.get(site, {}).get("obs_curves")
        if curves is None:
            return {}

        best_pos = self.best_position(site)
        if best_pos is None:
            raise ValueError(f"No best position available for site '{site}'")

        position = np.exp(best_pos)
        
        # Get the model used for this site to determine the mode
        site_result = self.result.get(site, {})
        if not site_result:
            return {}
            
        # Try to get model information from the cost function
        costfunc = self.costfunction if not isinstance(self.costfunction, dict) else self.costfunction.get(site)
        
        # Check if we have gradient mode cost function with fixed thickness
        if (costfunc is not None and
            hasattr(costfunc, '_mode') and
            getattr(costfunc, '_mode', None) == "gradient" and
            hasattr(costfunc, '_fixed_thickness') and
            getattr(costfunc, '_fixed_thickness', None) is not None):
            # Gradient mode: position contains only vs values, need fixed thickness
            model_matrix = Model.model(position, mode="gradient", fixed_thickness=costfunc._fixed_thickness)
        else:
            # Global mode: position contains [thickness, vs]
            model_matrix = Model.model(position)
        
        curves_dict = defaultdict(list)

        for item in curves:
            period, vel = forward_disp(model_matrix, item.period, mode=item.mode, wave_type=item.wave_type)
            curves_dict[f"{item.wave_type[0]}mode{item.mode}"].append(
                Curve(
                    freq=np.flipud(1 / period), 
                    velocity=np.flipud(vel), 
                    wave_type=item.wave_type, 
                    mode=item.mode,
                    x=getattr(item, 'x', None),  # Preserve x coordinate if available
                    y=getattr(item, 'y', None)   # Preserve y coordinate if available
                )
            )

        return dict(curves_dict)

    def history_positions(self, site: str = "site0", threshold: float = 1e3) -> Optional[np.ndarray]:
        """
        Return historical positions below a fitness threshold.

        Args:
            site: Site identifier (default: "site0")
            threshold: Fitness threshold for filtering positions

        Returns:
            Array of historical positions below threshold, or None if not available
        """
        optimizer = self.optimizer.get(site)
        if optimizer is not None and hasattr(optimizer, "position_below"):
            return optimizer.position_below(threshold)
        return None

    def history_curves(self, site: str = "site0", threshold: float = 1e3) -> Dict[str, List[Curve]]:
        """
        Generate predicted curves from historical positions below threshold.

        Args:
            site: Site identifier (default: "site0")
            threshold: Fitness threshold for filtering positions

        Returns:
            Dictionary of Curve objects grouped by mode
        """
        positions = self.history_positions(site, threshold=threshold)
        if positions is None or len(positions) == 0:
            return {}

        # Convert positions to linear scale
        positions = np.exp(positions)
        obs_curves = self.result.get(site, {}).get("obs_curves")
        if obs_curves is None:
            return {}

        # Get the model used for this site to determine the mode
        costfunc = self.costfunction if not isinstance(self.costfunction, dict) else self.costfunction.get(site)
        
        curves_dict = defaultdict(list)

        for p in positions:
            # Check if we have gradient mode cost function with fixed thickness
            if (costfunc is not None and
                hasattr(costfunc, '_mode') and
                getattr(costfunc, '_mode', None) == "gradient" and
                hasattr(costfunc, '_fixed_thickness') and
                getattr(costfunc, '_fixed_thickness', None) is not None):
                # Gradient mode: position contains only vs values, need fixed thickness
                model_matrix = Model.model(p, mode="gradient", fixed_thickness=costfunc._fixed_thickness)
            else:
                # Global mode: position contains [thickness, vs]
                model_matrix = Model.model(p)
                
            for item in obs_curves:
                period, vel = forward_disp(model_matrix, item.period, mode=item.mode, wave_type=item.wave_type)
                curves_dict[f"{item.wave_type[0]}mode{item.mode}"].append(
                    Curve(
                        freq=np.flipud(1 / period), 
                        velocity=np.flipud(vel), 
                        wave_type=item.wave_type, 
                        mode=item.mode,
                        x=getattr(item, 'x', None),  # Preserve x coordinate if available
                        y=getattr(item, 'y', None)   # Preserve y coordinate if available
                    )
                )

        return dict(curves_dict)

    def run(self, model: Model) -> None:
        """
        Run inversion using the provided model and optimizer.

        Supports both single-site and multi-site processing. The inversion results
        are stored in the `result` attribute.

        Args:
            model: Model instance containing parameter bounds and dimensions

        Raises:
            ValueError: If model bounds are invalid or optimization fails
            RuntimeError: If optimizer initialization or execution fails
        """
        # Validate model bounds
        lb = model.lower_bounds
        ub = model.upper_bounds

        if lb is None or ub is None:
            raise ValueError("Model bounds cannot be None")
        if len(lb) != len(ub):
            raise ValueError("Lower and upper bounds must have the same length")
        if np.any(lb >= ub):
            raise ValueError("Lower bounds must be less than upper bounds")

        # Set fixed thickness for gradient mode in cost function
        if not isinstance(self.costfunction, dict):
            if hasattr(self.costfunction, 'set_fixed_thickness') and model.mode == "gradient":
                self.costfunction.set_fixed_thickness(model.fixed_thickness)

        # Prepare optimization directives
        directives = {"lower_bound": lb, "upper_bound": ub, "dimension": model.dimension}
        directives.update(self.directives)

        # Filter optimizer arguments based on its __init__ signature
        try:
            sig = inspect.signature(self.optimizer_cls.__init__)
            filter_kwargs = {k: v for k, v in directives.items() if k in sig.parameters}
            # print('filter_kwargs', filter_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to inspect optimizer signature: {e}")

        try:
            if isinstance(self.costfunction, dict):
                # Multi-site inversion
                self._run_multi_site(model, filter_kwargs)
            else:
                # Single-site inversion
                self._run_single_site(filter_kwargs)
        except Exception as e:
            raise RuntimeError(f"Inversion failed: {e}")

    def _run_single_site(self, filter_kwargs: Dict[str, Any]) -> None:
        """Run single-site inversion."""
        optimizer = self.optimizer_cls(self.costfunction, **filter_kwargs)
        self.optimizer["site0"] = optimizer
        optimizer.optimize()
        self.result["site0"] = {
            "best_position": optimizer.best_position,
            "best_fitness": optimizer.best_fitness,
            "obs_curves": self.costfunction.curves,
            "best_chi_factor": optimizer.chi_factor,
            "best_model_norm": optimizer.model_norm,
            "chi_factor_history": optimizer.chi_factor_history,
            "model_norm_history": optimizer.model_norm_history,
            "total_misfit_history": optimizer.total_misfit_history,
        }

    def _run_multi_site(self, model: Model, filter_kwargs: Dict[str, Any]) -> None:
        """Run multi-site inversion."""
        for site, costfunc in self.costfunction.items():
            # Set fixed thickness for gradient mode
            if hasattr(costfunc, 'set_fixed_thickness') and model.mode == "gradient":
                costfunc.set_fixed_thickness(model.fixed_thickness)
                
            optimizer = self.optimizer_cls(costfunc, **filter_kwargs)
            self.optimizer[site] = optimizer
            optimizer.optimize()
            self.result[site] = {
                "best_position": optimizer.best_position,
                "best_fitness": optimizer.best_fitness,
                "obs_curves": costfunc.curves,
                "best_chi_factor": optimizer.chi_factor,
                "best_model_norm": optimizer.model_norm,
                "chi_factor_history": optimizer.chi_factor_history,
                "model_norm_history": optimizer.model_norm_history,
                "total_misfit_history": optimizer.total_misfit_history,
            }
