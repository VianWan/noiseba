"""
Scipy-based gradient optimization for surface wave dispersion curve inversion.

This module provides gradient-based optimization using scipy.optimize.minimize
with support for various algorithms including L-BFGS-B, TNC, and SLSQP.
"""

import numpy as np
import inspect
from typing import Optional, Dict, Any, Callable, Union
from scipy.optimize import minimize, approx_fprime

def inherit_signature(base_class):
    def decorator(subclass):
        base_sig = inspect.signature(base_class.__init__)
        sub_sig = inspect.signature(subclass.__init__)
        new_params = []
        
        new_params.append(list(base_sig.parameters.values())[0])
        
        # collect base parameter（except self and **kwargs）
        base_kwargs_param = None
        for name, param in base_sig.parameters.items():
            if name == 'self':
                continue
            elif param.kind == param.VAR_KEYWORD:
                base_kwargs_param = param
            else:
                new_params.append(param)
        
        # collect child parameter（except self and **kwargs）
        sub_kwargs_param = None
        for name, param in sub_sig.parameters.items():
            if name == 'self' or name in base_sig.parameters:
                continue
            elif param.kind == param.VAR_KEYWORD:
                sub_kwargs_param = param
            else:
                new_params.append(param)
        
        # Finally, add the **kwargs parameter (use the subclass if available; otherwise, use the base class)
        if sub_kwargs_param:
            new_params.append(sub_kwargs_param)
        elif base_kwargs_param:
            new_params.append(base_kwargs_param)
        
        
        new_sig = base_sig.replace(parameters=new_params)
        subclass.__init__.__signature__ = new_sig
        
        return subclass
    return decorator




class ScipyOptimizer:
    """
    Gradient-based optimizer using scipy.optimize.minimize for surface wave inversion.

    This optimizer supports various scipy optimization methods and can use either
    analytical gradients (if provided by the cost function) or numerical gradients.

    Attributes:
        cost_function: Objective function to minimize
        method: Optimization method ('L-BFGS-B', 'TNC', 'SLSQP', etc.)
        bounds: Parameter bounds as list of (lower, upper) tuples
        options: Additional optimization options
        use_analytical_gradient: Whether to use analytical gradients if available
    """

    def __init__(
        self,
        cost_function: Callable,
        method: str = "L-BFGS-B",
        lower_bound: Union[np.ndarray, float] = 0.0,
        upper_bound: Union[np.ndarray, float] = 1.0,
        dimension: Optional[int] = None,
        max_iterations: int = 1000,
        chi_factor: Optional[float] = None,
        model_norm: Optional[float] = None,
        tolerance: float = 1e-6,
        use_analytical_gradient: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the ScipyOptimizer.

        Args:
            cost_function: Objective function to minimize
            method: Optimization method from scipy.optimize.minimize
            lower_bound: Lower bounds for parameters
            upper_bound: Upper bounds for parameters
            dimension: Number of parameters (required)
            max_iterations: Maximum number of iterations
            chi_factor: Early stopping criterion
            tolerance: Convergence tolerance
            use_analytical_gradient: Use analytical gradients if available
            verbose: Print optimization progress
            **kwargs: Additional options passed to scipy.optimize.minimize
        """
        if dimension is None:
            raise ValueError("Dimension must be specified")

        # fmt: off
        self.cost_function      = cost_function
        self.method             = method
        self.dimension          = dimension
        self.max_iterations     = max_iterations
        self.chi_factor         = chi_factor
        self.model_norm         = model_norm                 
        self.tolerance          = tolerance
        self.use_analytical_gradient = use_analytical_gradient
        self.verbose = verbose

        # Set up bounds with proper type handling
        if np.isscalar(lower_bound):
            self.lb = np.full(dimension, float(lower_bound)) # type: ignore
        else:
            self.lb = np.asarray(lower_bound, dtype=float)

        if np.isscalar(upper_bound):
            self.ub = np.full(dimension, float(upper_bound)) # type: ignore
        else:
            self.ub = np.asarray(upper_bound, dtype=float)

        if self.lb.shape[0] != dimension or self.ub.shape[0] != dimension:
            raise ValueError("lower_bound and upper_bound must be scalars or length `dimension` arrays")

        self.bounds = list(zip(self.lb, self.ub))

        # Set up optimization options
        self._options = {"maxiter": max_iterations, "disp": verbose, **kwargs}

        # History tracking
        self.total_misfit_history   = []
        self.chi_factor_history     = []
        self.model_norm_history     = []
        self.position_history       = []

        # Results storage
        self.best_position   = None
        self.best_fitness    = None
        self.result          = None

        # Store additional attributes for compatibility
        self.total_misfit      = np.nan
        self._track_chi        = hasattr(cost_function, "chi_factor") and chi_factor is not None
        self._track_model_norm = hasattr(cost_function, "model_norm") and model_norm is not None
        # fmt: on

    def optimize(self, initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run the optimization.

        Args:
            initial_guess: Initial parameter guess (random if None)

        Returns:
            Dictionary containing optimization results
        """
        # Set initial guess
        if initial_guess is None:
            # Random initial guess within bounds
            initial_guess = np.random.uniform(self.lb, self.ub)
        else:
            initial_guess = np.asarray(initial_guess, dtype=float)

        # Validate initial guess
        if len(initial_guess) != self.dimension:
            raise ValueError(f"Initial guess must have length {self.dimension}")

        # Simplex Method do not need gradient information
        if self.method == "Nelder-Mead":
            jac = None
        else:
            jac = self._gradient_wrapper
        

        # Clip initial guess to bounds
        initial_guess = np.clip(initial_guess, self.lb, self.ub)

        if self.verbose:
            print(f"Starting {self.method} optimization with {self.dimension} parameters")
            if self.method in ["L-BFGS-B", "TNC", "SLSQP"]:
                print(f"Using {'analytical' if self._has_gradient else 'numerical'} gradients")

        # Early stopping
        def callback(x):
            if self._track_chi:
                curr_chi = self.chi_factor_history[-1]
                if curr_chi < self.chi_factor:
                    if self.verbose:
                        print(f"Early stopping with chi_factor={curr_chi:.2e}")
                        return True
            return False

        # Run optimization
        try:
            self.result = minimize(
                fun=self._objective_wrapper,
                x0=initial_guess,
                method=self.method,
                jac=jac,
                bounds=self.bounds,
                options=self.options,
                callback=callback,
            )

            # Store results
            self.best_position = self.result.x
            self.best_fitness = self.result.fun

            # Update compatibility attributes
            self.total_misfit = self.best_fitness

            if self.verbose:
                print(f"Optimization completed in {self.result.nit} iterations")
                print(f"Final fitness: {self.best_fitness:.6e}")
                if hasattr(self.result, "success"):
                    print(f"Success: {self.result.success}")
                if hasattr(self.result, "message"):
                    print(f"Message: {self.result.message}")

        except Exception as e:
            print(f"Optimization failed: {e}")
            # Use best found position
            if len(self.position_history) > 0:
                best_idx = np.argmin(self.total_misfit_history)
                self.best_position = self.position_history[best_idx]
                self.best_fitness = self.total_misfit_history[best_idx]
            else:
                self.best_position = initial_guess
                self.best_fitness = self._objective_wrapper(initial_guess)

            # Update compatibility attributes
            self.total_misfit = self.best_fitness

        return {
            "best_position": self.best_position,
            "best_fitness": self.best_fitness,
            "result": self.result,
            "total_misfit_history": np.array(self.total_misfit_history),
            "chi_factor_history": np.array(self.chi_factor_history),
            "model_norm_history": np.array(self.model_norm_history),
            "position_history": np.array(self.position_history),
        }

    @property
    def _has_gradient(self) -> bool:
        """Check if cost function has gradient method."""
        return (
            self.use_analytical_gradient
            and self.method in ["L-BFGS-B", "TNC", "SLSQP"] 
            and hasattr(self.cost_function, "gradient")
            and callable(getattr(self.cost_function, "gradient"))
        )

    @property
    def options(self):
        opts = self._options.copy()
        self._map_tolerance(opts)
        return opts

    def _map_tolerance(self, opts: dict) -> None:
        raise NotImplementedError("Subclass must implement _map_tolerance")

    def _objective_wrapper(self, x: np.ndarray) -> float:
        """
        Wrapper for the objective function that tracks history.

        Args:
            x: Parameter vector

        Returns:
            Objective function value
        """
        fitness = float(self.cost_function(x))

        # Track history if cost function has the required properties
        chi_val = np.nan
        model_val = np.nan

        if self._track_chi and self.cost_function.chi_factor is not None:
            chi_val = float(self.cost_function.chi_factor)
            self.chi_factor_history.append(chi_val)

        if self._track_model_norm and self.cost_function.model_norm is not None:
            model_val = float(self.cost_function.model_norm)
            self.model_norm_history.append(model_val)

        self.total_misfit_history.append(fitness)
        self.position_history.append(x.copy())

        return fitness

    def _gradient_wrapper(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self.use_analytical_gradient and hasattr(self.cost_function, "gradient"):
            try:
                grad = self.cost_function.gradient(x)
                if grad is not None:
                    grad = np.asarray(grad, dtype=float)
                    if grad.shape != x.shape:
                        raise ValueError(f"Gradient shape {grad.shape} != input shape {x.shape}")
                    if not np.all(np.isfinite(grad)):
                        raise ValueError("Non-finite gradient values detected.")
                    return grad
            except Exception as e:
                print(f"Analytical gradient failed: {e}, falling back to numerical.")

        if self.method in {"L-BFGS-B", "TNC", "SLSQP"}:
            eps = np.sqrt(np.finfo(float).eps)
            grad = approx_fprime(x, self.cost_function, eps)
            if grad.shape != x.shape:
                raise ValueError(f"Numerical gradient shape {grad.shape} != input shape {x.shape}")
            return grad

        return None

    def position_below(self, threshold: float = 1e3) -> Optional[np.ndarray]:
        """
        Return stored positions whose fitness is below threshold.

        Args:
            threshold: Fitness threshold

        Returns:
            Array of positions below threshold
        """
        if len(self.position_history) > 0:
            positions = np.array(self.position_history)
            fitnesses = np.array(self.total_misfit_history)
            mask = fitnesses < threshold
            if np.any(mask):
                return positions[mask]
        return None



# ------------------------------------------------------------------
# LBFGS
# ------------------------------------------------------------------
@inherit_signature(ScipyOptimizer)
class LBFGSOptimizer(ScipyOptimizer):
    """L-BFGS-B optimizer (default for bound-constrained problems)."""

    def __init__(self, *args, gtol: float = 1e-5, **kwargs):
        self.gtol = gtol
        kwargs["method"] = "L-BFGS-B"
        super().__init__(*args, **kwargs)

    def _map_tolerance(self, opts):
        opts.setdefault("ftol", self.tolerance) # fatol: Relative change in function values
        opts.setdefault("gtol", self.gtol) # gtol: Relative change in gradient norm


# ------------------------------------------------------------------
# TNC
# ------------------------------------------------------------------
@inherit_signature(ScipyOptimizer)
class TNCOptimizer(ScipyOptimizer):
    def __init__(self, *args, xtol: float = 1e-8, **kwargs):
        self.xtol = xtol
        kwargs["method"] = "TNC"
        super().__init__(*args, **kwargs)

    def _map_tolerance(self, opts: dict) -> None:
        opts.setdefault("ftol", self.tolerance)   
        opts.setdefault("xtol", self.xtol)        # Relative parameter change


# ------------------------------------------------------------------
# SLSQP
# ------------------------------------------------------------------
@inherit_signature(ScipyOptimizer)
class SLSQPOptimizer(ScipyOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["method"] = "SLSQP"
        super().__init__(*args, **kwargs)

    def _map_tolerance(self, opts: dict) -> None:
        opts.setdefault("ftol", self.tolerance)


# ------------------------------------------------------------------
# Nelder-Mead
# ------------------------------------------------------------------
@inherit_signature(ScipyOptimizer)
class NelderMeadOptimizer(ScipyOptimizer):
    def __init__(self, *args, xatol: float = 1e-4, **kwargs):
        self.xatol = xatol
        kwargs["method"] = "Nelder-Mead"
        kwargs.setdefault("use_analytical_gradient", False)
        super().__init__(*args, **kwargs)

    def _map_tolerance(self, opts: dict) -> None:
        opts.setdefault("fatol", self.tolerance)  # Absolute change in function values
        opts.setdefault("xatol", self.xatol)      # Pure form point distance
