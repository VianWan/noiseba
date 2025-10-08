import numpy as np
import inspect
from collections import defaultdict
from typing import List, Dict

from .model import Model, Curve
from noiseba.utils.forward import forward_disp



class Inversion:
    def __init__(self, costfunction, optimizer, directives) -> None:
        """
        Inversion class to manage dispersion curves optimization.
        """
        self.costfunction = costfunction
        self.optimizer_cls = optimizer
        self.directives = directives
        self.result = {}
        self.optimizer = {}

    def best_position(self, site='site0'):
        """Return best model parameters for a given site."""
        return self.result.get(site, {}).get('best_position')

    def best_fitness(self, site='site0'):
        """Return best fitness value for a given site."""
        return self.result.get(site, {}).get('best_fitness')

    def best_curves(self, site='site0') -> Dict[str, List[Curve]]:
        """
        Generate predicted dispersion curves from best model parameters.
        Returns a dictionary of Curve objects grouped by mode.
        """
        curves = self.result.get(site, {}).get('obs_curves')
        if curves is None:
            return {}

        model = Model.model(self.best_position(site))
        curves_dict = defaultdict(list)

        for item in curves:
            period, vel = forward_disp(model, item.period, mode=item.mode, wave_type=item.wave_type)
            curves_dict[f'mode{item.mode}'].append(
                Curve(np.flipud(1 / period), np.flipud(vel), item.wave_type, item.mode)
            )

        return dict(curves_dict)

    def convergence_curve(self, site='site0'):
        """Return convergence history for a given site."""
        return self.result.get(site, {}).get('convergence_curve')

    def history_positions(self, site='site0', threshold=1e3):
        """
        Return historical positions below a fitness threshold.
        Requires optimizer to implement `position_below(threshold)`.
        """
        optimizer = self.optimizer.get(site)
        if optimizer is not None and hasattr(optimizer, 'position_below'):
            return optimizer.position_below(threshold)
        return None

    def history_curves(self, site='site0', threshold=1e3):
        """
        Generate predicted curves from historical positions below threshold.
        Returns a dictionary of Curve objects grouped by mode.
        """
        positions = self.history_positions(site, threshold=threshold)
        if positions is None:
            return {}

        obs_curves = self.result.get(site, {}).get('obs_curves')
        if obs_curves is None:
            return {}

        curves_dict = defaultdict(list)

        for p in positions:
            model = Model.model(p)
            for item in obs_curves:
                period, vel = forward_disp(model, item.period, mode=item.mode, wave_type=item.wave_type)
                curves_dict[f'mode{item.mode}'].append(
                    Curve(np.flipud(1 / period), np.flipud(vel), item.wave_type, item.mode)
                )

        return dict(curves_dict)

    def run(self, model: Model):
        """
        Run inversion using the provided model and optimizer.
        Supports single-site and multi-site processing.
        """
        lb = model.lower_bounds
        ub = model.upper_bounds
        directives = {
            'lower_bound': lb,
            'upper_bound': ub,
            'dimension': model.dimension
        }
        directives.update(self.directives)

        # Filter optimizer arguments based on its __init__ signature
        sig = inspect.signature(self.optimizer_cls.__init__)
        filter_kwargs = {k: v for k, v in directives.items() if k in sig.parameters}

        if isinstance(self.costfunction, dict):
            # Multi-site inversion
            for site, costfunc in self.costfunction.items():
                optimizer = self.optimizer_cls(costfunc, **filter_kwargs)
                self.optimizer[site] = optimizer
                optimizer.optimize()
                self.result[site] = {
                    'best_position': optimizer.best_position,
                    'best_fitness': optimizer.best_fitness,
                    'convergence_curve': optimizer.convergence_curve,
                    'obs_curves': costfunc.curves
                }
        else:
            # Single-site inversion
            optimizer = self.optimizer_cls(self.costfunction, **filter_kwargs)
            self.optimizer['site0'] = optimizer
            optimizer.optimize()
            self.result['site0'] = {
                'best_position': optimizer.best_position,
                'best_fitness': optimizer.best_fitness,
                'convergence_curve': optimizer.convergence_curve,
                'obs_curves': self.costfunction.curves
            }
