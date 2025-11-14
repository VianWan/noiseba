import math
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange
from typing import Union, Optional


class CDOL:
    def __init__(
        self,
        objective_function,
        population_size: int = 50,
        max_iterations: int = 200,
        lower_bound: Union[np.ndarray, float] = 0.0,
        upper_bound: Union[np.ndarray, float] = 1.0,
        dimension: Optional[int] = None,
        njobs: int = 1,
        rng_seed: Optional[int] = None,
        chi_factor: Optional[float] = None,
        model_norm: Optional[float] = None,
        total_misfit_threshold: Optional[float] = None,
    ):
        r"""
        Cloud Drift Optimization with Levy flight (CDOL)

        Parameters
        ----------
        objective_function : callable
            Must accept (N, dim) array and return (N,) scalars, or (dim,) -> scalar.
        population_size : int, default=50
            Swarm size.
        max_iterations : int, default=200
            Max iterations.
        lower_bound : array-like or float, default=0.0
            Lower bounds.
        upper_bound : array-like or float, default=1.0
            Upper bounds.
        dimension : int, optional
            Number of decision variables (required).
        njobs : int, default=1
            Parallel jobs for fitness evaluation.
        rng_seed : int, optional
            Random seed.
        chi_factor : float, optional
            Target data misfit (chi).
        model_norm : float, optional
            Target model norm.
        total_misfit_threshold : float, optional
            Global fitness threshold for early stopping..
        """
        if dimension is None:
            raise ValueError("Dimension must be specified")

        self.N = population_size
        self.max_iter = max_iterations
        self.lb = np.asarray(lower_bound)
        self.ub = np.asarray(upper_bound)
        self.dim = dimension
        self.fobj = objective_function
        self.n_jobs = njobs
        self.rng = np.random.default_rng(rng_seed)
        self.levy_threshold = 5  # 连续未改善迭代数触发 Levy
        self.mutation_factor = 0.05  # 高斯突变基准强度
        self.levy_beta = 1.5  # Levy 指数
        self.stag_counter = np.zeros(self.N, dtype=int)  # 个体停滞计数器

        # --- stopping criteria ---
        self.chi_factor = chi_factor
        self.model_norm = model_norm
        self.total_misfit_threshold = total_misfit_threshold if total_misfit_threshold is not None else 1e-300
        self._track_chi = hasattr(self.fobj, "chi_factor") and self.chi_factor is not None
        self._track_model = hasattr(self.fobj, "model_norm") and self.model_norm is not None

        # --- swarm ---
        self.particle = self.rng.uniform(low=self.lb, high=self.ub, size=(self.N, self.dim))
        self.weights = np.ones((self.N, self.dim))
        self._half = self.N // 2

        # --- personal & global best ---
        self.pbest_fitness = self._evaluate_fitness(self.particle)
        idx_best = np.argmin(self.pbest_fitness)
        self.gbest_position = self.particle[idx_best].copy()
        self.gbest_fitness = self.pbest_fitness[idx_best]

        # --- history ---
        self.total_misfit_history = np.full(self.max_iter, np.nan)
        self.chi_factor_history = np.full(self.max_iter, np.nan)
        self.model_norm_history = np.full(self.max_iter, np.nan)
        self.position_history = np.full((self.N, self.dim, self.max_iter), np.nan)
        self.misfit_history = np.full((self.N, self.max_iter), np.nan)

    # ---------------- properties ----------------
    @property
    def best_position(self):
        return self.gbest_position

    @property
    def best_fitness(self):
        return self.gbest_fitness

    @property
    def total_misfit(self):
        return self.total_misfit_history[-1]

    @property
    def best_chi_factor(self):
        return self.chi_factor_history[-1]

    @property
    def best_model_norm(self):
        return self.model_norm_history[-1]

    # ---------------- main loop ----------------
    def optimize(self):
        pbar = trange(
            self.max_iter,
            dynamic_ncols=True,
            leave=True,
            desc="CDO Optimizing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
        )

        for iteration in pbar:
            # ---- 1. evaluate ----
            fitness = self._evaluate_fitness(self.particle)
            self.particle = np.clip(self.particle, self.lb, self.ub)

            # ---- 2. sort ----
            sort_idx = np.argsort(fitness)
            fitness = fitness[sort_idx]
            self.particle = self.particle[sort_idx]

            # ---- 3. update global best ----
            if fitness[0] < self.gbest_fitness:
                self.gbest_fitness = fitness[0].copy()
                self.gbest_position = self.particle[0].copy()

            # ---- 4. adaptive weights ----
            best, worst = fitness[0], fitness[-1]
            spread = worst - best + 1e-20
            self._update_weights_vector(fitness, best, spread)

            # ---- 5. early stopping (same as PSO) ----
            curr_chi = np.nan
            curr_model_norm = np.nan
            if self._track_chi or self._track_model:
                _ = self.fobj(self.gbest_position)
                if self._track_chi:
                    curr_chi = self.fobj.chi_factor
                if self._track_model:
                    curr_model_norm = self.fobj.model_norm

            criteria1 = self.gbest_fitness < self.total_misfit_threshold
            criteria2 = self._track_chi and curr_chi < self.chi_factor  # type: ignore
            criteria3 = self._track_model and curr_model_norm < self.model_norm  # type: ignore
            if criteria1 or criteria2 or criteria3:
                self._fill_tail_history(iteration, curr_chi, curr_model_norm)
                break

            # ---- 6. record history ----
            self._record_iteration(iteration, curr_chi, curr_model_norm, fitness)

            # ---- 7. move particles ----
            a = np.arctanh(-(iteration + 1) / self.max_iter + 1)
            b = 1 - (iteration + 1) / self.max_iter
            self._move_particles(fitness, a, b, iteration)

            # ---- 8. progress bar ----
            if self._track_chi:
                pbar.set_postfix({"RMS": f"{curr_chi:.6e}", "Model_norm": f"{curr_model_norm:.6e}"})
            else:
                pbar.set_postfix({"data_misfit": f"{self.gbest_fitness:.6e}"})

    # ---------------- fitness evaluation ----------------
    def _evaluate_fitness(self, positions: np.ndarray) -> np.ndarray:
        try:
            result = self.fobj(positions)
            result = np.asarray(result, dtype=float)
            if result.ndim == 1 and result.shape[0] == positions.shape[0]:
                return result
        except Exception:
            pass

        if self.n_jobs == 1:
            return np.array([float(self.fobj(pos)) for pos in positions], dtype=float)
        else:
            fitness_list = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.fobj)(pos) for pos in positions)
            return np.array(fitness_list, dtype=float)

    # ---------------- vectorized weights update ----------------
    def _update_weights_vector(self, fitness, best, spread):
        rank = np.arange(self.N)
        term = np.where(rank <= self._half, (best - fitness) / spread + 1, (fitness - best) / spread + 1)
        factor = (0.3 + 0.7 * self.rng.random((self.N, 1))) * np.log10(term.reshape(-1, 1))
        self.weights = np.where(rank.reshape(-1, 1) <= self._half, 1 + factor, 1 - factor)

    # ---------------- vectorized position update ----------------

    def _levy_flight(self, n):
        sigma = (
            math.gamma(1 + self.levy_beta)
            * np.sin(np.pi * self.levy_beta / 2)
            / (math.gamma((1 + self.levy_beta) / 2) * self.levy_beta * 2 ** ((self.levy_beta - 1) / 2))
        ) ** (1 / self.levy_beta)
        u = self.rng.normal(0, sigma, (n, self.dim))
        v = self.rng.normal(0, 1, (n, self.dim))
        step = u / (np.abs(v) ** (1 / self.levy_beta))
        # map to bounded space
        return np.clip(self.gbest_position + 0.01 * step * (self.ub - self.lb), self.lb, self.ub)

    def _move_particles(self, fitness, a, b, iteration):
        """
        Enhanced late-stage exploration:
        1. Dual-behaviour: elite vs. diversified
        2. Adaptive Gaussian mutation
        3. Levy-flight restart for stagnated particles
        4. Dynamic weight re-scaling
        """
        z = 0.002 + 0.003 * b
        mask_restart = self.rng.random(self.N) < z
        self.particle[mask_restart] = self.rng.uniform(self.lb, self.ub, (mask_restart.sum(), self.dim))
        self.stag_counter[mask_restart] = 0  # reset stagnation

        # --- Levy restart for long-stagnant individuals ---
        levy_mask = self.stag_counter >= self.levy_threshold
        if np.any(levy_mask):
            self.particle[levy_mask] = self._levy_flight(levy_mask.sum())
            self.stag_counter[levy_mask] = 0

        # --- dual-behaviour update ---
        active = ~(mask_restart | levy_mask)
        idx_active = np.where(active)[0]
        if idx_active.size == 0:
            return

        p = np.tanh(np.abs(fitness[idx_active] - self.gbest_fitness))
        # mutation variance increases towards the end
        mut_strength = self.mutation_factor * (1 + 5 * iteration / self.max_iter)

        vb = self.rng.uniform(-0.2 * a, 0.2 * a, (idx_active.size, self.dim))
        vc = self.rng.uniform(-0.2 * b, 0.2 * b, (idx_active.size, self.dim))
        A = self.rng.integers(0, self.N, idx_active.size)
        B = self.rng.integers(0, self.N, idx_active.size)

        # elite vs diversified strategy
        elite_mask = idx_active <= self._half
        delta = 0.8 * vb * (self.weights[idx_active] * self.particle[A] - self.particle[B])

        new_pos = np.empty_like(self.particle[idx_active])
        new_pos[elite_mask] = self.gbest_position + delta[elite_mask]
        new_pos[~elite_mask] = vc[~elite_mask] * self.particle[idx_active][~elite_mask]

        # adaptive Gaussian mutation
        noise = self.rng.normal(0, mut_strength, new_pos.shape)
        new_pos += noise * (1 - np.abs(new_pos - self.gbest_position) / (self.ub - self.lb + 1e-20))

        # late-stage micro-perturbation
        if iteration > 0.9 * self.max_iter:
            new_pos *= 1 - self.rng.random(new_pos.shape) * 1e-12

        self.particle[idx_active] = np.clip(new_pos, self.lb, self.ub)

        # --- update stagnation counter ---
        improved = fitness < self.pbest_fitness
        self.stag_counter[improved] = 0
        self.stag_counter[~improved] += 1

    # ---------------- history helpers ----------------
    def _record_iteration(self, iteration, curr_chi, curr_model_norm, fitness):
        self.total_misfit_history[iteration] = self.gbest_fitness
        self.chi_factor_history[iteration] = curr_chi
        self.model_norm_history[iteration] = curr_model_norm
        self.position_history[:, :, iteration] = self.particle
        self.misfit_history[:, iteration] = fitness

    def _fill_tail_history(self, iteration, curr_chi, curr_model_norm):
        sl = slice(iteration, None)
        self.total_misfit_history[sl] = self.gbest_fitness
        self.chi_factor_history[sl] = curr_chi
        self.model_norm_history[sl] = curr_model_norm
        self.position_history[:, :, sl] = np.nan
        self.misfit_history[:, sl] = np.nan

    # ---------------- restart placeholder ----------------
    def restart_particles(self, iteration):
        """Reserved for future use; CDO currently needs no restart."""
        pass

    # ---------------- extract history ----------------
    def position_below(self, threshold: float = 1e3):
        if self.position_history is not None and self.misfit_history is not None:
            fit = self.misfit_history.T.ravel()
            pos = self.position_history.transpose(2, 0, 1).reshape(-1, self.dim)
            mask = fit < threshold
            return pos[mask]
        return None
