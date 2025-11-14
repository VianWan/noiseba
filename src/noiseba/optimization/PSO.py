from typing import Optional, Union
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed


class PSO:
    def __init__(
        self,
        objective_function,
        population_size: int = 50,
        max_iterations: int = 200,
        c1: float = 2.1,
        c2: float = 2.1,
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
        Particle Swarm Optimization (PSO)
        """
        if dimension is None:
            raise ValueError("Dimension must be specified")

        self.N = population_size
        self.max_iter = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.phi = self.c1 + self.c2
        assert self.phi > 4, "c1 + c2 must be greater than 4"
        self.ksi = 2.0 / np.abs(2.0 - self.phi - np.sqrt(self.phi**2 - 4.0 * self.phi))
        self.lb = np.asarray(lower_bound)
        self.ub = np.asarray(upper_bound)
        self.dim = dimension
        self.fobj = objective_function
        self.njobs = njobs
        self.rng = np.random.default_rng(rng_seed)

        # Stop criteria
        self.chi_factor = chi_factor
        self.model_norm = model_norm
        self.total_misfit_threshold = total_misfit_threshold if total_misfit_threshold is not None else 1e-300

        # Initialize population and velocity
        self.particle = self.rng.uniform(low=self.lb, high=self.ub, size=(self.N, self.dim))
        self.vmax = 0.2 * np.abs(self.ub - self.lb)
        self.vel = self.rng.uniform(-1.0, 1.0, size=(self.N, self.dim)) * self.vmax

        # calculate fitness
        self.pbest_fitness = self._evaluate_fitness(self.particle)
        idx = np.argmin(self.pbest_fitness)

        # initialize personal and global best
        self.pbest_position = self.particle.copy()
        self.gbest_position = self.pbest_position[idx].copy()
        self.gbest_fitness = self.pbest_fitness[idx]

        # storing global best fitness
        self.total_misfit_history = np.full(self.max_iter, np.nan)
        self.chi_factor_history = np.full(self.max_iter, np.nan)
        self.model_norm_history = np.full(self.max_iter, np.nan)
        self.position_history = np.full((self.N, self.dim, self.max_iter), np.nan)
        self.misfit_history = np.full((self.N, self.max_iter), np.nan)

        # stop criteria
        self._track_chi = hasattr(self.fobj, "chi_factor") and self.chi_factor is not None
        self._track_model = hasattr(self.fobj, "model_norm") and self.model_norm is not None

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
            desc="PSO Optimizing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
        )

        for iteration in pbar:
            # 1. update particle velocity
            w = 0.9 - 0.5 * (iteration / self.max_iter)
            self.vel = self._update_vel(
                self.ksi,
                w,
                self.c1,
                self.c2,
                self.vel,
                self.particle,
                self.pbest_position,
                self.gbest_position,
            )

            # 2. update particle position
            self.particle += self.vel
            out_of_bounds = (self.particle < self.lb) | (self.particle > self.ub)
            self.vel[out_of_bounds] *= 0.5
            self.particle = np.clip(self.particle, self.lb, self.ub).copy()

            # 2.1 random jump for avoiding premature convergence
            jump_prob = 0.005 + 0.005 * (1.0 - iteration / self.max_iter)
            if self.rng.random() < jump_prob:
                mask = self.rng.random(self.N) < 0.02  # ~2 % of swarm
                if np.any(mask):
                    k = np.count_nonzero(mask)
                    self.particle[mask, :] = self.rng.uniform(self.lb, self.ub, (k, self.dim))

            # 2.2 In the later stages of optimization, keep exploring
            if iteration > 0.9 * self.max_iter:
                noise = 1e-12 * self.rng.normal(size=self.particle.shape)
                self.particle *= 1 - noise

            # 3. update particle best position
            fitness = self._evaluate_fitness(self.particle)
            better = fitness < self.pbest_fitness
            self.pbest_fitness[better] = fitness[better].copy()
            self.pbest_position[better] = self.particle[better].copy()

            # 4. update global best position
            gid = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[gid] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[gid]
                self.gbest_position = self.pbest_position[gid].copy()

            # 5. restart
            self.restart_particles(iteration)

            # 6. meet stop criteria and stop in advance
            need_call = self._track_chi or self._track_model
            if need_call:
                _ = self.fobj(self.best_position)

            curr_chi = np.nan
            curr_model_norm = np.nan
            if self._track_chi:
                curr_chi = self.fobj.chi_factor
            if self._track_model:
                curr_model_norm = self.fobj.model_norm

            criteria1 = self.gbest_fitness < self.total_misfit_threshold
            criteria2 = self._track_chi and curr_chi < self.chi_factor # type: ignore # 
            criteria3 = self._track_model and curr_model_norm < self.model_norm # type: ignore
            if criteria1 or criteria2 or criteria3:
                self._fill_tail_history(iteration, curr_chi, curr_model_norm)
                break

            # 7. store particle history
            self._record_iteration(iteration, curr_chi, curr_model_norm, fitness)

            # 8. progress bar
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

        if self.njobs == 1:
            fitness = np.array([float(self.fobj(pos)) for pos in positions], dtype=float)
        else:
            fitness_list = Parallel(n_jobs=self.njobs, backend="threading")(
                delayed(self.fobj)(pos) for pos in positions
            )
            fitness = np.array(fitness_list, dtype=float)
        return fitness

    # ---------------- velocity update ----------------
    def _update_vel(self, ksi, w, c1, c2, vel, particle, p_best, g_best):
        N, dim = particle.shape
        r1 = self.rng.random((N, dim))
        r2 = self.rng.random((N, dim))
        vel = w * vel + ksi * (c1 * r1 * (p_best - particle) + c2 * r2 * (g_best - particle))
        return vel

    # ---------------- restart ----------------
    def restart_particles(self, iteration):
        r"""Restart worst particles if swarm radius is too small."""
        popsize, ndim = self.particle.shape
        diffs = self.particle - self.gbest_position[np.newaxis, :]
        norms = np.linalg.norm(diffs, axis=1)
        radius = np.max(norms) / np.sqrt(4.0 * self.dim)

        delta = np.log(1.0 + 0.003 * popsize) / max(0.2, np.log(0.01 * self.max_iter))

        if radius < delta:
            inorm = iteration / self.max_iter
            gamma = 1.0
            nw = int((popsize - 1.0) / (1.0 + np.exp(1.0 / 0.09 * (inorm - gamma + 0.5))))
            if nw > 0:
                idx = self.pbest_fitness.argsort()[-nw:]  # worst particles
                self.vel[idx] = 0.0
                self.particle[idx] = self.rng.uniform(self.lb, self.ub, (nw, ndim))
                self.pbest_position[idx] = self.particle[idx]
                self.pbest_fitness[idx] = self._evaluate_fitness(self.particle[idx])

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

    # ---------------- extract history ----------------
    def position_below(self, threshold: float = 1e3):
        if self.position_history is not None and self.misfit_history is not None:
            fit = self.misfit_history.T.ravel()
            pos = self.position_history.transpose(2, 0, 1).reshape(-1, self.dim)
            mask = fit < threshold
            return pos[mask]
        return None


if __name__ == "__main__":

    def rastrigin(x):
        x = np.asarray(x)
        n = x.size
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)

    dimensions = 2
    lb = np.ones(dimensions) * -5.12
    ub = np.ones(dimensions) * 5.12
    max_iteratons = 200
    config = {
        "population_size": 20,
        "max_iterations": max_iteratons,
        "c1": 2.1,
        "c2": 2.5,
        "lower_bound": lb,
        "upper_bound": ub,
        "dimension": dimensions,
        "njobs": 1,
    }

    pso = PSO(rastrigin, **config)
    pso.optimize()
    positions = pso.position_history

    import matplotlib.pyplot as plt

    def rastriginp(X, Y):
        return 20 + X**2 + Y**2 - 10 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))

    x = np.linspace(-5.12, 5.12, 400)
    y = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x, y)
    Z = rastriginp(X, Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    scat = ax.scatter([], [], c="red", s=50)
    ax.set_xlim(-5.12, 5.12)
    ax.set_ylim(-5.12, 5.12)
    fig.colorbar(contour)
    ax.set_title("Rastrigin Function (2D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.grid(True)

    for i in range(max_iteratons)[::10]:
        scat.set_offsets(positions[:, :, i])
        plt.pause(0.1)

    plt.show()

    # Testing functions
    def ackley(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)

    def griewank(x):
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1

    def schwefel(x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def levy(x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        return term1 + sum_term + term3
