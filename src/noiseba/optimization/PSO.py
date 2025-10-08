from typing import Callable, Optional, Union
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed

class PSO:
    def __init__(self, 
                objective_function,
                population_size: int = 50,
                max_iterations: int = 200, 
                c1: float = 2.1, 
                c2: float = 2.1, 
                lower_bound: Union[np.ndarray, float] = 0., 
                upper_bound: Union[np.ndarray, float] = 1., 
                dimension: Optional[int] = None, 
                njobs: int =1,
                rng_seed: Optional[int] = None):
        r"""
        Particle Swarm Optimizer with constriction coefficient.

        Parameters
        ----------
        objective_function : Callable
            Either a function that accepts a single particle (1D array) and returns scalar fitness,
            or a function that accepts a batch (shape (pop_size, dim)) and returns 1D array of fitness.
        swarm_size : int
        max_iterations : int
        c1, c2 : float
            Cognitive and social coefficients. Require c1 + c2 > 4 for constriction variant.
        lower_bound, upper_bound : float or ndarray
            Bounds may be scalars or length-d arrays.
        dim : int
            Dimensionality of the search space.
        n_jobs : int
            Number of jobs for joblib fallback (useful when objective is not vectorized).
        rng_seed : Optional[int]
            Random seed for reproducibility.
    """
        if dimension is None:
            raise ValueError("Dimension must be specified")
        
        self.N = population_size
        self.max_iter = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.phi = self.c1 + self.c2
        assert self.phi > 4, "c1 + c2 must be greater than 4"
        self.ksi = 2.0 / np.abs(2.0 - self.phi - np.sqrt(self.phi ** 2 - 4.0 * self.phi))
        self.lb = np.asarray(lower_bound)
        self.ub = np.asarray(upper_bound)
        self.dim = dimension
        self.fobj = objective_function
        self.n_jobs = njobs
        self.rng = np.random.default_rng(rng_seed)

        # Initialize population and velocity
        self.particle = self.rng.uniform(
            low=self.lb, high=self.ub, size=(self.N, self.dim)
        )
        self.vmax = 0.2 * np.abs(self.ub - self.lb)
        self.vel = self.rng.uniform(-1.0, 1.0, size=(self.N, self.dim)) * self.vmax

        # calculate fitness  
        # self.pbest_fitness = np.array([self.fobj(p) for p in self.particle])
        self.pbest_fitness = self._evaluate_fitness(self.particle)
        idx = np.argmin(self.pbest_fitness)
        
        # initialize personal and global best
        self.pbest_position = self.particle.copy()
        self.gbest_position = self.pbest_position[idx].copy()
        self.gbest_fitness = self.pbest_fitness[idx]
        self.stop_threshold = 1e-300

        # store history
        self.convergence_curve = np.full(self.max_iter, np.nan) # storing globle best fitness
        self.position_history = np.full((self.N, self.dim, self.max_iter), np.nan)
        self.fit_history = np.full((self.N, self.max_iter), np.nan)

    @property
    def best_position(self):
        return self.gbest_position
    @property
    def best_fitness(self):
        return self.gbest_fitness

    def optimize(self):
        """
        Optimize the objective function.
        """

        pbar = trange(
            self.max_iter,
            dynamic_ncols=True,
            leave=True,
            desc="PSO Optimizing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}"
            )

        for iteration in pbar:
            # 1. update particle velocity
            w = 0.9 - 0.5 * (iteration / self.max_iter) 
            self.vel = self._update_vel(self.ksi, w, self.c1, self.c2, self.vel, self.particle, self.pbest_position, self.gbest_position)
            
            # vmax = self.vmax * (1 - iteration / self.max_iter)
            # self.vel = np.clip(self.vel, -vmax, vmax) # clip velocity

            # 2. update particle position
            self.particle += self.vel
            out_of_bounds = (self.particle < self.lb) | (self.particle > self.ub)
            self.vel[out_of_bounds] *= 0.5
            self.particle = np.clip(self.particle, self.lb, self.ub).copy()

            # 2.1 random jump for avoiding premature convergence
            jump_prob = 0.005 + 0.005 * (1.0 - iteration / self.max_iter)
            # for i in range(self.N):
            #     if np.random.rand() < jump_prob:
            #         self.particle[i] = np.random.uniform(low=self.lb, high=self.ub, size=self.dim)
            if self.rng.random() < jump_prob:
                # random subset to reinitialize
                mask = self.rng.random(self.N) < 0.02  # e.g., ~2% of swarm
                if np.any(mask):
                    k = np.count_nonzero(mask)
                    self.particle[mask, :] = self.rng.uniform(self.lb, self.ub, (k, self.dim))

            # 2.2 In the later stages of optimization, keep exploring
            if iteration > 0.9 * self.max_iter:
                noise = 1e-12 * np.random.randn(*self.particle.shape)
                self.particle *= (1 - noise)

            # 3. update particle best position
            # fitness = self._evaluate_fitness(self.particle)
            fitness = np.array([self.fobj(p) for p in self.particle])
            better = fitness < self.pbest_fitness               
            self.pbest_fitness[better] = fitness[better].copy()
            self.pbest_position[better] = self.particle[better].copy()

            # 4. update global best position
            gid = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[gid] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[gid]
                self.gbest_position = self.pbest_position[gid].copy()

            # 5. restrat 
            self.restart_particles(iteration)
            
            # meet stop criteria and stop in advance
            if self.gbest_fitness < self.stop_threshold:
                self.convergence_curve[iteration:] = self.gbest_fitness
                self.position_history[:, :, iteration] = self.pbest_position.copy()
                self.fit_history[:, iteration] = self.pbest_fitness.copy()
                break;
            
            # store particle history action
            self.convergence_curve[iteration] = self.gbest_fitness
            self.position_history[:, :, iteration] = self.pbest_position.copy()
            self.fit_history[:, iteration] = self.pbest_fitness.copy()

            pbar.set_postfix({"misfit": f"{self.gbest_fitness:.6e}"})


    def _evaluate_fitness(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for an array of positions:
        - If objective accepts batch array (positions shape (N, dim) -> returns (N,)), use it.
        - Otherwise fallback to joblib Parallel over rows (or list comprehension if n_jobs==1).
        """
        try:
            # try batch call
            result = self.fobj(positions)
            result = np.asarray(result, dtype=float)
            if result.ndim == 1 and result.shape[0] == positions.shape[0]:
                return result
            # otherwise fall through to per-particle evaluation
        except Exception:
            # fall back to per-particle evaluation
            pass

        # fallback: per-particle evaluation (parallel if requested)
        if self.n_jobs == 1:
            fitness = np.array([float(self.fobj(pos)) for pos in positions], dtype=float)
        else:
            # joblib parallel
            fitness_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.fobj)(pos) for pos in positions
            )
            fitness = np.array(fitness_list, dtype=float)
        return fitness

    @staticmethod
    def _update_vel(ksi, w, c1, c2, vel, particle, p_best, g_best):
        N, dim = particle.shape
        r1 = np.random.rand(N, dim)
        r2 = np.random.rand(N, dim)
        # w = 0
        vel =  w * vel + ksi * (c1 * r1 * (p_best - particle) + c2 * r2 * (g_best - particle))
        return vel
    
    def restart_particles(self, iteration):
        r"""Restart worst particles if swarm radius is too small."""
        popsize, ndim = self.particle.shape
        diffs = self.particle - self.gbest_position[np.newaxis, :]
        norms = np.linalg.norm(diffs, axis=1)
        radius = np.max(norms) / np.sqrt(4.0 * self.dim)

        # Adaptive threshold
        delta = np.log(1.0 + 0.003 * popsize) / max(0.2, np.log(0.01 * self.max_iter))

        if radius < delta:
            inorm = iteration / self.max_iter
            gamma = 1.0  # competitivity factor
            nw = int((popsize - 1.0) / (1.0 + np.exp(1.0 / 0.09 * (inorm - gamma + 0.5))))

            if nw > 0:
                idx = self.pbest_fitness.argsort()[-nw:]  # worst particles
                self.vel[idx] = np.zeros((nw, ndim))
                self.particle[idx] = np.random.uniform(self.lb, self.ub, (nw, ndim))
                self.pbest_position[idx] = self.particle[idx].copy()
                self.pbest_fitness[idx] = self._evaluate_fitness(self.particle[idx])

    def position_below(self, threshold: float=1e3):
        if self.position_history is not None and self.fit_history is not None:
            fit = self.fit_history.T.ravel()
            pos = self.position_history.transpose(2, 0, 1).reshape(-1, self.dim)
            mask = fit < threshold
            return pos[mask]
        else:
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
        "population_size" : 20,
        "max_iterations" : max_iteratons,
        "c1": 2.1,
        "c2": 2.5,
        "lower_bound" : lb,
        "upper_bound" : ub,
        "dimension": dimensions,
        "njobs": 10
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
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    scat = ax.scatter([], [], c='red', s=50)
    ax.set_xlim(-5.12, 5.12)
    ax.set_ylim(-5.12, 5.12)
    fig.colorbar(contour)
    ax.set_title('Rastrigin Function (2D)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid(True)

    for i in range(max_iteratons)[::10]:
        scat.set_offsets(positions[:,:, i])
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
        term1 = np.sin(np.pi * w[0])**2
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        sum_term = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        return term1 + sum_term + term3
