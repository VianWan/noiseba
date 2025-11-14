
from typing import Callable, Optional, Union
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import trange


class APSO:
    """
    Particle Swarm Optimizer with several practical optimizations and options.

    Key features
    - Constriction coefficient (stable velocity scaling)
    - Vectorized RNG per-iteration (fewer RNG calls)
    - Optional local-best (lbest) neighborhood to improve exploration
    - Adaptive inertia weight (non-linear/exponential decay)
    - Light-weight restart strategy for stagnation
    - Optional thin-history saving to reduce memory footprint

    Parameters
    ----------
    objective_function : Callable
        Either a function that accepts a single particle (1D array) and returns a scalar fitness,
        or a function that accepts a batch (shape (pop_size, dim)) and returns 1D array of fitness.
    population_size : int
        Number of particles in the swarm.
    max_iterations : int
        Maximum number of iterations.
    c1, c2 : float
        Cognitive and social coefficients. Recommend c1 + c2 > 4 for constriction variant.
    lower_bound, upper_bound : float or ndarray
        Search bounds (scalars or arrays of length `dimension`).
    dimension : int
        Dimensionality of the search space (required).
    njobs : int
        Number of jobs for joblib fallback for non-vectorized objectives.
    rng_seed : Optional[int]
        RNG seed for reproducibility.
    use_lbest : bool
        Whether to use local-best neighborhood (lbest) instead of global best (gbest).
    lbest_k : int
        Neighborhood half-size for lbest (each particle considers indices i-k..i+k modulo N).
    history_thinning : int
        Store history every `history_thinning` iterations (1 means store all). Use larger values to save memory.
    callback : Optional[Callable]
        Optional callback called as callback(iteration, gbest_fitness, gbest_position).
    verbose : bool
        If True, tqdm progress bar will show misfit updates.
    """

    def __init__(
        self,
        objective_function: Callable,
        population_size: int = 50,
        max_iterations: int = 200,
        c1: float = 2.1,
        c2: float = 2.1,
        lower_bound: Union[np.ndarray, float] = 0.0,
        upper_bound: Union[np.ndarray, float] = 1.0,
        dimension: Optional[int] = None,
        njobs: int = 1,
        rng_seed: Optional[int] = None,
        use_lbest: bool = False,
        lbest_k: int = 3,
        history_thinning: int = 1,
        callback: Optional[Callable] = None,
        verbose: bool = True,
    ):
        if dimension is None:
            raise ValueError("`dimension` must be specified")

        self.fobj = objective_function
        self.N = int(population_size)
        self.max_iter = int(max_iterations)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.phi = self.c1 + self.c2
        assert self.phi > 4.0, "c1 + c2 must be greater than 4 for constriction variant"
        # constriction coefficient
        self.ksi = 2.0 / np.abs(2.0 - self.phi - np.sqrt(self.phi ** 2 - 4.0 * self.phi))

        # bounds (broadcast to shape (dim,))
        self.lb = np.full(dimension, float(lower_bound)) if np.isscalar(lower_bound) else np.asarray(lower_bound, dtype=float)
        self.ub = np.full(dimension, float(upper_bound)) if np.isscalar(upper_bound) else np.asarray(upper_bound, dtype=float)
        if self.lb.shape[0] != dimension or self.ub.shape[0] != dimension:
            raise ValueError("lower_bound and upper_bound must be scalars or length `dimension` arrays")

        self.dim = int(dimension)
        self.njobs = int(njobs)
        self.rng = np.random.default_rng(rng_seed)

        self.use_lbest = bool(use_lbest)
        self.lbest_k = int(lbest_k)

        # initial swarm positions and velocities
        # uniform in bounds: generate shape (N, dim)
        self.particle = self.rng.uniform(self.lb, self.ub, size=(self.N, self.dim))
        # vmax based on search range per-dimension
        self.vmax = 0.2 * (self.ub - self.lb)
        # initial velocities drawn in [-vmax, vmax]
        self.vel = (self.rng.uniform(-1.0, 1.0, size=(self.N, self.dim)) * self.vmax)

        # evaluate initial fitness (support batch fobj)
        self.pbest_fitness = self._evaluate_fitness(self.particle)
        self.pbest_position = self.particle.copy()  # personal best positions
        idx = int(np.argmin(self.pbest_fitness))
        self.gbest_position = self.pbest_position[idx]  # global best position (or used if no lbest)
        self.gbest_fitness = float(self.pbest_fitness[idx])

        # stopping and bookkeeping
        self.stop_threshold = 1e-300
        self.convergence_curve = np.full(self.max_iter, np.nan, dtype=float)

        # history thinning: store only every `history_thinning` iterations
        self.history_thinning = max(1, int(history_thinning))
        history_len = (self.max_iter + self.history_thinning - 1) // self.history_thinning
        # position history stores pbest positions at saved iterations
        self.position_history = np.full((self.N, self.dim, history_len), np.nan, dtype=float)
        self.fit_history = np.full((self.N, history_len), np.nan, dtype=float)

        self.callback = callback
        self.verbose = bool(verbose)

    @property
    def best_position(self):
        return self.gbest_position

    @property
    def best_fitness(self):
        return self.gbest_fitness

    def optimize(self):
        """Run the PSO optimization loop."""
        pbar = trange(self.max_iter, dynamic_ncols=True, leave=True, desc="PSO Optimizing") if self.verbose else range(self.max_iter)

        saved_idx = 0
        for iteration in pbar:
            # --- adaptive inertia weight (exponential-like schedule) ---
            # early iterations: larger w (more exploration), late iterations: smaller w (exploitation)
            t = iteration / max(1, self.max_iter - 1)
            w = 0.9 * np.exp(-2.5 * t) + 0.4 * (1.0 - np.exp(-2.5 * t))

            # --- vectorized random matrices for velocity update ---
            r1 = self.rng.random((self.N, self.dim))
            r2 = self.rng.random((self.N, self.dim))

            # choose social attractor: global best or local best per particle
            if self.use_lbest:
                gbest_matrix = self._get_local_best_matrix()
            else:
                # broadcast global best to shape (N, dim) without copying repeatedly
                gbest_matrix = np.broadcast_to(self.gbest_position, (self.N, self.dim))

            # velocity update using constriction coefficient and vectorized operations
            # vel = w * vel + ksi * (c1*r1*(pbest - particle) + c2*r2*(gbest - particle))
            cognitive = self.c1 * r1 * (self.pbest_position - self.particle)
            social = self.c2 * r2 * (gbest_matrix - self.particle)
            self.vel = w * self.vel + self.ksi * (cognitive + social)

            # optional velocity clipping per-dimension
            vmax_mat = np.broadcast_to(self.vmax, (self.N, self.dim))
            np.clip(self.vel, -vmax_mat, vmax_mat, out=self.vel)

            # position update
            self.particle += self.vel

            # handle out-of-bounds particles: dampen velocity and clip positions
            below_lb = self.particle < self.lb
            above_ub = self.particle > self.ub
            out_of_bounds = below_lb | above_ub
            if np.any(out_of_bounds):
                # damp velocity where out of bounds
                self.vel[out_of_bounds] *= 0.5
                # clip positions in-place
                np.clip(self.particle, self.lb, self.ub, out=self.particle)

            # occasional probabilistic random reinitialization to maintain exploration
            # chance ~ 0.02 * decreasing factor across iterations
            reinit_prob = 0.02 * (0.005 + 0.005 * (1.0 - t))
            mask = self.rng.random(self.N) < reinit_prob
            if np.any(mask):
                k = int(mask.sum())
                self.particle[mask] = self.rng.uniform(self.lb, self.ub, size=(k, self.dim))

            # tiny multiplicative noise late in optimization to avoid premature freeze
            if iteration > 0.9 * self.max_iter:
                noise = 1e-12 * self.rng.standard_normal(size=self.particle.shape)
                self.particle *= (1.0 - noise)

            # --- evaluate fitness for all particles (batch or fallback) ---
            fitness = self._evaluate_fitness(self.particle)

            # update personal bests where improved
            better_mask = fitness < self.pbest_fitness
            if np.any(better_mask):
                self.pbest_fitness[better_mask] = fitness[better_mask]
                # assign slices directly (no copy unless other code mutates them later)
                self.pbest_position[better_mask] = self.particle[better_mask]

            # update global best from personal bests
            gid = int(np.argmin(self.pbest_fitness))
            if float(self.pbest_fitness[gid]) < float(self.gbest_fitness):
                self.gbest_fitness = float(self.pbest_fitness[gid])
                self.gbest_position = self.pbest_position[gid]

            # adaptive restart if swarm radius collapses
            self.restart_particles(iteration)

            # early stopping check
            self.convergence_curve[iteration] = self.gbest_fitness
            if self.gbest_fitness < self.stop_threshold:
                # fill remaining convergence curve with same value
                self.convergence_curve[iteration + 1 :] = self.gbest_fitness
                # store last history slice
                if iteration % self.history_thinning == 0:
                    self.position_history[:, :, saved_idx] = self.pbest_position
                    self.fit_history[:, saved_idx] = self.pbest_fitness
                break

            # store thinned history
            if iteration % self.history_thinning == 0:
                self.position_history[:, :, saved_idx] = self.pbest_position
                self.fit_history[:, saved_idx] = self.pbest_fitness
                saved_idx += 1

            # optional callback (user hook)
            if self.callback is not None:
                try:
                    self.callback(iteration, float(self.gbest_fitness), self.gbest_position)
                except Exception:
                    # avoid user callback breaking optimizer
                    pass

            # update progress bar
            if self.verbose:
                try:
                    pbar.set_postfix({"misfit": f"{self.gbest_fitness:.6e}"})
                except Exception:
                    pass

        return self.gbest_position, self.gbest_fitness

    def _evaluate_fitness(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for an array of positions.
        - Try batch call first. If result shape matches, return it.
        - Otherwise fallback to per-particle evaluation using joblib when njobs > 1.
        """
        positions = np.asarray(positions, dtype=float)
        try:
            result = self.fobj(positions)
            result = np.asarray(result, dtype=float)
            if result.ndim == 1 and result.shape[0] == positions.shape[0]:
                return result
            # else fall through to per-particle evaluation
        except Exception:
            # fall back to per-particle evaluation
            pass

        if self.njobs == 1:
            fitness = np.array([float(self.fobj(pos)) for pos in positions], dtype=float)
        else:
            fitness_list = Parallel(njobs=self.njobs, backend="threading")(
                delayed(self.fobj)(pos) for pos in positions
            )
            fitness = np.array(fitness_list, dtype=float)
        return fitness

    def _get_local_best_matrix(self) -> np.ndarray:
        """
        Compute a per-particle local-best matrix (shape (N, dim)).
        Each particle considers a neighborhood of indices [i-k, ..., i+k] (mod N)
        and adopts the best personal best among them as its social attractor.
        Efficient for moderate N; avoids heavy spatial nearest-neighbor computations.
        """
        # For small k relative to N this is cheap. Implemented in index space.
        local_best = np.empty((self.N, self.dim), dtype=float)
        idxs = np.arange(self.N)
        for i in range(self.N):
            # neighborhood indices with wrap-around
            neighbors = np.mod(np.arange(i - self.lbest_k, i + self.lbest_k + 1), self.N)
            # choose neighbor with best pbest fitness
            best_neighbor = neighbors[int(np.argmin(self.pbest_fitness[neighbors]))]
            local_best[i] = self.pbest_position[best_neighbor]
        return local_best

    def restart_particles(self, iteration: int):
        """Restart worst particles when swarm radius is too small (adaptive threshold).

        The method updates pbest positions/fitness for restarted particles by re-evaluating.
        """
        # compute per-particle distance to current best (Euclidean norm)
        diffs = self.particle - self.gbest_position
        norms = np.linalg.norm(diffs, axis=1)
        radius = float(np.max(norms) / np.sqrt(4.0 * max(1, self.dim)))

        # adaptive threshold (empirical formula)
        delta = np.log(1.0 + 0.003 * self.N) / max(0.2, np.log(max(1, 0.01 * self.max_iter)))

        if radius < delta:
            inorm = iteration / max(1, self.max_iter - 1)
            gamma = 1.0
            # compute number of worst particles to restart (smooth schedule)
            nw = int((self.N - 1.0) / (1.0 + np.exp(1.0 / 0.09 * (inorm - gamma + 0.5))))
            if nw > 0:
                # pick worst indexes by pbest fitness
                idx = np.argsort(self.pbest_fitness)[-nw:]
                # zero velocities and reinitialize positions
                self.vel[idx] = 0.0
                self.particle[idx] = self.rng.uniform(self.lb, self.ub, size=(idx.size, self.dim))
                # update personal bests for restarted particles
                self.pbest_position[idx] = self.particle[idx]
                self.pbest_fitness[idx] = self._evaluate_fitness(self.particle[idx])

    def position_below(self, threshold: float = 1e3):
        """
        Return stored positions (from history) whose recorded fitness is below `threshold`.
        Uses thinned history saved during optimization.
        """
        # flatten stored fit history (only filled slots are meaningful)
        fits = self.fit_history.ravel()
        pos = self.position_history.transpose(2, 0, 1).reshape(-1, self.dim)
        mask = fits < threshold
        if np.any(mask):
            return pos[mask]
        return np.empty((0, self.dim), dtype=float)


# End of PSO class
