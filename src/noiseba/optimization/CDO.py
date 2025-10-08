import numpy as np

from joblib import Parallel, delayed
from tqdm import trange
from typing import Union, Optional 

class CDO:
    def __init__(self, 
                objective_function,
                population_size: int = 50,
                max_iterations: int = 200, 
                lower_bound: Union[np.ndarray, float] = 0., 
                upper_bound: Union[np.ndarray, float] = 1., 
                dimension: Optional[int] = None,
                track_history: bool = True, 
                njobs: int =-1):        
        """
        Cloud Drift Optimization (CDO) is nature-inspired optimization algorithm. It mathmatically model the cloud formation influenced by weather patterns and climate regulation. 
        In their initial stage, cloud form slowly and move gradually under the influence of local weather patterns. As they mature, Clouds naturally gravitate towards areas with higher 
        moisture content and suitable temperature gradients. The highlight character is (1): dynamic adaptive weighting; (2) a probabilistic two-phase strategy with dynamic thresholds to exploration and exploitation.

        Parameters
        ----------
        population_size : `int`
            Number of coyotes in the population.
        max_iterations : `int`
            Maximum number of iterations to run.
        lower_bound : `array`
            Lower bounds for each dimension.
        upper_bound : `array`
            Upper bounds for each dimension.
        dimension : `int`
            Number of decision variables.
        objective_function : `callable`
            Function accepting a 1D array (dim,) and returning a scalar fitness.
        track_history : `bool, optional`
            If True, record all positions at each iteration.
        n_jobs : `int`,`optional`
            Number of parallel jobs for fitness evaluation (Joblib).
        """
        self.N = population_size
        self.max_iter = max_iterations
        self.lb = np.asarray(lower_bound)
        self.ub = np.asarray(upper_bound)
        self.dim = dimension
        self.fobj = objective_function
        self.track = track_history
        self.n_jobs = njobs

        # Initialize population and weights
        self.particle = np.random.uniform(low=self.lb, high=self.ub, size=(self.N, self.dim))
        self.weights = np.ones((self.N, self.dim))
        self.best_position = np.zeros(self.dim)
        self.best_fitness = np.inf
        self.convergence_curve = np.zeros(self.max_iter)
        self.stop_threshold = 1e-300
        self.z = 0.005
        self._half = self.N / 2

        if self.track:
            self.position_history = np.full((self.N, self.dim, self.max_iter), np.nan)
            self.fit_history = np.full((self.N, self.max_iter), np.nan)
        else:
            self.position_history = None
            self.fit_history = None

    def optimize(self):
        pbar = trange(
            self.max_iter,
            dynamic_ncols=True,
            leave=True,
            desc="CDO Optimizing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}"
            )

        for iteration in pbar:
            # Step 1: evulate fitness
            self.particle = np.clip(self.particle, self.lb, self.ub).copy()
            fitness_values = np.array([self.fobj(ind) for ind in self.particle]) 
            # fitness_list = self._parllel_fitness()
            # fitness_values = np.array(fitness_list)

            # Step 2: sort by fitness, ascending, two tendencies
            sorted_indices = np.argsort(fitness_values)
            fitness_values = fitness_values[sorted_indices]

            self.particle = self.particle[sorted_indices].copy()
            
            best = fitness_values[0]
            worst = fitness_values[-1]
            spread = worst - best + 1e-20
            
            # Step 3: update cloud weight
            # based on its fitness value relative to other particles
            # how cloud influenced by its own position and the positions of others
            self._update_weights(fitness_values, best, spread)

            # Step 4: update global best
            if best < self.best_fitness:
                self.best_fitness = best.copy()
                self.best_position = self.particle[0].copy()

            if best < self.stop_threshold:
                self.convergence_curve[iteration:] = self.best_fitness
                if self.track:
                    self.position_history[:, :, iteration:] = np.nan
                    self.fit_history[:, iteration:] = np.nan
                break

            # Step 5: Radnom pertubation
            a = np.arctanh(-(iteration + 1) / self.max_iter + 1)
            b = 1 - (iteration + 1) / self.max_iter
            self.z = 0.002 + 0.003 * b

            # Step 6: update positions
            self._update_particle(fitness_values, a, b, iteration)

            self.convergence_curve[iteration] = self.best_fitness

            if self.track:
                self.position_history[:, :, iteration] = self.particle
                self.fit_history[:, iteration] = fitness_values
                
            pbar.set_postfix({"misfit": f"{self.best_fitness:.6e}"})

        # if self.track:
        #     return self.best_position, self.best_fitness, self.convergence_curve, self.position_history
        # else:
        #     return self.best_position, self.best_fitness, self.convergence_curve
    

    def _parllel_fitness(self):
        with Parallel(n_jobs=self.n_jobs) as parallel:
            fitness_values = parallel(delayed(self.fobj)(ind) for ind in self.particle)
        return fitness_values


    def _update_weights(self, fitness_values, best, spread):
        """
        Update each cloud weight based on its fitness ranking.
        """
        for i in range(self.N):
            term = ((best - fitness_values[i]) if i <= self._half else (fitness_values[i] - best)) / spread + 1
            for j in range(self.dim):
                factor = (0.3 + 0.7 * np.random.rand()) * np.log10(term)
                self.weights[i, j] = 1 + factor if i <= self._half else 1 - factor

    def _update_particle(self, fitness_values, a, b, iteration):
        for i in range(self.N):
            # cloud shift
            if np.random.rand() < self.z:
                self.particle[i] = np.random.uniform(low=self.lb, high=self.ub, size=self.dim)
            else:
                p = np.tanh(abs(fitness_values[i] - self.best_fitness))
                vb = np.random.uniform(-0.2 * a, 0.2 * a, self.dim)
                vc = np.random.uniform(-0.2 * b, 0.2 * b, self.dim)
                for j in range(self.dim):
                    A, B = np.random.randint(self.N), np.random.randint(self.N) # [0, N)
                    r = np.random.rand()
                    if r < p:
                        self.particle[i, j] = self.best_position[j] + 0.8 * vb[j] * (self.weights[i, j] * self.particle[A, j] - self.particle[B, j])
                    else:
                        self.particle[i, j] = vc[j] * self.particle[i, j]
                        
                    if iteration > 0.9 * self.max_iter:
                        self.particle[i, j] *= (1 - np.random.rand() * 1e-12)

                # print(self.particle[i])
                self.particle[i] = np.clip(self.particle[i], self.lb, self.ub)



    def position_below(self, threshold: float= 1e3):
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

    dimensions = 20
    lb = np.ones(dimensions) * -5.12
    ub = np.ones(dimensions) * 5.12
    max_iteratons = 20
    config = {
        "population_size" : 50,
        "max_iterations" : max_iteratons,
        "lower_bound" : lb,
        "upper_bound" : ub,
        "dimension": dimensions,
        "track_history": True,
        "njob": -1,
    }

    cdo = CDO(rastrigin, **config)
    *_, curve, positions = cdo.optimize()

    import matplotlib.pyplot as plt
    def rastrigin_p(X, Y):
        return 20 + X**2 + Y**2 - 10 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))

    x = np.linspace(-5.12, 5.12, 400)
    y = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_p(X, Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    scat = ax.scatter([], [], c='red', s=10)
    ax.set_xlim(-5.12, 5.12)
    ax.set_ylim(-5.12, 5.12)
    fig.colorbar(contour)
    ax.set_title('Rastrigin Function (2D)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid(True)

    for i in range(max_iteratons)[::1]:
        scat.set_offsets(positions[:,:, i])
        plt.pause(0.1)

    plt.show()