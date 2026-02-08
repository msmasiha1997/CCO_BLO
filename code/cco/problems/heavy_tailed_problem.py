import numpy as np
from cco.problems.problem import Problem
from scipy import stats


class HeavyTailedProblem(Problem):

    def __init__(
        self,
        initial_x: float | list | np.ndarray,
        samples_num: int,
        max_iter_s: int,
        max_iter_x: int,
        theta: float,
        epsilon: float = 0.01,
        delta: float = 1.0 - 0.95,
        lr: float = 0.001,
        mu: float = 1,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        abstol: float | None = None,
    ):
        super().__init__(
            initial_x,
            samples_num,
            max_iter_s,
            max_iter_x,
            theta,
            epsilon,
            delta,
            lr,
            mu,
            grad_clip,
            update_clip,
            abstol,
        )

        # Store initial_x before dimension computation to avoid mutation
        self._compute_dimension(initial_x)
        self.samples_num = samples_num
        self.max_iter_s = max_iter_s
        self.max_iter_x = max_iter_x
        self.theta = theta
        self.epsilon = epsilon
        self.delta = delta
        self.lr = lr
        self.mu = mu
        self.grad_clip = grad_clip
        self.update_clip = update_clip
        self.abstol = abstol
        self.name = "heavy_tailed_problem"

    def chance_function(
        self, x: np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        return x * z - 1

    def partial_chance_function(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        return z

    def f_function(self, x: np.ndarray) -> np.ndarray:
        return (x - 2.0) ** 2

    def partial_f_function(self, x: float | np.ndarray) -> float | np.ndarray:
        return 2 * x - 4

    def z_samples(self, n: int | None = None) -> np.ndarray:
        if n is None:
            n = self.samples_num
        mu = 1.0
        sigma = 4.0
        return stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n)

    # cpp SCIP (pyscipopt) functions
    def cpp_objective_function(self, x):
        return (x - 2) ** 2

    def cpp_chance_function(self, x, z):
        return x * z - 1

