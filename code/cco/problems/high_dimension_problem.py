from __future__ import annotations

import numpy as np
from cco.problems.problem import Problem


class HighDimensionProblem(Problem):

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
        self.eps = epsilon
        self.delta = delta
        self.lr = lr
        self.mu = mu
        self.grad_clip = grad_clip
        self.update_clip = update_clip
        self.abstol = abstol
        self.name = "simple_high_dimension_problem"

    def f_function(self, x: np.ndarray) -> float | np.ndarray:
        if x.ndim == 1:
            return np.linalg.norm(x - 2.0, ord=2) ** 2
        return np.linalg.norm(x - 2.0, ord=2, axis=-1) ** 2

    def partial_f_function(self, x: np.ndarray) -> float | np.ndarray:
        return 2 * (x - 2.0)

    def chance_function(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return x @ z.T - 1

    def partial_chance_function(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return z.T

    def z_samples(self, n: int | None = None) -> np.ndarray:
        if n is None:
            n = self.samples_num
        mean = np.zeros(self.dimension[0])
        cov = np.eye(self.dimension[0])
        return np.random.multivariate_normal(mean, cov, size=n)

    # cpp SCIP (pyscipopt) functions
    def cpp_objective_function(self, x):
        sum = 0.0
        for i in range(self.dimension[0]):
            sum += (x[i] - 2.0) ** 2
        return sum

    def cpp_chance_function(self, x, z):
        sum = -1.0
        for i in range(self.dimension[0]):
            sum += x[i] * z[i]
        return sum
