from __future__ import annotations

import numpy as np
from cco.problems.problem import Problem

try:
    # Used when building SCIP expressions (CPP baselines).
    from pyscipopt import exp as scip_exp  # type: ignore
except Exception:  # pragma: no cover
    # Fallback so the rest of the codebase can run without pyscipopt installed.
    scip_exp = np.exp

"""
Dimensionality of the problem:
- x is a vector of dimension d
- z is a vector of dimension d
- s is a scalar
- alpha is a scalar
"""


class CPPProblem(Problem):

    def __init__(
        self,
        initial_x: float | list | np.ndarray,
        samples_num: int,
        max_iter_s: int,
        max_iter_x: int,
        theta: float,
        epsilon: float = 0.01,
        delta: float = 1.0 - 0.9,
        lr: float = 0.001,
        mu: float = 1,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        abstol: float | None = None
    ):
        """
        Initialize the Problem class with all necessary parameters.

        Args:
            initial_x: initial x value used by the algorithm (can be scalar, list, or numpy array)
            I: number of samples z for the chance constraint
            max_iter: number of iterations for the whole algorithm
            theta_G: 1-theta quantiles; scales down sums in the G function and its partial derivatives (must be in (0,1))
            learning_rate_SGLD: learning rate for the SGLD algorithm
            learning_rate_GD: learning rate for the GD algorithm
            mu: used in the H function
            delta: used in the h function to determine the interval where we approximate the ReLU function
            initial_alpha: initial value for the alpha variable before the GA algorithm
            K: used for the zeroth order H gradient function
            update_clipping: if not None, we clip the gradient of GD to this value

        Raises:
            ValueError: if theta_G is not in the valid range (0, 1)
        """
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
            abstol
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
        self.name = "cpp_problem"

    def chance_function(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 1:
            return 50 * z * np.exp(float(x.reshape(-1)[0])) - 5
        return 50 * z * np.exp(x) - 5

    def partial_chance_function(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 1:
            return 50 * z * np.exp(float(x.reshape(-1)[0]))
        return 50 * z * np.exp(x)

    def z_samples(self, n: int = -1) -> np.ndarray:
        if n == -1:
            n = self.samples_num
        # In the slides/paper we use Z ~ Exp(1/3), i.e. mean=3 => scale=3.
        return np.random.exponential(scale=3.0, size=(n,))

    def f_function(self, x: np.ndarray) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        vals = x**3 * np.exp(x)
        if x.ndim == 0:
            return float(vals)
        if x.ndim == 1:
            return float(vals.sum())
        return vals.sum(axis=-1)

    def partial_f_function(self, x: np.ndarray) -> float | np.ndarray:
        return x**2 * (x + 3) * np.exp(x)

    # cpp SCIP (pyscipopt) functions
    def cpp_objective_function(self, x):
        return (x ** 3) * scip_exp(x)
    
    def cpp_chance_function(self, x, z):
        return scip_exp(x) * (50 * z) - 5

    
