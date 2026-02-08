import numpy as np
from scipy.stats import norm

"""
Dimensionality of the problem:
- x is a vector of dimension d
- z is a vector of dimension d
- s is a scalar
- alpha is a scalar
"""


class Problem:

    def __init__(
        self,
        initial_x: float | list | np.ndarray,
        samples_num: int,
        max_iter_s: int,
        max_iter_x: int,
        theta: float,
        eps: float = 0.01,
        delta: float = 1 - 0.95,
        lr: float = 0.001,
        mu: float = 1,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        abstol: float | None = None,
    ):
        """
        Initialize the Problem class with all necessary parameters.

        Args:
            initial_x: initial x value used by the algorithm (can be scalar, list, or numpy array)
            I: number of samples z for the chance constraint
            max_iter: number of iterations for the whole algorithm
            delta: 1-delta quantiles; scales down sums in the G function and its partial derivatives (must be in (0,1))
            mu: used in the H function
            delta: used in the h function to determine the interval where we approximate the ReLU function
            update_clipping: if not None, we clip the gradient of GD to this value

        Raises:
            ValueError: if theta_G is not in the valid range (0, 1)
        """

        # Store initial_x before dimension computation to avoid mutation
        self._compute_dimension(initial_x)
        self.samples_num = samples_num
        self.max_iter_s = max_iter_s
        self.max_iter_x = max_iter_x
        self.theta = theta
        self.epsilon = eps
        self.delta = delta
        self.lr = lr
        self.mu = mu
        self.grad_clip = grad_clip
        self.update_clip = update_clip
        self.abstol = abstol
        self.name = "simple_problem"

        if self.__class__ == Problem:
            ppf_value = norm.ppf(1 - self.delta)
            self.optimal_solution = 1 / (ppf_value + 1)
            self.optimal_value = self.f_function(self.optimal_solution)

        # cpp SCIP (pyscipopt) inputs

    def _compute_dimension(self, initial_x) -> None:
        """
        Compute and store the dimension of the problem based on initial_x.

        Args:
            initial_x: Initial value for the decision variable

        Note: This method stores initial_x and computes its dimension.
        """
        if isinstance(initial_x, list):
            self.initial_x = np.array(initial_x)
            self.dimension = self.initial_x.shape
        elif isinstance(initial_x, np.ndarray):
            self.initial_x = initial_x.copy()
            self.dimension = self.initial_x.shape
        else:
            self.initial_x = np.array([initial_x])
            self.dimension = self.initial_x.shape
            # TODO: check the following
            # # Convert scalar to array for consistent handling
            # self.initial_x = np.array([initial_x])
            # self.dimension = self.initial_x.shape

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
        return np.random.normal(1, 1, n)

    def h_function(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x <= -self.theta,
            0,
            np.where(
                x >= self.theta,
                x,
                (-1 / (16 * self.theta**3)) * x**4
                + (3 / (8 * self.theta)) * x**2
                + (1 / 2) * x
                + (3 / 16) * self.theta,
            ),
        )

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x <= -self.theta,
            0,
            np.where(
                x >= self.theta,
                1,
                (-1 / (4 * self.theta**3)) * x**3
                + (3 / (4 * self.theta)) * x
                + (1 / 2),
            ),
        )

    def xx_partial_h_function(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x <= -self.theta,
            0,
            np.where(
                x >= self.theta,
                0,
                (-3 / (4 * self.theta**3)) * x**2 + (3 / (4 * self.theta)),
            ),
        )

    # cpp SCIP (pyscipopt) functions
    def cpp_objective_function(self, x):
        return (x - 2) ** 2

    def cpp_chance_function(self, x, z):
        return x * z - 1
