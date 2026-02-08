import numpy as np
from scipy.stats import norm
from cco.problems.problem import Problem
from scipy import stats

"""
Dimensionality of the problem:
- x is a vector of dimension d
- z is a vector of dimension d
- s is a scalar
- alpha is a scalar
"""


class TacoProblem(Problem):

    def __init__(
        self,
        initial_x: float | list | np.ndarray,
        samples_num: int,
        max_iter_s: int,
        max_iter_x: int,
        theta: float,
        epsilon: float = 0.01,
        delta: float = 1 - 1/3,
        lr: float = 0.001,
        mu: float = 1,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        abstol: float | None = None
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
        # TODO
        self.chance_dimension = ()
        self.name = "taco_problem"

        self.a = np.array([2.0, 2.0], dtype=np.float64)
        r = np.sqrt(2) / 2
        rot = np.array([[r, -1.0 * r], [r, r]], dtype=np.float64)
        inv_rot = np.array([[r, r], [-1.0 * r, r]], dtype=np.float64)
        mat_in = np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float64)
        self.Q = np.dot(inv_rot, np.dot(mat_in, rot))
        self.w = np.array([1.0, 1.0], dtype=np.float64)

    def f_function(self, x: np.ndarray) -> float | np.ndarray:

        diff = x - self.a
        if diff.ndim == 1:
            return 0.5 * float(np.dot(diff, np.dot(self.Q, diff)))

        vals = 0.5 * np.einsum('ij,jk,ik->i',diff, self.Q, diff)
        # tmp = diff @ self.geo_a
        # vals = 0.5 * np.sum(diff * tmp, axis=1)
        
        return vals

    def partial_f_function(self, x: np.ndarray) -> float | np.ndarray:
        diff = x - self.a
        if diff.ndim == 1:
            return np.dot(self.Q, diff)

        return diff @ self.Q.T
    
    def chance_function(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        if z.ndim == 1:
            # Single sample (vector)
            a = np.dot(z.T, np.dot(self.mat_w(x), z))
        else:
            # Multiple samples (matrix: each row is a sample)
            W = self.mat_w(x)
            a = np.einsum('ij,jk,ik->i', z, W, z)
        b = np.dot(z, self.w)
        c = -1.0
        res = a + b + c
        return res

    def partial_chance_function(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        if z.ndim == 1:
            g0 = 2 * x[0] * z[0] ** 2
            g1 = 3 * np.sign(x[1] - 1.0) * (z[1] * (x[1] - 1.0)) ** 2
        else:
            g0 = 2 * x[0] * z[:, 0] ** 2
            g1 = 3 * np.sign(x[1] - 1.0) * (z[:, 1] * (x[1] - 1.0)) ** 2
        res = np.array([g0, g1])
        return res

    def z_samples(self, n: int = -1) -> np.ndarray:
        if n == -1:
            n = self.samples_num
        mean = np.array([1.0, 1.0])
        cov = 20 * np.eye(2)
        # TODO: #samples = self.I * self.max_iter? what to do?
        nb_samples = 10000
        return np.random.multivariate_normal(mean, cov, size=n)

    @staticmethod
    def mat_w(x):
        d1 = x[0] ** 2 + 0.5
        d2 = abs((x[1] - 1)) ** 3 + 0.2
        res = np.diag(np.array([d1, d2], dtype=np.float64))
        return res
    
    def cpp_mat_w(self, x):
        d1 = x[0] ** 2 + 0.5
        d2 = abs((x[1] - 1)) ** 3 + 0.2
        res = [[d1, 0.0],
               [0.0, d2]]
        return res

    # cpp SCIP (pyscipopt) functions
    def cpp_objective_function(self, x):
        x_minus_a = [x[i] - self.a[i] for i in range(2)]

        quad_expr = 0.0
        for i in range(2):
            for j in range(2):
                quad_expr += 0.5 * self.Q[i,j] * x_minus_a[i] * x_minus_a[j]

        return quad_expr
    
    def cpp_chance_function(self, x, z):
        W = self.cpp_mat_w(x)
        a = 0.0
        for i in range(2):
            for j in range(2):
                a += z[i] * z[j] * W[i][j]

        b = 0.0
        for i in range(2):
            b += z[i] * self.w[i]
        c = -1.0
        res = a + b + c
        return res

    


