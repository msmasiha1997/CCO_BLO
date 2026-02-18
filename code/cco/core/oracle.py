from __future__ import annotations

import numpy as np
from ..problems.problem import Problem
from tqdm import tqdm
import json
import os


class Oracle:
    """Oracle class, containing all the useful functions"""

    def __init__(self, problem: Problem):
        self.problem = problem

    def G_function(self, x: np.ndarray, s: float, z_samples: np.ndarray) -> float:
        """empirical superquantile convex formulation - vectorized version"""

        chance_values = self.problem.chance_function(x, z_samples) - s
        expected_estimation = np.mean(self.problem.h_function(chance_values))

        return (
            s
            + expected_estimation / self.problem.delta
            + self.problem.epsilon * (s**2 / 2)
        )

    def s_partial_G_function(
        self, x: np.ndarray, s: float, z_samples: np.ndarray
    ) -> float:
        """differentiation of the g function, with respect to s"""

        chance_values = self.problem.chance_function(x, z_samples) - s
        expected_estimation = np.mean(self.problem.partial_h_function(chance_values))

        return 1 - expected_estimation / self.problem.delta + self.problem.epsilon * s

    def sx_partial_G_function(
        self, x: np.ndarray, s: float, z_samples: np.ndarray
    ) -> np.ndarray:
        """Partial derivative of G function with respect to s and x"""

        chance_values = self.problem.chance_function(x, z_samples) - s
        xx_partial_h_values = self.problem.xx_partial_h_function(chance_values)
        partial_chance_values = self.problem.partial_chance_function(x, z_samples)
        expected_estimation = np.mean(xx_partial_h_values * partial_chance_values)

        return -expected_estimation / self.problem.delta

    def ss_partial_G_function(
        self, x: np.ndarray, s: float, z_samples: np.ndarray
    ) -> np.ndarray:
        """Partial derivative of G function with respect to s and s"""

        chance_values = self.problem.chance_function(x, z_samples) - s
        expected_estimation = np.mean(self.problem.xx_partial_h_function(chance_values))

        return expected_estimation / self.problem.delta + self.problem.epsilon

    def F_function(self, x: np.ndarray, min_s: float, mu: float | None = None) -> float:
        """F function"""
        mu_val = float(self.problem.mu if mu is None else mu)
        return self.problem.f_function(x) + (min_s / 2) * np.maximum(
            min_s / mu_val, 0
        )

    def x_partial_F_function(
        self, x: np.ndarray, min_s: float, z_samples, mu: float | None = None
    ) -> np.ndarray:
        """Differentiation of F function with respect to x."""
        mu_val = float(self.problem.mu if mu is None else mu)
        return self.problem.partial_f_function(x) + self.x_partial_s(
            x, min_s, z_samples
        ) * np.maximum(min_s / mu_val, 0)

    def x_partial_s(self, x, s, z_samples) -> np.ndarray:
        """Partial derivative of s."""
        # TODO: possiblity of division by zero
        # print("=" * 20, self.ss_partial_G_function(x, s, z_samples))

        return -self.sx_partial_G_function(x, s, z_samples) / (
            self.ss_partial_G_function(x, s, z_samples)
        )
        # return -self.sx_partial_G_function(x, s, z_samples) / (
        #     self.ss_partial_G_function(x, s, z_samples) + 1e-10
        # )

    def GD_one_step(self, x: np.ndarray, func: callable) -> np.ndarray:
        """one step Gradient Descent algorithm"""
        gradient = func(x)

        if self.problem.grad_clip:
            gradient = self.clip_gradient(gradient, self.problem.grad_clip)

        return x - self.problem.lr * gradient

    def verify_quantiles(
        self,
        x,
        p: float,
        z_samples: np.ndarray,
    ) -> float:
        samples = self.problem.chance_function(x, z_samples)
        empirical_p_quantile = np.percentile(samples, 100 * p)

        return empirical_p_quantile

    def save_results(
        self, x, f_value: float, empirical_coverage: float, problem_name: str
    ) -> None:
        """Save optimization results to JSON file"""
        json_path = os.path.join(os.path.dirname(__file__), "results.json")

        # Handle file not existing or being empty
        try:
            with open(json_path, "r") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = {}

        results[problem_name] = {
            "x": x.tolist() if isinstance(x, np.ndarray) else x,
            "f_value": f_value,
            "empirical_coverage": empirical_coverage,
        }

        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

    @staticmethod
    def empirical_coverage_history(
        problem: Problem, x_history: np.ndarray, N=5000
    ) -> np.ndarray:
        ec_history = []
        for x in x_history:
            ec = Oracle.empirical_coverage(problem, x, N=N)
            ec_history.append(ec)
        return np.array(ec_history)

    @staticmethod
    def empirical_coverage(problem: Problem, x: float | np.ndarray, N=5000) -> float:
        z_samples = problem.z_samples(n=N)
        chance_values = problem.chance_function(x, z_samples)
        return np.mean(chance_values <= 0)

    @staticmethod
    def suboptimality_f(problem: Problem, x_history) -> np.ndarray:
        # TODO: for --> vectorize
        return np.array(
            [
                abs(problem.f_function(x) - problem.optimal_value)
                / abs(problem.optimal_value)
                for x in x_history
            ]
        )

    @staticmethod
    def ec(problem: Problem, empirical_coverage_history) -> np.ndarray:
        P = 1 - problem.delta
        return np.array(empirical_coverage_history)
        # return np.array([(ec - P) / abs(P) for ec in empirical_coverage_history])

    @staticmethod
    def clip_gradient(gradient, max_norm):
        """
        Clips a single gradient tensor to a maximum norm.

        Args:
            gradient (np.ndarray): Gradient tensor to clip.
            max_norm (float): Maximum allowed L2 norm for this gradient.

        Returns:
            np.ndarray: Clipped gradient.
        """
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_norm:
            gradient = gradient * (max_norm / grad_norm)
        return gradient
