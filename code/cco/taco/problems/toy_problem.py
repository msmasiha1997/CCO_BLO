from __future__ import annotations

from cco.problems.problem import Problem

import numpy as np

class ToyProblem():
    def __init__(self, problem: Problem):
        self.problem = problem
        self.name = self.problem.name
        self.data = self.problem.z_samples()

    def objective_func(self, x: np.ndarray):
        return self.problem.f_function(x)

    def objective_grad(self, x: np.ndarray):
        return self.problem.partial_f_function(x)

    def constraint_func(self, x: np.ndarray, z: float | np.ndarray):
        return self.problem.chance_function(x, z)

    def constraint_grad(self, x: np.ndarray, z: float | np.ndarray):
        return self.problem.partial_chance_function(x, z)
