import time
import numpy as np
from pyscipopt import Model

from cco.problems.problem import Problem

class KKTBL:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.time_limit = 3000
        self.f = self.problem.cpp_objective_function
        self.f_value = self.problem.f_function
        self.g = self.problem.cpp_chance_function
        self.g_value = self.problem.chance_function
        self.x_dim = self.problem.dimension[0]
        self.delta = self.problem.delta
        self.samples = self.problem.z_samples()
        self.samples_num = len(self.samples)
        self.model = Model("model")

    def solve(self):
        # Initialize the decision variable.
        if type(self.x_dim) == int and self.x_dim == 1:
            x = self.model.addVar(lb=None, ub=None, vtype="C", name="x")
        elif type(self.x_dim) == int and self.x_dim > 1:
            x = {}
            for i in range(self.x_dim):
                x[i] = self.model.addVar(lb=0, ub=None, vtype="C", name="x(%s)" % (i))
        elif type(self.x_dim) == tuple or type(self.x_dim) == list:
            x = {}
            for i in range(self.x_dim[0]):
                for j in range(self.x_dim[1]):
                    x[i, j] = self.model.addVar(lb=None, ub=None, vtype="C", name="x(%s, %s)" % (i, j))
        else:
            raise Exception("The dimension of the decision variable is not supported.")
        # Set the time limit.
        self.model.setRealParam("limits/time", self.time_limit)
        # Encode chance constraint.
        if callable(self.g):
            self.encode(x)
        else:
            raise Exception("g not callable")
        # Add cost function.
        objective = self.model.addVar(lb=None, ub=None, vtype="C", name="obj")
        self.model.addCons(self.f(x) <= objective)
        self.model.setObjective(objective, "minimize")
        # Solve the model.
        self.model.hideOutput()
        time_start = time.time()
        self.model.optimize()
        time_end = time.time()
        if self.model.getStatus() == "optimal":
            sol = self.model.getBestSol()
            if type(self.x_dim) == int and self.x_dim == 1:
                return sol[x], time_end - time_start
            elif type(self.x_dim) == int and self.x_dim > 1:
                return [sol[x[i]] for i in range(self.x_dim)], time_end - time_start
            else:
                opt_sol = []
                for i in range(self.x_dim[0]):
                    row = []
                    for j in range(self.x_dim[1]):
                        row.append(sol[x[i, j]])
                    opt_sol.append(row)
                return opt_sol, time_end - time_start
        else:
            return self.model.getStatus(), time_end - time_start  # Denotes error (or infeasibility)
        
    def encode(self, x):
        """
        Encode the chance constraint via CPP with KKT reformulation.
        """

        # Initialize variables.
        s = self.model.addVar(lb = None, ub = None, vtype = "C", name = "s")

        lambdas, betas, ys = {}, {}, {}
        for i in range(self.samples_num):
            lambdas[i], betas[i], ys[i]= [self.model.addVar(lb=None, ub=None, vtype="C") for j in range(3)]

        # Define the constraints.
        self.model.addCons(s <= 0)

        summation = 0
        for i in range(self.samples_num):
            summation += betas[i]
        self.model.addCons(1 - summation == 0)

        for i in range(self.samples_num):
            self.model.addCons(1/(self.samples_num * self.delta)  - betas[i] - lambdas[i] == 0)
            self.model.addCons(ys[i] >= 0)
            self.model.addCons(ys[i] + s - self.g(x, self.samples[i]) >= 0)
            self.model.addCons(lambdas[i] >= 0)
            self.model.addCons(betas[i] >= 0)
            self.model.addCons(lambdas[i] * ys[i] == 0)
            self.model.addCons(betas[i] * (ys[i] + s - self.g(x, self.samples[i])) == 0)
