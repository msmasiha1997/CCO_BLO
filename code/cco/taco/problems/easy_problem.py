# # Author: Yassine Laguel
# # License: GNU GPL V3

# import numpy as np
# from .toy_problem import ToyProblem


# class EasyProblem(ToyProblem):
#     def __init__(self, data):
#         self.data = data

#     def objective_func(self, x):
#         return (x - 2) ** 2

#     def objective_grad(self, x):
#         return 2 * x - 4

#     def constraint_func(self, x, z):
#         return x * z - 1

#     @staticmethod
#     def constraint_grad(x, z):
#         return z