import numpy as np
from benchmarking_framework import Problem, Hyperplane

# Define the objective function (e.g., a simple 3D quadratic function)
def function(x):
    return np.sum((np.array(x) - 1./3) ** 2)

def objective(y):
    return y

quadratic_hyperplane_domain = Hyperplane(bounds=[(0, 1), (0, 1), (0, 1)])
quadratic_hyperplane_problem = Problem(function,
                                    quadratic_hyperplane_domain,
                                       objective,
                                       name='Quadratic Hyperplane',
                                       n_starting_points = 5)

