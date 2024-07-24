import numpy as np
from benchmarking_framework import Problem, DiscreteDomain

# Define the objective function (e.g., a simple 3D quadratic function)
def function(x):
    return np.sum((np.array(x) - 1./3) ** 2)

def objective(y):
    return y

discrete_quadratic_domain = DiscreteDomain(discrete_dimensions=[np.array(range(0, 100))/100, np.array(range(0, 100))/100,np.array(range(0, 100))/100])
discrete_quadratic_problem = Problem(function,
                                    discrete_quadratic_domain,
                                       objective,
                                       name='Discrete Quadratic',
                                       n_starting_points = 5)


