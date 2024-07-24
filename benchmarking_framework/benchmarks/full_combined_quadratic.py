
from benchmarking_framework.domains.categorical import CategoricalDomain
from benchmarking_framework.domains.discrete import DiscreteDomain
from benchmarking_framework.domains.hyperplane import Hyperplane
from benchmarking_framework.domains.combined import CombinedDomain
from benchmarking_framework import Problem
import numpy as np

# Define continuous domain with constraint that they sum to 1
continuous_domain = Hyperplane(bounds=[(0, 1)] * 4, total=1)

# Define discrete domain
discrete_domain = DiscreteDomain(discrete_dimensions=[np.array([0, 0.25, 0.5, 0.75, 1])]*4)

# Define categorical domain
categorical_dimensions = [
    [0, 1, 2] for _ in range(3)
]
categorical_domain = CategoricalDomain(categorical_dimensions=categorical_dimensions)

combined_quadratic_domain = CombinedDomain(domains=[discrete_domain, categorical_domain, continuous_domain])

def combined_function(x):
    def l2_function(x):
        return -np.sum((np.array(x) - 0.25) ** 2)

    def manhattan_function(x):
        target_point = [0] * 5
        encode = categorical_domain.encode
        def manhattan_distance_one_hot(x, y):
            return sum(abs(xi - yi) for xi, yi in zip(x, y))

        encoded_x = encode(x)
        encoded_target = encode(target_point)
        distance = manhattan_distance_one_hot(encoded_x, encoded_target)
        return -distance ** 2

    # Extract the inputs based on their domains
    discrete_input = x[:2]
    categorical_input = x[2:7]
    continuous_input = x[7:]

    return l2_function(continuous_input) + l2_function(discrete_input) + manhattan_function(categorical_input)

def combined_objective(y):
    return y


combined_quadratic_problem = Problem(
    function=combined_function,
    domain=combined_quadratic_domain,
    objective=combined_objective,
    name='Combined Quadratic',
    max_objective = 0+0+0,
    min_objective = -0.75**2-4*0.75**2-9
)