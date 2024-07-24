import numpy as np
from benchmarking_framework import Problem, CategoricalDomain

# Define the Manhattan distance function for one-hot encoded points
def manhattan_distance_one_hot(x, y):
    return sum(abs(xi - yi) for xi, yi in zip(x, y))/2

# Define the function that quadratically decreases with Manhattan distance
def quadratic_manhattan_function(x, target, domain):
    encoded_x = domain.encode(x)
    encoded_target = domain.encode(target)
    distance = manhattan_distance_one_hot(encoded_x, encoded_target)
    return -distance ** 2

# Define the objective function for the benchmark problem
def objective(y):
    return y

# Define the target point (for simplicity, let's use the first value of each dimension)
target_point = [0] * 20  # Example target point for the given domain

# Define the discrete domain with 20 dimensions, each of size 5
categorical_dimensions = [
    [0, 1, 2, 3, 4] for _ in range(5)
]
categorical_quadratic_domain = CategoricalDomain(categorical_dimensions=categorical_dimensions)

# Define the function for the problem
def function(x):
    return float(quadratic_manhattan_function(x, target_point, categorical_quadratic_domain))

# Create the problem instance
categorical_quadratic_problem = Problem(
    function=function,
    domain=categorical_quadratic_domain,
    objective=objective,
    name='Categorical Quadratic Manhattan',
    n_starting_points=5
)

if __name__ == "__main__":
    print("Categorical Quadratic Manhattan Problem:")
    print("Target Point:", target_point)
    print("Domain Dimensions:", categorical_quadratic_domain.categorical_dimensions)
    # Example usage
    print("Categorical Domain Samples:")
    print(categorical_quadratic_domain.sample(n_samples=10))

    # Example of evaluating the function
    sample_point = categorical_quadratic_domain.sample(n_samples=1)[0]
    print("Sample Point:", sample_point)
    print("Function Value:", function(sample_point))

