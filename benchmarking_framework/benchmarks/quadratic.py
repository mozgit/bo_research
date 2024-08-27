import numpy as np
from benchmarking_framework import Problem
from benchmarking_framework.domains import CombinedDomain, Hypercube, CategoricalDomain


# Domain Generators

def generate_hypercube_domain(n_continuous_dims):
    """Generate a hypercube domain with the specified number of continuous dimensions."""
    return Hypercube(bounds=[(0, 1)] * n_continuous_dims)


def generate_categorical_domain(n_categorical_dims, n_categories=5):
    """Generate a categorical domain with the specified number of categorical dimensions."""
    categorical_dimensions = [[i for i in range(n_categories)] for _ in range(n_categorical_dims)]
    return CategoricalDomain(categorical_dimensions=categorical_dimensions)


# Problem Generator

def generate_quadratic_problem(n_categorical_dims = 0, n_continuous_dims = 5, n_starting_points = None):
    """
    Generate a quadratic problem with the specified number of categorical and continuous dimensions.

    The problem will be a combination of a quadratic function on a hypercube domain and a quadratic
    function with Manhattan distance on a categorical domain.
    """
    domains = []

    if n_categorical_dims > 0:
        categorical_domain = generate_categorical_domain(n_categorical_dims)
        domains.append(categorical_domain)
    if n_continuous_dims > 0:
        hypercube_domain = generate_hypercube_domain(n_continuous_dims)
        domains.append(hypercube_domain)

    if n_starting_points is None:
        n_starting_points = n_categorical_dims + n_continuous_dims

    combined_domain = CombinedDomain(domains=domains)

    # Define the combined function
    def combined_function(x):
        # Split the input into categorical and continuous parts
        categorical_input = x[:n_categorical_dims]
        continuous_input = x[n_categorical_dims:]

        # Manhattan distance for categorical part
        manhattan_value = 0
        if n_categorical_dims > 0:
            target_point = [0] * n_categorical_dims
            encoded_x = categorical_domain.encode(categorical_input)
            encoded_target = categorical_domain.encode(target_point)
            manhattan_distance = sum(abs(xi - yi) for xi, yi in zip(encoded_x, encoded_target)) / 2
            manhattan_value = -manhattan_distance ** 2

        # L2 norm for continuous part
        l2_value = 0
        if n_continuous_dims > 0:
            l2_value = -np.sum((np.array(continuous_input) - 0.5) ** 2)

        return manhattan_value + l2_value

    name = ""
    if n_categorical_dims > 0:
        name += f"{n_categorical_dims}D Categorical "
    if n_continuous_dims > 0:
        name += f"{n_continuous_dims}D Continuous "
    name +=f"{n_starting_points} Starting Points"

    # Define the problem
    problem = Problem(
        function=combined_function,
        domain=combined_domain,
        objective=lambda y: y,
        name=name,
        n_starting_points=n_starting_points,
        min_objective=-n_categorical_dims ** 2 - n_continuous_dims * 0.25,
        max_objective=0
    )

    return problem


# Example usage
if __name__ == "__main__":
    # Generate a problem with 3 categorical dimensions and 2 continuous dimensions
    problem = generate_quadratic_problem(n_categorical_dims=3, n_continuous_dims=2)

    print("Generated Problem:")
    print("Domain Details:", problem.domain)

    # Sample a point and evaluate the function
    sample_point = problem.domain.sample(n_samples=1)[0]
    print("Sample Point:", sample_point)
    print("Function Value:", problem.function(sample_point))
