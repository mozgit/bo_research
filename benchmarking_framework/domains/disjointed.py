from .base import ProtoDomain
import numpy as np

class DisjointedDomains(ProtoDomain):
    def __init__(self, bounds, categorical_values):
        self.bounds = bounds
        self.categorical_values = categorical_values
        self.n_dimensions = len(bounds) + len(categorical_values)

    def is_within_domain(self, x):
        in_bounds = all(lower <= xi <= upper for xi, (lower, upper) in zip(x[:-len(self.categorical_values)], self.bounds))
        valid_categorical = all(xi in values for xi, values in zip(x[-len(self.categorical_values):], self.categorical_values))
        return in_bounds and valid_categorical

    def sample(self, n_samples=1):
        samples = []
        for _ in range(n_samples):
            continuous_part = np.array([np.random.uniform(lower, upper) for lower, upper in self.bounds])
            categorical_part = np.array([np.random.choice(values) for values in self.categorical_values])
            samples.append(np.concatenate((continuous_part, categorical_part)))
        return np.array(samples)


if __name__ == "__main__":
    # Disjointed domains example
    disjointed_domains = DisjointedDomains(bounds=[(0, 1), (0, 1)], categorical_values=[[0, 1], ['A', 'B']])
    disjointed_samples = disjointed_domains.sample(n_samples=10)
    print("\nDisjointed Domain Samples:")
    print(disjointed_samples)
