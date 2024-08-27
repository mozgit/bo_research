import numpy as np
from .base import ProtoDomain
import torch


class Hyperplane(ProtoDomain):
    def __init__(self, bounds, total=1.0):
        self.bounds = bounds
        self.bounds_tensor = torch.tensor(self.bounds, dtype=torch.double).T
        self.total = total
        self.equalities = self._define_equalities()
        self.n_dimensions = len(self.bounds)

    def is_within_domain(self, x):
        lower_bounds, upper_bounds = zip(*self.bounds)
        in_bounds = np.all(np.array(lower_bounds) <= np.array(x)) and np.all(np.array(x) <= np.array(upper_bounds))
        on_hyperplane = np.isclose(np.sum(x), self.total, rtol=1e-01)
        return in_bounds and on_hyperplane

    def _define_equalities(self):
        n = len(self.bounds)
        A_eq = np.ones((1, n))
        b_eq = np.array([self.total])

        return torch.tensor(A_eq, dtype=torch.double), torch.tensor(b_eq, dtype=torch.double)

    def get_equality_constraints(self):
        A_eq, b_eq = self.equalities
        constraints = []
        for i in range(A_eq.size(0)):
            indices = torch.nonzero(A_eq[i]).squeeze()
            coefficients = A_eq[i, indices]
            rhs = b_eq[i].item()
            constraints.append((indices, coefficients, rhs))
        return constraints


    def sample(self, n_samples=1):
        samples = []
        n_dim = len(self.bounds)
        lower_bounds, upper_bounds = zip(*self.bounds)

        while len(samples) < n_samples:
            # Sample each dimension uniformly within its bounds
            sample = np.array([np.random.uniform(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)])
            # Scale the sample to ensure the sum equals to self.total
            sample = sample * self.total / np.sum(sample)

            # Check if the scaled sample is within the original bounds
            if self.is_within_domain(sample):
                samples.append(sample)

        return np.array(samples)



# Example usage
if __name__ == "__main__":
    hyperplane = Hyperplane(bounds=[(0, 1), (0, 1), (0, 1)], total=1.0)
    hyperplane_samples = hyperplane.sample(n_samples=10)
    print("\nHyperplane Samples:")
    print(hyperplane_samples)
