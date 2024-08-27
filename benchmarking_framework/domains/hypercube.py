from .base import ProtoDomain
import numpy as np
import torch

class Hypercube(ProtoDomain):
    def __init__(self, bounds):
        self.bounds = bounds
        self.bounds_tensor = torch.tensor(self.bounds, dtype=torch.double).T
        self.n_dimensions = len(bounds)

    def is_within_domain(self, x):
        lower_bounds, upper_bounds = zip(*self.bounds)
        in_bounds = np.all(np.array(lower_bounds) <= np.array(x)) and np.all(np.array(x) <= np.array(upper_bounds))
        return in_bounds

    def sample(self, n_samples=1):
        return np.array([[np.random.uniform(lower, upper) for lower, upper in self.bounds] for _ in range(n_samples)])

if __name__ == "__main__":
    # Hypercube domain example
    hypercube = Hypercube(bounds=[(0, 1), (0, 1), (0, 1)])
    hypercube_samples = hypercube.sample(n_samples=10)
    print("Hypercube Samples:")
    print(hypercube_samples)
