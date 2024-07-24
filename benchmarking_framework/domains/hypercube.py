from .base import ProtoDomain
import numpy as np

class Hypercube(ProtoDomain):
    def __init__(self, bounds):
        self.bounds = bounds
        self.n_dimensions = len(bounds)

    def is_within_domain(self, x):
        in_bounds = all(lower <= xi <= upper for xi, (lower, upper) in zip(x, self.bounds))
        return in_bounds

    def sample(self, n_samples=1):
        return np.array([[np.random.uniform(lower, upper) for lower, upper in self.bounds] for _ in range(n_samples)])

if __name__ == "__main__":
    # Hypercube domain example
    hypercube = Hypercube(bounds=[(0, 1), (0, 1), (0, 1)])
    hypercube_samples = hypercube.sample(n_samples=10)
    print("Hypercube Samples:")
    print(hypercube_samples)
