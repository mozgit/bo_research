import logging
import numpy as np
from itertools import product
from .base import ProtoDomain

class DiscreteDomain(ProtoDomain):
    def __init__(self, discrete_dimensions):
        """
        Initializes the DiscreteDomain with specific discrete dimensions.

        Args:
            discrete_dimensions (list of list): A list where each element is a list of possible values for that dimension.
                                                For example, [[0, 1, 2, 3, 4], [-1, 0, 1]].
        """
        self.discrete_dimensions = discrete_dimensions
        self.bounds = None
        self.n_dimensions = len(discrete_dimensions)
        self.logger = logging.getLogger(__name__)

    def is_within_domain(self, x):
        """
        Checks if the given point is within the discrete domain.

        Args:
            x (array-like): The point to check.

        Returns:
            bool: True if the point is within the domain, False otherwise.
        """
        for xi, values in zip(x, self.discrete_dimensions):
            if xi not in values:
                return False
        return True

    def sample(self, n_samples=1):
        """
        Samples points from the discrete domain.

        Args:
            n_samples (int): Number of points to sample.

        Returns:
            np.ndarray: An array of sampled points.
        """
        samples = []
        for _ in range(n_samples):
            sample = [np.random.choice(values) for values in self.discrete_dimensions]
            samples.append(sample)
        return np.array(samples)

    def generate_choices(self):
        """
        Generates all possible combinations of the discrete dimensions.

        Returns:
            list: A list of all possible combinations of the discrete dimensions.
        """
        return list(product(*self.discrete_dimensions))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Discrete domain example
    discrete_domain = DiscreteDomain(discrete_dimensions=[[0, 1, 2, 3, 4], [-1, 0, 1], [10, 20, 30]])
    discrete_samples = discrete_domain.sample(n_samples=10)
    print("Discrete Domain Samples:")
    print(discrete_samples)

    sample_point = discrete_samples[0]
    encoded_point = discrete_domain.encode(sample_point)
    decoded_point = discrete_domain.decode(encoded_point)

    print("Sample Point:", sample_point)
    print("Encoded Point:", encoded_point)
    print("Decoded Point:", decoded_point)
