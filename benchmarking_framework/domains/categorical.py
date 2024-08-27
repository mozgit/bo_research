import logging
import numpy as np
from itertools import product
from .base import ProtoDomain

class CategoricalDomain(ProtoDomain):
    def __init__(self, categorical_dimensions):
        """
        Initializes the CategoricalDomain with specific categorical dimensions.

        Args:
            categorical_dimensions (list of list): A list where each element is a list of possible categories for that dimension.
                                                   For example, [['cat', 'dog'], ['red', 'green', 'blue']].
        """
        self.categorical_dimensions = categorical_dimensions
        self.bounds = None
        self.n_dimensions = len(categorical_dimensions)
        self.logger = logging.getLogger(__name__)
        self.category_mask = np.ones(self.n_dimensions)


    def is_within_domain(self, x):
        """
        Checks if the given point is within the categorical domain.

        Args:
            x (array-like): The point to check.

        Returns:
            bool: True if the point is within the domain, False otherwise.
        """
        for xi, values in zip(x, self.categorical_dimensions):
            if xi not in values:
                return False
        return True

    def sample(self, n_samples=1):
        """
        Samples points from the categorical domain.

        Args:
            n_samples (int): Number of points to sample.

        Returns:
            np.ndarray: An array of sampled points.
        """
        samples = []
        for _ in range(n_samples):
            sample = [np.random.choice(values) for values in self.categorical_dimensions]
            samples.append(sample)
        return np.array(samples)

    def encode(self, x):
        """
        Encodes the categorical inputs using one-hot encoding.

        Args:
            x (array-like): The point to encode.

        Returns:
            list: One-hot encoded representation of the point.
        """
        encoded = []
        for xi, values in zip(x, self.categorical_dimensions):
            one_hot = [1 if xi == value else 0 for value in values]
            encoded.extend(one_hot)
        return encoded

    def decode(self, x):
        """
        Decodes the one-hot encoded inputs back to the original categorical values.

        Args:
            x (array-like): The one-hot encoded point to decode.

        Returns:
            list: Original categorical representation of the point.
        """
        self.logger.debug(f"Starting decode with input: {x}")

        # Flatten the input if it's a list of lists
        if isinstance(x, list) and isinstance(x[0], list):
            self.logger.debug("Flattening input list of lists")
            x = np.array(x).flatten()
        else:
            x = np.array(x)

        decoded = []
        index = 0
        for values in self.categorical_dimensions:
            one_hot_length = len(values)
            one_hot = x[index:index + one_hot_length]
            self.logger.debug(f"One-hot segment: {one_hot}, Values: {values}")
            if len(one_hot) == 0:
                self.logger.error("Empty one-hot segment detected. Aborting decode.")
                raise ValueError("Empty one-hot segment detected.")
            if np.sum(one_hot) != 1:
                self.logger.error(f"Invalid one-hot segment: {one_hot}. Aborting decode.")
                raise ValueError(f"Invalid one-hot segment: {one_hot}")
            decoded.append(values[np.argmax(one_hot)])
            index += one_hot_length
        self.logger.debug(f"Decoded output: {decoded}")
        return decoded

    def generate_choices(self):
        """
        Generates all possible combinations of the categorical dimensions.

        Returns:
            list: A list of all possible combinations of the categorical dimensions.
        """
        return list(product(*self.categorical_dimensions))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Categorical domain example
    categorical_domain = CategoricalDomain(categorical_dimensions=[['cat', 'dog', 'mouse'], ['red', 'green', 'blue']])
    categorical_samples = categorical_domain.sample(n_samples=10)
    print("Categorical Domain Samples:")
    print(categorical_samples)

    sample_point = categorical_samples[0]
    encoded_point = categorical_domain.encode(sample_point)
    decoded_point = categorical_domain.decode(encoded_point)

    print("Sample Point:", sample_point)
    print("Encoded Point:", encoded_point)
    print("Decoded Point:", decoded_point)
