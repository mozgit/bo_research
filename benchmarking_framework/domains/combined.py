import logging
import numpy as np
from itertools import product
from benchmarking_framework.domains.base import ProtoDomain
from benchmarking_framework.domains.categorical import CategoricalDomain
from benchmarking_framework.domains.discrete import DiscreteDomain
from benchmarking_framework.domains.hyperplane import Hyperplane
from benchmarking_framework.domains.hypercube import Hypercube

class CombinedDomain(ProtoDomain):
    def __init__(self, domains):
        """
        Initializes the CombinedDomain with a list of domains.

        Args:
            domains (list of ProtoDomain): A list of domain instances (e.g., CategoricalDomain, DiscreteDomain, Hyperplane, Hypercube).
        """
        self.domains = domains
        self.logger = logging.getLogger(__name__)

        # Ensure only one continuous domain with bounds
        continuous_domains = [domain for domain in domains if hasattr(domain, 'bounds') and domain.bounds]
        if len(continuous_domains) > 1:
            raise ValueError("Only one domain with non-empty 'bounds' is allowed.")

        # Move the continuous domain to the end if it exists
        if continuous_domains:
            continuous_domain = continuous_domains[0]
            self.domains = [domain for domain in domains if domain != continuous_domain]
            self.domains.append(continuous_domain)
            self.bounds = continuous_domain.bounds
            self.bounds_tensor = continuous_domain.bounds_tensor
            self.get_equality_constraints = continuous_domain.get_equality_constraints
            self.get_inequality_constraints = continuous_domain.get_inequality_constraints
        else:
            self.bounds = None
            self.bounds_tensor = None

        self.category_mask = self.generate_category_mask()
        self.discrete_mask = self.generate_discrete_mask()
        self.n_dimensions = self.get_dimensionality()

    def is_within_domain(self, x):
        """
        Checks if the given point is within the combined domain.

        Args:
            x (array-like): The point to check.

        Returns:
            bool: True if the point is within the domain, False otherwise.
        """
        index = 0
        for domain in self.domains:
            sub_x = x[index:index + domain.get_dimensionality()]
            if not domain.is_within_domain(sub_x):
                return False
            index += domain.get_dimensionality()
        return True

    def sample(self, n_samples=1):
        """
        Samples points from the combined domain.

        Args:
            n_samples (int): Number of points to sample.

        Returns:
            np.ndarray: An array of sampled points.
        """
        samples = []
        for _ in range(n_samples):
            sample = []
            for domain in self.domains:
                sub_sample = domain.sample(1)[0]
                sample.extend(sub_sample)
            samples.append(sample)
        return np.array(samples)

    def generate_category_mask(self):
        category_mask = []
        for d in self.domains:
            if isinstance(d, CategoricalDomain):
                category_mask.append(np.ones(d.n_dimensions))
            else:
                category_mask.append(np.zeros(d.n_dimensions))
        return np.concatenate(category_mask)

    def generate_category_dict(self):
        category_dict = {}
        dimension_index = 0
        for d in self.domains:
            for i in range(d.n_dimensions):
                if not isinstance(d, CategoricalDomain):
                    category_dict[dimension_index] = 0
                else:
                    category_dict[dimension_index] = d.categorical_dimensions[i]
                dimension_index += 1
        return category_dict

    def generate_discrete_dict(self):
        discrete_dict = {}
        dimension_index = 0
        for d in self.domains:
            for i in range(d.n_dimensions):
                if not isinstance(d, DiscreteDomain):
                    discrete_dict[dimension_index] = 0
                else:
                    discrete_dict[dimension_index] = d.discrete_dimensions[i]
                dimension_index += 1
        return discrete_dict

    def generate_discrete_mask(self):
        discrete_mask = []
        for d in self.domains:
            if isinstance(d, DiscreteDomain):
                discrete_mask.append(np.ones(d.n_dimensions))
            else:
                discrete_mask.append(np.zeros(d.n_dimensions))
        return np.concatenate(discrete_mask)

    def generate_choices(self):
        """
        Generates all possible combinations of the choices from the categorical and discrete domains.

        Returns:
            list: A list of all possible combinations of the choices from the categorical and discrete domains as numpy arrays.
        """
        choices_list = []
        for domain in self.domains:
            if isinstance(domain, (CategoricalDomain, DiscreteDomain)):
                choices_list.append(domain.generate_choices())

        if not choices_list:
            return []

        product_choices = list(product(*choices_list))
        combined_choices = np.array([np.concatenate(choice).flatten() for choice in product_choices])
        return combined_choices

    def generate_choices_encoded(self):
        """
        Generates all possible choices and does one-hot encoding for categorical dimensions.

        Returns:
            np.ndarray: A 2D NumPy array where each row is a possible combination of the choices from the categorical and discrete domains.
        """
        choices_list = []
        for domain in self.domains:
            if isinstance(domain, CategoricalDomain):
                # Encode categorical choices
                encoded_choices = [domain.encode(x) for x in domain.generate_choices()]
                choices_list.append(np.array(encoded_choices))
            elif isinstance(domain, DiscreteDomain):
                # Add discrete choices directly
                choices_list.append(np.array(domain.generate_choices()))

        if len(choices_list)==0:
            return np.array([])

        # Generate all possible combinations of the choices
        product_choices = list(product(*choices_list))

        # Flatten the combinations and return as a NumPy array
        combined_choices = np.array([np.concatenate(choice).flatten() for choice in product_choices])

        return combined_choices


    def encode(self, x):
        """
        Encodes the combined inputs using one-hot encoding for categorical dimensions.

        Args:
            x (array-like): The point to encode.

        Returns:
            list: One-hot encoded representation of the point.
        """
        encoded = []
        index = 0
        for domain in self.domains:
            sub_x = x[index:index + domain.get_dimensionality()]
            encoded.extend(domain.encode(sub_x))
            index += domain.get_dimensionality()
        return encoded

    def decode(self, x):
        """
        Decodes the one-hot encoded inputs back to the original categorical values.

        Args:
            x (array-like): The one-hot encoded point to decode.

        Returns:
            list: Original categorical representation of the point.
        """
        decoded = []
        index = 0
        for domain in self.domains:
            sub_x = x[index:index + domain.get_dimensionality()]
            decoded.extend(domain.decode(sub_x))
            index += domain.get_dimensionality()
        return decoded

    def get_dimensionality(self):
        """
        Returns the total dimensionality of the combined domain.

        Returns:
            int: Total dimensionality of the combined domain.
        """
        return sum(domain.n_dimensions for domain in self.domains)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from benchmarking_framework.domains.categorical import CategoricalDomain
    from benchmarking_framework.domains.discrete import DiscreteDomain
    from benchmarking_framework.domains.hyperplane import Hyperplane
    from benchmarking_framework.domains.hypercube import Hypercube

    # Define individual domains
    categorical_domain = CategoricalDomain(categorical_dimensions=[['cat', 'dog', 'mouse'], ['red', 'green', 'blue']])
    discrete_domain = DiscreteDomain(discrete_dimensions=[[0, 1, 2, 3, 4], [-1, 0, 1]])
    hyperplane_domain = Hyperplane(bounds=[(0, 1), (0, 1)], total=1)
    hypercube_domain = Hypercube(bounds=[(0, 1), (0, 1)])

    # Combine domains
    combined_domain = CombinedDomain(domains=[categorical_domain, discrete_domain, hyperplane_domain])
    print("Category mask:", combined_domain.generate_category_mask())

    # Sample points from the combined domain
    combined_samples = combined_domain.sample(n_samples=10)
    print("Combined Domain Samples:")
    print(combined_samples)

    sample_point = combined_samples[0]

    print("Sample Point:", sample_point)
