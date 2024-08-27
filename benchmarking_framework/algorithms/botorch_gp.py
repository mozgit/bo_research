import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf_mixed, optimize_acqf, optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
import logging
import numpy as np
from typing import List, Tuple
from benchmarking_framework.domains import CombinedDomain, CategoricalDomain, DiscreteDomain


class GPOptimization:
    def __init__(self, domain, name="GPs"):
        self.domain = domain
        self.name = name
        self.logger = logging.getLogger(__name__)
        self._initialize_domain_properties()

    def _initialize_domain_properties(self):
        """Initialize properties related to the domain."""
        self.category_mask = getattr(self.domain, "category_mask", [])
        self.discrete_mask = getattr(self.domain, "discrete_mask", [])
        self.n_cat_dims = self._calculate_dimensionality(CategoricalDomain)
        self.n_disc_dims = self._calculate_dimensionality(DiscreteDomain)
        self.n_cont_dims = self.domain.get_dimensionality() - self.n_cat_dims - self.n_disc_dims

    def _calculate_dimensionality(self, domain_type):
        """Calculate the dimensionality for a specific domain type."""
        return sum(domain.get_dimensionality() for domain in self.domain.domains if isinstance(domain, domain_type))

    def train(self, history):
        """Train the Gaussian Process model using the provided history."""
        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)
        cat_dims = [i for i, mask in enumerate(self.category_mask) if mask]

        if cat_dims:
            self.gp = MixedSingleTaskGP(X, Y, cat_dims=cat_dims)
        else:
            self.gp = SingleTaskGP(X, Y)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self

    def adjust_indices_for_new_dimensions(self, constraints: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        """Adjust indices in constraints to account for new dimensions."""
        num_new_dimensions = self.n_cat_dims + self.n_disc_dims
        return [(indices + num_new_dimensions, coefficients, rhs) for indices, coefficients, rhs in constraints]

    def step(self):
        """Perform an optimization step to find the next candidate point."""
        equality_constraints = self._get_constraints('get_equality_constraints')
        inequality_constraints = self._get_constraints('get_inequality_constraints')

        acq_function = LogExpectedImprovement(self.gp, best_f=self.gp.train_targets.max())

        choices = torch.tensor(self.domain.generate_choices(), dtype=torch.double)
        fixed_features_list = self._generate_fixed_features_list(choices)

        if self.n_cont_dims == 0:
            return self._optimize_discrete(acq_function, choices)
        else:
            return self._optimize_mixed(acq_function, choices, fixed_features_list, equality_constraints, inequality_constraints)

    def _get_constraints(self, method_name):
        """Retrieve and adjust constraints from the domain."""
        if hasattr(self.domain, method_name):
            constraints = getattr(self.domain, method_name)()
            adjusted_constraints = self.adjust_indices_for_new_dimensions(constraints)
            self.logger.info(f"Applied domain constraints: {adjusted_constraints}")
            return adjusted_constraints
        return None

    def _generate_fixed_features_list(self, choices):
        """Generate a list of fixed features for mixed optimization."""
        fixed_features_list = []
        num_fixed_dims = sum(domain.get_dimensionality() for domain in self.domain.domains if isinstance(domain, (CategoricalDomain, DiscreteDomain)))

        for choice in choices:
            fixed_features = {i: choice[i] for i in range(num_fixed_dims)}
            fixed_features_list.append(fixed_features)

        return fixed_features_list

    def _optimize_discrete(self, acq_function, choices):
        """Optimize the acquisition function for discrete domains."""
        self.logger.info("Discrete optimization")
        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_function,
            choices=choices,
            max_batch_size=2048,
            q=1,
            unique=True
        )
        return self._process_candidate(candidate)

    def _optimize_mixed(self, acq_function, choices, fixed_features_list, equality_constraints, inequality_constraints):
        """Optimize the acquisition function for mixed domains."""
        self.logger.info("Mixed optimization")
        continuous_bounds = torch.tensor(self.domain.bounds, dtype=torch.double).T
        all_bounds = self._combine_bounds(continuous_bounds)
        opt_kwargs = self._generate_opt_kwargs(acq_function, all_bounds, fixed_features_list, equality_constraints, inequality_constraints)

        candidate, acq_value = optimize_acqf_mixed(**opt_kwargs) if fixed_features_list else optimize_acqf(**opt_kwargs)
        return self._process_candidate(candidate)

    def _combine_bounds(self, continuous_bounds):
        """Combine discrete, categorical, and continuous bounds."""
        discrete_categorical_bounds = self._generate_discrete_categorical_bounds()
        return torch.cat((discrete_categorical_bounds, continuous_bounds), dim=1)

    def _generate_discrete_categorical_bounds(self):
        """Generate bounds for discrete and categorical dimensions."""
        bounds = []
        for domain in self.domain.domains:
            if isinstance(domain, DiscreteDomain):
                bounds.extend([(min(dim), max(dim)) for dim in domain.discrete_dimensions])
            elif isinstance(domain, CategoricalDomain):
                bounds.extend([(min(dim), max(dim)) for dim in domain.categorical_dimensions])
        return torch.tensor(bounds, dtype=torch.double).T

    def _generate_opt_kwargs(self, acq_function, bounds, fixed_features_list, equality_constraints,
                             inequality_constraints):
        """Generate keyword arguments for the optimization function."""
        opt_kwargs = {
            "acq_function": acq_function,
            "bounds": bounds,
            "num_restarts": 5,
            "raw_samples": 20,
            "q": 1,  # Set default value for q
        }
        if fixed_features_list:
            opt_kwargs["fixed_features_list"] = fixed_features_list
        if equality_constraints:
            opt_kwargs["equality_constraints"] = equality_constraints
        if inequality_constraints:
            opt_kwargs["inequality_constraints"] = inequality_constraints

        return opt_kwargs

    def _process_candidate(self, candidate):
        """Process and return the candidate from the optimization."""
        candidate_array = candidate.detach().numpy()
        self.logger.debug(f"Candidate array: {candidate_array}")

        if candidate_array.size == 0:
            self.logger.error("Candidate array is empty. Cannot decode an empty array.")
            raise ValueError("Candidate array is empty. Cannot decode an empty array.")

        return candidate_array


# Example usage
if __name__ == '__main__':
    from benchmarking_framework.domains import CombinedDomain, CategoricalDomain, DiscreteDomain, Hyperplane, Hypercube
    from benchmarking_framework import History

    # Define individual domains
    categorical_domain = CategoricalDomain(categorical_dimensions=[[91, 92, 93], [101, 102, 103]])
    discrete_domain = DiscreteDomain(discrete_dimensions=[[0, 1, 2, 3, 4], [-1, 0, 1]])
    hyperplane_domain = Hyperplane(bounds=[(0, 1), (0, 1)], total=1)

    # Combine domains
    combined_domain = CombinedDomain(domains=[categorical_domain, discrete_domain, hyperplane_domain])

    # Generate synthetic training data
    np.random.seed(42)
    num_train_points = 20

    history = History()
    for i in range(num_train_points):
        cat_point = categorical_domain.sample()
        dis_point = discrete_domain.sample()
        hyp_point = hyperplane_domain.sample()
        point = np.concatenate([cat_point[0], dis_point[0], hyp_point[0]])
        score = np.random.random()
        history.add_record(step=i, point=point, function_value=score, score=score)

    # Initialize the Bayesian optimization
    optimizer = GPOptimization(combined_domain)

    # Train the model with the historical data
    optimizer.train(history)

    # Perform an optimization step to find the next candidate
    next_point = optimizer.step()

    print("Next candidate point:", next_point)
