import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from benchmarking_framework.domains import CombinedDomain, CategoricalDomain, DiscreteDomain
import logging
import numpy as np

class CombinedBayesianOptimization:
    def __init__(self, domain, name="Bayesian Optimization"):
        self.domain = domain
        self.name = name
        self.logger = logging.getLogger(__name__)

    def train(self, history):
        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)

        self.gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self

    def step(self, domain):
        if not isinstance(domain, CombinedDomain):
            raise ValueError("Domain must be an instance of CombinedDomain for mixed optimization.")

        self.logger.debug("Starting step function for Bayesian Optimization.")

        ei = ExpectedImprovement(self.gp, best_f=self.gp.train_targets.max())

        # Generate all possible combinations of discrete and categorical choices
        choices = domain.generate_choices()
        self.logger.debug(f"Generated {len(choices)} choices from the domain.")

        # Encode the choices to torch tensors
        fixed_features_list = []
        num_fixed_dims = sum(domain.get_dimensionality() for domain in domain.domains if isinstance(domain, (CategoricalDomain, DiscreteDomain)))

        for choice in choices:
            fixed_features = {i: choice[i] for i in range(num_fixed_dims)}
            fixed_features_list.append(fixed_features)

        # Calculate bounds for discrete and categorical dimensions
        discrete_categorical_bounds = []
        for domain in domain.domains:
            if isinstance(domain, DiscreteDomain):
                for dim in domain.discrete_dimensions:
                    discrete_categorical_bounds.append((min(dim), max(dim)))
            elif isinstance(domain, CategoricalDomain):
                for dim in domain.categorical_dimensions:
                    discrete_categorical_bounds.append((min(dim), max(dim)))

        # Convert bounds to tensors
        discrete_categorical_bounds_tensor = torch.tensor(discrete_categorical_bounds, dtype=torch.double).T

        # Extract bounds for the continuous domain
        continuous_bounds = torch.tensor(domain.bounds, dtype=torch.double).T
        self.logger.debug(f"Continuous bounds: {continuous_bounds}")

        # Combine all bounds
        all_bounds = torch.cat((discrete_categorical_bounds_tensor, continuous_bounds), dim=1)
        self.logger.debug(f"All bounds: {all_bounds}")

        # Optimize the acquisition function
        self.logger.debug("Optimizing the acquisition function.")
        candidate, acq_value = optimize_acqf_mixed(
            acq_function=ei,
            bounds=all_bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
            fixed_features_list=fixed_features_list
        )

        self.logger.debug(f"Optimization completed: candidate={candidate}, acq_value={acq_value}")

        return candidate.detach().numpy()
