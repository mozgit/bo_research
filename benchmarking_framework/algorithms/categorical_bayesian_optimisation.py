import torch
import logging
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from benchmarking_framework.domains import CategoricalDomain
import numpy as np

class CategoricalBayesianOptimization:
    def __init__(self, domain, name="Categorical Bayesian Optimization"):
        self.domain = domain
        self.name = name
        self.logger = logging.getLogger(__name__)

    def train(self, history):
        self.logger.debug("Starting training process")
        X = torch.tensor([self.domain.encode(point) for point in history.get_points()], dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)

        self.gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        self.logger.debug("Training completed")
        return self

    def step(self, domain):
        self.logger.debug("Starting step process")
        if not isinstance(domain, CategoricalDomain):
            raise ValueError("Domain must be an instance of CategoricalDomain for categorical optimization.")

        LogEI = LogExpectedImprovement(self.gp, best_f=torch.max(self.gp.train_targets))

        self.logger.debug("Generating choices")
        # Generate all possible combinations of the discrete dimensions and encode them
        choices = torch.tensor([domain.encode(point) for point in self.domain.generate_choices()], dtype=torch.double)
        self.logger.debug(f"Choices generated: {choices.shape}")

        try:
            candidate, acq_value = optimize_acqf_discrete(
                acq_function=LogEI,
                q=1,
                choices=choices,
                max_batch_size=2048,  # Adjust based on your memory and computational constraints
                unique=True  # Ensure the returned candidate is unique
            )
            self.logger.debug(f"Optimization completed: candidate={candidate}, acq_value={acq_value}")
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

        self.logger.debug(f"Raw candidate tensor: {candidate}")
        candidate_array = candidate.detach().numpy()
        self.logger.debug(f"Candidate as numpy array: {candidate_array}")

        # Check if the candidate array is empty before decoding
        if candidate_array.size == 0:
            self.logger.error("Candidate array is empty. Cannot decode an empty array.")
            raise ValueError("Candidate array is empty. Cannot decode an empty array.")

        decoded_candidates =np.array([domain.decode(c) for c in candidate_array])
        self.logger.debug(f"Decoded candidates: {decoded_candidates}")

        return decoded_candidates

    def _generate_choices(self, categorical_dimensions):
        """
        Generates all possible combinations of the discrete dimensions.

        Args:
            discrete_dimensions (list of list): A list where each element is a list of possible values for that dimension.

        Returns:
            list: A list of all possible combinations of the discrete dimensions.
        """
        from itertools import product
        choices = list(product(*categorical_dimensions))
        return choices
