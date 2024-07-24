import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from benchmarking_framework.domains import DiscreteDomain

class DiscreteBayesianOptimization:
    def __init__(self, domain, name="Bayesian Optimization"):
        self.domain = domain
        self.name = name

    def train(self, history):
        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)

        self.gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self

    def step(self, domain):
        if not isinstance(domain, DiscreteDomain):
            raise ValueError("Domain must be an instance of DiscreteDomain for discrete optimization.")

        LogEI = LogExpectedImprovement(self.gp, best_f=torch.max(self.gp.train_targets))

        # Convert the discrete domain to a tensor of possible choices
        choices = self._generate_choices(domain.discrete_dimensions)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=LogEI,
            q=1,
            choices=choices,
            max_batch_size=2048,
            unique=True
        )

        return candidate.detach().numpy()[0]

    def _generate_choices(self, discrete_dimensions):
        """
        Generates all possible combinations of the discrete dimensions.

        Args:
            discrete_dimensions (list of list): A list where each element is a list of possible values for that dimension.

        Returns:
            torch.Tensor: A tensor containing all possible combinations of the discrete dimensions.
        """
        from itertools import product
        choices = list(product(*discrete_dimensions))
        return torch.tensor(choices, dtype=torch.double)


