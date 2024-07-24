import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from benchmarking_framework.domains import Hyperplane

class BayesianOptimization:
    def __init__(self, domain, name="Bayesian Optimization"):
        self.domain = domain
        self.bounds = torch.tensor(domain.bounds, dtype=torch.double).T
        self.name = name

        self.total = None
        if isinstance(self.domain, Hyperplane):
            self.total = self.domain.total


    def train(self, history):
        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)

        self.gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self

    def step(self, domain):
        equality_constraints = None
        inequality_constraints = None

        if self.total is not None:
            indices = torch.arange(self.bounds.shape[1], dtype=torch.long)  # All indices in X
            coefficients = torch.ones_like(indices, dtype=torch.double)  # Coefficients are all 1
            equality_constraints = [(indices, coefficients,torch.tensor(self.total, dtype=torch.double))]

        if hasattr(domain, 'get_equality_constraints'):
            equality_constraints = domain.get_equality_constraints()

        if hasattr(domain, 'get_inequality_constraints'):
            inequality_constraints = domain.get_inequality_constraints()

        LogEI = LogExpectedImprovement(self.gp, best_f=torch.max(self.gp.train_targets))

        candidate, _ = optimize_acqf(
            LogEI,
            bounds=self.bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints
        )

        return candidate.detach().numpy()[0]