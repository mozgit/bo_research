import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll as  fit_gpytorch_model
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf_mixed, optimize_acqf, optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from benchmarking_framework.domains import CombinedDomain, CategoricalDomain, DiscreteDomain
import logging
from typing import List, Tuple
import numpy as np

class CombinedBayesianOptimization:
    def __init__(self, domain, name="GPs"):
        self.domain = domain
        self.category_mask = []
        if hasattr(domain, "category_mask"):
            self.category_mask = domain.category_mask
        self.discrete_mask = []
        if hasattr(domain, "discrete_mask"):
            self.discrete_mask = domain.discrete_mask
        self.n_cat_dims = sum([_domain.get_dimensionality() for _domain in self.domain.domains if isinstance(_domain, CategoricalDomain)])
        self.n_disc_dims = sum([_domain.get_dimensionality() for _domain in self.domain.domains if isinstance(_domain, DiscreteDomain)])
        self.n_cont_dims = self.domain.get_dimensionality() - self.n_cat_dims - self.n_disc_dims
        self.name = name
        self.logger = logging.getLogger(__name__)

    def train(self, history):
        X = torch.tensor(history.get_points(), dtype=torch.double)
        # X = torch.tensor([self.domain.encode(point) for point in history.get_points()], dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)

        # index of categorical dimensions
        cat_dims = [i for i, mask in enumerate(self.category_mask) if mask == 1]
        if len(cat_dims)>0:
            self.gp = MixedSingleTaskGP(X, Y, cat_dims = [i for i, mask in enumerate(self.category_mask) if mask == 1])
        else:
            self.gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self

    def adjust_indices_for_new_dimensions(self, constraints: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        num_new_dimensions = self.n_cat_dims + self.n_disc_dims
        adjusted_constraints = []
        for indices, coefficients, rhs in constraints:
            adjusted_indices = indices + num_new_dimensions
            adjusted_constraints.append((adjusted_indices, coefficients, rhs))
        return adjusted_constraints

    def step(self):
        equality_constraints = None
        inequality_constraints = None

        if hasattr(self.domain, 'get_equality_constraints'):
            equality_constraints = self.adjust_indices_for_new_dimensions(self.domain.get_equality_constraints())
            self.logger.info(f"Applied domain equality constraints: {equality_constraints}")

        if hasattr(self.domain, 'get_inequality_constraints'):
            inequality_constraints =self.adjust_indices_for_new_dimensions(self.domain.get_inequality_constraints())
            self.logger.info(f"Applied domain inequality constraints: {inequality_constraints}")

        LogEI = LogExpectedImprovement(self.gp, best_f=self.gp.train_targets.max())

        # Generate all possible combinations of discrete and categorical choices
        choices = torch.tensor(self.domain.generate_choices(),dtype=torch.double)
        self.logger.info(f"Generated {len(choices)} choices from the domain.")

        # Encode the choices to torch tensors
        fixed_features_list = []
        num_fixed_dims = sum(domain.get_dimensionality() for domain in self.domain.domains if isinstance(domain, (CategoricalDomain, DiscreteDomain)))

        for choice in choices:
            fixed_features = {i: choice[i] for i in range(num_fixed_dims)}
            fixed_features_list.append(fixed_features)

        # print("Fixed features list:", fixed_features_list)
        # Calculate bounds for discrete and categorical dimensions
        discrete_categorical_bounds = []
        for domain in self.domain.domains:
            if isinstance(domain, DiscreteDomain):
                for dim in domain.discrete_dimensions:
                    discrete_categorical_bounds.append((min(dim), max(dim)))
            elif isinstance(domain, CategoricalDomain):
                for dim in domain.categorical_dimensions:
                    discrete_categorical_bounds.append((min(dim), max(dim)))
                    # Categorical variables are one-hot encoded
                    # for i in range(len(dim)):
                    #     discrete_categorical_bounds.append((0, 1))
                    # discrete_categorical_bounds.append((min(dim), max(dim)))

        # Convert bounds to tensors
        discrete_categorical_bounds_tensor = torch.tensor(discrete_categorical_bounds, dtype=torch.double).T

        opt_kwargs = {"acq_function": LogEI, "q":1}
        if self.n_cont_dims == 0:
            self.logger.info(f"Discrete optimisation")
            opt_func = optimize_acqf_discrete
            opt_kwargs["choices"] = choices
            opt_kwargs["max_batch_size"] = 2048
            opt_kwargs["unique"] = True
        else:
            self.logger.info(f"Mixed optimisation")
            # Extract bounds for the continuous domain
            continuous_bounds = torch.tensor(domain.bounds, dtype=torch.double).T
            self.logger.debug(f"Continuous bounds: {continuous_bounds}")
            # Combine all bounds
            all_bounds = torch.cat((discrete_categorical_bounds_tensor, continuous_bounds), dim=1)
            self.logger.debug(f"All bounds: {all_bounds}")

            # Optimize the acquisition function
            self.logger.debug("Optimizing the acquisition function.")
            opt_func = optimize_acqf
            opt_kwargs["bounds"] =  all_bounds
            opt_kwargs["num_restarts"] =  5
            opt_kwargs["raw_samples"] =  20
            if fixed_features_list:
                opt_func = optimize_acqf_mixed
                opt_kwargs["fixed_features_list"] = fixed_features_list
            if equality_constraints:
                opt_kwargs["equality_constraints"] = equality_constraints
            if inequality_constraints:
                opt_kwargs["inequality_constraints"] = inequality_constraints

        # self.logger.debug(opt_kwargs)

        candidate, acq_value = opt_func(**opt_kwargs)

        self.logger.debug(f"Optimization completed: candidate={candidate}, acq_value={acq_value}")

        candidate_array = candidate.detach().numpy()
        self.logger.debug(f"Candidate as numpy array: {candidate_array}")

        # Check if the candidate array is empty before decoding
        if candidate_array.size == 0:
            self.logger.error("Candidate array is empty. Cannot decode an empty array.")
            raise ValueError("Candidate array is empty. Cannot decode an empty array.")


        return candidate_array
        # decoded_candidates =np.array([domain.decode(c) for c in candidate_array])
        # self.logger.debug(f"Decoded candidates: {decoded_candidates}")
        #
        # return decoded_candidates

# Example usage
if __name__ == '__main__':
    import numpy as np
    from benchmarking_framework.domains import CombinedDomain
    from benchmarking_framework import History
    from benchmarking_framework.domains.categorical import CategoricalDomain
    from benchmarking_framework.domains.discrete import DiscreteDomain
    from benchmarking_framework.domains.hyperplane import Hyperplane
    from benchmarking_framework.domains.hypercube import Hypercube

    # Define individual domains
    categorical_domain = CategoricalDomain(categorical_dimensions=[[91,92, 93], [101, 102, 103]])
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
        # print(f"Generated point: {point}, Score: {score}")
        history.add_record(step=i, point=point, function_value=score, score=score)

    # Initialize the Bayesian optimization
    optimizer = CombinedBayesianOptimization(combined_domain)

    # Train the model with the historical data
    optimizer.train(history)

    # Perform an optimization step to find the next candidate
    next_point = optimizer.step()

    print("Next candidate point:", next_point)