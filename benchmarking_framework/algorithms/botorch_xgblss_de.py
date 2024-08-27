from benchmarking_framework.domains.categorical import CategoricalDomain
from benchmarking_framework.domains.discrete import DiscreteDomain
import torch
import logging
import time  # Import time for timing
from botorch.acquisition import LogExpectedImprovement
from benchmarking_framework.algorithms.optimize import optimize_acqf_mixed_de
from botorch.optim import optimize_acqf_discrete
from xgb_models import XGBLSSModel
from typing import List, Tuple

class XGBLSSDEOptimization:
    def __init__(self, domain, opt_params = None,  name=" XGBoostLSS+DE"):
        self.domain = domain
        self.opt_params = opt_params
        self.bounds = None
        if hasattr(domain, "bounds_tensor"):
            self.bounds = domain.bounds_tensor

        self.category_mask = []
        if hasattr(domain, "category_mask"):
            self.category_mask = domain.category_mask
        self.discrete_mask = []
        if hasattr(domain, "discrete_mask"):
            self.discrete_mask = domain.discrete_mask
        self.n_cat_dims = sum(
            domain.get_dimensionality() for domain in domain.domains if isinstance(domain, CategoricalDomain))
        self.n_disc_dims = sum(
            domain.get_dimensionality() for domain in domain.domains if isinstance(domain, DiscreteDomain))
        self.name = name
        self.logger = logging.getLogger(__name__)

    def train(self, history):
        start_time = time.time()

        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)
        self.train_x = X
        self.train_targets = Y
        self.model = XGBLSSModel(X, Y, self.bounds, category_mask=self.category_mask, discrete_mask=self.discrete_mask, opt_params = self.opt_params )

        self.logger.info("Training xgblss model with data points and scores.")
        self.logger.debug(f"Train X: {X}")
        self.logger.debug(f"Train Y: {Y}")

        end_time = time.time()
        self.logger.info(f"Training completed in {end_time - start_time:.4f} seconds.")
        return self

    def _eq_c_converter(self, eq_c):
        dim = self.bounds.size(-1)
        n_eq_con = len(eq_c)
        C = torch.zeros((n_eq_con, dim), dtype=torch.float)
        d = torch.zeros((n_eq_con, 1), dtype=torch.float)

        for i, (indices, coefficients, rhs) in enumerate(eq_c):
            C[i, indices] = coefficients.float()
            d[i] = float(rhs)
        return C, d

    def _ineq_c_converter(self, ineq_c):
        dim = self.bounds.size(-1)
        n_ineq_con = len(ineq_c)
        A = torch.zeros((n_ineq_con, dim), dtype=torch.float)
        b = torch.zeros((n_ineq_con, 1), dtype=torch.float)

        for i, (indices, coefficients, rhs) in enumerate(ineq_c):
            A[i, indices] = -coefficients.float()
            b[i] = -float(rhs)
        return A, b

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

        LogEI = LogExpectedImprovement(self.model, best_f=torch.max(self.train_targets))

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
        if self.domain.get_dimensionality() == self.n_cat_dims + self.n_disc_dims:
            opt_func = optimize_acqf_discrete
            opt_kwargs["choices"] = choices
            opt_kwargs["max_batch_size"] = 2048
            opt_kwargs["unique"] = True
        else:

            # Extract bounds for the continuous domain
            continuous_bounds = torch.tensor(self.domain.bounds, dtype=torch.double).T
            self.logger.debug(f"Continuous bounds: {continuous_bounds}")
            # Combine all bounds
            all_bounds = torch.cat((discrete_categorical_bounds_tensor, continuous_bounds), dim=1)
            self.logger.debug(f"All bounds: {all_bounds}")

            # Optimize the acquisition function
            self.logger.debug("Optimizing the acquisition function.")
            opt_func = optimize_acqf_mixed_de
            opt_kwargs["bounds"] =  all_bounds
            opt_kwargs["num_restarts"] =  5
            opt_kwargs["raw_samples"] =  20
            if fixed_features_list:
                opt_kwargs["fixed_features_list"] = fixed_features_list
            if equality_constraints:
                opt_kwargs["equality_constraints"] = equality_constraints
            if inequality_constraints:
                opt_kwargs["inequality_constraints"] = inequality_constraints

        candidate, acq_value = opt_func(**opt_kwargs)

        self.logger.debug(f"Optimization completed: candidate={candidate}, acq_value={acq_value}")

        candidate_array = candidate.detach().numpy()
        self.logger.debug(f"Candidate as numpy array: {candidate_array}")

        # Check if the candidate array is empty before decoding
        if candidate_array.size == 0:
            self.logger.error("Candidate array is empty. Cannot decode an empty array.")
            raise ValueError("Candidate array is empty. Cannot decode an empty array.")


        return candidate_array

