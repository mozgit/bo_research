import torch
import logging
import time
from botorch.acquisition import LogExpectedImprovement
from botorch.utils.sampling import HitAndRunPolytopeSampler
from xgb_models import XGBLSSModel
import numpy as np


class XGBLSSOptimization:
    def __init__(self, domain, name="XGBoostLSS", opt_params=None):
        self.domain = domain
        self.bounds = getattr(domain, "bounds_tensor", None)
        self.category_mask = getattr(domain, "category_mask", [])
        self.discrete_mask = getattr(domain, "discrete_mask", [])
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.opt_params = opt_params

    def train(self, history):
        start_time = time.time()

        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)
        self.train_x = X
        self.train_targets = Y
        self.model = XGBLSSModel(X, Y, self.bounds, category_mask=self.category_mask, discrete_mask=self.discrete_mask, opt_params=self.opt_params)

        self.logger.info("Training XGBLSS model with data points and scores.")
        self.logger.debug(f"Train X: {X}")
        self.logger.debug(f"Train Y: {Y}")

        end_time = time.time()
        self.logger.info(f"Training completed in {end_time - start_time:.4f} seconds.")
        return self

    def _convert_constraints(self, constraints, converter_type="eq"):
        dim = self.bounds.size(-1)
        n_constraints = len(constraints)
        C_or_A = torch.zeros((n_constraints, dim), dtype=torch.float)
        d_or_b = torch.zeros((n_constraints, 1), dtype=torch.float)

        for i, (indices, coefficients, rhs) in enumerate(constraints):
            C_or_A[i, indices] = coefficients.float()
            d_or_b[i] = float(rhs)

        if converter_type == "ineq":
            C_or_A = -C_or_A
            d_or_b = -d_or_b

        return C_or_A, d_or_b

    def step(self):
        # Handle equality and inequality constraints
        equality_constraints = self._handle_constraints("get_equality_constraints")
        inequality_constraints = self._handle_constraints("get_inequality_constraints", constraint_type="ineq")

        acq_function = LogExpectedImprovement(self.model, best_f=torch.max(self.train_targets))
        choices_tensor = torch.tensor(self.domain.generate_choices(), dtype=torch.double)

        opt_kwargs = {
            "acq_function": acq_function,
            "q": 1,
        }

        if self._is_discrete_only():
            return self._optimize_discrete(choices_tensor, opt_kwargs)
        else:
            return self._optimize_mixed(choices_tensor, equality_constraints, inequality_constraints, opt_kwargs)

    def _handle_constraints(self, method_name, constraint_type="eq"):
        if hasattr(self.domain, method_name):
            constraints = getattr(self.domain, method_name)()
            if constraints:
                adjusted_constraints = self.adjust_indices_for_new_dimensions(constraints)
                self.logger.info(f"Applied domain {constraint_type} constraints: {adjusted_constraints}")
                return self._convert_constraints(adjusted_constraints, constraint_type=constraint_type)
        return None

    def _get_partition_bounds(self):
        partitions_start_time = time.time()
        partitions = self.model.get_partitions()
        partition_bounds = self.model.partition_splits_to_bounds(partitions)
        self.logger.info(f"Num generated partition bounds: {len(partition_bounds)}")
        self.logger.info(f"Partitioning took {time.time() - partitions_start_time:.4f} seconds.")
        return partition_bounds

    def _sample_continuous(self, partition_bounds, inequality_constraints, equality_constraints):
        continuous_samples = []

        for bounds in partition_bounds:
            bounds_tensor = torch.tensor(bounds, dtype=torch.float).T
            self.logger.debug(f"Bounds tensor: {bounds_tensor}, inequality constraints: {inequality_constraints}")

            if self._no_constraints(inequality_constraints):
                samples = self._uniform_sampling(bounds_tensor)
            else:
                samples = self._sample_with_constraints(bounds_tensor, inequality_constraints, equality_constraints)

            if samples is not None:
                continuous_samples.append(samples)

        return continuous_samples

    def _no_constraints(self, inequality_constraints):
        A, b = inequality_constraints if inequality_constraints else (torch.tensor([]), torch.tensor([]))
        return A.numel() == 0 and b.numel() == 0

    def _uniform_sampling(self, bounds_tensor):
        try:
            lower_bounds = bounds_tensor[0]
            upper_bounds = bounds_tensor[1]
            sampling_start_time = time.time()
            samples = torch.rand(1, len(lower_bounds)) * (upper_bounds - lower_bounds) + lower_bounds
            self.logger.debug(f"Uniform sampling took {time.time() - sampling_start_time:.4f} seconds for bounds {bounds_tensor}.")
            return samples
        except Exception as e:
            self.logger.debug(f"Sampling failed for bounds {bounds_tensor}: {e}")
            return None

    def _sample_with_constraints(self, bounds_tensor, inequality_constraints, equality_constraints):
        try:
            sampling_start_time = time.time()
            sampler = HitAndRunPolytopeSampler(
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                bounds=bounds_tensor,
            )
            samples = sampler.draw(n=1)
            self.logger.info(f"Sampling took {time.time() - sampling_start_time:.4f} seconds for bounds {bounds_tensor}.")
            return samples
        except ValueError as e:
            self.logger.debug(f"Empty polytope for bounds {bounds_tensor}: {e}")
            return None

    def _evaluate_samples(self, continuous_samples):
        if continuous_samples:
            samples_concat_start_time = time.time()
            continuous_samples_tensor = torch.cat(continuous_samples, dim=0)
            self.logger.info(f"Concatenation of samples took {time.time() - samples_concat_start_time:.4f} seconds.")

            choices = self.domain.generate_choices()
            choices_tensor = torch.tensor(choices, dtype=torch.float)

            # Create all possible combinations of categorical and continuous features
            full_points_list = [
                torch.cat((c.unsqueeze(0).repeat(len(continuous_samples_tensor), 1), continuous_samples_tensor), dim=1)
                for c in choices_tensor
            ]
            full_points = torch.cat(full_points_list, dim=0)

            # Ensure that full_points has the shape (batch_size, q, d)
            full_points = full_points.unsqueeze(1)  # Shape (batch_size, 1, d)

            best_value, best_candidate = None, None

            evaluation_start_time = time.time()
            for full_points_batch in torch.split(full_points, 10000):
                batch_values = LogExpectedImprovement(self.model, best_f=torch.max(self.train_targets))(
                    full_points_batch)
                batch_values_np = batch_values.detach().numpy()  # Convert to numpy array

                # Find the best value in the current batch
                batch_best_value = np.max(batch_values_np)
                if best_value is None or batch_best_value > best_value:
                    best_value = batch_best_value
                    best_candidate = full_points_batch[np.argmax(batch_values_np)]

            self.logger.info(f"Evaluation of LogEI took {time.time() - evaluation_start_time:.4f} seconds.")
            return best_candidate.squeeze(1)  # Remove the extra dimension added earlier

        return None

    def _handle_categorical(self, LogEI):
        choices_start_time = time.time()
        choices = self.domain.generate_choices()
        self.logger.debug(f"Generating choices took {time.time() - choices_start_time:.4f} seconds.")

        tensor_conversion_start_time = time.time()
        choices_tensor = torch.tensor(choices, dtype=torch.float).unsqueeze(1)
        self.logger.debug(f"Converting choices to tensor took {time.time() - tensor_conversion_start_time:.4f} seconds.")

        logei_computation_start_time = time.time()
        values = LogEI(choices_tensor).detach().numpy()
        self.logger.debug(f"Computing LogEI took {time.time() - logei_computation_start_time:.4f} seconds.")

        best_index = values.argmax()
        best_candidate = choices_tensor[best_index]

        self.logger.info(f"Found best candidate with value: {values[best_index]}, candidate: {best_candidate}")
        return best_candidate.detach().numpy()

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

    opt_params = {
            "eta": 0.5,
            "max_depth": 2,
            "min_child_weight": 5,
            "subsample": 0.5,
            "colsample_bytree": 0.9,
            "gamma": 6e-6,
            "n_rounds": 5,
            "booster": "dart"
        }

    # Initialize the Bayesian optimization
    optimizer = XGBLSSOptimization(combined_domain, opt_params = opt_params)

    # Train the model with the historical data
    optimizer.train(history)

    # Perform an optimization step to find the next candidate
    next_point = optimizer.step()

    print("Next candidate point:", next_point)