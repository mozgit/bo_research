import torch
import logging
import time
import numpy as np
from botorch.acquisition import LogExpectedImprovement
from botorch.utils.sampling import HitAndRunPolytopeSampler
from botorch.optim import optimize_acqf
from xgb_models import XGBLSSModel

class XGBLSSOptimization:
    def __init__(self, domain, name="XGBoostLSS", opt_params=None):
        self.domain = domain
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.opt_params = opt_params
        self.bounds = getattr(domain, "bounds_tensor", None)
        self.category_mask = getattr(domain, "category_mask", [])
        self.discrete_mask = getattr(domain, "discrete_mask", [])

    def train(self, history):
        start_time = time.time()

        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)
        self.train_targets = Y
        self.model = XGBLSSModel(X, Y, self.bounds, category_mask=self.category_mask, discrete_mask=self.discrete_mask, opt_params=self.opt_params)

        self.logger.info(f"Training XGBLSS model with {X.shape[0]} points.")
        self.logger.debug(f"Train X: {X}, Train Y: {Y}")
        self.logger.info(f"Training completed in {time.time() - start_time:.4f} seconds.")
        return self

    def _convert_constraints(self, constraints, constraint_type="eq"):
        dim = self.bounds.size(-1)
        n_constraints = len(constraints)
        tensor_A_or_C = torch.zeros((n_constraints, dim), dtype=torch.float)
        tensor_b_or_d = torch.zeros((n_constraints, 1), dtype=torch.float)

        for i, (indices, coefficients, rhs) in enumerate(constraints):
            tensor_A_or_C[i, indices] = coefficients.float()
            tensor_b_or_d[i] = float(rhs)

        if constraint_type == "ineq":
            tensor_A_or_C = -tensor_A_or_C
            tensor_b_or_d = -tensor_b_or_d

        return tensor_A_or_C, tensor_b_or_d

    def _get_constraints(self):
        equality_constraints = self._convert_constraints(self.domain.get_equality_constraints(), "eq") if hasattr(self.domain, 'get_equality_constraints') else None
        inequality_constraints = self._convert_constraints(self.domain.get_inequality_constraints(), "ineq") if hasattr(self.domain, 'get_inequality_constraints') else None
        return equality_constraints, inequality_constraints

    def _sample_continuous_points(self, partition_bounds, equality_constraints, inequality_constraints):
        continuous_samples = []
        for bounds in partition_bounds:
            bounds_tensor = torch.tensor(bounds, dtype=torch.float).T
            try:
                if inequality_constraints and inequality_constraints[0].numel() != 0:
                    sampler = HitAndRunPolytopeSampler(
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                        bounds=bounds_tensor,
                    )
                    samples = sampler.draw(n=1)
                else:
                    lower_bounds, upper_bounds = bounds_tensor[0], bounds_tensor[1]
                    samples = torch.rand(1, len(lower_bounds)) * (upper_bounds - lower_bounds) + lower_bounds

                continuous_samples.append(samples)
                self.logger.debug(f"Sampled points within bounds: {bounds_tensor}")

            except ValueError as e:
                self.logger.debug(f"Sampling failed for bounds {bounds_tensor}: {e}")

        return continuous_samples

    def _evaluate_acquisition(self, full_points, batch_size=10000):
        num_batches = (full_points.size(0) + batch_size - 1) // batch_size
        best_value, best_candidate = None, None

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, full_points.size(0))
            full_points_batch = full_points[batch_start:batch_end]
            batch_values = LogExpectedImprovement(self.model, best_f=torch.max(self.train_targets))(full_points_batch.unsqueeze(1)).detach().numpy()

            batch_best_value = np.max(batch_values)
            if best_value is None or batch_best_value > best_value:
                best_value = batch_best_value
                best_candidate = full_points_batch[np.argmax(batch_values)]

        return best_candidate, best_value

    def step(self):
        step_start_time = time.time()

        choices = self.domain.generate_choices()
        choices_tensor = torch.tensor(choices, dtype=torch.float)

        if self.bounds is not None:
            equality_constraints, inequality_constraints = self._get_constraints()

            partitions_start_time = time.time()
            partitions = self.model.get_partitions()
            partition_bounds = self.model.partition_splits_to_bounds(partitions)
            self.logger.info(f"Partitioning took {time.time() - partitions_start_time:.4f} seconds.")

            continuous_samples = self._sample_continuous_points(partition_bounds, equality_constraints, inequality_constraints)

            continuous_samples_tensor = torch.cat(continuous_samples, dim=0)

            if len(choices) > 0:
                # Create all possible combinations of categorical and continuous features
                full_points_list = [
                    torch.cat((c.unsqueeze(0).repeat(len(continuous_samples_tensor), 1), continuous_samples_tensor),
                              dim=1)
                    for c in choices_tensor
                ]
                full_points = torch.cat(full_points_list, dim=0)
            else:
                full_points = continuous_samples_tensor

            best_candidate, best_value = self._evaluate_acquisition(full_points)
            self.logger.info(f"Evaluation of LogEI completed. Best value: {best_value}, Best candidate: {best_candidate}")

        else:
            best_candidate = self._evaluate_discrete_only(choices_tensor)

        if best_candidate is not None:
            self.logger.info(f"Step completed in {time.time() - step_start_time:.4f} seconds.")
            return best_candidate.detach().numpy()

        self.logger.error("No valid candidate found satisfying the constraints.")
        self.logger.info(f"Step completed in {time.time() - step_start_time:.4f} seconds with no valid candidate found.")
        return None

    def _evaluate_discrete_only(self, choices_tensor):
        self.logger.info("Evaluating acquisition function for discrete choices only.")
        values = LogExpectedImprovement(self.model, best_f=torch.max(self.train_targets))(choices_tensor.unsqueeze(1)).detach().numpy()
        best_index = values.argmax()
        best_candidate = choices_tensor[best_index]
        return best_candidate

# Example usage
if __name__ == '__main__':
    import numpy as np
    from benchmarking_framework.domains import CombinedDomain, CategoricalDomain, DiscreteDomain, Hyperplane
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
    opt_params = {
        "eta": 0.5,
        "max_depth": 10,
        "min_child_weight": 5,
        "subsample": 0.5,
        "colsample_bytree": 0.9,
        "gamma": 6e-6,
        "n_rounds": 10,
        "booster": "dart"
    }
    optimizer = XGBLSSOptimization(combined_domain, opt_params = opt_params)

    # Train the model with the historical data
    optimizer.train(history)

    # Perform an optimization step to find the next candidate
    next_point = optimizer.step()

    print("Next candidate point:", next_point)
