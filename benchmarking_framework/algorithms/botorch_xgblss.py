import torch
import logging
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim import gen_batch_initial_conditions
from benchmarking_framework.domains import Hyperplane
from xgb_models import XGBLSSModel
from botorch.utils.sampling import HitAndRunPolytopeSampler
from itertools import product
import numpy as np
import scipy

import torch
import logging
import time  # Import time for timing
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import HitAndRunPolytopeSampler
from xgb_models import XGBLSSModel
import numpy as np


class XGBLSSCombinedBayesianOptimization:
    def __init__(self, domain, name="XGBoostLSS"):
        self.domain = domain
        self.bounds = None
        if hasattr(domain, "bounds_tensor"):
            self.bounds = domain.bounds_tensor

        self.category_mask = []
        if hasattr(domain, "category_mask"):
            self.category_mask = domain.category_mask
        self.discrete_mask = []
        if hasattr(domain, "discrete_mask"):
            self.discrete_mask = domain.discrete_mask
        self.name = name
        self.logger = logging.getLogger(__name__)

    def train(self, history):
        start_time = time.time()

        X = torch.tensor(history.get_points(), dtype=torch.double)
        Y = torch.tensor(history.get_scores(), dtype=torch.double).unsqueeze(-1)
        self.train_x = X
        self.train_targets = Y
        self.model = XGBLSSModel(X, Y, self.bounds, category_mask=self.category_mask, discrete_mask=self.discrete_mask)

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

    def step(self):
        step_start_time = time.time()

        equality_constraints = None
        inequality_constraints = None
        best_value = -float('inf')
        best_candidate = None

        LogEI = LogExpectedImprovement(self.model, best_f=torch.max(self.train_targets))

        if self.bounds is not None:
            self.logger.debug(f"Handling continuous variables within bounds {self.bounds}.")
            if hasattr(self.domain, 'get_equality_constraints'):
                eq_start_time = time.time()
                equality_constraints = self._eq_c_converter(self.domain.get_equality_constraints())
                self.logger.info(f"Applied domain equality constraints: {equality_constraints}")
                self.logger.info(f"Equality constraints conversion took {time.time() - eq_start_time:.4f} seconds.")

            if hasattr(self.domain, 'get_inequality_constraints'):
                ineq_start_time = time.time()
                inequality_constraints = self._ineq_c_converter(self.domain.get_inequality_constraints())
                self.logger.info(f"Applied domain inequality constraints: {inequality_constraints}")
                self.logger.info(f"Inequality constraints conversion took {time.time() - ineq_start_time:.4f} seconds.")

            partitions_start_time = time.time()
            partitions = self.model.get_partitions()
            self.logger.debug(f"Partitions {partitions}")
            partition_bounds = self.model.partition_splits_to_bounds(partitions)
            self.logger.info(f"Num generated partition bounds: {len(partition_bounds)}")
            self.logger.debug(f"Bounds {partition_bounds}")
            self.logger.info(f"Partitioning took {time.time() - partitions_start_time:.4f} seconds.")

            continuous_samples = []
            choices = self.domain.generate_choices()
            choices_tensor = torch.tensor(choices, dtype=torch.float)

            for bounds in partition_bounds:
                bounds_tensor = torch.tensor(bounds, dtype=torch.float).T
                self.logger.debug(f"Bounds tensor: {bounds_tensor}, inequality constraints: {inequality_constraints}")

                # Unpack inequality constraints
                A, b = inequality_constraints

                # Check if the inequality constraints are effectively empty
                no_constraints = A.numel() == 0 and b.numel() == 0

                if no_constraints:
                    # No constraints, just sample uniformly from the hypercube defined by bounds
                    lower_bounds = bounds_tensor[0]
                    upper_bounds = bounds_tensor[1]
                    try:
                        sampling_start_time = time.time()
                        samples = torch.rand(len(choices), len(lower_bounds)) * (
                                    upper_bounds - lower_bounds) + lower_bounds
                        continuous_samples.append(samples)
                        self.logger.debug(
                            f"Uniform sampling took {time.time() - sampling_start_time:.4f} seconds for bounds {bounds_tensor}.")
                    except Exception as e:
                        self.logger.debug(f"Sampling failed for bounds {bounds_tensor}: {e}")
                        continue
                else:
                    try:
                        sampling_start_time = time.time()
                        sampler = HitAndRunPolytopeSampler(
                            inequality_constraints=inequality_constraints,
                            equality_constraints=equality_constraints,
                            bounds=bounds_tensor,
                        )
                        samples = sampler.draw(n=max(1, len(choices)))
                        continuous_samples.append(samples)
                        self.logger.info(
                            f"Sampling took {time.time() - sampling_start_time:.4f} seconds for bounds {bounds_tensor}.")
                    except ValueError as e:
                        self.logger.debug(f"Empty polytope for bounds {bounds_tensor}: {e}")
                        continue

            if continuous_samples:
                self.logger.info(
                    f"No. of full points: {len(continuous_samples)}x{choices_tensor.size(0)} == {len(continuous_samples) * choices_tensor.size(0)}")
                samples_concat_start_time = time.time()
                continuous_samples_tensor = torch.cat(continuous_samples, dim=0)
                full_points = torch.cat((choices_tensor.repeat(len(continuous_samples), 1), continuous_samples_tensor),
                                        dim=1)
                self.logger.info(
                    f"Concatenation of samples took {time.time() - samples_concat_start_time:.4f} seconds.")
                self.logger.info(f"Created {full_points.size(0)} full points")

                # Assuming full_points is already defined and contains all your data points
                batch_size = 10000  # Adjust based on your memory capacity
                num_batches = (full_points.size(0) + batch_size - 1) // batch_size  # Calculate the number of batches

                best_value = None
                best_candidate = None

                evaluation_start_time = time.time()

                for i in range(num_batches):
                    # Define the batch range
                    batch_start = i * batch_size
                    batch_end = min((i + 1) * batch_size, full_points.size(0))

                    # Extract the current batch
                    full_points_batch = full_points[batch_start:batch_end]

                    # Evaluate LogEI on the current batch
                    batch_values = LogEI(full_points_batch.unsqueeze(1)).detach().numpy()

                    # Find the best value in the current batch
                    batch_best_value = np.max(batch_values)  # or np.min(batch_values) for minimization problems
                    if best_value is None or batch_best_value > best_value:  # or < for minimization
                        best_value = batch_best_value
                        best_candidate = full_points_batch[np.argmax(batch_values)]  # or np.argmin for minimization

                self.logger.info(f"Evaluation of LogEI took {time.time() - evaluation_start_time:.4f} seconds.")
                self.logger.info(f"The best value found: {best_value}")
                self.logger.info(f"The sample yielding the best value: {best_candidate}")
                # self.logger.info(f"Values: {values}")
                #
                # if values.ndim == 0:  # It's a scalar
                #     self.logger.debug(f"It's a scalar")
                #     best_value = values
                #     best_candidate = full_points
                # else:
                #     self.logger.debug(f"It's a vector")
                #     best_index = values.argmax()
                #     self.logger.debug(f"best_index: {best_index}")
                #     best_value = values[best_index]
                #     best_candidate = full_points[best_index]
                self.logger.debug(
                    f"Evaluation of acquisition function took {time.time() - evaluation_start_time:.4f} seconds.")

            if best_candidate is not None:
                self.logger.info(f"Found best candidate with value: {best_value}, candidate: {best_candidate}")
                step_end_time = time.time()
                self.logger.info(f"Step completed in {step_end_time - step_start_time:.4f} seconds.")
                return best_candidate.detach().numpy()

        else:
            choices_start_time = time.time()

            # Generate choices
            choices_gen_start_time = time.time()
            choices = self.domain.generate_choices()
            self.logger.debug(f"Generating choices took {time.time() - choices_gen_start_time:.4f} seconds.")

            # Convert choices to tensor
            tensor_conversion_start_time = time.time()
            choices_tensor = torch.tensor(choices, dtype=torch.float)
            self.logger.debug(f"Converting choices to tensor took {time.time() - tensor_conversion_start_time:.4f} seconds.")

            # Evaluate the acquisition function for all choices
            acq_evaluation_start_time = time.time()

            # Unsqueeze the tensor for compatibility
            tensor_unsqueeze_start_time = time.time()
            choices_tensor = choices_tensor.unsqueeze(1)
            self.logger.debug(f"Unsqueezing tensor took {time.time() - tensor_unsqueeze_start_time:.4f} seconds.")

            # Compute the Log Expected Improvement
            logei_computation_start_time = time.time()
            values_tf = LogEI(choices_tensor)
            self.logger.debug(f"Computing LogEI took {time.time() - logei_computation_start_time:.4f} seconds.")

            # Detach the tensor and convert to numpy
            detach_conversion_start_time = time.time()
            values = values_tf.detach().numpy()
            self.logger.debug(f"Detaching and converting to numpy took {time.time() - detach_conversion_start_time:.4f}             seconds.")

            self.logger.debug(f"Evaluating acquisition function for choices took {time.time() -             acq_evaluation_start_time:.4f} seconds.")

            # Find the best index and value
            best_index_start_time = time.time()
            best_index = values.argmax()
            best_value = values[best_index]
            best_candidate = choices_tensor[best_index]
            self.logger.debug(f"Finding the best index and candidate took {time.time() - best_index_start_time:.4f} seconds.")

            self.logger.debug(f"Handling categorical variables took {time.time() - choices_start_time:.4f} seconds in total.")

            if best_candidate is not None:
                self.logger.info(f"Found best candidate with value: {best_value}, candidate: {best_candidate}")
                step_end_time = time.time()
                self.logger.info(f"Step completed in {step_end_time - step_start_time:.4f} seconds.")
                return best_candidate.detach().numpy()

        self.logger.error("No valid candidate found satisfying the constraints.")
        step_end_time = time.time()
        self.logger.info(
            f"Step completed in {step_end_time - step_start_time:.4f} seconds with no valid candidate found.")
        return None


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
    optimizer = XGBLSSCombinedBayesianOptimization(combined_domain)

    # Train the model with the historical data
    optimizer.train(history)

    # Perform an optimization step to find the next candidate
    next_point = optimizer.step()

    print("Next candidate point:", next_point)