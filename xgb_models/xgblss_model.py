import copy

import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from typing import Optional, List, Any, Union, Callable
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import xgboost as xgb
import pandas as pd
import logging
from scipy.optimize import linprog
from itertools import product


class XGBLSSModel(Model):
    def __init__(self, train_X: Tensor, train_Y: Tensor, bounds: Tensor, category_mask = [], discrete_mask = [], opt_params = None):
        super(XGBLSSModel, self).__init__()
        self.train_X = train_X.numpy()
        self.train_Y = train_Y.numpy()
        self.bounds = bounds

        # Convert train_X to a DataFrame and create column names
        num_features = self.train_X.shape[1]
        column_names = [f'feature_{i}' for i in range(num_features)]
        self.categorical_columns = []
        self.discrete_columns = []
        self.train_df = pd.DataFrame(self.train_X, columns=column_names)
        for i, disc in enumerate(discrete_mask):
            if disc == 1:
                self.discrete_columns.append(column_names[i])
        for i, cat in enumerate(category_mask):
            if cat == 1:
                self.categorical_columns.append(column_names[i])
                self.train_df[column_names[i]] = self.train_df[column_names[i]].astype('category')

        self.dtrain = xgb.DMatrix(self.train_df, label=self.train_Y, enable_categorical=True)
        self.xgblss = XGBoostLSS(
            Gaussian(stabilization="None", response_fn="exp", loss_fn="nll")
        )
        if opt_params is None:
            self.opt_params, self.n_rounds = self._get_optimized_parameters()
        else:
            self.opt_params = copy.copy(opt_params)
            self.n_rounds = self.opt_params.pop("n_rounds")
        self._train_model()

    def _get_optimized_parameters(self):
        param_dict = {
            "eta": ["float", {"low": 1e-5, "high": 1, "log": True}],
            "max_depth": ["int", {"low": 1, "high": 10, "log": False}],
            "gamma": ["float", {"low": 1e-8, "high": 40, "log": True}],
            "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}],
            "rate_drop": ["float", {"low": 0, "high": 1, "log": False}],
            "booster": ["categorical", ["dart"]]
        }

        opt_param = self.xgblss.hyper_opt(param_dict,
                                 self.dtrain,
                                 num_boost_round=30,
                                 nfold=5,
                                 early_stopping_rounds=10,
                                 max_minutes=5,
                                 n_trials=1000,
                                 silence=True
                                 )

        opt_params = opt_param.copy()
        n_rounds = opt_params.pop("opt_rounds")

        return opt_params, n_rounds

    def _train_model(self):
        self.xgblss.train(
            self.opt_params,
            self.dtrain,
            num_boost_round=self.n_rounds
        )

    def posterior(
            self,
            X: Tensor,
            output_indices: Optional[List[int]] = None,
            observation_noise: Union[bool, Tensor] = False,
            posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
            **kwargs: Any,
    ) -> Posterior:
        batch_shape, q, d = X.shape
        X = X.view(-1, d)

        X_df = pd.DataFrame(X.numpy(), columns=self.train_df.columns)
        dtest = xgb.DMatrix(X_df)
        pred_params = self.xgblss.predict(dtest, pred_type="parameters")

        loc = torch.tensor(pred_params["loc"])#.view(batch_shape, q, -1)
        scale = torch.tensor(pred_params["scale"])#.view(batch_shape, q, -1)
        covar_matrix = torch.diag_embed(scale)
        mvn = MultivariateNormal(loc, covar_matrix)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self.train_Y.shape[-1]

    def subset_output(self, idcs: List[int]) -> Model:
        subset_train_Y = self.train_Y[..., idcs]
        return XGBLSSModel(self.train_X, subset_train_Y)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        new_train_X = torch.cat([self.train_X, X], dim=-2)
        new_train_Y = torch.cat([self.train_Y, Y], dim=-2)
        return XGBLSSModel(new_train_X, new_train_Y)

    def get_partitions(self):
        booster = self.xgblss.booster
        partitions = {}
        for feature in self.train_df.columns:
            if (not feature in self.categorical_columns) and (not feature in self.discrete_columns):
                histogram = booster.get_split_value_histogram(feature)
                splits = histogram['SplitValue'].values
                partitions[feature] = splits
        return partitions

    def partition_splits_to_bounds(self, partitions):
        split_ranges = []
        for i, (feature, splits) in enumerate(partitions.items()):
            # Include the min and max bounds for each feature
            min_bound = self.bounds[0,i].item()
            max_bound = self.bounds[1,i].item()
            feature_bounds = [min_bound] + splits.tolist() + [max_bound]
            feature_bounds = sorted(list(set(feature_bounds)))
            # Drop bounds outside the min and max bounds
            feature_bounds = [bound for bound in feature_bounds if min_bound <= bound <= max_bound]
            feature_bounds = [(feature_bounds[j], feature_bounds[j + 1]) for j in range(len(feature_bounds) - 1)]
            split_ranges.append(feature_bounds)

        # Generate all possible combinations of partition bounds using Cartesian product
        all_bounds = list(product(*split_ranges))
        return all_bounds

if __name__=='__main__':

    # Example usage of XGBLSSModel
    import numpy as np
    num_train_points = 20
    low = 0
    high = 10
    x_dim = 3
    train_X = torch.tensor(np.random.uniform(low=low, high=high, size=(num_train_points, x_dim)), dtype=torch.float32)
    train_Y = torch.tensor(np.random.uniform(low=low, high=high, size=(num_train_points, 1)), dtype=torch.float32)

    bounds = torch.tensor([[low-1]*x_dim,[high+1]*x_dim], dtype=torch.double)
    # Instantiate the model
    xgblss_model = XGBLSSModel(train_X, train_Y, bounds)

    # Test the model
    num_test_points = 5
    test_X = torch.tensor(np.random.uniform(low=0, high=10, size=(num_test_points, 3)), dtype=torch.float32).unsqueeze(1)
    print("Shape of test_X: ", test_X.shape)
    posterior = xgblss_model.posterior(test_X)

    print("Mean Predictions: ", posterior.mean)
    print("Variance Predictions: ", posterior.variance)
    print("Covariance Predictions: ", posterior.covariance_matrix)
    print("Partition Splits: ", xgblss_model.get_partitions())
    # Example usage of partition_splits_to_bounds and sample_from_partition functions
    partitions = xgblss_model.get_partitions()
    partition_bounds = xgblss_model.partition_splits_to_bounds(partitions)
    print(f"Partition Bounds: {len(partition_bounds)} partitions found. Example: {partition_bounds[0]}")

    from botorch.acquisition import LogExpectedImprovement

    LogEI = LogExpectedImprovement(xgblss_model, best_f=torch.max(train_Y))
    print("Log Expected Improvement acquisition function initialized.")
    print("Getting LogEI for the training data")
    # Iterate over train_X and get LogEI values there
    for i in range(num_train_points):
        train_point = train_X[i].unsqueeze(0).unsqueeze(0)  # Reshape to [batch_shape, q, d]
        log_ei = LogEI(train_point)
        print(f"LogEI at train point {train_point}: {log_ei.item()}")