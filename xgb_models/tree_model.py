import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from typing import Optional, List, Any, Union, Callable, Tuple
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import logging
from itertools import product
from sklearn.tree import export_text

class DecisionTreeProbabilisticModel(Model):
    def __init__(self, train_X: Tensor, train_Y: Tensor, bounds: Tensor, category_dict={}, discrete_dict={}):
        super(DecisionTreeProbabilisticModel, self).__init__()
        self.bounds = bounds
        self.category_dict = category_dict
        self.discrete_dict = discrete_dict

        # Convert train_X to a DataFrame and create column names
        num_features = train_X.shape[1]
        column_names = [f'feature_{i}' for i in range(num_features)]
        self.train_X = pd.DataFrame(train_X.numpy(), columns=column_names)
        self.train_Y = train_Y.numpy()
        self.categorical_columns = []
        self.encoded_categorical_columns = []
        self.discrete_columns = []
        self.category_encoders = {}

        for i, disc_values in discrete_dict.items():
            if disc_values != 0:
                self.discrete_columns.append(column_names[i])
        for i, cat_values in category_dict.items():
            if cat_values!=0:
                self.categorical_columns.append(column_names[i])
                for v in cat_values:
                    self.encoded_categorical_columns.append(f'{column_names[i]}_{v}')
                self.category_encoders[column_names[i]] = cat_values
                self.train_X[column_names[i]] = self.train_X[column_names[i]].astype('category')

        self.continuous_columns = [col for col in column_names if col not in self.categorical_columns and col not in self.discrete_columns]

        # Perform custom one-hot encoding for categorical variables
        self.train_X = self._custom_one_hot_encode(self.train_X)

        # Update bounds to reflect the new dimensions after one-hot encoding
        discrete_bounds = []
        for i, disc_values in self.discrete_dict.items():
            if  disc_values != 0:
                discrete_bounds.append((min(disc_values), max(disc_values)))

        new_bounds = []
        if self.bounds is not None:
            new_bounds = self.bounds.T.tolist()
        for col in self.train_X.columns:
            if col in self.encoded_categorical_columns:
                new_bounds.append((0, 1))

        new_bounds = discrete_bounds + new_bounds

        self.bounds = torch.tensor(new_bounds, dtype=torch.double).T

        # Train the model
        self.model = DecisionTreeRegressor()
        self._train_model(self.train_X, self.train_Y)

        # Calculate the variance for each leaf node after training
        self.leaf_variances = self._calculate_leaf_variances(self.train_X, self.train_Y)

    def _custom_one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col, categories in self.category_encoders.items():
            for cat in categories:
                df_encoded[f"{col}_{cat}"] = (df_encoded[col] == cat).astype(int)
            df_encoded.drop(columns=[col], inplace=True)
        return df_encoded

    def _custom_one_hot_encode_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col, categories in self.category_encoders.items():
            for cat in categories:
                df_encoded[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            if col not in self.train_X.columns:
                df_encoded.drop(columns=[col], inplace=True)

        # Reorder the columns to match the order in self.train_X
        df_encoded = df_encoded.reindex(columns=self.train_X.columns, fill_value=0)

        return df_encoded

    def _train_model(self, X, Y):
        self.model.fit(X, Y)

    def _calculate_leaf_variances(self, X, Y):
        # Predict the leaf index for each training point
        leaf_indices = self.model.apply(X)

        # Calculate the variance of the target values for each leaf node
        leaf_variances = {}
        for leaf in np.unique(leaf_indices):
            leaf_targets = Y[leaf_indices == leaf]
            leaf_variances[leaf] = np.var(leaf_targets) if len(leaf_targets) > 1 else 0.0

        return leaf_variances

    def _encode_input(self, X: Tensor) -> pd.DataFrame:
        # Convert to DataFrame
        X_df = pd.DataFrame(X.numpy(), columns=[f'feature_{i}' for i in range(X.shape[1])])
        # Perform custom one-hot encoding for inference
        X_df_encoded = self._custom_one_hot_encode_inference(X_df)
        return X_df_encoded

    def get_leaf_bounds(self, X: Tensor) -> Tuple[List[Tuple[float, float]], List[List[int]], List[List[int]]]:
        """
        Returns the bounds of the leaf containing the given training point, adjusted for continuous, categorical,
        and discrete variables.

        Args:
            X (Tensor): A single training point as a Tensor.

        Returns:
            Tuple[List[Tuple[float, float]], List[List[int]], List[List[int]]]:
            - Continuous bounds as a list of tuples.
            - Categorical possible values as a list of lists.
            - Discrete possible values as a list of lists.
        """
        # Start with a copy of the original bounds
        leaf_bounds = [(self.bounds[0, i].item(), self.bounds[1, i].item()) for i in range(len(self.train_X.columns))]

        # Encode the input X
        X_df = self._encode_input(X.view(1, -1))

        # Get the leaf index for the input point
        leaf_index = self.model.apply(X_df)[0]
        tree = self.model.tree_

        # Traverse the tree to update bounds based on the leaf
        node = 0  # Start at the root
        while node != leaf_index:
            feature = tree.feature[node]
            threshold = tree.threshold[node]
            if X_df.iloc[0, feature] <= threshold:
                leaf_bounds[feature] = (leaf_bounds[feature][0], min(leaf_bounds[feature][1], threshold))
                node = tree.children_left[node]
            else:
                leaf_bounds[feature] = (max(leaf_bounds[feature][0], threshold), leaf_bounds[feature][1])
                node = tree.children_right[node]

        # Decouple bounds for categorical variables and determine possible values
        categorical_values = []
        for col in self.categorical_columns:
            one_hot_columns = [_col for _col in self.train_X.columns if _col.startswith(col)]
            one_hot_bounds = [leaf_bounds[self.train_X.columns.get_loc(col)] for col in one_hot_columns]
            possible_values = []
            for val in self.category_dict[int(col.split('_')[1])]:
                # Check if this value fits in the one-hot bounds
                fits = True
                for i, bound in enumerate(one_hot_bounds):
                    if val == i and not (bound[0] <= 1 <= bound[1]):
                        fits = False
                        break
                    elif val != i and not (bound[0] <= 0 <= bound[1]):
                        fits = False
                        break
                if fits:
                    possible_values.append(val)
            categorical_values.append(possible_values)

        # Decouple bounds for discrete variables and determine possible values
        discrete_values = []
        for col in self.discrete_columns:
            feature_index = self.train_X.columns.get_loc(col)
            bound = leaf_bounds[feature_index]
            possible_values = [val for val in self.discrete_dict[int(col.split("_")[1])] if bound[0] <= val <= bound[1]]
            discrete_values.append(possible_values)

        # Extract bounds for continuous columns
        continuous_bounds = [leaf_bounds[self.train_X.columns.get_loc(col)] for col in self.continuous_columns]

        return continuous_bounds, categorical_values, discrete_values

    def posterior(
            self,
            X: Tensor,
            output_indices: Optional[List[int]] = None,
            observation_noise: Union[bool, Tensor] = False,
            posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
            **kwargs: Any,
    ) -> Posterior:
        X = X.view(-1, X.shape[-1])

        # One-hot encode the input before prediction
        X_df = self._encode_input(X)

        # Make predictions
        pred_mean = self.model.predict(X_df)
        pred_mean = torch.tensor(pred_mean)

        # Determine which leaf each prediction falls into
        leaf_indices = self.model.apply(X_df)

        # Assign variance based on the leaf node variance
        pred_variance = torch.tensor([self.leaf_variances[leaf] for leaf in leaf_indices])

        # Ensure positive variance by adding a small jitter
        jitter = 1e-6
        pred_variance = torch.clamp(pred_variance, min=jitter)

        # Create a multivariate normal distribution using the mean and variance
        covar_matrix = torch.diag_embed(pred_variance)
        mvn = MultivariateNormal(pred_mean, covar_matrix)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self.train_Y.shape[-1]

    def subset_output(self, idcs: List[int]) -> Model:
        subset_train_Y = self.train_Y[..., idcs]
        return DecisionTreeProbabilisticModel(self.train_X, subset_train_Y, self.bounds, category_dict=self.category_dict)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        new_train_X = torch.cat([self.train_X, X], dim=-2)
        new_train_Y = torch.cat([self.train_Y, Y], dim=-2)
        return DecisionTreeProbabilisticModel(new_train_X, new_train_Y, self.bounds, category_dict=self.category_dict)


    def get_partitions(self):
        # Extract partitions (splits) from the decision tree model
        tree = self.model.tree_
        partitions = {}

        for feature_index in range(tree.n_features):
            feature = self.train_X.columns[feature_index]
            # Only extract partitions for continuous features
            if feature in self.continuous_columns:
                threshold = tree.threshold[tree.feature == feature_index]
                partitions[feature] = threshold
        return partitions

    def partition_splits_to_bounds(self, partitions):
        split_ranges = []
        for i, (feature, splits) in enumerate(partitions.items()):
            # Get the index of the current feature in continuous columns
            feature_index = self.continuous_columns.index(feature)

            # Include the min and max bounds for each feature
            min_bound = self.bounds[0, feature_index].item()
            max_bound = self.bounds[1, feature_index].item()
            feature_bounds = [min_bound] + list(splits) + [max_bound]
            feature_bounds = [(feature_bounds[j], feature_bounds[j + 1]) for j in range(len(feature_bounds) - 1)]
            split_ranges.append(feature_bounds)

        # Generate all possible combinations of partition bounds using Cartesian product
        all_bounds = list(product(*split_ranges))
        return all_bounds

if __name__ == '__main__':
    # Example usage of DecisionTreeProbabilisticModel
    num_train_points = 20
    low = 0
    high = 10
    x_dim = 3
    # Generate the first feature as categorical (0, 1, or 2)
    first_feature = np.random.choice([0., 1., 2.], size=(num_train_points, 1))

    # Generate the remaining features as continuous values
    remaining_features = np.random.uniform(low=low, high=high, size=(num_train_points, x_dim))

    # Combine the categorical first feature with the continuous features
    train_X_np = np.hstack((first_feature, remaining_features))

    # Convert to Torch tensors
    train_X = torch.tensor(train_X_np, dtype=torch.float32)
    train_Y = torch.tensor(np.random.uniform(low=low, high=high, size=(num_train_points, 1)), dtype=torch.float32)

    print("Shape of train_X: ", train_X.shape)
    print("Shape of train_Y: ", train_Y.shape)
    print("Train X", train_X)

    bounds = torch.tensor([[low - 1] * x_dim, [high + 1] * x_dim], dtype=torch.double)

    # Assume the first feature is categorical with 3 possible values
    category_dict = {0: [0, 1, 2]}

    # Instantiate the model
    dt_model = DecisionTreeProbabilisticModel(train_X, train_Y, bounds, category_dict=category_dict)

    # Test point
    test_point = torch.tensor([1.0, 2.5, 7.5, 4.0], dtype=torch.float32)  # Assuming the first feature is categorical

    # Use the get_leaf_bounds method to find the bounds for the leaf containing this test point
    continuous_bounds, categorical_values, discrete_values = dt_model.get_leaf_bounds(test_point)

    # Print the results
    print("Continuous Bounds:")
    for i, bounds in enumerate(continuous_bounds):
        print(f"Feature {i + 1}: {bounds}")

    print("\nCategorical Values:")
    for i, values in enumerate(categorical_values):
        print(f"Categorical Feature {i + 1}: {values}")

    print("\nDiscrete Values:")
    for i, values in enumerate(discrete_values):
        print(f"Discrete Feature {i + 1}: {values}")

    # Test the model
    num_test_points = 5
    first_feature = np.random.choice([0., 1., 2.], size=(num_test_points, 1))
    remaining_features = np.random.uniform(low=0, high=10, size=(num_test_points, x_dim))
    test_X_np = np.hstack((first_feature, remaining_features))
    test_X = torch.tensor(test_X_np, dtype=torch.float32)

    print("Shape of test_X: ", test_X.shape)
    posterior = dt_model.posterior(test_X)

    print("Mean Predictions: ", posterior.mean)
    print("Variance Predictions: ", posterior.variance)
    print("Covariance Predictions: ", posterior.covariance_matrix)
    print("Partition Splits: ", dt_model.get_partitions())

    # Example usage of partition_splits_to_bounds and sample_from_partition functions
    partitions = dt_model.get_partitions()
    partition_bounds = dt_model.partition_splits_to_bounds(partitions)
    print(f"Partition Bounds: {len(partition_bounds)} partitions found. Example: {partition_bounds[0]}")
