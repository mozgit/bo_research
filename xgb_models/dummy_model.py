import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from typing import Optional, List, Any, Union, Callable

class DummyModel(Model):
    def __init__(self, train_X: Tensor, train_Y: Tensor):
        super(DummyModel, self).__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.mean = train_Y.mean()
        self.variance = train_Y.var()

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        mean = torch.full(X.shape[:-1], self.mean)
        variance = torch.full(X.shape[:-1], self.variance)
        covar_matrix = torch.diag_embed(variance)
        mvn = MultivariateNormal(mean, covar_matrix)
        return GPyTorchPosterior(mvn)

    @property
    def batch_shape(self) -> torch.Size:
        # Return the shape of the batch dimensions of the training data
        return self.train_X.shape[:-2]

    @property
    def num_outputs(self) -> int:
        # Return the number of outputs (m) of the model
        return self.train_Y.shape[-1]

    def subset_output(self, idcs: List[int]) -> Model:
        subset_train_Y = self.train_Y[..., idcs]
        return DummyModel(self.train_X, subset_train_Y)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        new_train_X = torch.cat([self.train_X, X], dim=-2)
        new_train_Y = torch.cat([self.train_Y, Y], dim=-2)
        return DummyModel(new_train_X, new_train_Y)


if __name__ == '__main__':
    # Example usage
    train_X = torch.rand(2, 20, 1)  # batch_shape x n x d
    train_Y = torch.sin(train_X * (2 * torch.pi)).expand(2, 20, 1)  # batch_shape x n x m

    # Instantiate the model
    dummy_model = DummyModel(train_X, train_Y)

    # Test the model
    test_X = torch.rand(2, 5, 1)
    posterior = dummy_model.posterior(test_X)
    print("Mean Predictions: ", posterior.mean)
    print("Variance Predictions: ", posterior.variance)
