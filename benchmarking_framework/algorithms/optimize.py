import torch
from botorch.acquisition import AcquisitionFunction
from scipy.optimize import differential_evolution, NonlinearConstraint, LinearConstraint
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def optimize_acqf_mixed_de(
        acq_function: AcquisitionFunction,
        bounds: torch.Tensor,
        q: int,
        num_restarts: int,
        fixed_features_list: Optional[List[Dict[int, float]]]=[],
        raw_samples: Optional[int] = None,
        options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
        inequality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
        post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_initial_conditions: Optional[torch.Tensor] = None,
        ic_generator: Optional[Callable[..., torch.Tensor]] = None,
        ic_gen_kwargs: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize over a list of fixed_features using differential_evolution and returns the best solution.

    This function replicates the signature and core logic of `optimize_acqf_mixed` but uses
    `scipy.optimize.differential_evolution` for optimization.
    """

    def acq_func_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        # Apply fixed features
        for fixed_features in fixed_features_list:
            for idx, value in fixed_features.items():
                x_tensor[0, idx] = value

        # Evaluate the acquisition function
        return -acq_function(x_tensor).item()  # Note: scipy.optimize minimizes, so we negate the value

    # Convert bounds to a format acceptable by scipy
    bounds_list = [(bounds[0, i].item(), bounds[1, i].item()) for i in range(bounds.shape[1])]

    # Prepare constraints
    constraints = []

    if inequality_constraints:
        for (indices, coefficients, rhs) in inequality_constraints:
            def lin_ineq_constraint(x, indices=indices, coefficients=coefficients, rhs=rhs):
                return torch.sum(torch.tensor(x)[indices] * coefficients) - rhs

            constraints.append(LinearConstraint(coefficients.numpy(), rhs, np.inf))

    if equality_constraints:
        for (indices, coefficients, rhs) in equality_constraints:
            def lin_eq_constraint(x, indices=indices, coefficients=coefficients, rhs=rhs):
                return torch.sum(torch.tensor(x)[indices] * coefficients) - rhs

            constraints.append(NonlinearConstraint(lin_eq_constraint, 0, 0))

    if nonlinear_inequality_constraints:
        for constraint, intra_point in nonlinear_inequality_constraints:
            constraints.append(NonlinearConstraint(constraint, 0, np.inf))

    # Optimize using differential_evolution
    result = differential_evolution(acq_func_wrapper, bounds_list, constraints=constraints, **(options or {}))

    # Get the optimal point and value
    optimal_x = torch.tensor(result.x, dtype=torch.float32).unsqueeze(0)
    optimal_value = -result.fun  # Revert the negation to get the actual acquisition value

    # Post-process the result if a post_processing_func is provided
    if post_processing_func is not None:
        optimal_x = post_processing_func(optimal_x)

    return optimal_x, torch.tensor(optimal_value)

if __name__ == '__main__':
    # Example usage (assuming you've set up your GP model and acquisition function):
    from botorch.models import SingleTaskGP
    from botorch.acquisition import ExpectedImprovement
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch import fit_gpytorch_mll as fit_gpytorch_model

    # Example data
    train_X = torch.rand(10, 2)
    train_Y = (train_X ** 2).sum(dim=-1, keepdim=True)

    # Fit a GP model
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # Define the acquisition function (e.g., Expected Improvement)
    acq_func = ExpectedImprovement(gp, best_f=train_Y.max())

    # Define bounds
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    # Define fixed features
    fixed_features_list = [{0: 0.5}]

    # Define constraints (as an example)
    inequality_constraints = [(torch.tensor([0, 1]), torch.tensor([1.0, -2.0]), -0.5)]
    equality_constraints = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0)]

    # Optimize using the custom differential evolution function
    optimal_point, optimal_value = optimize_acqf_mixed_de(
        acq_func, bounds, 1, 10, fixed_features_list,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints
    )

    print(f"Optimal Point: {optimal_point}")
    print(f"Optimal Acquisition Value: {optimal_value}")

