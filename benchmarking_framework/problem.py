import numpy as np
from scipy.optimize import minimize, Bounds
import logging
from .history import History
from benchmarking_framework.domains import CategoricalDomain, DiscreteDomain, Hypercube, Hyperplane

class Problem:
    """
    Function optimization object for benchmarking.

    Attributes:
        function (callable): The function to be optimized.
        domain (Domain): The domain within which the function is defined.
        objective (callable): The objective function to evaluate the function's output.
        starting_points (list of array-like): Optional list of starting points.
        name (str): Optional name for the problem object.
        max_factor (float): Maximum allowed normalized objective value for starting points.
    """

    def __init__(self, function, domain, objective, objective_meta={}, n_starting_points=5, starting_points=None, name=None, max_objective=None, min_objective=None, max_factor=0.5):
        self.logger = logging.getLogger(__name__)
        self.function = function
        self.domain = domain
        self.objective = objective
        self.objective_meta = objective_meta
        self.n_starting_points = n_starting_points
        self.max_factor = max_factor
        self.starting_points = starting_points
        self.max_objective = max_objective if max_objective is not None else self.maximize_objective()
        self.min_objective = min_objective if min_objective is not None else self.minimize_objective()
        self.name = name or f"Problem {id(self)}"

    def evaluate(self, x):
        """
        Evaluates the function and objective at a given point.

        Args:
            x (array-like): Point at which to evaluate.

        Returns:
            tuple: Function value and objective score at the given point.
        """
        if not self.domain.is_within_domain(x):
            raise ValueError("Input is out of domain bounds")

        y = self.function(x)
        score = self.objective(y)
        return self._convert_to_native_type(y), self._convert_to_native_type(score)

    def get_starting_points(self):
        """
        Returns the starting points for optimization.
        """
        if self.starting_points is None:
            self.starting_points = self.sample_starting_points()
        return self.starting_points

    def sample_starting_points(self):
        """
        Samples starting points within the domain that meet the max_factor criterion.

        Returns:
            np.ndarray: Array of sampled starting points.
        """
        samples = []
        while len(samples) < self.n_starting_points:
            candidate = self.domain.sample(1)[0]
            normalized_value = self._normalize_objective(self.objective(self.function(candidate)))
            self.logger.debug(f"Candidate value: {normalized_value}")
            if normalized_value <= self.max_factor:
                samples.append(candidate)
        return np.array(samples)

    def evaluate_starting_points(self):
        """
        Evaluates the starting points and records their function values and scores.

        Returns:
            History: History object containing records of the evaluations.
        """
        history = History()
        for i, point in enumerate(self.get_starting_points()):
            y, score = self.evaluate(point)
            history.add_record(-self.n_starting_points + i, point, y, score)
        return history

    def _optimize_objective(self, sign=1):
        def objective(x):
            return sign * self.objective(self.function(x))

        best_value = float('inf') if sign == 1 else float('-inf')
        best_result = None
        choices = self.domain.generate_choices()

        self.logger.info("Optimizing over the combined domain")
        for choice in choices:
            if self.domain.bounds:
                result = self._optimize_continuous(objective, choice, sign)
                if result and ((sign == 1 and result[1] < best_value) or (sign == -1 and result[1] > best_value)):
                    best_result, best_value = result
            else:
                current_value = objective(choice)
                if (sign == 1 and current_value < best_value) or (sign == -1 and current_value > best_value):
                    best_result, best_value = choice, current_value

        self.logger.info(f"{'Maximum' if sign == -1 else 'Minimum'} at {best_result}, {best_value}")
        return best_value

    def _optimize_continuous(self, objective, fixed_choice, sign):
        bounds = Bounds([b[0] for b in self.domain.bounds], [b[1] for b in self.domain.bounds])
        constraints = self.get_scipy_constraints()
        initial_point = np.array([np.random.uniform(b[0], b[1]) for b in self.domain.bounds])

        def fixed_objective(continuous_vars):
            combined_point = np.concatenate((fixed_choice, continuous_vars))
            return objective(combined_point)

        result = minimize(fixed_objective, x0=initial_point, bounds=bounds, constraints=constraints, method='SLSQP')
        if result.success:
            current_value = sign * result.fun
            return np.concatenate((fixed_choice, result.x)), current_value
        return None

    def maximize_objective(self):
        return self._optimize_objective(sign=-1)

    def minimize_objective(self):
        return self._optimize_objective(sign=1)

    def get_scipy_constraints(self):
        # Convert BoTorch constraints to scipy constraints
        inequality_constraints = self.domain.get_inequality_constraints()
        equality_constraints = self.domain.get_equality_constraints()

        constraints = []
        for indices, coefficients, rhs in inequality_constraints:
            def constraint_func(x, indices=indices, coefficients=coefficients, rhs=rhs):
                return np.dot(x[indices], coefficients) - rhs
            constraints.append({'type': 'ineq', 'fun': constraint_func})

        for indices, coefficients, rhs in equality_constraints:
            def constraint_func(x, indices=indices, coefficients=coefficients, rhs=rhs):
                return np.dot(x[indices], coefficients) - rhs
            constraints.append({'type': 'eq', 'fun': constraint_func})

        return constraints


    def _normalize_objective(self, value):
        """
        Normalizes the objective value between the known min and max objectives.

        Args:
            value (float): Objective value to normalize.

        Returns:
            float: Normalized objective value.
        """
        return (value - self.min_objective) / (self.max_objective - self.min_objective)

    @staticmethod
    def _convert_to_native_type(value):
        """
        Converts numpy types to native Python types.

        Args:
            value: The value to convert.

        Returns:
            The converted value in native Python types.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        return value
