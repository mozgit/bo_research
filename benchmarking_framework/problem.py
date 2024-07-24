from .history import History
import numpy as np
from scipy.optimize import minimize, Bounds
from benchmarking_framework.domains import CategoricalDomain, DiscreteDomain, Hypercube, Hyperplane
import logging

class Problem:
    """
    Function optimization object for benchmarking.

    Attributes:
        function (callable): The function to be optimized.
        domain (Domain): The domain within which the function is defined.
        objective (callable): The objective function to evaluate the function's output.
        starting_points (list of array-like): Optional list of starting points.
        name (str): Optional name for the problem object.
    """
    def __init__(self, function, domain, objective, objective_meta = {}, n_starting_points = 5, starting_points=None, name=None, max_objective=None, min_objective=None, max_factor = 0.5):
        self.logger = logging.getLogger(__name__)
        self.function = function
        self.max_factor = max_factor
        self.domain = domain
        self.objective = objective
        self.objective_meta = objective_meta
        self.starting_points = starting_points
        self.n_starting_points = n_starting_points
        if max_objective is not None:
            self.max_objective = max_objective
        else:
            self.max_objective = self.maximize_objective()
        if min_objective is not None:
            self.min_objective = min_objective
        else:
            self.min_objective = self.minimize_objective()
        self.name = f"Problem {id(self)}" if name is None else name

    def evaluate(self, x):
        """
        Evaluates the function and objective at a given point.

        Args:
            x (array-like): Point at which to evaluate.

        Returns:
            tuple: Function value and objective score at the given point.
        """
        if self.domain.is_within_domain(x):
            y = self.function(x)
            score = self.objective(y)
            if isinstance(y, (np.float32, np.float64)):
                y = float(y)
            if isinstance(score, (np.float32, np.float64)):
                score = float(score)
            # print("x:",x,"Y: ", y, "Score: ", score)
            return y, score
        else:
            raise ValueError("Input is out of domain bounds")

    def get_starting_points(self):
        # print("Getting starting points")
        if self.starting_points is None:
            self.starting_points = self.sample_starting_points()
        return self.starting_points

    def sample_starting_points(self):
        samples = []
        while len(samples) < self.n_starting_points :
            candidate = self.domain.sample(1)[0]
            candidate_value = self.objective(self.function(candidate))
            normalized_value = (candidate_value - self.min_objective) / (self.max_objective - self.min_objective)
            self.logger.debug(f"Candidate value: {candidate_value}, Min objective {self.min_objective}, Max objective {self.max_objective}, Normalized value: {normalized_value},")
            if normalized_value <= self.max_factor:
                samples.append(candidate)

        # Ensure at least one sample has normalized value exactly equal to the factor


        return np.array(samples)


    def evaluate_starting_points(self):
        """
        Evaluates the starting points and records their function values and scores.

        Returns:
            History: History object containing records of the evaluations.
        """
        history = History()
        starting_points = self.get_starting_points()
        for i, point in enumerate(starting_points):
            y, score = self.evaluate(point)
            history.add_record(-len(starting_points) + i, point, y, score)
        return history

    def _optimize_objective(self, sign=1):
        def objective(x):
            return sign * self.objective(self.function(x))

        def continuous_optimization(fixed_choice):
            nonlocal best_value, best_result
            bounds = Bounds([b[0] for b in self.domain.bounds], [b[1] for b in self.domain.bounds])
            constraints = self.get_scipy_constraints()

            initial_point = np.array([np.random.uniform(b[0], b[1]) for b in self.domain.bounds])

            def fixed_objective(continuous_vars):
                combined_point = np.concatenate((fixed_choice, continuous_vars))
                return objective(combined_point)

            result = minimize(fixed_objective,
                              x0=initial_point,
                              bounds=bounds,
                              constraints=constraints,
                              method='SLSQP')
            current_value = sign * result.fun
            if (sign == 1 and current_value < best_value) or (sign == -1 and current_value > best_value):
                best_value = current_value
                best_result = np.concatenate((fixed_choice, result.x))

        best_value = float('inf') if sign == 1 else float('-inf')
        best_result = None

        # Generate choices from the combined domain
        choices = self.domain.generate_choices()

        self.logger.info("Optimizing over the combined domain")
        for choice in choices:
            if self.domain.bounds is not None:
                continuous_optimization(choice)
            else:
                current_value = self.objective(self.function(choice))
                if (sign == 1 and current_value < best_value) or (sign == -1 and current_value > best_value):
                    best_value = current_value
                    best_result = choice
        self.logger.info("Maximum at" if sign == -1 else "Minimum at", best_result, best_value)
        return best_value

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
