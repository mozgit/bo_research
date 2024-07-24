import time
import logging
from .history import History
from .results_storage import ResultsStorage
import numpy as np
import os
import json
import multiprocess
import torch



class BenchmarkFramework:
    """
    A framework for benchmarking black-box function optimization algorithms.

    Attributes:
        problems (list): List of function optimization objects.
        algorithms (list): List of optimization algorithms.
        steps (int): Maximum number of optimization steps.
        directory (str): Directory to save results.
        overwrite (bool): Whether to overwrite existing results.
        verbosity (int): Logging verbosity level.
    """
    def __init__(self, problems, algorithms, steps=50, directory=None, overwrite=False, verbosity=logging.INFO):
        self.problems = problems
        self.algorithms = algorithms
        self.steps = steps
        self.directory = directory
        self.overwrite = overwrite
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """
        Set up the logging configuration.

        Args:
            verbosity (int): Logging verbosity level.
        """
        logging.basicConfig(level=self.verbosity)
        handler = logging.StreamHandler()
        handler.setLevel(self.verbosity)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(self.verbosity)


    def run(self, run_number = 0):
        """
        Run the benchmarking for all combinations of algorithms and function optimization objects.

        Returns:
            dict: A dictionary containing the results of the optimizations.
        """
        results = {}
        for problem in self.problems:
            self.logger.info(f"Checking for existing starting points for {problem.name}")
            run_path = os.path.join(self.directory, problem.name, f"run_{run_number}")
            starting_points_storage = ResultsStorage('StartingPoints', problem.name, run_path,
                                                     overwrite=self.overwrite)

            if starting_points_storage.data:
                self.logger.info(f"Loading existing starting points for {problem.name}")
                problem.starting_points = starting_points_storage.to_history().get_points()
            else:
                self.logger.info(f"Generating new starting points for {problem.name}")
                problem.starting_points = problem.get_starting_points()
                self.logger.info(f"Saving starting points for {problem.name}")
                starting_history = History()
                for i, point in enumerate(problem.starting_points):
                    y, score = problem.evaluate(point)
                    starting_history.add_record(-len(problem.starting_points) + i, point, y, score)
                starting_points_storage.add_history(starting_history)

            for algo in self.algorithms:
                key = f"{algo.__class__.__name__}_{problem.name}"
                self.logger.info(f"Running {key}")
                result_storage = ResultsStorage(algo.__class__.__name__, problem.name, run_path,
                                                overwrite=self.overwrite)
                results[key] = self._optimize(algo, problem, result_storage)
        return results

    def _optimize(self, algorithm, problem, results_storage):
        """
        Perform the optimization for a given algorithm and function optimization object.

        Args:
            algorithm: The optimization algorithm.
            problem: The function optimization object.

        Returns:
            History: History object containing the records of the optimization process.
        """
        # Get and evaluate starting points
        history = problem.evaluate_starting_points()

        # Initialize the budget
        budget = self.steps

        # Set step to 0
        step = 0

        # Start optimization cycle
        while budget > 0:
            try:
                self.logger.info(f"{algorithm.__class__.__name__} {problem.name} step {step}")
                # Call algorithm's function 'train' and 'step' with current data and domain
                start_time = time.time()
                algorithm.train(history)
                self.logger.info("*** Training complete ***")
                train_time = time.time()
                next_points = algorithm.step(problem.domain)
                self.logger.info("*** Next points defined ***")
                end_time = time.time()


                # If the algorithm returns multiple points, handle them
                if isinstance(next_points, np.ndarray):
                    if next_points.ndim == 1:
                        # If it's a 1D array, make it a list with one element
                        next_points = [next_points]
                    elif next_points.ndim == 2:
                        # If it's a 2D array, convert each row to a separate list element
                        next_points = [next_points[i] for i in range(next_points.shape[0])]
                else:
                    raise ValueError("algorithm.step must return a 2D np.array")

                for next_point in next_points:
                    # Evaluate the suggested points with methods of problem object
                    self.logger.info(f"Evaluating next point {next_point}")
                    y, score = problem.evaluate(next_point)

                    self.logger.info(f"Next point evaluated. x: {next_point}, y: {y}, score: {score}")
                    if hasattr(algorithm, 'property_handler'):
                        _obj = algorithm.property_handler.objective
                        _scaler = algorithm.property_handler._scaler
                        _y = _scaler(np.array(y))
                        _torch_y = torch.tensor(_y)
                        __y = torch.reshape(_torch_y, (1, 1, 1, len(_y)))
                        self.logger.info(f"Unravel objective {_obj(__y)}")

                    history.add_record(step, next_point, y, score, train_time - start_time, end_time - train_time)

                    results_storage.add_history(history)

                    # Reduce the optimization budget by the number of suggested points
                    budget -= 1

                    # If the budget is exhausted, break the loop
                    if budget <= 0:
                        break

                step += 1

            except Exception as e:
                self.logger.error(f"Error during optimization at step {step}: {e}")
                break

        self.logger.info("Returning history")

        return history


    def run_parallel(self, num_runs, num_processes=None):
        """
        Run multiple instances of the benchmarking framework in parallel.

        Args:
            num_runs (int): Number of parallel runs.
            num_processes (int): Number of processes to run in parallel.
        """
        import numpy as np
        multiprocess.set_start_method("spawn", force=True)

        if num_processes is None:
            num_processes = multiprocess.cpu_count()

        with multiprocess.Pool(processes=num_processes) as pool:
            args = [(self.problems, self.algorithms, self.steps, self.directory, self.overwrite, self.verbosity, i) for i in range(num_runs)]
            pool.starmap(self._run_single_instance, args)

    @staticmethod
    def _run_single_instance(problems, algorithms, steps, directory, overwrite, verbosity, run_number):

        framework = BenchmarkFramework(problems, algorithms, steps, directory, overwrite, verbosity)
        return framework.run(run_number = run_number)

