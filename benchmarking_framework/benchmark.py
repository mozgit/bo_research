import time
import logging
import os
import multiprocess
import numpy as np
from .history import History
from .results_storage import ResultsStorage


class BenchmarkFramework:
    """
    A framework for benchmarking black-box function optimization algorithms.

    Attributes:
        problems (list): List of function optimization objects.
        algorithms (list): List of optimization algorithms.
        steps (int): Maximum number of optimization steps.
        directory (str): Directory to save results.
        overwrite (bool): Whether to overwrite existing results.
        resume (bool): Whether to resume from the last saved state.
        verbosity (int): Logging verbosity level.
    """

    def __init__(self, problems, algorithms, steps=50, directory=None, overwrite=False, resume=False, verbosity=logging.INFO):
        self.problems = problems
        self.algorithms = algorithms
        self.steps = steps
        self.directory = directory
        self.overwrite = overwrite
        self.resume = resume
        self.verbosity = verbosity
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        Set up the logging configuration.

        Returns:
            Logger: Configured logger.
        """
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logging.basicConfig(level=self.verbosity)
            handler = logging.StreamHandler()
            handler.setLevel(self.verbosity)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.verbosity)
        return logger

    def run(self, run_number=0):
        """
        Run the benchmarking for all combinations of algorithms and function optimization objects.

        Args:
            run_number (int): The run number for this instance.

        Returns:
            dict: A dictionary containing the results of the optimizations.
        """
        results = {}
        for problem in self.problems:
            run_path = os.path.join(self.directory, problem.name, f"run_{run_number}")
            problem.starting_points = self._get_starting_points(problem, run_path)

            for algo in self.algorithms:
                key = f"{algo.__class__.__name__}_{problem.name}"
                self.logger.info(f"Running {key}")
                result_storage = ResultsStorage(algo.__class__.__name__, problem.name, run_path, overwrite=self.overwrite)

                history, step, budget = self._prepare_for_optimization(algo, problem, result_storage)

                results[key] = self._optimize(algo, problem, result_storage, history, step, budget)
        return results

    def _get_starting_points(self, problem, run_path):
        """
        Retrieve or generate starting points for a problem.

        Args:
            problem: The problem object.
            run_path (str): The path where results are stored.

        Returns:
            list: List of starting points.
        """
        starting_points_storage = ResultsStorage('StartingPoints', problem.name, run_path, overwrite=self.overwrite)

        if starting_points_storage.data:
            self.logger.info(f"Loading existing starting points for {problem.name}")
            return starting_points_storage.to_history().get_points()
        else:
            self.logger.info(f"Generating new starting points for {problem.name}")
            starting_points = problem.get_starting_points()
            self._save_starting_points(problem, starting_points_storage, starting_points)
            return starting_points

    def _save_starting_points(self, problem, starting_points_storage, starting_points):
        """
        Save the starting points for a problem.

        Args:
            problem: The problem object.
            starting_points_storage: The storage object for starting points.
            starting_points: List of starting points to save.
        """
        self.logger.info(f"Saving starting points for {problem.name}")
        starting_history = History()
        for i, point in enumerate(starting_points):
            y, score = problem.evaluate(point)
            starting_history.add_record(-len(starting_points) + i, point, y, score)
        starting_points_storage.add_history(starting_history)

    def _prepare_for_optimization(self, algorithm, problem, result_storage):
        """
        Prepare for the optimization by loading or initializing the history.

        Args:
            algorithm: The optimization algorithm.
            problem: The problem object.
            result_storage: The storage object for results.

        Returns:
            tuple: The history object, starting step, and remaining budget.
        """
        if self.resume and result_storage.data:
            self.logger.info(f"Resuming from the last saved state for {algorithm.__class__.__name__}_{problem.name}")
            history = result_storage.to_history()
            step = history.records[-1]['step'] + 1
            budget = self.steps - step
        else:
            history = problem.evaluate_starting_points()
            step = 0
            budget = self.steps
        return history, step, budget

    def _optimize(self, algorithm, problem, results_storage, history, step, budget):
        """
        Perform the optimization for a given algorithm and function optimization object.

        Args:
            algorithm: The optimization algorithm.
            problem: The function optimization object.
            history (History): The history of the optimization process.
            step (int): The current optimization step.
            budget (int): The remaining optimization budget.

        Returns:
            History: History object containing the records of the optimization process.
        """
        while budget > 0:
            self.logger.info(f"{algorithm.__class__.__name__} {problem.name} step {step}")
            train_start = time.time()
            algorithm.train(history)
            train_end = time.time()
            next_points = self._get_next_points(algorithm)

            for next_point in next_points:
                y, score = self._evaluate_point(problem, algorithm, next_point)
                history.add_record(step, next_point, y, score, train_end - train_start, time.time() - train_end)
                results_storage.add_history(history)
                budget -= 1
                if budget <= 0:
                    break
            step += 1

        self.logger.info("Optimization complete. Returning history.")
        return history

    def _get_next_points(self, algorithm):
        """
        Retrieve the next points to evaluate from the algorithm.

        Args:
            algorithm: The optimization algorithm.

        Returns:
            list: List of next points to evaluate.
        """
        next_points = algorithm.step()
        if isinstance(next_points, np.ndarray):
            return next_points if next_points.ndim == 2 else [next_points]
        else:
            raise ValueError("algorithm.step must return a 2D np.array")

    def _evaluate_point(self, problem, algorithm, next_point):
        """
        Evaluate a single point using the problem's objective function.

        Args:
            problem: The problem object.
            algorithm: The optimization algorithm.
            next_point: The point to evaluate.

        Returns:
            tuple: The function value and score of the evaluated point.
        """
        y, score = problem.evaluate(next_point)
        self.logger.info(f"Evaluating next point {next_point} yielded y: {y}, score: {score}")

        return y, score

    def run_parallel(self, num_runs, num_processes=None):
        """
        Run multiple instances of the benchmarking framework in parallel.

        Args:
            num_runs (int): Number of parallel runs.
            num_processes (int): Number of processes to run in parallel.
        """
        multiprocess.set_start_method("spawn", force=True)
        num_processes = num_processes or multiprocess.cpu_count()

        with multiprocess.Pool(processes=num_processes) as pool:
            args = [(self.problems, self.algorithms, self.steps, self.directory, self.overwrite, self.resume, self.verbosity, i) for i in range(num_runs)]
            pool.starmap(self._run_single_instance, args)

    def run_sequential(self, num_runs):
        """
        Run multiple instances of the benchmarking framework sequentially.

        Args:
            num_runs (int): Number of sequential runs.
        """
        for i in range(num_runs):
            self._run_single_instance(self.problems, self.algorithms, self.steps, self.directory, self.overwrite, self.resume, self.verbosity, i)

    @staticmethod
    def _run_single_instance(problems, algorithms, steps, directory, overwrite, resume, verbosity, run_number):
        """
        Run a single instance of the benchmarking framework.

        Args:
            problems: List of problem objects.
            algorithms: List of optimization algorithms.
            steps: Number of steps in the optimization.
            directory: Directory to save results.
            overwrite: Whether to overwrite existing results.
            resume: Whether to resume from the last saved state.
            verbosity: Logging verbosity level.
            run_number: The run number for this instance.

        Returns:
            dict: Results of the optimization run.
        """
        framework = BenchmarkFramework(problems, algorithms, steps, directory, overwrite, resume, verbosity)
        return framework.run(run_number=run_number)
