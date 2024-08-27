import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResultSummary:
    def __init__(self, results):
        self.results = results

    def summarize(self):
        """
        Summarizes the results by computing convergence and average time for each algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the summarized results.
        """
        summary = {}

        for algo_name, result in self.results.items():
            convergence = self._extract_convergence(result)
            average_time = self._calculate_average_time(result)

            summary[algo_name] = {
                'convergence': convergence,
                'average_time': average_time
            }

        return pd.DataFrame(summary)

    def _extract_convergence(self, result):
        """
        Extracts the convergence data from the result.

        Args:
            result (list): List of results for an algorithm.

        Returns:
            list: List of convergence scores.
        """
        convergence = []

        for res in result:
            if isinstance(res, list):
                convergence.extend(entry['best_score'] for entry in res if 'best_score' in entry)

        return convergence

    def _calculate_average_time(self, result):
        """
        Calculates the average time from the result.

        Args:
            result (list): List of results for an algorithm.

        Returns:
            float: The average time, or None if no time data is available.
        """
        times = [res['time'] for res in result if isinstance(res, dict) and 'time' in res]
        return np.mean(times) if times else None

    def plot_convergence(self):
        """
        Plots the convergence of the optimization algorithms over iterations.
        """
        plt.figure(figsize=(10, 6))

        for algo_name, result in self.results.items():
            for i, res in enumerate(result):
                if isinstance(res, list):
                    steps = [entry['step'] for entry in res]
                    best_scores = [entry['best_score'] for entry in res]
                    plt.plot(steps, best_scores, label=f"{algo_name} (run {i + 1})")

        plt.xlabel('Iteration')
        plt.ylabel('Best Objective Score')
        plt.title('Convergence Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    results = {
        'RandomSearch': [
            [{'step': 0, 'point': [0], 'score': 0, 'best_score': 0}],
            {'time': 1.0}
        ]
    }
    summary = ResultSummary(results).summarize()
    print(summary)
    ResultSummary(results).plot_convergence()
