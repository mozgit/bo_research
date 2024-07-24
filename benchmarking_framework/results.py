import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ResultSummary:
    def __init__(self, results):
        self.results = results

    def summarize(self):
        summary = {}
        for algo_name, result in self.results.items():
            convergence = []
            times = []

            for res in result:
                if isinstance(res, list):
                    for entry in res:
                        if 'best_score' in entry:
                            convergence.append(entry['best_score'])
                if isinstance(res, dict) and 'time' in res:
                    times.append(res['time'])

            summary[algo_name] = {
                'convergence': convergence,
                'average_time': np.mean(times) if times else None
            }

        return pd.DataFrame(summary)

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))

        for algo_name, result in self.results.items():
            for res in result:
                if isinstance(res, list):
                    steps = [entry['step'] for entry in res]
                    best_scores = [entry['best_score'] for entry in res]
                    plt.plot(steps, best_scores, label=f"{algo_name} (run {result.index(res)})")

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
    summary.plot_convergence()
