import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'plotting_config.json')


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_results(directory):
    results = defaultdict(list)
    for run_dir in os.listdir(directory):
        run_path = os.path.join(directory, run_dir)
        if os.path.isdir(run_path):
            for file_name in os.listdir(run_path):
                if file_name.endswith('.json') and "StartingPoints" not in file_name:
                    algo_name = file_name.split('_')[0]
                    file_path = os.path.join(run_path, file_name)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    results[algo_name].append(data)
    return results


def plot_results(results, config_path=CONFIG_PATH, name=None):
    config = load_config(config_path)
    plt.figure(figsize=(7, 4))
    color_map = plt.get_cmap('tab10')

    for i, (algo_name, histories) in enumerate(results.items()):
        if algo_name not in config:
            config[algo_name] = {"color": list(color_map(i % 10)), "name": algo_name}

        color = config[algo_name]["color"]
        all_steps = []
        all_best_scores = []

        for history in histories:
            history = [record for record in history if record['step'] >= -1]
            steps = [record['step'] for record in history]
            best_scores = [record['best_score'] for record in history]
            all_steps.append(steps)
            all_best_scores.append(best_scores)
            plt.plot(steps, best_scores, color=color, alpha=0.1)

        max_steps = max(max(steps) for steps in all_steps)
        common_steps = np.arange(-1, max_steps + 1)
        interpolated_scores = np.zeros((len(histories), len(common_steps)))

        for j, (steps, best_scores) in enumerate(zip(all_steps, all_best_scores)):
            interpolated_scores[j, :] = np.interp(common_steps, steps, best_scores)

        avg_best_scores = np.mean(interpolated_scores, axis=0)
        p20_best_scores = np.percentile(interpolated_scores, 20, axis=0)
        p80_best_scores = np.percentile(interpolated_scores, 80, axis=0)

        plt.plot(common_steps, avg_best_scores, color=color, label=config[algo_name]["name"])
        plt.fill_between(common_steps, p20_best_scores, p80_best_scores, color=color, alpha=0.2)

    plt.xlabel('Iteration')
    plt.ylabel('Best Objective Score')
    if name:
        plt.title('Convergence Plot for ' + name)
    else:
        plt.title('Convergence Plot')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    save_config(config, config_path)


def get_time_stats(histories):
    all_training_times = []
    all_inference_times = []

    for history in histories:
        training_times = [record['training_time'] for record in history if
                          'training_time' in record and record['training_time'] is not None]
        inference_times = [record['inference_time'] for record in history if
                           'inference_time' in record and record['inference_time'] is not None]
        all_training_times.extend(training_times)
        all_inference_times.extend(inference_times)

    avg_training_time = np.nanmean(all_training_times)
    var_training_time = np.nanvar(all_training_times)
    avg_inference_time = np.nanmean(all_inference_times)
    var_inference_time = np.nanvar(all_inference_times)

    return avg_training_time, var_training_time, avg_inference_time, var_inference_time


def plot_time_stats(results, config_path=CONFIG_PATH, name=None):
    config = load_config(config_path)

    avg_training_times = []
    var_training_times = []
    avg_inference_times = []
    var_inference_times = []

    algo_names = list(results.keys())

    for algo_name in algo_names:
        histories = results[algo_name]
        avg_train_time, var_train_time, avg_infer_time, var_infer_time = get_time_stats(histories)

        avg_training_times.append(avg_train_time)
        var_training_times.append(var_train_time)

        avg_inference_times.append(avg_infer_time)
        var_inference_times.append(var_infer_time)

    x = np.arange(len(algo_names))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))

    for i, algo_name in enumerate(algo_names):
        color = config[algo_name]["color"]

        # Plot training times
        axes[0].errorbar(x[i], avg_training_times[i], yerr=np.sqrt(var_training_times[i]), fmt='o', capsize=5,
                         color=color)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([config[algo_name]["name"] for algo_name in algo_names], rotation=45)
        axes[0].set_ylabel('Time (s)')
        axes[0].set_title('Training')
        axes[0].grid(True)

        # Plot inference times
        axes[1].errorbar(x[i], avg_inference_times[i], yerr=np.sqrt(var_inference_times[i]), fmt='x', capsize=5,
                         color=color)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([config[algo_name]["name"] for algo_name in algo_names], rotation=45)
        axes[1].set_ylabel('Time (s)')
        axes[1].set_title('Optimization step')
        axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    save_config(config, config_path)


def plot_comparison(results, config_path=CONFIG_PATH, name=None):
    algo_names = list(results.keys())
    n_algos = len(algo_names)
    if n_algos>2:
        fig, axes = plt.subplots(n_algos, n_algos, figsize=(4*n_algos, 4*n_algos))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4 * 1))
    color_map = plt.get_cmap('tab10')

    with open(config_path, 'r') as f:
        config = json.load(f)

    n_algos_j = 1
    if n_algos>2:
        n_algos_j = n_algos
    for i in range(n_algos):
        for j in range(n_algos_j):
            if n_algos>2:
                if i == j:
                    axes[i, j].axis('off')
                    continue
                _ax = axes[i, j]


                algo_name_1 = algo_names[i]
                algo_name_2 = algo_names[j]

            else:
                _ax = axes[i]
                algo_name_1 = algo_names[i]
                algo_name_2 = algo_names[1 if i==0 else 0]

            color_1 = config[algo_name_1]["color"]
            color_2 = config[algo_name_2]["color"]

            all_diffs = []
            all_steps = []

            for history_1, history_2 in zip(results[algo_name_1], results[algo_name_2]):
                steps_1 = [record['step'] for record in history_1]
                steps_2 = [record['step'] for record in history_2]
                common_steps = np.intersect1d(steps_1, steps_2)

                scores_1 = np.interp(common_steps, steps_1, [record['best_score'] for record in history_1])
                scores_2 = np.interp(common_steps, steps_2, [record['best_score'] for record in history_2])

                diffs = scores_1 - scores_2
                all_diffs.append(diffs)
                all_steps.append(common_steps)

            max_steps = max(max(steps) for steps in all_steps)
            common_steps = np.arange(-1, max_steps + 1)
            interpolated_diffs = np.zeros((len(all_diffs), len(common_steps)))

            for k, diffs in enumerate(all_diffs):
                interpolated_diffs[k, :] = np.interp(common_steps, all_steps[k], diffs)

            avg_diffs = np.mean(interpolated_diffs, axis=0)
            p20_diffs = np.percentile(interpolated_diffs, 20, axis=0)
            p80_diffs = np.percentile(interpolated_diffs, 80, axis=0)

            _ax.plot(common_steps, avg_diffs, color='black')
            _ax.fill_between(common_steps, p20_diffs, p80_diffs, color='grey', alpha=0.2)
            _ax.fill_between(common_steps, avg_diffs, 0, where=(avg_diffs >= 0), color=color_1, alpha=0.5, interpolate=True)
            _ax.fill_between(common_steps, avg_diffs, 0, where=(avg_diffs < 0), color=color_2, alpha=0.5, interpolate=True)

            if name:
                _ax.set_title(f'{name}')

            _ax.set_xlabel("Optimization step")
            _ax.set_ylabel("Difference")

            handles = [
                # plt.Line2D([0], [0], color='black', lw=2),
                plt.Line2D([0], [0], color=color_1, lw=2, label=f'{algo_name_1} is better'),
                plt.Line2D([0], [0], color=color_2, lw=2, label=f'{algo_name_2} is better')
            ]

            _ax.legend(handles=handles, loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

def plot_comparison_stats(results, config_path=CONFIG_PATH, name=None):
    algo_names = list(results.keys())
    n_algos = len(algo_names)

    if n_algos>2:
        fig, axes = plt.subplots(n_algos, n_algos, figsize=(4*n_algos, 4*n_algos))#, sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4 * 1))  # , sharex=True, sharey=True)
    color_map = plt.get_cmap('tab10')

    with open(config_path, 'r') as f:
        config = json.load(f)

    n_algos_j = 1
    if n_algos > 2:
        n_algos_j = n_algos
    for i in range(n_algos):
        for j in range(n_algos_j):
            if n_algos > 2:
                if i == j:
                    axes[i, j].axis('off')
                    continue
                _ax = axes[i, j]

                algo_name_1 = algo_names[i]
                algo_name_2 = algo_names[j]

            else:
                _ax = axes[i]
                algo_name_1 = algo_names[i]
                algo_name_2 = algo_names[1 if i==0 else 0]

            color_1 = config[algo_name_1]["color"]
            color_2 = config[algo_name_2]["color"]

            all_diffs = []
            all_steps = []

            for history_1, history_2 in zip(results[algo_name_1], results[algo_name_2]):
                steps_1 = [record['step'] for record in history_1]
                steps_2 = [record['step'] for record in history_2]
                common_steps = np.intersect1d(steps_1, steps_2)

                scores_1 = np.interp(common_steps, steps_1, [record['best_score'] for record in history_1])
                scores_2 = np.interp(common_steps, steps_2, [record['best_score'] for record in history_2])

                diffs = scores_1 - scores_2
                all_diffs.append(diffs)
                all_steps.append(common_steps)

            max_steps = max(max(steps) for steps in all_steps)
            common_steps = np.arange(-1, max_steps + 1)
            interpolated_diffs = np.zeros((len(all_diffs), len(common_steps)))

            for k, diffs in enumerate(all_diffs):
                interpolated_diffs[k, :] = np.interp(common_steps, all_steps[k], diffs)

            probs = []
            conf_intervals = []

            for step in range(len(common_steps)):
                diff_at_step = interpolated_diffs[:, step]
                n_better = np.sum(diff_at_step > 0)
                n_total = len(diff_at_step)
                a, b = n_better + 1, n_total - n_better + 1  # Beta-Binomial with non-informative prior
                mean = a / (a + b)
                std = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
                probs.append(mean)
                conf_intervals.append((mean - std, mean + std))

            probs = np.array(probs)
            conf_intervals = np.array(conf_intervals)

            _ax.plot(common_steps, probs, color='black', label='Probability')
            _ax.fill_between(common_steps, conf_intervals[:, 0], conf_intervals[:, 1], color='grey', alpha=0.2)
            _ax.fill_between(common_steps, probs, 0.5, where=(probs >= 0.5), color=color_1, alpha=0.5, interpolate=True)
            _ax.fill_between(common_steps, probs, 0.5, where=(probs < 0.5), color=color_2, alpha=0.5, interpolate=True)

            # if i == n_algos - 1:
            _ax.set_xlabel("Optimization step")
#             if j == 0:
            _ax.set_ylabel("Probability")

            # Adding text to each subplot
            _ax.text(0.5, 0.95, f"Is {algo_name_1}\nbetter than {algo_name_2}?",
                            horizontalalignment='center', verticalalignment='top', transform=_ax.transAxes, fontsize=10)

            handles = [
                plt.Line2D([0], [0], color='black', lw=2, label='Probability'),
                plt.Line2D([0], [0], color=color_1, lw=2, label='Likely'),
                plt.Line2D([0], [0], color=color_2, lw=2, label='Unlikely')
            ]
            legend = _ax.legend(handles=handles, loc='lower right', fontsize='small')
            _ax.add_artist(legend)

            # Draw horizontal lines
            for y in [0.5, 0.9, 0.95, 1.0]:
                _ax.axhline(y=y, color='grey', linestyle='--', alpha=0.2)

            if name:
                _ax.set_title(f'{name}')
            _ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    base_directory = "results/Quadratic Hyperplane"
    results = load_results(base_directory)
    plot_results(results)
    plot_time_stats(results)
    plot_comparison(results)

