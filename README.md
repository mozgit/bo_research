# Bayesian Optimization Research

This repository contains research and experiments related to Bayesian optimization using various models and techniques, including Gaussian Processes (GP) and XGBoost with distributional learning (XGBLSS). The repository is designed to benchmark different optimization algorithms on various types of domains.

## Installation

To get started with this repository, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/mozgit/bo_research
    cd bo_research
    ```

2. **Set up a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install ipykernel
    ```

4. **Set up the Jupyter kernel for the environment:**

    ```bash
    python -m ipykernel install --user --name=bo_research_env
    ```

5. **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

## Running Benchmarks

The repository includes several Jupyter notebooks for running benchmarks with different optimization algorithms. You can find these notebooks in the `notebooks/` folder.

### Steps to Run Benchmarks:

1. **Navigate to the `notebooks/` folder:**

    From the Jupyter Notebook interface, navigate to the `notebooks/` folder to explore the available benchmark notebooks.

2. **Run the notebooks:**

    Open any of the benchmark notebooks and execute the cells to run the optimization experiments.

### Available Notebooks:

- **`Categorical_benchmark.ipynb`**: Benchmarking on categorical domains.
- **`Combined_benchmark.ipynb`**: Benchmarking on combined (categorical + continuous) domains.
- **`Continuous_benchmark.ipynb`**: Benchmarking on continuous domains.

## Project Structure

- **`benchmarking_framework/`**: Core framework for benchmarking optimization algorithms.
  - **`algorithms/`**: Contains the implementation of various optimization algorithms.
  - **`benchmarks/`**: Contains benchmark-specific code.
  - **`domains/`**: Defines different types of domains (e.g., categorical, discrete, continuous).
  - **`utils/`**: Utility functions and helper classes.
  - **`benchmark.py`**: Main script to run benchmarks.
  - **`history.py`**: Tracks the history of optimization runs.
  - **`problem.py`**: Defines the optimization problem.
  - **`results.py`**: Handles result summary and plotting.
  - **`results_storage.py`**: Manages storage and retrieval of benchmark results.
  
- **`xgb_models/`**: Implementation of XGBoost models used in the optimization process.
  - **`dummy_model.py`**: Dummy model for testing purposes.
  - **`tree_model.py`**: Implements tree-based models.
  - **`xgblss_model.py`**: XGBoost model with distributional learning (LSS).
  
- **`notebooks/`**: Contains Jupyter notebooks for running benchmarks.
  
- **`requirements.txt`**: List of Python dependencies required for the project.
- **`README.md`**: This file.

## Contributions

Contributions to this repository are welcome. If you have any improvements, bug fixes, or new features to add, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
