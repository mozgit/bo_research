import numpy as np

class History:
    """
    Records the history of the optimization process.

    Attributes:
        records (list): List of dictionaries storing the records of the optimization process.
        best_score (float): The best score encountered during the optimization.
    """
    def __init__(self):
        self.records = []
        self.best_score = float('-inf')

    def add_record(self, step, point, function_value, score, training_time=None, inference_time=None):
        """
        Adds a record of the optimization step.

        Args:
            step (int): The step number.
            point (array-like): The point in the domain.
            function_value (array-like): The function value at the point.
            score (float): The score of the function value.
            training_time (float, optional): Time taken for the training step.
            inference_time (float, optional): Time taken for the inference step.
        """
        # Validation
        assert isinstance(step, int), f"step must be an integer, but {type(step)} is found"
        assert isinstance(point, (list, float, np.ndarray)), f"point must be a list, float, or numpy.ndarray, but {type(point)} is found"
        assert isinstance(function_value, (list, float, np.ndarray)), f"function_value must be a list, float, or numpy.ndarray, but {type(function_value)} is found"
        assert isinstance(score, float), f"score must be a float, but {type(score)} is found"

        self.best_score = max(self.best_score, score)
        record = {
            'step': step,
            'point': self._to_native_type(np.array(point).flatten().tolist() if isinstance(point, np.ndarray) else point),
            'function_value': self._to_native_type(np.array(function_value).tolist() if isinstance(function_value, np.ndarray) else function_value),
            'score': float(score),
            'best_score': float(self.best_score),
            'training_time': float(training_time) if training_time is not None else None,
            'inference_time': float(inference_time) if inference_time is not None else None,
        }
        self.records.append(record)

    def _to_native_type(self, value):
        if isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return [self._to_native_type(v) for v in value]
        return value

    def get_records(self):
        """
        Returns all records.

        Returns:
            list: List of all records.
        """
        return self.records

    def __iter__(self):
        return iter(self.records)

    def get_steps(self):
        return [record['step'] for record in self.records]

    def get_points(self):
        return [record['point'] for record in self.records]

    def get_function_values(self):
        return [record['function_value'] for record in self.records]

    def get_scores(self):
        return [record['score'] for record in self.records]

    def get_best_scores(self):
        return [record['best_score'] for record in self.records]

    def get_training_times(self):
        return [record['training_time'] for record in self.records if 'training_time' in record]

    def get_inference_times(self):
        return [record['inference_time'] for record in self.records if 'inference_time' in record]
