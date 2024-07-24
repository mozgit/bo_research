import os
import json
import logging
import pandas as pd
from deepdiff import DeepDiff
from .history import History


class ResultsStorage:
    def __init__(self, algorithm_name, problem_name, directory='results', overwrite=False):
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.directory = directory
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)

        if not isinstance(self.directory, str):
            raise ValueError("The directory parameter must be a valid string.")

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.logger.warning(f"The directory {self.directory} did not exist and was created.")

        self.filename = self._get_filename()
        self.data = self._load_existing_data()

    def add_history(self, history):
        """
        Adds a history object to the results storage.

        Args:
            history (History): The history object containing optimization records.
        """
        new_records = history.get_records()

        if self.data and not self.overwrite:
            existing_records = self.data
            if not self._is_extension(existing_records, new_records):
                diff = DeepDiff(existing_records, new_records[:len(existing_records)])
                raise ValueError(f"New history is not an extension of existing records. Differences:\n{diff}")

        self._save_data(new_records)

    def _get_filename(self):
        return os.path.join(self.directory, f"{self.algorithm_name}_{self.problem_name}.json")

    def _load_existing_data(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        return None

    def _is_extension(self, existing_records, new_records):
        len_existing = len(existing_records)
        len_new = len(new_records)
        if len_existing > len_new:
            return False

        for existing, new in zip(existing_records, new_records):
            if existing != new:
                return False

        return True

    def _save_data(self, records):
        with open(self.filename, 'w') as f:
            json.dump(records, f, indent=2)
        self.data = records

    def to_dataframe(self):
        if self.data is not None:
            return pd.DataFrame(self.data)
        return pd.DataFrame()

    def save_to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

    def to_history(self):
        """
        Converts the stored data back into a History object.

        Returns:
            History: A History object containing the optimization records.
        """
        if self.data is None:
            raise ValueError("No data available to convert to History.")

        history = History()
        for record in self.data:
            history.add_record(
                step=record['step'],
                point=record['point'],
                function_value=record['function_value'],
                score=record['score'],
                training_time=record.get('training_time'),
                inference_time=record.get('inference_time')
            )

        return history