import unittest
import mlflow
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .mlflow_repository import MlflowRepository
import pandas as pd
import numpy as np
from dotenv import load_dotenv

class TestMlflowRepository(unittest.TestCase):
    def setUp(self):
        load_dotenv(dotenv_path=".env")

        self.tracking_uri = "http://localhost:5001"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.repo = MlflowRepository(tracking_uri=self.tracking_uri)
        self.model_name = "test-model"

        # Create a simple model and data for testing
        self.model = LogisticRegression()
        X = pd.DataFrame(np.random.rand(100, 5))
        y = np.random.randint(0, 2, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

        self.params = {"C": 1.0, "solver": "liblinear"}
        self.metrics = {"accuracy": self.accuracy}
        self.scores = {"test_accuracy": self.accuracy}

    def test_save_and_load_model(self):
        # Save the model
        self.repo.save(
            model=self.model,
            name=self.model_name,
            version="1.0.0",
            metrics=self.metrics,
            params=self.params,
            scores=self.scores,
            # signature=(self.y_pred, self.y_test)
        )

        # Load the model
        loaded_model = self.repo.load(name=self.model_name, version="1.0.0")

        # Assert that the loaded model is not None
        self.assertIsNotNone(loaded_model)

        # Assert that the loaded model is the same as the original model
        print(type(loaded_model))
        self.assertIsInstance(loaded_model, mlflow.pyfunc.PyFuncModel)
        

    # def tearDown(self):
    #     # Clean up the MLflow experiment after the test
    #     client = mlflow.tracking.MlflowClient()
    #     experiments = client.search_experiments()
    #     for exp in experiments:
    #         runs = client.search_runs(exp.experiment_id)
    #         for run in runs:
    #             client.delete_run(run.info.run_id)

if __name__ == '__main__':
    unittest.main()
