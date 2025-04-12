from typing import Any, Dict
import mlflow
from sklearn.base import BaseEstimator, ClassifierMixin
from . import ModelRepository

DEFAULT_NAME = "rain-predictor"

class MlflowRepository(ModelRepository):
    def load(self, name=DEFAULT_NAME):
        try:
            # Assuming "prod" tag, you might need to adjust based on your MLflow setup.
            mlflow.set_tracking_uri("http://your-mlflow-server:5000")
            stage = "Production" #or Staging, or None, or Archived.
            model_uri = f"models:/{name}/{stage}"

            loaded_model = mlflow.pyfunc.load_model(model_uri)
            return loaded_model
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return None


    def save(
        self,
        model: ClassifierMixin,
        name=DEFAULT_NAME,
        metrics: Dict[str,float] = {},
        params: Dict[str,Any] = {},
        scores: Dict[str,float] = {}
    ) -> None:
        mlflow.set_tracking_uri("http://your-mlflow-server:5000") 
        with mlflow.start_run():
            for k,v in params:
                mlflow.log_param(k, v)
            for k,v in metrics:
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model, name)
            for k,v in scores:
                mlflow.log_metric(k, v)


