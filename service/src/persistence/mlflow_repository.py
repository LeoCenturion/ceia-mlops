from typing import Any, Dict, Tuple
import mlflow
from pandas import Series
from sklearn.base import BaseEstimator, ClassifierMixin
from . import ModelRepository
from mlflow.models import infer_signature

DEFAULT_NAME = "rain-predictor"

class MlflowRepository(ModelRepository):
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri

    def model_uri(self, name, version):
        return f"models:/{name}/{version}"

    def load(self, name=DEFAULT_NAME, version="latest"):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            loaded_model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
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
        scores: Dict[str,float] = {},
    ) -> None:
        with mlflow.start_run():
            mlflow.set_tracking_uri(self.tracking_uri)
            for k,v in params.items():
                mlflow.log_param(k, v)
            for k,v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model, name, registered_model_name=name, signature=False)
            for k,v in scores.items():
                mlflow.log_metric(k, v)


