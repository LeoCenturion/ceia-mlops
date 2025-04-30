from typing import Any, Dict, Tuple
import mlflow
from pandas import Series
from sklearn.base import BaseEstimator, ClassifierMixin
from . import ModelRepository
from mlflow.models import infer_signature

DEFAULT_NAME = "rain-predictor"

class MlflowRepository(ModelRepository):
    def __init__(self, tracking_uri="http://your-mlflow-server:5000"):
        self.tracking_uri = tracking_uri

    def model_uri(self, name, version):
        return f"models:/{name}/{version}"

    def load(self, name=DEFAULT_NAME, version="1.0.0"):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            model_uri = self.model_uri(name,version)

            loaded_model = mlflow.sklearn.load_model(f"models:/{name}/latest")
            return loaded_model
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return None


    def save(
        self,
        model: ClassifierMixin,
        # signature: Tuple[Series, Series],
        name=DEFAULT_NAME,
        version="1.0.0",
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
            model_uri = self.model_uri(name, version)
            mlflow.sklearn.log_model(model, name, registered_model_name=name, signature=False)
            for k,v in scores.items():
                mlflow.log_metric(k, v)


