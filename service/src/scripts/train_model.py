from src.persistence.pickle_repository import PickleRepository
from src.persistence.mlflow_repository import MlflowRepository
import pandas as pd
import numpy as np
from src.model import Model
import os
import requests

rains = pd.read_csv('./data/weatherAUS.csv')
rains = rains.dropna(subset=['RainTomorrow'])
rains_x = rains.drop(columns=['RainTomorrow'])  # Drop the target column from features
rains_y = np.where(rains['RainTomorrow'] == "Yes", 1, 0)
mlflow_url = os.getenv("MLFLOW_URL")
persistence = MlflowRepository(mlflow_url)
model = Model(persistence)
model.train(rains_x, rains_y)
