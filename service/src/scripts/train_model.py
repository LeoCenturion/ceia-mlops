from src.persistence.pickle_repository import PickleRepository
from src.persistence.mlflow_repository import MlflowRepository
import pandas as pd
import numpy as np
from src.model import Model

rains = pd.read_csv('./data/weatherAUS.csv')
rains = rains.dropna(subset=['RainTomorrow'])
rains_x = rains.drop(columns=['RainTomorrow'])  # Drop the target column from features
rains_y = np.where(rains['RainTomorrow'] == "Yes", 1, 0)
persistence_local = MlflowRepository("http://localhost:5001")
model = Model(persistence_local)
model.train(rains_x, rains_y)
