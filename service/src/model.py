from pipeline import HierarchicalImputer, CoordinateTransformer, WindDirectionTransformer, DropColumnsTransformer, RainTodayTransformer, ExpandDateTransformer
import persistence
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

NAME: str = "num-minmax-xgb"

def read_coords() -> pd.DataFrame:
    coordinates: pd.DataFrame = pd.read_csv("./data/worldcities.csv")
    coordinates["Location"] = coordinates["city"]
    coordinates.drop(columns=["city"], inplace = True)
    coordinates = coordinates[coordinates["country"] == "Australia"]
    return coordinates

def city_coords():
    return {
        'Albury': (-36.0785, 146.9136),
        'BadgerysCreek': (-33.8813, 150.7282),
        'Cobar': (-31.8667, 145.7667),
        'CoffsHarbour': (-30.3026, 153.1137),
        'Moree': (-29.4706, 149.8392),
        'Newcastle': (-32.9283, 151.7817),
        'NorahHead': (-33.2202, 151.5433),
        'NorfolkIsland': (-29.0408, 167.9541),
        'Penrith': (-33.7675, 150.6931),
        'Richmond': (-33.5982, 150.7581),
        'Sydney': (-33.8688, 151.2093),
        'SydneyAirport': (-33.9399, 151.1753),
        'WaggaWagga': (-35.0433, 147.3587),
        'Williamtown': (-32.7951, 151.8118),
        'Wollongong': (-34.4278, 150.8931),
        'Canberra': (-35.2809, 149.1300),
        'Tuggeranong': (-35.4167, 149.1000),
        'MountGinini': (-35.4471, 148.9685),
        'Ballarat': (-37.5622, 143.8503),
        'Bendigo': (-36.7582, 144.2814),
        'Sale': (-38.1100, 147.0737),
        'MelbourneAirport': (-37.6692, 144.8411),
        'Melbourne': (-37.8136, 144.9631),
        'Mildura': (-34.1850, 142.1625),
        'Nhil': (-35.2060, 141.6450),
        'Portland': (-38.3516, 141.5878),
        'Watsonia': (-37.7139, 145.0875),
        'Dartmoor': (-37.7251, 141.2843),
        'Brisbane': (-27.4698, 153.0251),
        'Cairns': (-16.9203, 145.7710),
        'GoldCoast': (-28.0167, 153.4000),
        'Townsville': (-19.2589, 146.8183),
        'Adelaide': (-34.9285, 138.6007),
        'MountGambier': (-37.8321, 140.7807),
        'Nuriootpa': (-34.4973, 138.9966),
        'Woomera': (-31.1395, 136.7984),
        'Albany': (-35.0285, 117.8837),
        'Witchcliffe': (-33.7015, 115.0911),
        'PearceRAAF': (-31.9131, 115.9741),
        'PerthAirport': (-31.9402, 115.9676),
        'Perth': (-31.9505, 115.8605),
        'SalmonGums': (-33.3937, 121.2060),
        'Walpole': (-34.9639, 115.8106),
        'Hobart': (-42.8821, 147.3272),
        'Launceston': (-41.4391, 147.1349),
        'AliceSprings': (-23.6980, 133.8807),
        'Darwin': (-12.4634, 130.8456),
        'Katherine': (-14.4686, 132.2678),
        'Uluru': (-25.3444, 131.0369)
    }


def make_model(coordinates=read_coords(), city_coordinates=city_coords()):
    xgb_best_params = {'subsample': 0.8,
                       'n_estimators': 200,
                       'max_depth': 4,
                       'learning_rate': 0.5,
                       'gamma': 1,
                       'colsample_bytree': 1.0}

    model = XGBClassifier(
        **xgb_best_params,
        objective='binary:logistic',  # Usamos clasificación binaria
        random_state=42,
        n_jobs=12,
    )

    pipeline =  Pipeline(steps = [
        ("date_expander", ExpandDateTransformer()),
        ("imputer", HierarchicalImputer()),
        ("rain_today", RainTodayTransformer()),
        ("coordinates", CoordinateTransformer(coordinates.drop_duplicates(subset="Location"), city_coordinates)),
        ("wind_direction", WindDirectionTransformer()),
        ("drop_directions", DropColumnsTransformer(columns=["WindGustDir", "WindDir9am", "WindDir3pm"])),
        ("drop_date_location", DropColumnsTransformer(columns=["Date","Location"])),
        ("scaler", MinMaxScaler()),
        ("logistic_regression", model)
    ])
    return pipeline

def train(X, Y, model=make_model(), name=NAME):
    fitted = model.fit(X, Y)
    persistence.save(fitted, name)
    return fitted

def load(model=make_model(), name=NAME):
    return persistence.load(model, name)

if __name__=="__main__":
    rains = pd.read_csv('./data/weatherAUS.csv')
    rains = rains.dropna(subset=['RainTomorrow'])
    rains_x = rains.drop(columns=['RainTomorrow'])  # Drop the target column from features
    rains_y = np.where(rains['RainTomorrow'] == "Yes", 1, 0)
    train(rains_x, rains_y)

