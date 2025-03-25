from flask import Flask
from flask import request
import numpy as np
import pandas as pd
from model import load
app = Flask(__name__)
model = load()

@app.route("/liveness")
def liveness():
    return "live"

@app.route("/readiness")
def readiness():
    return f'Model trained: {model != None}'

@app.route("/predict",methods=['POST'])
def predict():
    data = [[
        "2008-12-01", "Albury", 13.4, 22.9, 0.6, None, None, "W", 44, "W", "WNW", 
        20, 24, 71, 22, 1007.7, 1007.1, 8, None, 16.9, 21.8, "No"
    ]]

    X = pd.DataFrame(data, columns=['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                                         'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
                                         'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                                         'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'])
    y = model.predict(X)

    return f'{y}\n'
